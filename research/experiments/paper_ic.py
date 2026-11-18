"""原文指标、品种自身标准化、板块秩 IC 与固定调仓缓冲对照。"""
from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research import paper_factors as F
from research import portfolio as P
from research.settings import XS_POSITION as XS

MIN_DAYS = 120
MIN_SECTOR_NAMES = 5
HAC_LAGS = 20
BUFFER_FACTORS = ("CROWD1", "POWER_20", "POWER_20_FILLED", "K_NET_60", "FLOW_CONTROL")


def _one_symbol(symbol, panel, contracts, dates, include_active=False):
    sides = D.MEMBER_SIDES if include_active else ("long", "short")
    ranks = D.load_member_rank_contracts(symbol, "research", sides)
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    position_ranks = ranks.loc[ranks.side.ne("volume")] if include_active and not ranks.empty else ranks
    result = F.symbol_factors(symbol, panel, contracts, dates, position_ranks)
    if include_active:
        result = result.join(F.kaiyuan_active(ranks, dates)[["SPIDER_ACTIVE", "SPIDER_SAME_COVERAGE"]])
    return result


def _signed(name):
    reverse = (name.startswith(("CROWD", "C_SHORT")) and not name.endswith("_TAIL")) or name == "K_DS"
    return -1 if reverse else 1


def _discrete(name):
    return (name.startswith(("SPIDER", "LONG_SYNC", "SHORT_SYNC", "LONG_RESONANCE", "SHORT_RESONANCE",
                             "BOTH_RESONANCE", "POWER_TAIL"))
            or name.endswith("_TAIL") or (name.startswith("K_NET_") and "_S" in name and name != "K_NET_SHARE"))


def _score_variants(name, raw):
    score = raw.replace([np.inf, -np.inf], np.nan) * _signed(name)
    if _discrete(name):
        return {"raw": score}
    prior = score.shift(1).rolling(XS["zscore_window"], min_periods=XS["zscore_min_periods"])
    delta = score - prior.mean()
    return {"raw": score, "delta": delta, "z": delta / prior.std().where(lambda value: value.gt(0))}


def _factor_family(name):
    if "FILLED" in name or name.endswith(("_PROXY", "_CONTROL", "_4D")):
        return "sensitivity"
    return "paper_signal" if _discrete(name) else "paper_indicator"


def _rank_corr(score, outcome, min_names):
    paired = np.isfinite(score) & np.isfinite(outcome)
    left = score.where(paired).rank(axis=1)
    right = outcome.where(paired).rank(axis=1)
    left = left.sub(left.mean(axis=1), axis=0)
    right = right.sub(right.mean(axis=1), axis=0)
    magnitude = np.sqrt(left.pow(2).sum(axis=1) * right.pow(2).sum(axis=1))
    corr = (left * right).sum(axis=1) / magnitude.where(magnitude.gt(0))
    count = paired.sum(axis=1)
    return corr.where(count.ge(min_names)), count.where(corr.notna() & count.ge(min_names))


def _hac_t(series):
    finite = series.notna()
    n = int(finite.sum())
    if n < MIN_DAYS:
        return np.nan
    mean = series.mean()
    # 缺失日的中心化贡献为零，保留真实交易日间隔。
    resid = (series - mean).fillna(0).to_numpy()
    var = np.dot(resid, resid)
    for lag in range(1, HAC_LAGS + 1):
        var += 2 * (1 - lag / (HAC_LAGS + 1)) * np.dot(resid[lag:], resid[:-lag])
    return mean * n / np.sqrt(var) if var > 0 else np.nan


def _bh(p):
    ordered = p.dropna().sort_values()
    adjusted = (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return adjusted.reindex(p.index)


def _stats(ic, count):
    t = _hac_t(ic)
    return {
        "mean_ic": ic.mean(), "hac_t": t, "p": erfc(abs(t) / sqrt(2)),
        "days": ic.count(), "avg_names": count.mean(),
        "positive_fraction": ic.dropna().gt(0).mean(),
        "ic_2016_2018": ic.loc[:"2018-12-31"].mean(),
        "ic_2019_2021": ic.loc["2019-01-01":].mean(),
    }


def _return_matrix(contract_map, bar_open):
    """逐合约匹配交易日开盘与下一交易日开盘，缺失日不跨越。"""
    dates = contract_map.index
    ids = contract_map.to_numpy().ravel()
    entry_date = dates.to_numpy().repeat(len(contract_map.columns))
    exit_date = dates.to_series().shift(-1).to_numpy().repeat(len(contract_map.columns))
    entry = bar_open.reindex(pd.MultiIndex.from_arrays([ids, entry_date])).to_numpy()
    exit_price = bar_open.reindex(pd.MultiIndex.from_arrays([ids, exit_date])).to_numpy()
    valid = (entry > 0) & (exit_price > 0)
    ret = np.divide(exit_price, entry, out=np.full(len(entry), np.nan), where=valid) - 1
    return pd.DataFrame(ret.reshape(contract_map.shape), index=dates, columns=contract_map.columns)


def buffered_tails(score, min_names, buffered, entry_gate=None):
    """两端各约 30% 入场。缓冲版穿过中位数退出。缺失值退出。"""
    ranks = score.rank(axis=1)
    counts = score.count(axis=1)
    tail_count = (counts * 0.30).apply(np.floor).clip(lower=1)
    enter_long = ranks.gt(counts - tail_count, axis=0)
    enter_short = ranks.le(tail_count, axis=0)
    if entry_gate is not None:
        enter_long &= entry_gate.ge(1)
        enter_short &= entry_gate.le(-1)
    enter_long = enter_long.to_numpy()
    enter_short = enter_short.to_numpy()
    centered = ranks.sub((counts + 1) / 2, axis=0).to_numpy()
    result = np.zeros(ranks.shape)
    status = np.zeros(ranks.shape[1])
    for i, count in enumerate(counts):
        hold_long = (status > 0) & (centered[i] > 0) & buffered
        hold_short = (status < 0) & (centered[i] < 0) & buffered
        long = enter_long[i] | hold_long
        short = enter_short[i] | hold_short
        if count >= min_names and long.any() and short.any():
            status = long.astype(float) - short.astype(float)
            result[i] = long / long.sum() - short / short.sum()
        else:
            status.fill(0)
    return pd.DataFrame(result, index=score.index, columns=score.columns)


def _buffer_rows(scores, returns, fee, rolls, sectors, start):
    for (name, norm), score in scores.items():
        score = score.loc[start:].reindex(score.index)
        for scope in ("all", "sector_neutral"):
            for buffered in (False, True):
                if scope == "all":
                    weights = buffered_tails(score, XS["min_names"], buffered)
                else:
                    books = [
                        buffered_tails(score.loc[:, cols], MIN_SECTOR_NAMES, buffered)
                        for cols in sectors.values()
                    ]
                    active = sum(book.abs().sum(axis=1).gt(0).astype(float) for book in books)
                    weights = pd.concat(books, axis=1).reindex(columns=score.columns)
                    weights = weights.div(active.where(active.gt(0)), axis=0).fillna(0)
                weights.loc[weights.index < start] = 0
                for delay in (0, 1):
                    held = weights.shift(1 + delay).fillna(0)
                    prev = held.shift().fillna(0)
                    new_trade = (held - prev).abs().where(~rolls, held.abs())
                    old_trade = prev.abs().where(rolls, 0)
                    old_fee = fee.shift(1)
                    turnover = (new_trade + old_trade).sum(axis=1)
                    valid = (~(held.ne(0) & returns.isna()).any(axis=1)
                             & ~(new_trade.gt(0) & fee.isna()).any(axis=1)
                             & ~(old_trade.gt(0) & old_fee.isna()).any(axis=1))
                    gross = (held * returns).sum(axis=1)
                    cost = ((new_trade * fee).sum(axis=1) + (old_trade * old_fee).sum(axis=1)) / 10_000
                    gross = gross.where(valid).loc[start:]
                    net = (gross - cost.loc[start:]).dropna()
                    stats = P.metrics(net, turnover)
                    yield {
                        "factor": name, "norm": norm, "scope": scope, "buffered": buffered, "delay": delay,
                        "ann_gross": gross.mean() * P.TD,
                        "ann_cost": cost.reindex(net.index).mean() * P.TD,
                        "net_ann": stats["ann_return"], "net_sharpe": stats["sharpe"],
                        "turnover": stats["daily_turnover"], "hac_t": _hac_t(gross - cost.loc[start:]),
                        "days": len(net), "excluded_days": int((~valid.loc[start:]).sum()),
                        "active_days": int(held.loc[net.index].abs().sum(axis=1).gt(0).sum()),
                    }


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int):
    C.assert_no_holdout_dates(panel["date"], "原文指标日面板")
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees["trading_date"], "原文指标手续费")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    eligible = P.eligibility(panel)
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    start = pd.Timestamp(XS["test_start"])
    calendar = dates[dates >= start]
    root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(root)
    symbols = sorted(s for s in panel.loc[eligible, "symbol"].unique() if (root / f"{s}_long.pkl").is_file())
    print(f"构建研报指标：{len(symbols)} 个品种", flush=True)
    contracts = D.load_contracts("research")[["date", "contract", "open", "open_interest", "symbol"]]
    by_symbol = dict(tuple(contracts.groupby("symbol", sort=False)))
    panel_by_symbol = dict(tuple(panel.groupby("symbol", sort=False)))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_one_symbol)(s, panel_by_symbol[s], by_symbol[s], dates) for s in symbols
    )
    factors = pd.concat(pieces, axis=1, keys=symbols).swaplevel(axis=1)
    contract_map = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(index=dates, columns=symbols)
    returns = _return_matrix(contract_map, contracts.set_index(["contract", "date"])["open"])
    rolls = contract_map.ne(contract_map.shift(1)) & contract_map.notna() & contract_map.shift(1).notna()
    fee = P.cost_rate_matrix(panel, fees, XS["slippage_ticks"]).reindex_like(returns)
    eligible = panel.assign(eligible=eligible).pivot(index="date", columns="symbol", values="eligible")
    eligible = eligible.reindex_like(returns).astype("boolean").fillna(False).astype(bool)
    sector_series = pd.Series({s: C.SECTOR[s] for s in symbols})
    sectors = {str(name): list(group.index) for name, group in sector_series.groupby(sector_series)}
    forward = {delay: returns.shift(-(1 + delay)).loc[calendar] for delay in (0, 1)}
    daily, rows, ts_rows, buffer_scores, coverage = {}, [], [], {}, []

    for name in pieces[0].columns:
        variants = _score_variants(name, factors[name])
        score = variants["raw"]
        if name in BUFFER_FACTORS:
            buffer_scores[(name, "z")] = variants["z"].where(eligible)
        if name == "K_DS":
            # 按 IC 筛选出的候选追加同一组执行规则，方向与缓冲参数沿用前述规格。
            buffer_scores.update({(name, norm): value.where(eligible) for norm, value in variants.items()})
        family = _factor_family(name)
        valid = (score.notna() & eligible).loc[calendar]
        coverage.append({
            "factor": name, "family": family, "sign": _signed(name),
            "available_pairs": int(valid.to_numpy().sum()), "eligible_pairs": int(eligible.loc[calendar].to_numpy().sum()),
            "days_with_data": int(valid.any(axis=1).sum()), "symbols_with_data": int(valid.any(axis=0).sum()),
            "avg_names": float(valid.sum(axis=1).mean()),
        })
        for norm, variant in variants.items():
            masked = variant.where(eligible).loc[calendar]
            for delay, outcome in forward.items():
                sector_ic = []
                for scope, cols in {"all": symbols, **sectors}.items():
                    least = XS["min_names"] if scope == "all" else MIN_SECTOR_NAMES
                    ic, count = _rank_corr(masked[cols], outcome[cols], least)
                    rows.append(dict(factor=name, family=family, norm=norm, scope=scope, delay=delay,
                                     **_stats(ic, count)))
                    daily[(name, norm, scope, delay)] = ic
                    if scope != "all":
                        sector_ic.append(ic.rename(scope))
                matrix = pd.concat(sector_ic, axis=1)
                n_sector = matrix.count(axis=1)
                ic = matrix.mean(axis=1).where(n_sector >= 2)
                rows.append(dict(factor=name, family=family, norm=norm, scope="sector_equal", delay=delay,
                                 **_stats(ic, n_sector.where(ic.notna()))))
                daily[(name, norm, "sector_equal", delay)] = ic
                if norm == "raw":
                    matched = masked.notna() & outcome.notna()
                    ic_ts, counts = _rank_corr(masked.T, outcome.T, MIN_DAYS)
                    ts_rows.extend(dict(factor=name, symbol=s, sector=C.SECTOR[s], delay=delay,
                                        days=int(matched[s].sum()), ic=ic_ts[s]) for s in counts.index)

    summary = pd.DataFrame(rows)
    primary = (summary["family"].eq("paper_indicator") & summary["norm"].eq("raw")
               & summary["scope"].eq("all") & summary["delay"].eq(0))
    summary["q_primary"] = _bh(summary["p"].where(primary))
    summary["q_all"] = _bh(summary["p"])
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "paper_ic.csv", index=False)
    pd.DataFrame(daily).to_csv(out / "paper_ic_daily.csv", index_label="date")
    pd.DataFrame(ts_rows).to_csv(out / "paper_ic_timeseries.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "paper_coverage.csv", index=False)
    pd.DataFrame(_buffer_rows(buffer_scores, returns, fee, rolls, sectors, start)).to_csv(
        out / "paper_buffer.csv", index=False)
    print(summary.loc[primary, ["factor", "mean_ic", "hac_t", "q_primary", "q_all", "days"]].round(4).to_string(index=False))

"""IC 检验合并模块：逐品种时序 IC（member-ic）、全市场与板块截面 IC（paper-ic）及 5 日预测与固定调仓（hold-five）。"""
from __future__ import annotations

from functools import reduce
from math import erfc, sqrt
from operator import mul
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research import execution as E, factors as F, portfolio as P
from research.settings import XS_POSITION as XS

LABEL_FIELDS = ("entry_date", "exit_date", "entry_contract", "entry_open", "exit_open", "forward_return")
MIN_DAYS = 120
MIN_SECTOR_NAMES = 5
HAC_LAGS = 20
BUFFER_FACTORS = ("CROWD1", "POWER_20", "POWER_20_FILLED", "K_NET_60", "FLOW_CONTROL")

HOLD_DAYS = 5
HORIZONS = {"d1": (1, 0), "d5": (5, 0), "d2_to_d5": (4, 1), "d5_delayed": (5, 1)}
BOOKS = {
    "K_DS_GATE": ("K_DS", "delta", "gated"),
    "CROWD1_Z": ("CROWD1", "z", "buffer"),
    "POWER_20_STRICT_Z": ("POWER_20", "z", "rank"),
    "POWER_20_FILLED_Z": ("POWER_20_FILLED", "z", "buffer"),
    "K_NET_60_Z": ("K_NET_60", "z", "buffer"),
    "CROWD2_TAIL_CHEM": ("CROWD2_TAIL", "raw", "signal:chemicals"),
    "SPIDER_ACTIVE": ("SPIDER_ACTIVE", "raw", "signal"),
    "FLOW_RAW": ("FLOW_CONTROL", "raw", "rank"),
    "RESONANCE_60_METALS": ("BOTH_RESONANCE_60_FILLED", "raw", "signal:base_metals"),
}


# ---------------------------------------------------------------- 公共工具

def forward_open_returns(dominant: pd.Series, opens: pd.Series) -> pd.DataFrame:
    """全市场交易日索引上，入场与退出都使用 t+1 的主力合约。"""
    dates = dominant.index.to_series()
    entry_day = dates.shift(-1)
    exit_day = dates.shift(-2)
    contract = dominant.shift(-1)
    entry = opens.reindex(pd.MultiIndex.from_arrays([contract, entry_day])).to_numpy(dtype=float)
    exit_price = opens.reindex(pd.MultiIndex.from_arrays([contract, exit_day])).to_numpy(dtype=float)
    valid = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_price) & (exit_price > 0)
    returns = np.divide(exit_price, entry, out=np.full(len(dates), np.nan), where=valid) - 1
    return pd.DataFrame({
        "entry_date": entry_day, "exit_date": exit_day, "entry_contract": contract,
        "entry_open": entry, "exit_open": exit_price, "forward_return": returns,
    }, index=dominant.index)


def forward_returns(returns: pd.DataFrame, horizon: int, delay: int) -> pd.DataFrame:
    """信号日之后的逐日主力开盘收益连乘，任何单日缺失则整段缺失。"""
    if horizon < 1 or delay < 0:
        raise ValueError("累计收益窗口或延迟无效")
    legs = (1 + returns.shift(-(1 + delay + offset)) for offset in range(horizon))
    return reduce(mul, legs) - 1


def _signed(name: str) -> int:
    reverse = (name.startswith(("CROWD", "C_SHORT")) and not name.endswith("_TAIL")) or name == "K_DS"
    return -1 if reverse else 1


def _discrete(name: str) -> bool:
    return (name.startswith(("SPIDER", "LONG_SYNC", "SHORT_SYNC", "LONG_RESONANCE", "SHORT_RESONANCE",
                             "BOTH_RESONANCE", "POWER_TAIL"))
            or name.endswith("_TAIL") or (name.startswith("K_NET_") and "_S" in name and name != "K_NET_SHARE"))


def _score_variants(name: str, raw: pd.Series) -> dict[str, pd.Series]:
    score = raw.replace([np.inf, -np.inf], np.nan) * _signed(name)
    if _discrete(name):
        return {"raw": score}
    prior = score.shift(1).rolling(XS["zscore_window"], min_periods=XS["zscore_min_periods"])
    delta = score - prior.mean()
    return {"raw": score, "delta": delta, "z": delta / prior.std().where(lambda value: value.gt(0))}


def _factor_family(name: str) -> str:
    if "FILLED" in name or name.endswith(("_PROXY", "_CONTROL", "_4D")):
        return "sensitivity"
    return "paper_signal" if _discrete(name) else "paper_indicator"


def _rank_corr(score: pd.DataFrame, outcome: pd.DataFrame, min_names: int) -> tuple[pd.Series, pd.Series]:
    paired = np.isfinite(score) & np.isfinite(outcome)
    left = score.where(paired).rank(axis=1)
    right = outcome.where(paired).rank(axis=1)
    left = left.sub(left.mean(axis=1), axis=0)
    right = right.sub(right.mean(axis=1), axis=0)
    magnitude = np.sqrt(left.pow(2).sum(axis=1) * right.pow(2).sum(axis=1))
    corr = (left * right).sum(axis=1) / magnitude.where(magnitude.gt(0))
    count = paired.sum(axis=1)
    return corr.where(count.ge(min_names)), count.where(corr.notna() & count.ge(min_names))


def _hac_t(series: pd.Series) -> float:
    finite = series.notna()
    n = int(finite.sum())
    if n < MIN_DAYS:
        return np.nan
    mean = series.mean()
    resid = (series - mean).fillna(0).to_numpy()
    var = np.dot(resid, resid)
    for lag in range(1, HAC_LAGS + 1):
        var += 2 * (1 - lag / (HAC_LAGS + 1)) * np.dot(resid[lag:], resid[:-lag])
    return mean * n / np.sqrt(var) if var > 0 else np.nan


def _bh(p: pd.Series) -> pd.Series:
    ordered = p.dropna().sort_values()
    adjusted = (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return adjusted.reindex(p.index)


def _stats(ic: pd.Series, count: pd.Series) -> dict:
    t = _hac_t(ic)
    return {
        "mean_ic": ic.mean(), "hac_t": t, "p": erfc(abs(t) / sqrt(2)),
        "days": ic.count(), "avg_names": count.mean(),
        "positive_fraction": ic.dropna().gt(0).mean(),
        "ic_2016_2018": ic.loc[:"2018-12-31"].mean(),
        "ic_2019_2021": ic.loc["2019-01-01":].mean(),
    }


def _return_matrix(contract_map: pd.DataFrame, bar_open: pd.Series) -> pd.DataFrame:
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


def buffered_tails(score: pd.DataFrame, min_names: int, buffered: bool,
                   entry_gate: pd.DataFrame | None = None) -> pd.DataFrame:
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


def _one_symbol_paper(symbol: str, panel: pd.DataFrame, contracts: pd.DataFrame,
                      dates: pd.DatetimeIndex, include_active: bool = False) -> pd.DataFrame:
    sides = D.MEMBER_SIDES if include_active else ("long", "short")
    ranks = D.load_member_rank_contracts(symbol, "research", sides)
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    position_ranks = ranks.loc[ranks.side.ne("volume")] if include_active and not ranks.empty else ranks
    result = F.symbol_factors(symbol, panel, contracts, dates, position_ranks)
    if include_active:
        result = result.join(F.kaiyuan_active(ranks, dates)[["SPIDER_ACTIVE", "SPIDER_SAME_COVERAGE"]])
    return result


# ---------------------------------------------------------------- 1) member-ic

def summarize_symbol(symbol: str, daily: pd.DataFrame, factors, min_days: int = 3) -> pd.DataFrame:
    """逐因子配对，缺失不填零，离散的 0 信号保留；不足 min_days 时 IC 置空。"""
    rows = []
    for name in factors:
        values = daily[name]
        outcome = daily["forward_return"]
        paired = np.isfinite(values) & np.isfinite(outcome)
        score, returns = values.loc[paired], outcome.loc[paired]
        n = len(score)
        coefficient = (score.rank().corr(returns.rank())
                       if n >= min_days and score.nunique() > 1 and returns.nunique() > 1 else np.nan)
        direction = _signed(name)
        rows.append({
            "symbol": symbol, "factor": name, "ic": coefficient, "direction": direction,
            "signed_ic": direction * coefficient, "n_pairs": n, "n_nonzero": int(score.ne(0).sum()),
            "n_factor_days": int(np.isfinite(values).sum()), "n_return_days": int(np.isfinite(outcome).sum()),
        })
    return pd.DataFrame(rows)


def _one_symbol_member_ic(symbol: str, panel: pd.DataFrame, bars: pd.DataFrame,
                          calendar: pd.DatetimeIndex, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """与全市场截面 IC 使用完全一致的流动性掩码、研报因子集、开盘收益标签与最少有效天数。"""
    features = _one_symbol_paper(symbol, panel, bars, calendar, include_active=True)
    eligible = panel.set_index("date")["eligible"].reindex(calendar).astype("boolean").fillna(False).astype(bool)
    features = features.where(eligible)
    dominant = panel.set_index("date")["c1_contract"].reindex(calendar)
    labels = forward_open_returns(dominant, bars.set_index(["contract", "date"])["open"])
    daily = labels.join(features).loc[calendar[calendar >= start]].rename_axis("signal_date")
    daily.insert(0, "symbol", symbol)
    return daily.reset_index(), summarize_symbol(symbol, daily, features.columns, min_days=MIN_DAYS)


def run_member_ic(panel: pd.DataFrame, out: Path, jobs: int) -> None:
    C.assert_no_holdout_dates(panel.date, "逐品种 IC 日面板")
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"]).copy()
    panel["eligible"] = P.eligibility(panel)
    calendar = pd.DatetimeIndex(panel.date.unique()).sort_values()
    start = pd.Timestamp(XS["test_start"])
    root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(root)
    symbols = sorted(
        s for s in panel.loc[panel["eligible"] & panel["date"].ge(start), "symbol"].unique()
        if (root / f"{s}_long.pkl").is_file()
    )
    if not symbols:
        raise RuntimeError("评价区间没有满足流动性条件与会员排名的商品期货")
    prices = D.load_contracts("research")[["date", "contract", "symbol", "open", "open_interest"]]
    panels = {name: part for name, part in panel.groupby("symbol")}
    bars = {name: part for name, part in prices.groupby("symbol")}
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_one_symbol_member_ic)(symbol, panels[symbol], bars[symbol], calendar, start)
        for symbol in symbols
    )
    daily = pd.concat([rows for rows, _ in pieces], ignore_index=True)
    summary = pd.concat([rows for _, rows in pieces], ignore_index=True)
    factors = [name for name in daily.columns if name not in {"signal_date", "symbol", *LABEL_FIELDS}]
    matrix = summary.pivot(index="symbol", columns="factor", values="ic").reindex(columns=factors)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily_factors_and_returns.csv", index=False)
    summary.to_csv(out / "ic_by_symbol.csv", index=False)
    matrix.to_csv(out / "ic_matrix.csv")
    print(f"{len(symbols)} 个流动性品种，{len(factors)} 个指标与信号，{len(daily)} 个品种日")
    print(f"可估 IC（≥{MIN_DAYS} 日）{summary.ic.notna().sum()} / {len(summary)}，结果写入 {out}")


# ---------------------------------------------------------------- 2) paper-ic

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


def run_paper_ic(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int) -> None:
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
        delayed(_one_symbol_paper)(s, panel_by_symbol[s], by_symbol[s], dates, include_active=True) for s in symbols
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


# ---------------------------------------------------------------- 3) hold-five

def study_inputs(panel: pd.DataFrame, jobs: int):
    """研究期品种池、信号分数与主力合约行情。"""
    C.assert_no_holdout_dates(panel.date, "会员持仓研究日面板")
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    eligible = P.eligibility(panel)
    root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(root)
    names = sorted(s for s in panel.loc[eligible, "symbol"].unique() if (root / f"{s}_long.pkl").is_file())
    start = pd.Timestamp(XS["test_start"])
    dates = pd.DatetimeIndex(panel.date.unique()).sort_values()
    sample = dates[dates >= start]
    bars = D.load_contracts("research")
    panels = dict(tuple(panel.groupby("symbol")))
    by_symbol = dict(tuple(bars.loc[bars.symbol.isin(names), ["symbol", "contract", "date", "open", "open_interest"]]
                           .groupby("symbol")))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_one_symbol_paper)(name, panels[name], by_symbol[name], dates, include_active=True) for name in names
    )
    factors = pd.concat(pieces, axis=1, keys=names).swaplevel(axis=1)
    eligible = panel.assign(eligible=eligible).pivot(index="date", columns="symbol", values="eligible")
    eligible = eligible.reindex(index=dates, columns=names).astype("boolean").fillna(False)
    contracts = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(index=dates, columns=names)
    scores = {
        factor: {norm: score.where(eligible).loc[sample]
                 for norm, score in _score_variants(factor, factors[factor]).items()}
        for factor in pieces[0].columns
    }
    return scores, eligible.loc[sample], contracts, bars


def _ic_summary_5d(scores, outcomes, sectors):
    rows = []
    for factor, variants in scores.items():
        family = _factor_family(factor)
        for norm, score in variants.items():
            for label, outcome in outcomes.items():
                per_sector = []
                for sector, names in {"all": score.columns, **sectors}.items():
                    minimum = XS["min_names"] if sector == "all" else MIN_SECTOR_NAMES
                    ic, count = _rank_corr(score[names], outcome[names], minimum)
                    rows.append(dict(factor=factor, family=family, norm=norm, scope=sector,
                                     window=label, **_stats(ic, count)))
                    if sector != "all":
                        per_sector.append(ic)
                board = pd.concat(per_sector, axis=1)
                count = board.count(axis=1)
                ic = board.mean(axis=1).where(count >= 2)
                rows.append(dict(factor=factor, family=family, norm=norm, scope="sector_equal",
                                 window=label, **_stats(ic, count.where(ic.notna()))))
    result = pd.DataFrame(rows)
    original = result.family.eq("paper_indicator") & result.norm.eq("raw") & result.scope.eq("all")
    result["q_primary"] = result["p"].where(original).groupby(result.window).transform(_bh)
    result["q_all"] = result.groupby("window")["p"].transform(_bh)
    return result


def _weights_5d(score, gate, rule, sectors):
    if rule.startswith("signal"):
        _, _, sector = rule.partition(":")
        if sector:
            score = score.loc[:, sectors[sector]]
        minimum = MIN_SECTOR_NAMES if sector else XS["min_names"]
        count = score.count(axis=1)
        return score.div(count.where(count.ge(minimum)), axis=0).fillna(0)
    return buffered_tails(score, XS["min_names"], buffered=rule != "rank", entry_gate=gate)


def _account_rows_5d(scores, contracts, market, start, sectors):
    for name, (factor, norm, rule) in BOOKS.items():
        score = scores[factor][norm]
        for interval in (1, HOLD_DAYS):
            for phase in range(interval):
                observations = score.iloc[phase::interval]
                gate = scores[factor]["z"].reindex(observations.index) if rule == "gated" else None
                weights = _weights_5d(observations, gate, rule, sectors)
                neutral = not rule.startswith("signal")
                for delay in (0, 1):
                    daily, fills = E.simulate(weights, contracts, market, start, delay, neutral=neutral,
                                              rebalance_every=interval, phase=phase)
                    stats = P.metrics(daily["return"], daily["turnover"])
                    peak = daily.equity.cummax().clip(lower=E.CAPITAL_CNY)
                    schedule = daily.index[phase + 1 + delay::interval]
                    outside = fills.loc[~fills.date.isin(schedule) & fills.date.ne(daily.index[-1])] if len(fills) else fills
                    yield {
                        "book": name, "interval": interval, "phase": phase, "delay": delay,
                        "end_equity": daily.equity.iloc[-1], "net_sharpe": stats["sharpe"],
                        "net_ann": stats["ann_return"], "turnover": stats["daily_turnover"],
                        "max_drawdown": (daily.equity / peak - 1).min(),
                        "gross_pnl": daily.gross_pnl.sum(), "fees": daily.fees.sum(), "slippage": daily.slippage.sum(),
                        "avg_gross_fraction": daily.gross_notional.div(daily.equity).mean(),
                        "avg_abs_net_fraction": daily.net_notional.abs().div(daily.equity).mean(),
                        "active_days": daily.open_contracts.gt(0).sum(), "fill_days": fills.date.nunique() if len(fills) else 0,
                        "off_schedule_fills": len(outside), "blocked_rolls": daily.blocked_rolls.sum(),
                        "terminal_contracts": daily.open_contracts.iloc[-1], "stale_marks": daily.stale_marks.sum(),
                        "net_ann_2016_2018": daily.loc[:"2018-12-31", "return"].mean() * P.TD,
                        "net_ann_2019_2021": daily.loc["2019-01-01":, "return"].mean() * P.TD,
                    }


def run_hold_five(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int) -> None:
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees.trading_date, "五日研究手续费")
    scores, eligible, contracts, bars = study_inputs(panel, jobs)
    sample = eligible.index
    start = sample[0]
    returns = _return_matrix(contracts, bars.set_index(["contract", "date"])["open"])
    outcomes = {
        window: forward_returns(returns, horizon, delay).loc[sample]
        for window, (horizon, delay) in HORIZONS.items()
    }
    sectors = {}
    for name in contracts.columns:
        sectors.setdefault(C.SECTOR[name], []).append(name)
    out.mkdir(parents=True, exist_ok=True)
    ic = _ic_summary_5d(scores, outcomes, sectors)
    ic.to_csv(out / "five_day_ic.csv", index=False)
    print(f"预测检验完成：{len(ic)} 项；开始整数手调仓比较", flush=True)
    ids = [contract for contract in pd.unique(contracts.loc[sample].to_numpy().ravel()) if isinstance(contract, str)]
    market = E.load_market(bars[["date", "contract", "open", "settlement", "volume"]], fees, ids, start)
    accounts = pd.DataFrame(_account_rows_5d(scores, contracts, market, start, sectors))
    accounts.to_csv(out / "five_day_accounts.csv", index=False)
    metrics = ["net_sharpe", "net_ann", "turnover", "end_equity", "avg_gross_fraction", "active_days"]
    phase_stats = accounts.groupby(["book", "interval", "delay"])[metrics].agg(["min", "median", "max"])
    phase_stats.columns = ["_".join(parts) for parts in phase_stats.columns]
    phase_stats.to_csv(out / "five_day_phase_summary.csv")
    print(phase_stats.loc[(slice(None), slice(None), 0), ["net_sharpe_median", "turnover_median", "end_equity_median"]]
          .round(4).to_string())

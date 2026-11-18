"""会员因子的五日预测与固定周期整数手调仓。"""
from __future__ import annotations

from functools import reduce
from operator import mul

import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from . import execution as E
from . import paper_ic as I
from research import portfolio as P
from research.settings import XS_POSITION as XS

HOLD_DAYS = 5
HORIZONS = {"d1": (1, 0), "d5": (5, 0), "d2_to_d5": (4, 1), "d5_delayed": (5, 1)}
# 方向与标准化沿用原规格。两类仓位控制唯一的区别为调仓日历。
BOOKS = {
    "K_DS_GATE": ("K_DS", "delta", "gated"),
    "CROWD1_Z": ("CROWD1", "z", "buffer"),
    "POWER_20_STRICT_Z": ("POWER_20", "z", "rank"),
    "POWER_20_FILLED_Z": ("POWER_20_FILLED", "z", "buffer"),
    "K_NET_60_Z": ("K_NET_60", "z", "buffer"),
    "CROWD2_TAIL_CHEM": ("CROWD2_TAIL", "raw", "signal:chemicals"),
    "SPIDER_ACTIVE": ("SPIDER_ACTIVE", "raw", "signal"),
    # 根据五日 IC 追加的执行对照，方向和调仓规则沿用前述设定。
    "FLOW_RAW": ("FLOW_CONTROL", "raw", "rank"),
    "RESONANCE_60_METALS": ("BOTH_RESONANCE_60_FILLED", "raw", "signal:base_metals"),
}


def forward_returns(returns, horizon, delay):
    """信号日之后的逐日主力开盘收益连乘，任何单日缺失则整段缺失。"""
    if horizon < 1 or delay < 0:
        raise ValueError("累计收益窗口或延迟无效")
    legs = (1 + returns.shift(-(1 + delay + offset)) for offset in range(horizon))
    return reduce(mul, legs) - 1


def _ic_summary(scores, outcomes, sectors):
    rows = []
    for factor, variants in scores.items():
        family = I._factor_family(factor)
        for norm, score in variants.items():
            for label, outcome in outcomes.items():
                # 五日及衰减对照使用各自窗口；d1 保留历史一天标签的完整覆盖。
                per_sector = []
                for sector, names in {"all": score.columns, **sectors}.items():
                    minimum = XS["min_names"] if sector == "all" else I.MIN_SECTOR_NAMES
                    ic, count = I._rank_corr(score[names], outcome[names], minimum)
                    rows.append(dict(factor=factor, family=family, norm=norm, scope=sector,
                                     window=label, **I._stats(ic, count)))
                    if sector != "all":
                        per_sector.append(ic)
                board = pd.concat(per_sector, axis=1)
                count = board.count(axis=1)
                ic = board.mean(axis=1).where(count >= 2)
                rows.append(dict(factor=factor, family=family, norm=norm, scope="sector_equal",
                                 window=label, **I._stats(ic, count.where(ic.notna()))))
    result = pd.DataFrame(rows)
    original = result.family.eq("paper_indicator") & result.norm.eq("raw") & result.scope.eq("all")
    result["q_primary"] = result["p"].where(original).groupby(result.window).transform(I._bh)
    result["q_all"] = result.groupby("window")["p"].transform(I._bh)
    return result


def _weights(score, gate, rule, sectors):
    if rule.startswith("signal"):
        _, _, sector = rule.partition(":")
        if sector:
            score = score.loc[:, sectors[sector]]
        minimum = I.MIN_SECTOR_NAMES if sector else XS["min_names"]
        count = score.count(axis=1)
        return score.div(count.where(count.ge(minimum)), axis=0).fillna(0)
    return I.buffered_tails(score, XS["min_names"], buffered=rule != "rank", entry_gate=gate)


def _account_rows(scores, contracts, market, start, sectors):
    for name, (factor, norm, rule) in BOOKS.items():
        score = scores[factor][norm]
        for interval in (1, HOLD_DAYS):
            for phase in range(interval):
                observations = score.iloc[phase::interval]
                gate = scores[factor]["z"].reindex(observations.index) if rule == "gated" else None
                weights = _weights(observations, gate, rule, sectors)
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


def study_inputs(panel, jobs):
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
        delayed(I._one_symbol)(name, panels[name], by_symbol[name], dates, include_active=True) for name in names
    )
    factors = pd.concat(pieces, axis=1, keys=names).swaplevel(axis=1)
    eligible = panel.assign(eligible=eligible).pivot(index="date", columns="symbol", values="eligible")
    eligible = eligible.reindex(index=dates, columns=names).astype("boolean").fillna(False)
    contracts = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(index=dates, columns=names)
    scores = {
        factor: {norm: score.where(eligible).loc[sample]
                 for norm, score in I._score_variants(factor, factors[factor]).items()}
        for factor in pieces[0].columns
    }
    return scores, eligible.loc[sample], contracts, bars


def run(panel, fee_path, out, jobs):
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees.trading_date, "五日研究手续费")
    scores, eligible, contracts, bars = study_inputs(panel, jobs)
    sample = eligible.index
    start = sample[0]
    returns = I._return_matrix(contracts, bars.set_index(["contract", "date"])["open"])
    outcomes = {
        window: forward_returns(returns, horizon, delay).loc[sample]
        for window, (horizon, delay) in HORIZONS.items()
    }
    sectors = {}
    for name in contracts.columns:
        sectors.setdefault(C.SECTOR[name], []).append(name)
    out.mkdir(parents=True, exist_ok=True)
    ic = _ic_summary(scores, outcomes, sectors)
    ic.to_csv(out / "five_day_ic.csv", index=False)
    print(f"预测检验完成：{len(ic)} 项；开始整数手调仓比较", flush=True)
    ids = [contract for contract in pd.unique(contracts.loc[sample].to_numpy().ravel()) if isinstance(contract, str)]
    market = E.load_market(bars[["date", "contract", "open", "settlement", "volume"]], fees, ids, start)
    accounts = pd.DataFrame(_account_rows(scores, contracts, market, start, sectors))
    accounts.to_csv(out / "five_day_accounts.csv", index=False)
    metrics = ["net_sharpe", "net_ann", "turnover", "end_equity", "avg_gross_fraction", "active_days"]
    phase_stats = accounts.groupby(["book", "interval", "delay"])[metrics].agg(["min", "median", "max"])
    phase_stats.columns = ["_".join(parts) for parts in phase_stats.columns]
    phase_stats.to_csv(out / "five_day_phase_summary.csv")
    print(phase_stats.loc[(slice(None), slice(None), 0), ["net_sharpe_median", "turnover_median", "end_equity_median"]]
          .round(4).to_string())

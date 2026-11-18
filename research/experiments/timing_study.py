"""单品种会员行为择时与同预算截面对照。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from . import execution as E
from . import hold_study as H
from . import paper_ic as I
from research import portfolio as P
from research.settings import XS_POSITION as XS

CONTINUOUS = ("POWER_20_FILLED", "POWER_20", "FLOW_CONTROL", "K_DS")
EVENTS = {
    "BOTH_RESONANCE_60_FILLED": ("base_metals", 5),
    "CROWD2_TAIL": ("chemicals", 1),
}
HORIZONS = {"d1": (1, 0), "d1_delayed": (1, 1), "d5": (5, 0), "d5_delayed": (5, 1)}


def timing_directions(z: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    """同向合力或流量且越过自身一倍标准差才入场，回到零或数据缺失退出。"""
    z_values = z.to_numpy()
    levels = level.to_numpy()
    held = np.zeros(z.shape[1], dtype=np.int8)
    result = np.zeros(z.shape, dtype=np.int8)
    for i, (score, raw) in enumerate(zip(z_values, levels)):
        long = (score > 0) & (raw > 0) & ((score >= 1) | (held > 0))
        short = (score < 0) & (raw < 0) & ((score <= -1) | (held < 0))
        held = long.astype(np.int8) - short.astype(np.int8)
        result[i] = held
    return pd.DataFrame(result, index=z.index, columns=z.columns)


def _symbol_ic(scores, outcomes, names):
    specifications = [(name, "z", names) for name in CONTINUOUS]
    specifications.extend((name, "raw", [s for s in names if C.SECTOR[s] == sector])
                          for name, (sector, _) in EVENTS.items())
    specifications.extend((name, norm, names) for name, norm in (("K_DS", "delta"), ("K_NET_60", "raw")))
    rows = []
    for factor, norm, columns in specifications:
        score = scores[factor][norm].loc[:, columns]
        for window, outcome in outcomes.items():
            returns = outcome[columns]
            ic, _ = I._rank_corr(score.T, returns.T, I.MIN_DAYS)
            triggers = score.ne(0) & score.notna() & returns.notna()
            rows.extend(dict(factor=factor, norm=norm, window=window, symbol=s, sector=C.SECTOR[s],
                             ic=ic[s], pairs=(score[s].notna() & returns[s].notna()).sum(),
                             nonzero=triggers[s].sum()) for s in columns)
    result = pd.DataFrame(rows)
    grouped = result.groupby(["factor", "norm", "window"])["ic"]
    aggregate = grouped.agg(symbols="count", median_ic="median")
    aggregate["positive_symbols"] = grouped.apply(lambda values: values.gt(0).sum())
    return result, aggregate.reset_index()


def _configurations(scores, eligible):
    for factor in CONTINUOUS:
        for phase in range(H.HOLD_DAYS):
            z = scores[factor]["z"].iloc[phase::H.HOLD_DAYS]
            level = scores[factor]["raw"].reindex(z.index)
            # 名义预算按流动性品种数分配，缺因子值的份额留在现金。
            count = eligible.reindex(z.index).sum(axis=1)
            budget = 2 / count.where(count.gt(0))
            eligible_gate = z.where(z.mul(level).gt(0))
            xs = I.buffered_tails(z, XS["min_names"], buffered=True, entry_gate=eligible_gate)
            for mode, direction in (("time_series", timing_directions(z, level)), ("cross_section", np.sign(xs))):
                weights = direction.mul(budget, axis=0).fillna(0)
                yield factor, mode, H.HOLD_DAYS, phase, weights
    for factor, (sector, interval) in EVENTS.items():
        names = [name for name in eligible.columns if C.SECTOR[name] == sector]
        for phase in range(interval):
            signal = scores[factor]["raw"].loc[:, names].iloc[phase::interval]
            count = eligible[names].reindex(signal.index).sum(axis=1)
            weights = signal.div(count.where(count.gt(0)), axis=0).fillna(0)
            yield factor, "event_timing", interval, phase, weights


def _account_rows(scores, eligible, contracts, market, start):
    for factor, mode, interval, phase, weights in _configurations(scores, eligible):
        for delay in (0, 1):
            daily, _ = E.simulate(weights, contracts, market, start, delay, neutral=mode == "cross_section",
                                  rebalance_every=interval, phase=phase)
            stats = P.metrics(daily["return"], daily["turnover"])
            peak = daily.equity.cummax().clip(lower=E.CAPITAL_CNY)
            yield {
                "factor": factor, "mode": mode, "interval": interval, "phase": phase, "delay": delay,
                "end_equity": daily.equity.iloc[-1], "net_sharpe": stats["sharpe"],
                "net_ann": stats["ann_return"], "turnover": stats["daily_turnover"],
                "max_drawdown": (daily.equity / peak - 1).min(),
                "avg_gross_fraction": daily.gross_notional.div(daily.equity).mean(),
                "avg_net_fraction": daily.net_notional.div(daily.equity).mean(),
                "avg_abs_net_fraction": daily.net_notional.abs().div(daily.equity).mean(),
                "active_days": daily.open_contracts.gt(0).sum(), "gross_pnl": daily.gross_pnl.sum(),
                "fees": daily.fees.sum(), "slippage": daily.slippage.sum(),
                "terminal_contracts": daily.open_contracts.iloc[-1], "stale_marks": daily.stale_marks.sum(),
                "net_ann_2016_2018": daily.loc[:"2018-12-31", "return"].mean() * P.TD,
                "net_ann_2019_2021": daily.loc["2019-01-01":, "return"].mean() * P.TD,
            }


def run(panel, fee_path, out, jobs):
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees.trading_date, "单品种择时手续费")
    scores, eligible, contracts, bars = H.study_inputs(panel, jobs)
    dates = eligible.index
    start = dates[0]
    returns = I._return_matrix(contracts, bars.set_index(["contract", "date"])["open"])
    outcomes = {name: H.forward_returns(returns, horizon, delay).loc[dates]
                for name, (horizon, delay) in HORIZONS.items()}
    ic, aggregate = _symbol_ic(scores, outcomes, eligible.columns)
    out.mkdir(parents=True, exist_ok=True)
    ic.to_csv(out / "timing_ic_by_symbol.csv", index=False)
    aggregate.to_csv(out / "timing_ic_summary.csv", index=False)
    ids = [name for name in pd.unique(contracts.loc[dates].to_numpy().ravel()) if isinstance(name, str)]
    market = E.load_market(bars[["date", "contract", "open", "settlement", "volume"]], fees, ids, start)
    accounts = pd.DataFrame(_account_rows(scores, eligible, contracts, market, start))
    accounts.to_csv(out / "timing_accounts.csv", index=False)
    metrics = ["net_sharpe", "net_ann", "turnover", "end_equity", "avg_gross_fraction", "avg_abs_net_fraction",
               "active_days", "net_ann_2016_2018", "net_ann_2019_2021"]
    result = accounts.groupby(["factor", "mode", "interval", "delay"])[metrics].agg(["min", "median", "max"])
    result.columns = ["_".join(col) for col in result.columns]
    result.to_csv(out / "timing_phase_summary.csv")
    view = ["net_sharpe_median", "turnover_median", "end_equity_median", "avg_gross_fraction_median"]
    print(result[view].round(4).to_string())

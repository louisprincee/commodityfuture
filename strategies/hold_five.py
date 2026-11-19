"""五日预测候选的固定周期调仓：1000 万元整数手账户，每日或每 5 日调仓并报告全部起始偏移。

    python -m backtest.engine hold-five

候选由 research 的 5 日 IC 检验事先选定，规则在看到账户结果之前固定。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as C
from backtest import engine as E
from research import ic_study as IC
from research.settings import XS_POSITION as XS
from strategies import rules as R

STRATEGY_ID = "hold_five"

HOLD_DAYS = 5
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


def _weights_5d(score, gate, rule, sectors):
    if rule.startswith("signal"):
        _, _, sector = rule.partition(":")
        if sector:
            score = score.loc[:, sectors[sector]]
        minimum = IC.MIN_SECTOR_NAMES if sector else XS["min_names"]
        count = score.count(axis=1)
        return score.div(count.where(count.ge(minimum)), axis=0).fillna(0)
    return R.buffered_tails(score, XS["min_names"], buffered=rule != "rank", entry_gate=gate)


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
                    stats = E.metrics(daily["return"], daily["turnover"])
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
                        "net_ann_2016_2018": daily.loc[:"2018-12-31", "return"].mean() * E.TD,
                        "net_ann_2019_2021": daily.loc["2019-01-01":, "return"].mean() * E.TD,
                    }


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int) -> None:
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees.trading_date, "五日调仓手续费")
    scores, eligible, contracts, bars = IC.study_inputs(panel, jobs)
    sample = eligible.index
    start = sample[0]
    sectors = {}
    for name in contracts.columns:
        sectors.setdefault(C.SECTOR[name], []).append(name)
    ids = [contract for contract in pd.unique(contracts.loc[sample].to_numpy().ravel()) if isinstance(contract, str)]
    market = E.load_market(bars[["date", "contract", "open", "settlement", "volume"]], fees, ids, start)
    accounts = pd.DataFrame(_account_rows_5d(scores, contracts, market, start, sectors))
    out.mkdir(parents=True, exist_ok=True)
    accounts.to_csv(out / "five_day_accounts.csv", index=False)
    metrics = ["net_sharpe", "net_ann", "turnover", "end_equity", "avg_gross_fraction", "active_days"]
    phase_stats = accounts.groupby(["book", "interval", "delay"])[metrics].agg(["min", "median", "max"])
    phase_stats.columns = ["_".join(parts) for parts in phase_stats.columns]
    phase_stats.to_csv(out / "five_day_phase_summary.csv")
    print(phase_stats.loc[(slice(None), slice(None), 0), ["net_sharpe_median", "turnover_median", "end_equity_median"]]
          .round(4).to_string())

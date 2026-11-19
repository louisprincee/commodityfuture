"""两端入场与中位数缓冲规则：单因子 z-score 截面多空，比较日频重判与缓冲退出的换手和净收益。

    python -m backtest.engine buffer

因子与方向沿用 research；入场约两端各 30%，缓冲版到达中位数才退出原方向。
手续费加每边 1 tick，换月按两腿计费，旧腿费率用昨日旧主力费率代理。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as C
from backtest import engine as E
from research import ic_study as IC
from research.settings import XS_POSITION as XS
from strategies import rules as R

STRATEGY_ID = "buffer"
# 预先选定的代表：拥挤、合力（严格与填补）、净头寸变化与持仓流的 z-score，另加空仓增量的三种得分
BUFFER_FACTORS = ("CROWD1", "POWER_20", "POWER_20_FILLED", "K_NET_60", "FLOW_CONTROL")
ALL_NORMS_FACTORS = ("K_DS",)


def _buffer_rows(scores, returns, fee, rolls, sectors, start):
    for (name, norm), score in scores.items():
        score = score.loc[start:].reindex(score.index)
        for scope in ("all", "sector_neutral"):
            for buffered in (False, True):
                if scope == "all":
                    weights = R.buffered_tails(score, XS["min_names"], buffered)
                else:
                    books = [
                        R.buffered_tails(score.loc[:, cols], IC.MIN_SECTOR_NAMES, buffered)
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
                    stats = E.metrics(net, turnover)
                    yield {
                        "factor": name, "norm": norm, "scope": scope, "buffered": buffered, "delay": delay,
                        "ann_gross": gross.mean() * E.TD,
                        "ann_cost": cost.reindex(net.index).mean() * E.TD,
                        "net_ann": stats["ann_return"], "net_sharpe": stats["sharpe"],
                        "turnover": stats["daily_turnover"], "hac_t": IC._hac_t(gross - cost.loc[start:]),
                        "days": len(net), "excluded_days": int((~valid.loc[start:]).sum()),
                        "active_days": int(held.loc[net.index].abs().sum(axis=1).gt(0).sum()),
                    }


def book_scores(scores: dict[str, dict[str, pd.DataFrame]], dates: pd.DatetimeIndex) -> dict:
    """参与缓冲对照的 (因子, 得分) 组合，按因子构建顺序排列并扩展到全部交易日。"""
    chosen = {}
    for name, variants in scores.items():
        if name in BUFFER_FACTORS:
            chosen[(name, "z")] = variants["z"].reindex(dates)
        if name in ALL_NORMS_FACTORS:
            chosen.update({(name, norm): value.reindex(dates) for norm, value in variants.items()})
    return chosen


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int) -> None:
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees["trading_date"], "缓冲对照手续费")
    scores, eligible, contracts, bars = IC.study_inputs(panel, jobs)
    start = eligible.index[0]
    returns = IC.return_matrix(contracts, bars.set_index(["contract", "date"])["open"])
    rolls = contracts.ne(contracts.shift(1)) & contracts.notna() & contracts.shift(1).notna()
    commodity = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    fee = E.cost_rate_matrix(commodity, fees, XS["slippage_ticks"]).reindex_like(returns)
    sector_series = pd.Series({s: C.SECTOR[s] for s in contracts.columns})
    sectors = {str(name): list(group.index) for name, group in sector_series.groupby(sector_series)}
    rows = pd.DataFrame(_buffer_rows(book_scores(scores, returns.index), returns, fee, rolls, sectors, start))
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "paper_buffer.csv", index=False)
    print(rows.loc[rows["delay"].eq(0), ["factor", "norm", "scope", "buffered", "net_ann", "turnover"]]
          .round(4).to_string(index=False))

"""研究期候选组合：会员持仓流、净持仓状态与期限结构/价格趋势横截面合成。"""
from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config as C
import data as D
from backtest import engine as E
from research.economic_ic import momentum_12_1
from research.ic_study import _one_symbol_paper
from research.settings import XS_POSITION as XS
from strategies import rules as R

STRATEGY_ID = "position_blend"
HORIZON = 5
TAIL = 0.20
MIN_SYMBOLS = XS["min_names"]
NAME_CAP = E.NAME_CAP
FLAT_MARGIN = 0.20
MARGIN_CAP = E.MARGIN_CAP
FACTOR_NAMES = ("K_DS", "FLOW_CONTROL", "K_NET_SHARE", "ROLL_YIELD", "MOM_12_1")
BOOK_LABELS = {
    "MEMBER_FLOW": "持仓流",
    "MEMBER_FLOW_STATE": "持仓流+净持仓状态",
    "POSITION_FLOW_TREND": "持仓流+状态+趋势/期限结构",
}


def blend_scores(
    factors: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    tail: float = TAIL,
    min_symbols: int = MIN_SYMBOLS,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """用共同完整截面形成等权经济腿分数及两端组合权重。"""
    if not 0 < tail < 0.5:
        raise ValueError("tail 必须在 (0, 0.5) 内")
    if min_symbols < 2:
        raise ValueError("min_symbols 至少为 2")
    missing = set(FACTOR_NAMES) - set(factors.columns.get_level_values(0))
    if missing:
        raise ValueError(f"因子输入缺少字段: {sorted(missing)}")

    frames = {name: factors[name].reindex_like(eligible) for name in FACTOR_NAMES}
    complete = eligible.astype(bool).copy()
    for frame in frames.values():
        complete &= np.isfinite(frame)

    oriented = {
        "K_DS": -frames["K_DS"],
        "FLOW_CONTROL": frames["FLOW_CONTROL"],
        "K_NET_SHARE": frames["K_NET_SHARE"],
        "ROLL_YIELD": frames["ROLL_YIELD"],
        "MOM_12_1": frames["MOM_12_1"],
    }
    ranks = {
        name: values.where(complete).rank(axis=1, method="average", pct=True)
        for name, values in oriented.items()
    }
    flow = (ranks["K_DS"] + ranks["FLOW_CONTROL"]) / 2
    state = ranks["K_NET_SHARE"]
    market = (ranks["ROLL_YIELD"] + ranks["MOM_12_1"]) / 2
    scores = {
        "MEMBER_FLOW": flow.where(complete),
        "MEMBER_FLOW_STATE": (0.5 * flow + 0.5 * state).where(complete),
        "POSITION_FLOW_TREND": (flow + state + market).div(3).where(complete),
    }

    weights = {}
    for name, score in scores.items():
        target = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        for date, row in score.iterrows():
            available = row.dropna().sort_values(kind="mergesort")
            count = len(available)
            if count < min_symbols:
                continue
            side_count = max(1, ceil(count * tail))
            short_names = available.index[:side_count]
            long_names = available.index[-side_count:]
            target.loc[date, long_names] = 0.5 / side_count
            target.loc[date, short_names] = -0.5 / side_count
        weights[name] = target
    return scores, pd.concat(weights, axis=1)


def _features(panel: pd.DataFrame, bars: pd.DataFrame, dates: pd.DatetimeIndex,
              symbols: list[str], jobs: int) -> pd.DataFrame:
    rank_root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(rank_root)
    symbols = [
        symbol for symbol in symbols
        if (rank_root / f"{symbol}_long.pkl").is_file()
        and (rank_root / f"{symbol}_short.pkl").is_file()
    ]
    if not symbols:
        raise RuntimeError("没有同时具备多、空会员排名的流动性商品")
    panel_by_symbol = dict(tuple(panel.groupby("symbol", sort=False)))
    bars_by_symbol = dict(tuple(bars.loc[bars["symbol"].isin(symbols)].groupby("symbol", sort=False)))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_one_symbol_paper)(
            symbol, panel_by_symbol[symbol], bars_by_symbol[symbol], dates, include_active=False
        )
        for symbol in symbols
    )
    holdings = pd.concat(pieces, axis=1, keys=symbols).swaplevel(axis=1).sort_index(axis=1)
    momentum = {}
    for symbol, rows in panel.groupby("symbol", sort=False):
        if symbol in symbols:
            returns = rows.set_index("date")["ret1"].reindex(dates)
            momentum[symbol] = momentum_12_1(returns)
    symbols = sorted(symbols)
    daily = panel.pivot(index="date", columns="symbol", values="roll_yield_ann")
    roll_yield = daily.reindex(index=dates, columns=symbols)
    momentum_frame = pd.DataFrame(momentum).reindex(index=dates, columns=symbols)
    features = {
        "K_DS": holdings["K_DS"].reindex(index=dates, columns=symbols),
        "FLOW_CONTROL": holdings["FLOW_CONTROL"].reindex(index=dates, columns=symbols),
        "K_NET_SHARE": holdings["K_NET_SHARE"].reindex(index=dates, columns=symbols),
        "ROLL_YIELD": roll_yield,
        "MOM_12_1": momentum_frame,
    }
    return pd.concat(features, axis=1).sort_index(axis=1)


def _summary(book: str, ticks: int, daily: pd.DataFrame, benchmark: pd.Series) -> dict:
    stats = E.metrics(daily["return"], daily["turnover"])
    peak = daily["equity"].cummax().clip(lower=E.CAPITAL_CNY)
    drawdown = float((daily["equity"] / peak - 1).min())
    ret = daily["return"].fillna(0.0)
    bmk = benchmark.reindex(ret.index).fillna(0.0)
    variance = float(bmk.var())
    beta = float(ret.cov(bmk) / variance) if variance > 0 else np.nan
    residual = ret - beta * bmk if np.isfinite(beta) else ret
    return {
        "book": book,
        "label": BOOK_LABELS.get(book, "等权基准"),
        "ticks": ticks,
        "end_equity": float(daily["equity"].iloc[-1]),
        "net_ann": float(stats["ann_return"]),
        "ann_vol": float(stats["ann_vol"]),
        "sharpe": float(stats["sharpe"]),
        "sortino": float(stats["sortino"]),
        "max_drawdown": drawdown,
        "turnover": float(stats["daily_turnover"]),
        "gross_pnl": float(daily["gross_pnl"].sum()),
        "fees": float(daily["fees"].sum()),
        "slippage": float(daily["slippage"].sum()),
        "gross_exposure": float(daily["gross_notional"].div(daily["equity"]).mean()),
        "signed_net_exposure": float(daily["net_notional"].div(daily["equity"]).mean()),
        "margin_mean": float(daily["margin_fraction"].mean()),
        "margin_max": float(daily["margin_fraction"].max()),
        "beta_to_bmk": beta,
        "market_adj_net_ann": float(residual.mean() * E.TD),
        "stale_marks": int(daily["stale_marks"].sum()),
        "terminal_contracts": int(daily["open_contracts"].iloc[-1]),
    }


def _yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book in equity.columns:
        series = equity[book].dropna()
        for year, values in series.groupby(series.index.year):
            previous = series.loc[series.index.year < year]
            base = float(previous.iloc[-1] if len(previous) else E.CAPITAL_CNY)
            rows.append({"book": book, "year": int(year), "return": float(values.iloc[-1] / base - 1)})
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int = 8) -> None:
    """评估固定候选组合，仅使用研究期数据与研究期执行参数。"""
    C.assert_no_holdout_dates(panel["date"], "持仓组合研究日面板")
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees["trading_date"], "持仓组合研究期手续费")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"]).copy()
    panel["eligible"] = D.eligibility(panel)
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    start = pd.Timestamp(XS["test_start"])
    symbols = sorted(panel.loc[panel["eligible"] & panel["date"].ge(start), "symbol"].unique())

    bars = D.load_contracts("research")
    factors = _features(panel, bars[["date", "contract", "open", "open_interest", "symbol"]],
                        dates, symbols, jobs)
    eligible = panel.pivot(index="date", columns="symbol", values="eligible").reindex(
        index=dates, columns=factors.columns.get_level_values(1).unique().sort_values()
    ).astype("boolean").fillna(False).astype(bool)
    scores, book_weights = blend_scores(factors, eligible)
    eligible_sample = eligible.loc[start:]
    raw_weights = {
        book: R.hold_cohorts(book_weights[book].reindex(index=eligible_sample.index), HORIZON)
        for book in BOOK_LABELS
    }
    dominant = panel.pivot(index="date", columns="symbol", values="c1_contract")
    contracts = dominant.shift(1).reindex(index=eligible_sample.index, columns=eligible_sample.columns)
    ids = {contract for contract in contracts.to_numpy().ravel() if isinstance(contract, str)}
    market = E.load_market(
        bars[["date", "contract", "open", "settlement", "volume"]],
        fees,
        ids,
        start,
        root=C.RESEARCH_EXECUTION_PARAMETERS / "contracts",
        flat_margin=FLAT_MARGIN,
        partition="research",
    )

    common = scores["POSITION_FLOW_TREND"].notna().reindex_like(eligible_sample).fillna(False)
    benchmark_weights = common.astype(float).div(common.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    benchmark_returns = {}
    for ticks in (1, 2):
        benchmark_daily, _ = E.simulate(
            benchmark_weights, contracts, market, start, neutral=False,
            slippage_ticks=ticks, margin_cap=MARGIN_CAP,
        )
        benchmark_returns[ticks] = benchmark_daily["return"]
        rows.append(_summary("LONG_ONLY_BMK", ticks, benchmark_daily, benchmark_returns[ticks]))
        if ticks == 1:
            curves["LONG_ONLY_BMK"] = benchmark_daily["equity"]
        for book, target in raw_weights.items():
            daily, _ = E.simulate(
                target, contracts, market, start, neutral=False, slippage_ticks=ticks,
                name_cap=NAME_CAP, margin_cap=MARGIN_CAP,
            )
            rows.append(_summary(book, ticks, daily, benchmark_returns[ticks]))
            if ticks == 1:
                curves[book] = daily["equity"]

    summary = pd.DataFrame(rows)
    equity_curves = pd.DataFrame(curves)
    summary.to_csv(out / "summary.csv", index=False)
    equity_curves.to_csv(out / "equity_curves_d0_t1.csv", index_label="date")
    _yearly_returns(equity_curves).to_csv(out / "yearly_returns.csv", index=False)
    figure = E.plot_equity_drawdown(
        equity_curves,
        out / "performance.png",
        strategy="POSITION_FLOW_TREND",
        benchmark="LONG_ONLY_BMK",
        period_label="研究期候选",
    )
    view = summary.loc[summary["ticks"].eq(1), [
        "book", "label", "end_equity", "net_ann", "sharpe", "max_drawdown", "turnover",
        "gross_exposure", "margin_max", "market_adj_net_ann",
    ]]
    print(view.round(4).to_string(index=False))
    print(f"候选组合汇总写入 {out / 'summary.csv'}")
    print(f"净值与回撤图写入 {figure}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m strategies.position_blend")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    run(D.load_panel(), C.RESEARCH_FEE_PATH, C.strategy_dir(STRATEGY_ID), jobs=args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

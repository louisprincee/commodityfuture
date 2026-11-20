"""STICKY_CORE：基差动量、低偏度与粘性席位净多头占比的三因子截面组合，两端等权多空与 10 日重叠持有。

构造顺序
  因子   t 日收盘后仅用当日及之前数据计算：
           1. BASIS_MOM_242   (+): 主力与次主力 242 日复合收益之差（期限结构斜率动量）
           2. NEG_SKEW_242    (-): 主力 242 日（至少 200 日）日收益偏度取反（厌恶彩票/尾部风险）
           3. STICKY_NET_SHARE(+): 剔除高换手活跃席位（CROWD1 门槛）后的前 20 名净多头占比
  合成   在当日流动性池且三因子均有效的品种内分别计算百分位秩，等权平均。
  选样   有效候选至少 8 个品种时，取合成得分前 20% 做多、后 20% 做空，不设拥挤度硬门槛否决。
  持有   每日新鲜多空各分配 0.50 名义权重，经 10 个交易日重叠持有形成当期组合，次日开盘成交。
  对照   LONG_ONLY_BMK；当前唯一策略组合为 2X（上限 20%）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import strategies  # noqa: F401
import backtest.engine as E
import config as C
import data as D
from research.factors import market_inputs, member_tables, sticky_net_share
from research.settings import XS_POSITION
from research.xs_robustness import basis_momentum, rolling_skew

STRATEGY_ID = "sticky_core"
FACTORS = ("BASIS_MOM_242", "NEG_SKEW_242", "STICKY_NET_SHARE")
HORIZON = 10
TAIL_SHARE = 0.20
MIN_NAMES = 8
MIN_SYMBOLS = 10
BASIS_WINDOW = 242
SKEW_WINDOW = 242
SKEW_MIN_PERIODS = 200
FLAT_MARGIN = 0.20
MARGIN_CAP = 0.50
BOOKS = {
    "STICKY_CORE_2X": (2.0, 0.20),
}
MAIN_BOOK = "STICKY_CORE_2X"
BENCHMARK = "LONG_ONLY_BMK"


def composite_score(factors: dict[str, pd.DataFrame], candidates: pd.DataFrame) -> pd.DataFrame:
    """仅在候选池内对固定因子分别做截面百分位秩，再等权平均。"""
    ranked = [factors[name].where(candidates).rank(axis=1, pct=True) for name in FACTORS]
    return pd.concat(ranked).groupby(level=0).mean().where(candidates)


def tail_weights(score: pd.DataFrame, tail: float = TAIL_SHARE,
                 min_names: int = MIN_NAMES) -> pd.DataFrame:
    """每日选前/后 20% 等权配资，多头合计 +0.50、空头合计 -0.50，不做否决留白。"""
    n = score.notna().sum(axis=1)
    ok = n.ge(min_names)
    k = np.ceil(n * tail)
    long_sel = score.rank(axis=1, ascending=False, method="first").le(k, axis=0).where(ok, False)
    short_sel = score.rank(axis=1, ascending=True, method="first").le(k, axis=0).where(ok, False)
    wl = long_sel.astype(float).div(long_sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    ws = -short_sel.astype(float).div(short_sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    both_ok = wl.sum(axis=1).gt(0) & ws.sum(axis=1).lt(0)
    return (wl + ws).where(both_ok, 0.0)


def hold_cohorts(weights: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """每日新开权重各自持有 horizon 日，汇总为当期目标权重。"""
    if horizon < 1:
        raise ValueError("持有期必须为正整数")
    return sum((weights.shift(lag, fill_value=0.0) for lag in range(horizon))) / horizon


def _sticky_series(symbol: str, panel_sym: pd.DataFrame, bars_sym: pd.DataFrame,
                   dates: pd.DatetimeIndex, partition: str) -> pd.Series:
    mkt = market_inputs(panel_sym, bars_sym, symbol).reindex(dates)
    tables = member_tables(symbol, dates, top20=True, partition=partition)
    return sticky_net_share(tables, mkt)["STICKY_NET_SHARE"]


def selection_summary(score: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """记录样本期每日候选池与实际入选多空数量及换仓规模。"""
    long_cnt = fresh.gt(0).sum(axis=1)
    short_cnt = fresh.lt(0).sum(axis=1)
    return pd.DataFrame([{
        "days": int(len(fresh)),
        "trading_days_with_book": int(fresh.abs().sum(axis=1).gt(0).sum()),
        "candidates_per_day": float(score.notna().sum(axis=1).mean()),
        "long_slots_per_day": float(long_cnt.mean()),
        "short_slots_per_day": float(short_cnt.mean()),
        "fresh_gross_per_day": float(fresh.abs().sum(axis=1).mean()),
    }])


def _yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    year_end = equity.resample("YE").last()
    year_end.index = year_end.index.year
    prior = pd.concat([
        pd.DataFrame(E.CAPITAL_CNY, index=[year_end.index[0]], columns=year_end.columns),
        year_end.iloc[:-1].set_axis(year_end.index[1:]),
    ])
    returns = year_end / prior - 1
    rows = returns.reset_index(names="year").melt(id_vars="year", var_name="book", value_name="return")
    return rows[["book", "year", "return"]]


def _summary(book: str, ticks: int, daily: pd.DataFrame) -> dict:
    metrics = E.metrics(daily["return"], daily["turnover"])
    peak = daily["equity"].cummax().clip(lower=E.CAPITAL_CNY)
    return {
        "book": book,
        "ticks": ticks,
        "end_equity": float(daily["equity"].iloc[-1]),
        "net_ann": float(metrics["ann_return"]),
        "ann_vol": float(metrics["ann_vol"]),
        "sharpe": float(metrics["sharpe"]),
        "sortino": float(metrics["sortino"]),
        "max_drawdown": float((daily["equity"] / peak - 1).min()),
        "turnover": float(metrics["daily_turnover"]),
        "gross_pnl": float(daily["gross_pnl"].sum()),
        "fees": float(daily["fees"].sum()),
        "slippage": float(daily["slippage"].sum()),
        "gross_exposure": float(daily["gross_notional"].div(daily["equity"]).mean()),
        "net_exposure": float(daily["net_notional"].div(daily["equity"]).mean()),
        "margin_mean": float(daily["margin_fraction"].mean()),
        "margin_max": float(daily["margin_fraction"].max()),
        "stale_marks": int(daily["stale_marks"].sum()),
        "terminal_contracts": int(daily["open_contracts"].iloc[-1]),
    }


def run_combined(base_out: Path) -> None:
    """将研究期与样本外净值按研究期终值承接为连续日序列。"""
    research_path = base_out / "research" / "equity_curves_d0_t1.csv"
    oos_path = base_out / "oos" / "equity_curves_d0_t1.csv"
    if not research_path.is_file() or not oos_path.is_file():
        raise FileNotFoundError("组合净值需要先分别生成研究期与样本外净值曲线")

    research = pd.read_csv(research_path, parse_dates=["date"]).set_index("date")
    oos = pd.read_csv(oos_path, parse_dates=["date"]).set_index("date")
    required = {BENCHMARK, MAIN_BOOK}
    missing = {book for book in required if book not in research.columns or book not in oos.columns}
    if missing:
        raise KeyError(f"组合净值缺少序列: {sorted(missing)}")

    scaled = oos.loc[:, sorted(required)].copy()
    if scaled.empty:
        raise ValueError("样本外净值曲线为空")
    for book in sorted(required):
        research_end = float(research[book].iloc[-1])
        oos_start = float(scaled[book].iloc[0])
        if not (research_end > 0 and oos_start > 0):
            raise ValueError("组合净值必须为正")
        scaled[book] = scaled[book] / oos_start * research_end

    combined = pd.concat([research.loc[:, sorted(required)], scaled]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    out = base_out / "combined"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "equity_curves.csv", index_label="date")
    _yearly_returns(combined).to_csv(out / "yearly_returns.csv", index=False)

    rows = []
    for book in sorted(required):
        daily = combined[book].pct_change(fill_method=None).dropna()
        metrics = E.metrics(daily)
        peak = combined[book].cummax().clip(lower=E.CAPITAL_CNY)
        rows.append({
            "book": book,
            "end_equity": float(combined[book].iloc[-1]),
            "net_ann": float(metrics["ann_return"]),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float((combined[book] / peak - 1).min()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    figure = E.plot_equity_drawdown(
        combined,
        out / "performance.png",
        strategy=MAIN_BOOK,
        benchmark=BENCHMARK,
        period_label="研究期 + 样本外",
    )
    print(summary.round(4).to_string(index=False))
    print(f"组合净值与逐年收益写入 {out}")
    print(f"组合净值与回撤图写入 {figure}")


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int,
        partition: str = "research") -> None:
    if partition not in C.PARTITIONS:
        raise ValueError(f"未知分区 {partition}")
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    if partition == "research":
        C.assert_no_holdout_dates(panel["date"], "STICKY_CORE 研究日面板")
        C.assert_research_only(fee_path)
        C.assert_no_holdout_dates(fees["trading_date"], "STICKY_CORE 研究期手续费")
        start = pd.Timestamp(XS_POSITION["test_start"])
        market_root = C.RESEARCH_EXECUTION_PARAMETERS / "contracts"
        period_label = "研究期"
    else:
        start = pd.Timestamp(C.OOS_START)
        C.assert_oos_dates(panel.loc[panel["date"].ge(start), "date"], "STICKY_CORE 样本外日面板")
        C.assert_oos_dates(fees["trading_date"], "STICKY_CORE 样本外手续费")
        market_root = C.OOS_EXECUTION_PARAMETERS / "contracts"
        period_label = "样本外"

    bars = D.load_contracts_span(partition)
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(
        ["date", "symbol"], kind="mergesort",
    )
    panel = panel.assign(eligible=D.eligibility(panel))
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    names = sorted(panel.loc[panel["eligible"] & panel["date"].ge(start), "symbol"].unique())
    rank_root = C.daily_dir(partition) / "member_rank"
    if partition == "research":
        C.assert_research_only(rank_root)
    names = [name for name in names if (rank_root / f"{name}_long.pkl").is_file()]
    if len(names) < MIN_SYMBOLS:
        raise RuntimeError("评价区间没有足够的流动性品种及会员排名")

    panel_by_symbol = dict(tuple(panel.groupby("symbol", sort=False)))
    bars_by_symbol = dict(tuple(bars.loc[bars["symbol"].isin(names)].groupby("symbol", sort=False)))
    sticky_cols = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_sticky_series)(name, panel_by_symbol[name], bars_by_symbol[name], dates, partition)
        for name in names
    )
    sticky_df = pd.DataFrame(dict(zip(names, sticky_cols))).reindex(dates)
    market = lambda col: panel.pivot(index="date", columns="symbol", values=col).reindex(index=dates, columns=names)

    factors = {
        "BASIS_MOM_242": basis_momentum(market("ret1"), market("ret2"), window=BASIS_WINDOW),
        "NEG_SKEW_242": -rolling_skew(market("ret1"), window=SKEW_WINDOW, min_periods=SKEW_MIN_PERIODS),
        "STICKY_NET_SHARE": sticky_df,
    }
    factors = {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in factors.items()}

    eligible = market("eligible").astype("boolean").fillna(False).astype(bool)
    candidates = eligible.copy()
    for frame in factors.values():
        candidates &= frame.notna()

    score = composite_score(factors, candidates)
    fresh = tail_weights(score)
    fresh.loc[fresh.index < start] = 0.0
    held = hold_cohorts(fresh).loc[start:]

    eligible = eligible.loc[start:]
    benchmark = eligible.astype(float).div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    contracts = market("c1_contract").shift(1).reindex(index=held.index, columns=names)
    ids = {contract for contract in contracts.to_numpy().ravel() if isinstance(contract, str)}
    quotes = E.load_market(
        bars, fees, ids, start, root=market_root, flat_margin=FLAT_MARGIN, partition=partition,
    )

    out.mkdir(parents=True, exist_ok=True)
    selection_summary(score.loc[start:], fresh.loc[start:]).to_csv(out / "selection_summary.csv", index=False)
    rows, curves = [], {}
    for ticks in (1, 2):
        daily = {
            BENCHMARK: E.simulate(
                benchmark, contracts, quotes, start, neutral=False, slippage_ticks=ticks,
                margin_cap=MARGIN_CAP,
            )[0],
        }
        for book, (multiplier, ncap) in BOOKS.items():
            daily[book] = E.simulate(
                held * multiplier, contracts, quotes, start, neutral=True, slippage_ticks=ticks,
                name_cap=ncap, margin_cap=MARGIN_CAP,
            )[0]
        rows.extend(_summary(book, ticks, frame) for book, frame in daily.items())
        if ticks == 1:
            curves = {book: frame["equity"] for book, frame in daily.items()}

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    equity = pd.DataFrame(curves)
    equity.to_csv(out / "equity_curves_d0_t1.csv", index_label="date")
    _yearly_returns(equity).to_csv(out / "yearly_returns.csv", index=False)
    figure = E.plot_equity_drawdown(
        equity, out / "performance.png", strategy=MAIN_BOOK, benchmark=BENCHMARK, period_label=period_label,
    )
    view = summary.loc[summary.ticks.eq(1), [
        "book", "end_equity", "net_ann", "sharpe", "max_drawdown", "turnover", "gross_exposure", "margin_max",
    ]]
    print(view.round(4).to_string(index=False))
    print(f"回测摘要写入 {out / 'summary.csv'}；净值与回撤图写入 {figure}")

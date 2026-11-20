"""FLOW_BASIS：10 日累计会员持仓流、净多头占比与基差动量的截面合成，两端多空、拥挤否决，10 日重叠持有。

本规格在已退役的 FLOW_STATE_1X 研究期诊断之后提出，是独立命名的规格，结果只算样本内证据。
规则在看到本规格回测结果之前固定：

  候选   当日在流动性池内、主力未换月、三个因子都有效的品种。
  得分   三个因子在候选内的截面百分位秩等权平均，持仓类占三分之二。
         FLOW_10       = 过去 10 个交易日 (ΣΔL − ΣΔS) 之和 / 当日商品总持仓，品种榜全部会员；
                         10 日中任一日缺失则无值。窗口与持有期一致。
         K_NET_SHARE   = 双边披露合约前 20 名净多头 / 这些合约的持仓量。
         BASIS_MOM_242 = 主力与次主力各自同合约日收益的 242 日复合收益之差。
  两端   候选不少于 8 个且得分不全相同时，最高、最低各 ceil(20%·N) 个，
         多空各 0.5 目标权重，侧内等权。
  拥挤   CROWD1 只作否决，分位用该品种过去 250 日、至少 60 个观测，不含当日。
         多头遇到 CROWD1 高于自身 90% 分位、空头遇到低于自身 10% 分位，或 CROWD1 缺失，
         该名额留成现金，不递补。
  持有   每日一批，重叠 10 个交易日，每批 1/10；研究期开始前的信号不进入持仓。
  两本   FLOW_BASIS_1X 为主报告；FLOW_BASIS_2X 把目标权重乘以 2，是下单前的输入，
         不代表账户已实现的杠杆。单品种名义上限 10% 与 50% 保证金上限对两本相同。
  执行   收盘信号，下一交易日开盘成交；整数手、20% 保证金率、历史手续费、
         每边 1 tick（2 tick 另存）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import config as C
import data as D
from backtest import engine as E
from research import factors as F
from research.settings import XS_POSITION
from research.xs_robustness import basis_momentum

STRATEGY_ID = "flow_basis"
BOOKS = {"FLOW_BASIS_1X": 1.0, "FLOW_BASIS_2X": 2.0}
MAIN_BOOK = "FLOW_BASIS_1X"
BENCHMARK = "LONG_ONLY_BMK"
FACTORS = ("FLOW_10", "K_NET_SHARE", "BASIS_MOM_242")
FLOW_WINDOW = 10
HORIZON = 10
TAIL = 0.20
MIN_SYMBOLS = XS_POSITION["min_names"]
WINDOW = 250
MIN_OBSERVATIONS = 60
CROWD_HIGH = 0.90
CROWD_LOW = 0.10
FLAT_MARGIN = 0.20
MARGIN_CAP = E.MARGIN_CAP


def cumulative_flow(flow: pd.DataFrame, open_interest: pd.DataFrame, window: int = FLOW_WINDOW) -> pd.DataFrame:
    """过去 window 日净持仓流之和除以当日总持仓；窗口内任一日缺失则无值。"""
    total = flow.rolling(window, min_periods=window).sum()
    return total.div(open_interest.where(open_interest.gt(0)))


def trailing_quantile(values: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """每个品种自身过去窗口的分位，不含当日，至少 MIN_OBSERVATIONS 个有效观测。"""
    return values.shift(1).rolling(WINDOW, min_periods=MIN_OBSERVATIONS).quantile(quantile)


def crowd_vetoes(crowd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """多头否决：CROWD1 高于自身 90% 分位；空头否决：低于 10% 分位。缺失一律否决。"""
    high = trailing_quantile(crowd, CROWD_HIGH)
    low = trailing_quantile(crowd, CROWD_LOW)
    long_ok = crowd.notna() & high.notna() & crowd.le(high)
    short_ok = crowd.notna() & low.notna() & crowd.ge(low)
    return ~long_ok, ~short_ok


def tail_weights(score: pd.DataFrame, veto_long: pd.DataFrame, veto_short: pd.DataFrame,
                 *, tail: float = TAIL, min_symbols: int = MIN_SYMBOLS) -> pd.DataFrame:
    """两端等权，多空各 0.5；被否决的名额留成现金，不递补。"""
    if not 0 < tail < 0.5:
        raise ValueError("tail 必须在 (0, 0.5) 内")
    if min_symbols < 2:
        raise ValueError("min_symbols 至少为 2")
    valid = score.replace([np.inf, -np.inf], np.nan)
    count = valid.notna().sum(axis=1)
    active = count.ge(min_symbols) & valid.nunique(axis=1).ge(2)
    side = np.ceil(count * tail).where(active)
    low = valid.rank(axis=1, method="first").le(side, axis=0)
    high = valid.rank(axis=1, method="first", ascending=False).le(side, axis=0)
    size = side.where(side.gt(0))
    long = high & ~veto_long.reindex_like(valid).fillna(True).astype(bool)
    short = low & ~veto_short.reindex_like(valid).fillna(True).astype(bool)
    per_name = (0.5 / size).to_numpy()[:, None]
    weights = np.where(long, per_name, 0.0) - np.where(short, per_name, 0.0)
    return pd.DataFrame(np.nan_to_num(weights), index=valid.index, columns=valid.columns)


def hold_cohorts(weights: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """重叠持有最近 horizon 个日度组合；不足 horizon 批时未用预算留在现金。"""
    if horizon < 1:
        raise ValueError("horizon 必须为正")
    return weights.fillna(0.0).rolling(horizon, min_periods=1).sum() / horizon


def composite_score(factors: dict[str, pd.DataFrame], candidates: pd.DataFrame) -> pd.DataFrame:
    """候选内各因子截面百分位秩的等权平均；候选已要求每个因子都有效。"""
    ranks = [factors[name].where(candidates).rank(axis=1, pct=True) for name in FACTORS]
    return sum(ranks) / len(ranks)


def _symbol_inputs(name: str, panel: pd.DataFrame, bars: pd.DataFrame,
                   dates: pd.DatetimeIndex) -> pd.DataFrame:
    ranks = D.load_member_rank_contracts(name, "research", ("long", "short"))
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    net_share = F.kaiyuan(ranks, bars, dates)["K_NET_SHARE"]
    tables = F.member_tables(name, dates, partition="research")
    market = F.market_inputs(panel, bars, name).reindex(dates)
    flow = tables["dL"].sum(axis=1, min_count=1) - tables["dS"].sum(axis=1, min_count=1)
    crowd = F.crowding(tables, market)["CROWD1"].reindex(dates)
    return pd.DataFrame({"flow": flow, "K_NET_SHARE": net_share, "crowd": crowd}, index=dates)


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


def _yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book, series in equity.items():
        series = series.dropna()
        for year, values in series.groupby(series.index.year):
            previous = series.loc[series.index.year < year]
            base = float(previous.iloc[-1] if len(previous) else E.CAPITAL_CNY)
            rows.append({"book": book, "year": int(year), "return": float(values.iloc[-1] / base - 1)})
    return pd.DataFrame(rows)


def selection_summary(score: pd.DataFrame, fresh: pd.DataFrame,
                      veto_long: pd.DataFrame, veto_short: pd.DataFrame) -> pd.DataFrame:
    """每日候选数、两端入选与否决情况的研究期均值。"""
    none = pd.DataFrame(False, index=score.index, columns=score.columns)
    raw = tail_weights(score, none, none)
    picked_long, picked_short = raw.gt(0), raw.lt(0)
    kept_long, kept_short = fresh.gt(0), fresh.lt(0)
    return pd.DataFrame([{
        "days": int(len(score)),
        "trading_days_with_book": int(raw.ne(0).any(axis=1).sum()),
        "candidates_per_day": float(score.notna().sum(axis=1).mean()),
        "long_slots_per_day": float(picked_long.sum(axis=1).mean()),
        "short_slots_per_day": float(picked_short.sum(axis=1).mean()),
        "long_veto_share": float(1 - kept_long.to_numpy().sum() / max(picked_long.to_numpy().sum(), 1)),
        "short_veto_share": float(1 - kept_short.to_numpy().sum() / max(picked_short.to_numpy().sum(), 1)),
        "fresh_gross_per_day": float(fresh.abs().sum(axis=1).mean()),
    }])


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int,
        partition: str = "research") -> None:
    if partition != "research":
        raise RuntimeError("封存样本外评价需先另行声明评估方案；本策略目前只实现研究期回测")
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(panel["date"], "持仓基差研究日面板")
    C.assert_research_only(fee_path)
    C.assert_no_holdout_dates(fees["trading_date"], "持仓基差研究期手续费")
    start = pd.Timestamp(XS_POSITION["test_start"])
    bars = D.load_contracts(partition)
    market_root = C.RESEARCH_EXECUTION_PARAMETERS / "contracts"

    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(
        ["date", "symbol"], kind="mergesort",
    )
    panel = panel.assign(eligible=D.eligibility(panel))
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    names = sorted(panel.loc[panel["eligible"] & panel["date"].ge(start), "symbol"].unique())
    rank_root = C.daily_dir(partition) / "member_rank"
    C.assert_research_only(rank_root)
    names = [name for name in names if (rank_root / f"{name}_long.pkl").is_file()]
    if len(names) < MIN_SYMBOLS:
        raise RuntimeError("评价区间没有足够的流动性品种及会员排名")

    panel_by_symbol = dict(tuple(panel.groupby("symbol", sort=False)))
    bars_by_symbol = dict(tuple(bars.loc[bars["symbol"].isin(names)].groupby("symbol", sort=False)))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_symbol_inputs)(name, panel_by_symbol[name], bars_by_symbol[name], dates)
        for name in names
    )
    inputs = dict(zip(names, pieces))
    member = lambda col: pd.DataFrame({name: inputs[name][col] for name in names}).reindex(dates)
    market = lambda col: panel.pivot(index="date", columns="symbol", values=col).reindex(index=dates, columns=names)

    factors = {
        "FLOW_10": cumulative_flow(member("flow"), market("oi_total")),
        "K_NET_SHARE": member("K_NET_SHARE"),
        "BASIS_MOM_242": basis_momentum(market("ret1"), market("ret2")),
    }
    factors = {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in factors.items()}

    dominant = market("c1_contract")
    previous = dominant.shift()
    rolled = (dominant.ne(previous) & previous.notna()) | dominant.isna()
    eligible = market("eligible").astype("boolean").fillna(False).astype(bool)
    candidates = eligible & ~rolled.fillna(True)
    for frame in factors.values():
        candidates &= frame.notna()

    score = composite_score(factors, candidates)
    veto_long, veto_short = crowd_vetoes(member("crowd"))
    fresh = tail_weights(score, veto_long, veto_short)
    fresh.loc[fresh.index < start] = 0.0
    held = hold_cohorts(fresh).loc[start:]
    eligible = eligible.loc[start:]
    benchmark = eligible.astype(float).div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    contracts = dominant.shift(1).reindex(index=held.index, columns=names)
    ids = {contract for contract in contracts.to_numpy().ravel() if isinstance(contract, str)}
    quotes = E.load_market(
        bars, fees, ids, start, root=market_root, flat_margin=FLAT_MARGIN, partition=partition,
    )

    out.mkdir(parents=True, exist_ok=True)
    selection_summary(score.loc[start:], fresh.loc[start:], veto_long.loc[start:], veto_short.loc[start:]).to_csv(
        out / "selection_summary.csv", index=False,
    )
    rows, curves = [], {}
    for ticks in (1, 2):
        daily = {
            BENCHMARK: E.simulate(
                benchmark, contracts, quotes, start, neutral=False, slippage_ticks=ticks,
                margin_cap=MARGIN_CAP,
            )[0],
        }
        for book, multiplier in BOOKS.items():
            daily[book] = E.simulate(
                held * multiplier, contracts, quotes, start, neutral=False, slippage_ticks=ticks,
                name_cap=E.NAME_CAP, margin_cap=MARGIN_CAP,
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
        equity, out / "performance.png", strategy=MAIN_BOOK, benchmark=BENCHMARK, period_label="研究期",
    )
    view = summary.loc[summary.ticks.eq(1), [
        "book", "end_equity", "net_ann", "sharpe", "max_drawdown", "turnover", "gross_exposure", "margin_max",
    ]]
    print(view.round(4).to_string(index=False))
    print(f"回测摘要写入 {out / 'summary.csv'}；净值与回撤图写入 {figure}")

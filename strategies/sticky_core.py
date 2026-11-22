"""STICKY_CORE：基差动量、低偏度与粘性席位净多头占比的三因子截面组合，两端等权多空与 10 日重叠持有。

构造顺序
  因子   使用 t 日及之前的日线和会员排名：
           1. BASIS_MOM_242   (+): 主力与次主力 242 日复合收益之差（期限结构斜率动量）
           2. NEG_SKEW_242    (-): 主力 242 日（至少 200 日）日收益偏度取反（厌恶彩票/尾部风险）
           3. STICKY_NET_SHARE(+): 剔除高换手活跃席位（CROWD1 门槛）后的前 20 名净多头占比
  合成   在当日流动性池且三因子均有效的品种内分别计算百分位秩，等权平均。
  选样   有效候选至少 8 个品种时，取合成得分前 20% 做多、后 20% 做空，不设拥挤度硬门槛否决。
  持有   每日新鲜多空各分配 0.50 名义权重，经 10 个交易日重叠持有形成当期组合，次日开盘成交。
  对照   LONG_ONLY_BMK；当前唯一策略组合为 2X（上限 20%）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from strategies import Targets
import config as C
import data as D
from research.factors import market_inputs, member_tables, sticky_net_share
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
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20
MAIN_BOOK = "STICKY_CORE_2X"
EQUITY_CURVE_FILE = "equity_curves_d0_t1.csv"


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


def build_targets(panel: pd.DataFrame, bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    """构造策略和流动性池基准的日频权重。"""
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
    dominant = market("c1_contract")
    selection = selection_summary(score.loc[start:], fresh.loc[start:])
    return Targets(held, dominant, benchmark=benchmark, selection=selection)

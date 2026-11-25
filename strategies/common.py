"""策略共用的横截面选样、持有和权重构造。"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import data as D
from research.factors.member import member_tables, sticky_net_share
from research.member.complementarity import ranked_halves, sticky_quintiles
from research.factors.price import load_price_factors
from strategies import Targets

HORIZON = 10
TAIL_SHARE = 0.10
MIN_NAMES = 8
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20
RANKED_FACTORS = ("BASIS_MOM_242", "NEG_SKEW_242", "STICKY_NET_SHARE")


def composite_score(factors: dict[str, pd.DataFrame], candidates: pd.DataFrame,
                    names: tuple[str, ...]) -> pd.DataFrame:
    """仅在候选池内计算截面百分位秩，再等权平均。"""
    if not names:
        raise ValueError("合成因子不能为空")
    ranked = [factors[name].where(candidates).rank(axis=1, pct=True).mul(1.0 / len(names)) for name in names]
    return pd.concat(ranked).groupby(level=0).sum().where(candidates)


def equal_leg_weights(long_sel: pd.DataFrame, short_sel: pd.DataFrame) -> pd.DataFrame:
    """两端均有入选品种时，各分配 0.50 名义权重。"""
    wl = long_sel.astype(float).div(long_sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    ws = -short_sel.astype(float).div(short_sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 0.5
    return (wl + ws).where(long_sel.any(axis=1) & short_sel.any(axis=1), 0.0)


def tail_weights(score: pd.DataFrame, tail_share: float = TAIL_SHARE) -> pd.DataFrame:
    """候选至少 8 个时，按固定品种顺序处理同分，两端各取 tail_share。"""
    n = score.notna().sum(axis=1)
    k = np.ceil(n * tail_share)
    long = score.rank(axis=1, ascending=False, method="first").le(k, axis=0).where(n.ge(MIN_NAMES), False)
    short = score.rank(axis=1, method="first").le(k, axis=0).where(n.ge(MIN_NAMES), False)
    return equal_leg_weights(long, short)


def confirmed_weights(sticky: pd.DataFrame, residual: pd.DataFrame,
                      tail_share: float | None = None) -> pd.DataFrame:
    """在粘性净多两端分别取一致性残差最高和最低的一半。"""
    buckets = sticky_quintiles(sticky)
    if tail_share is None:
        long, _ = ranked_halves(residual, buckets.eq(5))
        _, short = ranked_halves(residual, buckets.eq(1))
    else:
        n = sticky.notna().sum(axis=1)
        k = np.ceil(n * tail_share)
        rank = sticky.rank(axis=1, method="first")
        eligible = buckets.notna().any(axis=1).to_numpy()[:, None]
        long_pool = rank.gt(n - 2 * k, axis=0) & eligible
        short_pool = rank.le(2 * k, axis=0) & eligible
        long_rank = residual.where(long_pool).rank(axis=1, ascending=False, method="first")
        short_rank = residual.where(short_pool).rank(axis=1, method="first")
        long = long_pool & long_rank.le(k, axis=0)
        short = short_pool & short_rank.le(k, axis=0)
    return equal_leg_weights(long, short)


def hold_cohorts(weights: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """每日新开权重持有 horizon 日（默认 10 日），汇总为当期目标权重。"""
    return sum(weights.shift(lag, fill_value=0.0) for lag in range(horizon)) / horizon


def _sticky_series(symbol: str, market: pd.DataFrame, partition: str) -> pd.Series:
    tables = member_tables(symbol, market.index, top20=True, partition=partition)
    return sticky_net_share(tables, market)["STICKY_NET_SHARE"]


def ranked_targets(panel: pd.DataFrame, start: pd.Timestamp, partition: str, jobs: int,
                   score_factors: tuple[str, ...], horizon: int = HORIZON) -> Targets:
    """在流动性池及三因子均有效的品种内构建等权截面排序目标。"""
    eligible, dominant = D.member_universe(panel, start, partition)
    if len(eligible.columns) < 10:
        raise RuntimeError("评价区间没有足够的流动性品种及会员排名")
    eligible = eligible.loc[start:]
    dates, names = eligible.index, eligible.columns
    volume = panel.pivot(index="date", columns="symbol", values="volume_total").reindex_like(eligible)
    oi = panel.pivot(index="date", columns="symbol", values="oi_total").reindex_like(eligible)
    sticky_cols = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_sticky_series)(name, pd.DataFrame({"volume": volume[name], "oi": oi[name]}), partition)
        for name in names
    )
    sticky = pd.DataFrame(dict(zip(names, sticky_cols)), index=dates)
    basis, neg_skew = load_price_factors(dates, names, partition=partition, panel=panel)
    factors = dict(zip(RANKED_FACTORS, (basis, neg_skew, sticky)))
    factors = {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in factors.items()}
    candidates = eligible.copy()
    for frame in factors.values():
        candidates &= frame.notna()

    score = composite_score(factors, candidates, score_factors)
    fresh = tail_weights(score)
    selection = pd.DataFrame([{
        "days": len(fresh),
        "trading_days_with_book": int(fresh.abs().sum(axis=1).gt(0).sum()),
        "candidates_per_day": score.notna().sum(axis=1).mean(),
        "long_slots_per_day": fresh.gt(0).sum(axis=1).mean(),
        "short_slots_per_day": fresh.lt(0).sum(axis=1).mean(),
        "fresh_gross_per_day": fresh.abs().sum(axis=1).mean(),
    }])
    return Targets(hold_cohorts(fresh, horizon), dominant, selection=selection)

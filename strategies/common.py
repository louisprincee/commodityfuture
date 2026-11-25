"""四套当前策略共用的截面选样、权重与重叠持有。"""
import numpy as np
import pandas as pd

from research.factors.member import load_sticky_inputs
from research.member.complementarity import ranked_halves, sticky_quintiles
from research.factors.price import load_price_factors
from strategies import Targets

HORIZON = 10
TAIL_SHARE = 0.10
MIN_NAMES = 8
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20
RANKED_FACTORS = ("BASIS_MOM_242", "NEG_SKEW_242", "STICKY_NET_SHARE")


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


def hold_cohorts(weights: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """每日新开权重持有 horizon 日（默认 10 日），汇总为当期目标权重。"""
    return sum(weights.shift(lag, fill_value=0.0) for lag in range(horizon)) / horizon


def composite_score(factors: dict[str, pd.DataFrame], candidates: pd.DataFrame,
                    names: tuple[str, ...]) -> pd.DataFrame:
    """仅在候选池内计算截面百分位秩，再等权平均。"""
    if not names:
        raise ValueError("合成因子不能为空")
    ranked = [factors[name].where(candidates).rank(axis=1, pct=True).mul(1.0 / len(names)) for name in names]
    return pd.concat(ranked).groupby(level=0).sum().where(candidates)


def confirmed_weights(sticky: pd.DataFrame, residual: pd.DataFrame,
                      tail_share: float | None = None, min_names: int = 20) -> pd.DataFrame:
    """在粘性净多两端分别取二次排序因子最高和最低的一半。"""
    if tail_share is None:
        buckets = sticky_quintiles(sticky)
        long, _ = ranked_halves(residual, buckets.eq(5))
        _, short = ranked_halves(residual, buckets.eq(1))
        return equal_leg_weights(long, short)
    common = np.isfinite(sticky) & np.isfinite(residual)
    s, r = sticky.where(common), residual.where(common)
    n = common.sum(axis=1)
    k = np.ceil(n * tail_share)
    rank = s.rank(axis=1, method="first")
    long_pool = rank.gt(n - 2 * k, axis=0)
    short_pool = rank.le(2 * k, axis=0)
    long = long_pool & r.where(long_pool).rank(axis=1, ascending=False, method="first").le(k, axis=0)
    short = short_pool & r.where(short_pool).rank(axis=1, method="first").le(k, axis=0)
    valid = n.ge(min_names)
    return equal_leg_weights(long.where(valid, False), short.where(valid, False))


def ranked_targets(panel: pd.DataFrame, start: pd.Timestamp, partition: str, jobs: int,
                   score_factors: tuple[str, ...], horizon: int = HORIZON) -> Targets:
    """在流动性池及三因子均有效的品种内构建等权截面排序目标。"""
    sticky, dominant = load_sticky_inputs(panel, start, partition, jobs)
    if len(sticky.columns) < 10:
        raise RuntimeError("评价区间没有足够的流动性品种及会员排名")
    dates, names = sticky.index, sticky.columns
    basis, neg_skew = load_price_factors(dates, names, partition=partition, panel=panel)
    factors = dict(zip(RANKED_FACTORS, (basis, neg_skew, sticky)))
    factors = {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in factors.items()}
    candidates = pd.DataFrame(True, index=dates, columns=names)
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

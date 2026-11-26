"""会员持仓与价格复合策略共用的截面百分位复合与目标权重构造。"""
import numpy as np
import pandas as pd

from research.factors.member import load_sticky_inputs
from research.factors.price import load_price_factors
from strategies import Targets
from strategies.common import HORIZON, hold_cohorts, tail_weights

RANKED_FACTORS = ("BASIS_MOM_242", "NEG_SKEW_242", "STICKY_NET_SHARE")


def composite_score(factors: dict[str, pd.DataFrame], candidates: pd.DataFrame,
                    names: tuple[str, ...]) -> pd.DataFrame:
    """仅在候选池内计算截面百分位秩，再等权平均。"""
    if not names:
        raise ValueError("合成因子不能为空")
    ranked = [factors[name].where(candidates).rank(axis=1, pct=True).mul(1.0 / len(names)) for name in names]
    return pd.concat(ranked).groupby(level=0).sum().where(candidates)


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

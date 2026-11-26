"""BASIS_MOM_242、NEG_SKEW_242 与 STICKY_NET_SHARE 等权百分位排名，持有天数见 HOLDING_DAYS。"""
import numpy as np
import pandas as pd

from research.factors.member import load_sticky_inputs
from research.factors.price import load_price_factors
from strategies import Targets
from strategies.common import hold_cohorts, tail_weights

# 非 10 日的结果写入策略目录下 holding_<天数>/，不覆盖默认结果。
HOLDING_DAYS = 10

STRATEGY_ID = "sticky_basis_skew"
MAIN_BOOK = "STICKY_BASIS_SKEW"
LABEL = "粘性净多 + 基差动量 + 负偏度"
EQUITY_CURVE_FILE = "equity_curves.csv"

RANKED_FACTORS = ("BASIS_MOM_242", "NEG_SKEW_242", "STICKY_NET_SHARE")


def composite_score(factors: dict[str, pd.DataFrame], candidates: pd.DataFrame,
                    names: tuple[str, ...]) -> pd.DataFrame:
    """仅在候选池内计算截面百分位秩，再等权平均。"""
    if not names:
        raise ValueError("合成因子不能为空")
    ranked = [factors[name].where(candidates).rank(axis=1, pct=True).mul(1.0 / len(names)) for name in names]
    return pd.concat(ranked).groupby(level=0).sum().where(candidates)


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
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

    score = composite_score(factors, candidates, RANKED_FACTORS)
    fresh = tail_weights(score)
    selection = pd.DataFrame([{
        "days": len(fresh),
        "trading_days_with_book": int(fresh.abs().sum(axis=1).gt(0).sum()),
        "candidates_per_day": score.notna().sum(axis=1).mean(),
        "long_slots_per_day": fresh.gt(0).sum(axis=1).mean(),
        "short_slots_per_day": fresh.lt(0).sum(axis=1).mean(),
        "fresh_gross_per_day": fresh.abs().sum(axis=1).mean(),
    }])
    return Targets(hold_cohorts(fresh, HOLDING_DAYS), dominant, selection=selection)

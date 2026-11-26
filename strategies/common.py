"""会员持仓与复合策略共用的截面选样、权重与重叠持有工具。"""
import numpy as np
import pandas as pd

HORIZON = 10
TAIL_SHARE = 0.10
MIN_NAMES = 8
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20


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
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("持有天数必须是正整数")
    return sum(weights.shift(lag, fill_value=0.0) for lag in range(horizon)) / horizon

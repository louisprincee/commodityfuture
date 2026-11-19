"""信号到持仓的通用规则：重叠持有、分位事件、择时方向与预算等权分配。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.settings import CROWDING as CR

MIN_SLOTS = 10


def hold_cohorts(weights: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """重叠持有：t 日持仓为最近 H 个日度组合之和 / H；首个预测日之前的组合视为现金。"""
    return weights.fillna(0.0).rolling(horizon, min_periods=1).sum() / horizon


def short_shock(values: pd.Series, window: int = 250, minimum: int = 60) -> tuple[pd.Series, pd.Series]:
    """按此前窗口的分位开仓，并用上下分位间距衡量异常强度。"""
    past = values.shift(1).rolling(window, min_periods=minimum)
    hi, lo = past.quantile(CR["q_high"]), past.quantile(CR["q_low"])
    spread = hi - lo
    direction = values.gt(hi).astype(float) - values.lt(lo).astype(float)
    distance = values - (hi + lo) / 2
    strength = distance.div(spread.where(spread.gt(0)))
    flat = spread.eq(0) & distance.notna()
    strength = strength.mask(flat & distance.eq(0), 0)
    strength = strength.mask(flat & distance.gt(0), np.inf).mask(flat & distance.lt(0), -np.inf)
    return direction, strength


def timing_directions(z: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    """同向合力且越过自身一倍标准差才入场，回到零或数据缺失退出。"""
    z_values = z.to_numpy()
    levels = level.to_numpy()
    held = np.zeros(z.shape[1], dtype=np.int8)
    result = np.zeros(z.shape, dtype=np.int8)
    for i, (score, raw) in enumerate(zip(z_values, levels)):
        long = (score > 0) & (raw > 0) & ((score >= 1) | (held > 0))
        short = (score < 0) & (raw < 0) & ((score <= -1) | (held < 0))
        held = long.astype(np.int8) - short.astype(np.int8)
        result[i] = held
    return pd.DataFrame(result, index=z.index, columns=z.columns)


def active_weights(direction: pd.DataFrame, min_slots: int = MIN_SLOTS) -> pd.DataFrame:
    """按 max(当日激活品种数, min_slots) 等权；激活数不足时未用预算留在现金。"""
    active = direction.ne(0).sum(axis=1).clip(lower=min_slots)
    return direction.div(active, axis=0).fillna(0.0)

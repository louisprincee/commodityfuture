"""信号到持仓的通用规则：截面权重、重叠持有、两端缓冲、分位事件与择时方向、预算分配与对冲。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.settings import CROWDING as CR

MIN_SLOTS = 10
VOL_WINDOW = 60
VOL_MIN_PERIODS = 20
WEIGHT_FLOOR = 0.20


# ---------------------------------------------------------------- 截面权重与重叠持有

def score_to_weights(score: pd.DataFrame) -> pd.DataFrame:
    """score: 日期 × 品种（缺失 = 不在池内）。中位数以上做多、以下做空，腿内按 |得分-中位| 加权，每腿总敞口 1。"""
    q = score.rank(axis=1, pct=True)
    s = q.sub(q.mean(axis=1), axis=0)
    long = s.clip(lower=0)
    short = s.clip(upper=0)
    w = long.div(long.sum(axis=1).replace(0, np.nan), axis=0) \
        + short.div(short.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)


def hold_cohorts(weights: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """重叠持有：t 日持仓为最近 H 个日度组合之和 / H；首个预测日之前的组合视为现金。"""
    return weights.fillna(0.0).rolling(horizon, min_periods=1).sum() / horizon


def buffered_tails(score: pd.DataFrame, min_names: int, buffered: bool,
                   entry_gate: pd.DataFrame | None = None) -> pd.DataFrame:
    """两端各约 30% 入场。缓冲版穿过中位数退出。缺失值退出。"""
    ranks = score.rank(axis=1)
    counts = score.count(axis=1)
    tail_count = (counts * 0.30).apply(np.floor).clip(lower=1)
    enter_long = ranks.gt(counts - tail_count, axis=0)
    enter_short = ranks.le(tail_count, axis=0)
    if entry_gate is not None:
        enter_long &= entry_gate.ge(1)
        enter_short &= entry_gate.le(-1)
    enter_long = enter_long.to_numpy()
    enter_short = enter_short.to_numpy()
    centered = ranks.sub((counts + 1) / 2, axis=0).to_numpy()
    result = np.zeros(ranks.shape)
    status = np.zeros(ranks.shape[1])
    for i, count in enumerate(counts):
        hold_long = (status > 0) & (centered[i] > 0) & buffered
        hold_short = (status < 0) & (centered[i] < 0) & buffered
        long = enter_long[i] | hold_long
        short = enter_short[i] | hold_short
        if count >= min_names and long.any() and short.any():
            status = long.astype(float) - short.astype(float)
            result[i] = long / long.sum() - short / short.sum()
        else:
            status.fill(0)
    return pd.DataFrame(result, index=score.index, columns=score.columns)


# ---------------------------------------------------------------- 分位事件与择时方向

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
    """同向合力或流量且越过自身一倍标准差才入场，回到零或数据缺失退出。"""
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


# ---------------------------------------------------------------- 预算分配、对冲与配权

def active_weights(direction: pd.DataFrame, min_slots: int = MIN_SLOTS) -> pd.DataFrame:
    """在触发品种间等权分配 1 倍名义预算，激活品种不足 min_slots 时限制单品种名义占比。"""
    active = direction.ne(0).sum(axis=1).clip(lower=min_slots)
    return direction.div(active, axis=0).fillna(0.0)


def hedge_against_benchmark(weights: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """在组合层用当日流动性商品等权基准对冲净敞口，合并净头寸后执行。"""
    net = weights.sum(axis=1)
    n_elig = eligible.sum(axis=1).replace(0, np.nan)
    hedge = eligible.astype(float).div(n_elig, axis=0).mul(net, axis=0).fillna(0.0)
    return weights.sub(hedge, fill_value=0.0)


def inverse_vol_weights(ret_core: pd.Series, ret_sat: pd.Series,
                        window: int = VOL_WINDOW, min_periods: int = VOL_MIN_PERIODS,
                        floor: float = WEIGHT_FLOOR) -> tuple[pd.Series, pd.Series]:
    """严格因果的滚动波动率倒数配权（仅用 t-1 及以前日收益，前 min_periods 日等权）。"""
    vol_c = ret_core.shift(1).rolling(window, min_periods=min_periods).std()
    vol_s = ret_sat.shift(1).rolling(window, min_periods=min_periods).std()
    valid = vol_c.gt(0) & vol_s.gt(0)
    inv_c = (1.0 / vol_c.where(valid)).fillna(0.0)
    inv_s = (1.0 / vol_s.where(valid)).fillna(0.0)
    denom = inv_c + inv_s
    w_core = inv_c.div(denom.where(denom > 0)).clip(floor, 1.0 - floor).fillna(0.5)
    return w_core, 1.0 - w_core

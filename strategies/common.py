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


def buffered_tail_weights(score: pd.DataFrame, enter_share: float = TAIL_SHARE,
                          exit_share: float = 0.20, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """滞后排名缓冲带：进入前/后 enter_share 建仓，维持在前/后 exit_share 内则留仓。"""
    if not (0 < enter_share <= exit_share < 0.5):
        raise ValueError("enter_share 须小于等于 exit_share 且均小于 0.5")
    n = score.notna().sum(axis=1)
    r_desc = score.rank(axis=1, ascending=False, method="first")
    r_asc = score.rank(axis=1, method="first")
    long_sel = pd.DataFrame(False, index=score.index, columns=score.columns)
    short_sel = pd.DataFrame(False, index=score.index, columns=score.columns)
    prev_long = pd.Series(False, index=score.columns)
    prev_short = pd.Series(False, index=score.columns)
    for i in range(len(score.index)):
        cnt = int(n.iloc[i])
        if cnt < min_names:
            prev_long = pd.Series(False, index=score.columns)
            prev_short = pd.Series(False, index=score.columns)
            continue
        k_in, k_out = np.ceil(cnt * enter_share), np.ceil(cnt * exit_share)
        rd, ra = r_desc.iloc[i], r_asc.iloc[i]
        cur_long = rd.le(k_in) | (prev_long & rd.le(k_out))
        cur_short = ra.le(k_in) | (prev_short & ra.le(k_out))
        cur_long = cur_long.fillna(False) & ~ra.le(k_in).fillna(False)
        cur_short = cur_short.fillna(False) & ~rd.le(k_in).fillna(False)
        long_sel.iloc[i] = cur_long
        short_sel.iloc[i] = cur_short
        prev_long, prev_short = cur_long, cur_short
    return equal_leg_weights(long_sel, short_sel)


def hold_cohorts(weights: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """每日新开权重持有 horizon 日（默认 10 日），汇总为当期目标权重。"""
    return sum(weights.shift(lag, fill_value=0.0) for lag in range(horizon)) / horizon


def confirmed_weights(sticky: pd.DataFrame, residual: pd.DataFrame,
                      tail_share: float = TAIL_SHARE, min_names: int = 20) -> pd.DataFrame:
    """在粘性净多两端分别取二次排序因子最高和最低的一半。"""
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

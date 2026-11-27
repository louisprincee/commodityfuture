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


def gate_weights(score: pd.DataFrame, flow: pd.DataFrame, *, veto: bool, buffer: bool,
                 veto_share: float = 0.20, band_share: float = 0.20, horizon: int = HORIZON) -> pd.DataFrame:
    """身份尾部选样，持仓流只做开仓否决；可选按身份宽带保留即将到期批次的在任品种。

    veto 与 buffer 都关闭时与 tail_weights 完全一致。"""
    common = np.isfinite(score)
    n = common.sum(axis=1).to_numpy()
    rank_long = score.where(common).rank(axis=1, ascending=False, method="first").to_numpy()
    rank_short = score.where(common).rank(axis=1, method="first").to_numpy()
    k = np.ceil(n * TAIL_SHARE)
    band = np.ceil(n * band_share)
    usable = flow.where(common & np.isfinite(flow))
    percentile = usable.rank(axis=1, pct=True)
    percentile = percentile.where(usable.notna().sum(axis=1).ge(MIN_NAMES), axis=0).to_numpy()
    bad_long = (percentile <= veto_share) if veto else np.zeros_like(rank_long, dtype=bool)
    bad_short = (percentile >= 1 - veto_share) if veto else np.zeros_like(rank_long, dtype=bool)
    long = np.zeros(rank_long.shape, dtype=bool)
    short = np.zeros(rank_long.shape, dtype=bool)
    for i in range(len(score)):
        if n[i] < MIN_NAMES:
            continue
        for side, rank, bad in ((long, rank_long[i], bad_long[i]), (short, rank_short[i], bad_short[i])):
            kept = np.zeros(len(rank), dtype=bool)
            if buffer and i >= horizon:
                kept = side[i - horizon] & (rank <= band[i]) & ~bad
            fill = (rank <= k[i]) & ~bad & ~kept
            room = int(k[i]) - int(kept.sum())
            order = np.argsort(np.where(fill, rank, np.inf), kind="stable")[:max(room, 0)]
            chosen = kept.copy()
            chosen[order[fill[order]]] = True
            side[i] = chosen
    frame = lambda values: pd.DataFrame(values, index=score.index, columns=score.columns)
    return equal_leg_weights(frame(long), frame(short))

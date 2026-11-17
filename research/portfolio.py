"""截面得分 → 多空权重 → 日收益、换手与绩效指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

COST_BPS = 5.0
TD = C.TRADING_DAYS_PER_YEAR


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


def normalize_legs(weights: pd.DataFrame) -> pd.DataFrame:
    """多头部分缩放到总敞口 1，空头部分缩放到 -1。"""
    long = weights.clip(lower=0)
    short = weights.clip(upper=0)
    out = long.div(long.sum(axis=1).replace(0, np.nan), axis=0) \
        + short.div(short.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return out.fillna(0.0)


def backtest(weights: pd.DataFrame, returns: pd.DataFrame, rolls: pd.DataFrame,
             delay: int = 0, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """weights 为 t 日收盘目标权重，持有到 t+1+delay 日收盘取得 returns 的该日收益。

    换手 = Σ|Δw| + 换月日 2|w|（平旧主力、开新主力）；成本 = cost_bps × 换手。
    """
    dates = returns.index
    w = weights.reindex(index=dates, columns=returns.columns).fillna(0.0)
    held = w.shift(1 + delay).fillna(0.0)          # 当日收益对应的持仓
    r = returns.fillna(0.0)
    gross = (held * r).sum(axis=1)
    trade = held.diff().abs().sum(axis=1)
    trade.iloc[0] = held.iloc[0].abs().sum()
    roll = (held.abs() * rolls.reindex(index=dates, columns=returns.columns).fillna(False).astype(float)).sum(axis=1) * 2
    turnover = trade + roll
    net = gross - turnover * cost_bps / 1e4
    active = held.abs().sum(axis=1) > 0
    out = pd.DataFrame({"gross": gross, "net": net, "turnover": turnover, "roll_turnover": roll,
                        "gmv": held.abs().sum(axis=1), "nmv": held.sum(axis=1)})
    first = active.idxmax() if active.any() else dates[-1]
    return out.loc[first:]


def metrics(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Exhibit 5 口径：年化收益为日均 × 242，夏普不扣无风险利率（期货多空无资金占用）。"""
    r = r.dropna()
    ann = r.mean() * TD
    vol = r.std() * np.sqrt(TD)
    down = np.sqrt((r.clip(upper=0) ** 2).mean()) * np.sqrt(TD)
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    out = {
        "ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
        "sortino": ann / down if down > 0 else np.nan, "downside_vol": down,
        "var95": -np.quantile(r, 0.05), "success_rate": (r > 0).mean(), "max_drawdown": dd,
        "t_stat": r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan,
        "days": len(r),
    }
    if turnover is not None:
        out["daily_turnover"] = turnover.reindex(r.index).mean()
    return out


def exposures(weights: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """每日按板块的多头、空头与净敞口。"""
    sec = sector.reindex(weights.columns)
    long = weights.clip(lower=0).T.groupby(sec.values).sum().T
    short = weights.clip(upper=0).T.groupby(sec.values).sum().T
    net = weights.T.groupby(sec.values).sum().T
    return pd.concat({"long": long, "short": short, "net": net}, axis=1)

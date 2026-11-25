"""交易日网格上的 HAC 推断和固定假设族校正。"""
from __future__ import annotations

from math import erfc, sqrt
import numpy as np
import pandas as pd

def bh(p: pd.Series) -> pd.Series:
    ordered = p.dropna().sort_values()
    adjusted = (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return adjusted.reindex(p.index)

def rank_corr(score: pd.DataFrame, outcome: pd.DataFrame, min_names: int) -> tuple[pd.Series, pd.Series]:
    paired = np.isfinite(score) & np.isfinite(outcome)
    left = score.where(paired).rank(axis=1)
    right = outcome.where(paired).rank(axis=1)
    left = left.sub(left.mean(axis=1), axis=0)
    right = right.sub(right.mean(axis=1), axis=0)
    magnitude = np.sqrt(left.pow(2).sum(axis=1) * right.pow(2).sum(axis=1))
    corr = (left * right).sum(axis=1) / magnitude.where(magnitude.gt(0))
    count = paired.sum(axis=1)
    return corr.where(count.ge(min_names)), count.where(corr.notna() & count.ge(min_names))


def fixed_bh(p: pd.Series) -> pd.Series:
    """不可估计假设按1占据预先声明的名额，显示值仍缺失。"""
    return bh(p.fillna(1)).where(p.notna())


def rank_residual(score: pd.DataFrame, control: pd.DataFrame, min_names: int = 8,
                  *, pct: bool = False) -> pd.DataFrame:
    """同日共同品种的截面秩线性残差；控制量无变异时不可估计。"""
    common = np.isfinite(score) & np.isfinite(control)
    x = score.where(common).rank(axis=1, pct=pct)
    z = control.where(common).rank(axis=1, pct=pct)
    x, z = x.sub(x.mean(axis=1), axis=0), z.sub(z.mean(axis=1), axis=0)
    beta = (x * z).sum(axis=1) / z.pow(2).sum(axis=1).replace(0, np.nan)
    return x.sub(z.mul(beta, axis=0)).where(common.sum(axis=1).ge(min_names), axis=0)


def hac_variance(values: np.ndarray, lags: int) -> float:
    variance = float(values @ values)
    for lag in range(1, min(lags, len(values) - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(values[lag:] @ values[:-lag])
    return max(0., variance)


def mean_inference(series: pd.Series, lags: int, min_days: int = 120) -> dict:
    x = series.where(np.isfinite(series))
    n, mean = int(x.count()), x.mean()
    se = sqrt(hac_variance((x - mean).fillna(0).to_numpy(), lags)) / n if n >= min_days else np.nan
    t = mean / se if np.isfinite(se) and se > 0 else np.nan
    return dict(mean_ic=mean, hac_se=se, hac_t=t, days=n,
                ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se,
                p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan)


def hac_t(series: pd.Series, lags: int = 20, min_days: int = 120) -> float:
    t = mean_inference(series, lags, min_days)["hac_t"]
    return float(t) if np.isfinite(t) else np.nan


def rank_time_series(score: pd.Series, outcome: pd.Series, lags: int, min_days: int = 120) -> dict:
    """Spearman 点估计；标准化秩回归的 HAC 推断，不假设品种间独立。

    推断是对样本秩转换后回归的渐近近似；原交易日缺口保留在 HAC 网格上。
    """
    data = pd.concat([score.rename('x'), outcome.rename('y')], axis=1)
    paired = np.isfinite(data).all(axis=1)
    n = int(paired.sum())
    result = dict(ic=np.nan, n_pairs=n, hac_se=np.nan, hac_t=np.nan,
                  p=np.nan, ci_low=np.nan, ci_high=np.nan)
    if n < min_days:
        return result
    ranks = data.where(paired, axis=0).rank()
    std = ranks.std(ddof=0)
    if not std.gt(0).all():
        return result
    z = (ranks - ranks.mean()) / std
    rho = float((z.x * z.y).sum() / n)
    rho = float(np.clip(rho, -1., 1.))
    moment = (z.x * (z.y - rho * z.x)).fillna(0).to_numpy()
    se = sqrt(hac_variance(moment, lags)) / n
    t = rho / se if se > 1e-14 else np.nan
    result.update(ic=rho, hac_se=se, hac_t=t,
                  ci_low=max(-1., rho - 1.96 * se), ci_high=min(1., rho + 1.96 * se),
                  p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan)
    return result

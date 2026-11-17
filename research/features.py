"""论文九类特征 × 14 个回看窗口，及前瞻收益标签。全部只用 t 日收盘及以前的数据。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C

PROTOCOL = json.loads((C.CONFIG_DIR / "research" / "replication.json").read_text(encoding="utf-8"))
FAMILIES = PROTOCOL["features"]["families"]
LOOKBACKS = PROTOCOL["features"]["lookbacks"]
HORIZONS = PROTOCOL["targets"]["horizons"]
UNIVERSE = PROTOCOL["universe"]
FEATURES = [f"{fam}{L}" for fam in FAMILIES for L in LOOKBACKS]


def eligibility(panel: pd.DataFrame) -> pd.Series:
    """时点可得的品种池：过去 242 日成交额中位数、上市天数、当日主力有成交。"""
    g = panel.groupby("symbol", sort=False)
    med = g["turnover_total"].transform(
        lambda s: s.rolling(UNIVERSE["turnover_window_days"],
                            min_periods=UNIVERSE["turnover_min_periods"]).median())
    age = g.cumcount() + 1
    ok = (med >= UNIVERSE["min_median_daily_turnover_cny"]) & (age >= UNIVERSE["min_history_days"])
    if UNIVERSE["require_dominant_volume_positive"]:
        ok &= panel["c1_volume"] > 0
    ok &= panel["ret1"].notna()
    ok &= ~panel["symbol"].isin(UNIVERSE["excluded_symbols"])
    return ok


def _roll(s: pd.Series, L: int, how: str) -> pd.Series:
    r = s.rolling(L, min_periods=L)
    return getattr(r, how)()


def symbol_features(df: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """单品种时间序列特征。df 按日期排序，market 为同日期的等权市场收益。"""
    r = df["ret1"].fillna(0.0)
    lr = np.log1p(r)
    lr2 = np.log1p(df["ret2"])
    price = np.exp(lr.cumsum())
    oi = df["oi_total"].where(df["oi_total"] > 0)
    m = market.reindex(df.index).fillna(0.0)
    gain, loss = r.clip(lower=0), (-r).clip(lower=0)
    out = {}
    for L in LOOKBACKS:
        sum_lr = _roll(lr, L, "sum")
        out[f"MOM{L}"] = np.expm1(sum_lr)
        out[f"TSMOM{L}"] = sum_lr / (_roll(lr, L, "std") * np.sqrt(L))
        out[f"CARRY{L}"] = df["roll_yield_ann"].rolling(L, min_periods=L).mean()
        out[f"BASISMOM{L}"] = sum_lr - lr2.rolling(L, min_periods=L).sum()
        out[f"SKEW{L}"] = _roll(r, L, "skew")
        var_i = _roll(r, L, "var")
        cov = r.rolling(L, min_periods=L).cov(m)
        var_m = _roll(m, L, "var")
        out[f"IVOL{L}"] = np.sqrt((var_i - cov ** 2 / var_m).clip(lower=0))
        out[f"MA{L}"] = price / _roll(price, L, "mean") - 1.0
        up, dn = _roll(gain, L, "mean"), _roll(loss, L, "mean")
        out[f"RSI{L}"] = 100.0 * up / (up + dn)
        out[f"MAV{L}"] = np.log(oi / oi.rolling(L, min_periods=L).mean())
    feats = pd.DataFrame(out, index=df.index)
    # 收益窗口跨越首日：首日 ret1 缺失被置 0，故窗口需完全落在上市后
    return feats[FEATURES]


def forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """t 日收盘到 t+H 日收盘的主力连续累计收益；不足 H 日为缺失。"""
    lr = np.log1p(panel["ret1"])
    out = {}
    for H in HORIZONS:
        fwd = panel.assign(lr=lr).groupby("symbol", sort=False)["lr"].transform(
            lambda s: s[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1))
        out[f"fwd{H}"] = np.expm1(fwd)
    return pd.DataFrame(out, index=panel.index)


def cs_rank(frame: pd.DataFrame, dates: pd.Series, mask: pd.Series) -> pd.DataFrame:
    """在每日可交易品种内做截面百分位；池外置缺失。"""
    sub = frame.where(mask, np.nan)
    return sub.groupby(dates.values).rank(pct=True)


def build_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """日面板 → (date, symbol) 数据集：原始特征、截面百分位特征、标签、池标记。"""
    panel = panel.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    panel["eligible"] = eligibility(panel)
    market = panel[panel["eligible"]].groupby("date")["ret1"].mean()
    feats = []
    for _, df in panel.groupby("symbol", sort=False):
        idx = df.set_index("date")
        f = symbol_features(idx, market)
        f.index = df.index
        feats.append(f)
    raw = pd.concat(feats).sort_index()
    fwd = forward_returns(panel)
    base = panel[["date", "symbol", "sector", "eligible", "ret1", "roll", "c1_contract"]]
    ranked = cs_rank(raw, panel["date"], panel["eligible"])
    target = cs_rank(fwd, panel["date"], panel["eligible"])
    target.columns = [f"y{H}" for H in HORIZONS]
    ds = pd.concat([base, ranked, fwd, target], axis=1)
    return ds.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True), raw

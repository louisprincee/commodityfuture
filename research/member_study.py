"""会员持仓指标：兴业证券拥挤度 / 一致性，开源证券蜘蛛网。

指标只用 t 日收盘后已经公布的会员排名。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data as D
from research.settings import CONSISTENCY as CS
from research.settings import CROWDING as CR


# ---------------------------------------------------------------- 输入

def member_tables(symbol: str, dates: pd.DatetimeIndex, *, top20=False) -> dict[str, pd.DataFrame]:
    """日期 × 会员宽表：多、空持仓，成交量，多、空持仓变化（交易所披露的较前一日变化）。"""
    ranks = D.load_member_rank(symbol, "research")
    C.assert_no_holdout_dates(ranks["trading_date"], f"{symbol} 会员排名")
    if top20:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    members = sorted(ranks["member_name"].unique())
    out = {}
    for side, key in (("long", "L"), ("short", "S"), ("volume", "V")):
        sub = ranks[ranks["side"] == side]
        out[key] = sub.pivot(index="trading_date", columns="member_name", values="volume") \
            .reindex(index=dates, columns=members)
        out[f"d{key}"] = sub.pivot(index="trading_date", columns="member_name", values="volume_change") \
            .reindex(index=dates, columns=members)
    out["ranked"] = pd.Series(dates.isin(ranks["trading_date"].unique()), index=dates)
    return out


def market_inputs(panel: pd.DataFrame, contracts: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """单品种日度市场输入：主力合约、开盘对开盘收益、收盘对收盘收益、单边总成交量与总持仓。"""
    p = panel[panel["symbol"] == symbol].set_index("date").sort_index()
    sub = contracts[contracts["symbol"] == symbol]
    px = sub.set_index(["contract", "date"])
    next_open = sub.groupby("contract")["open"].shift(-1)
    next_open.index = px.index
    key = pd.MultiIndex.from_arrays([p["c1_contract"].values, p.index])
    open_today = px["open"].reindex(key).to_numpy(dtype=float)
    out = pd.DataFrame({
        "c1_contract": p["c1_contract"],
        # 当日主力合约：今日开盘 → 下一交易日开盘
        "ret_oo": next_open.reindex(key).to_numpy(dtype=float) / open_today - 1.0,
        "ret_cc": p["ret1"],
        "volume": p["volume_total"].astype(float),
        "oi": p["oi_total"].astype(float),
    }, index=p.index)
    out.loc[~(open_today > 0), "ret_oo"] = np.nan
    out["d_oi"] = out["oi"].diff()
    out["roll"] = out["c1_contract"].ne(out["c1_contract"].shift()) & out["c1_contract"].shift().notna()
    return out


# ---------------------------------------------------------------- 兴业：拥挤度

def crowding_indicators(t: dict[str, pd.DataFrame], mkt: pd.DataFrame) -> pd.DataFrame:
    """三类投机席位净多占比。会员成交量逐边计数，阈值换算为同口径的全市场平均。"""
    L, S, V, dL, dS = t["L"], t["S"], t["V"], t["dL"], t["dS"]
    complete = L.notna() & S.notna() & V.notna() & (L + S > 0)
    net, gross = L - S, L + S

    def share(spec: pd.DataFrame) -> pd.Series:
        num = net.where(spec).sum(axis=1, min_count=1)
        den = gross.where(spec).sum(axis=1, min_count=1)
        return num / den.where(den > 0)

    th1 = mkt["volume"] / mkt["oi"]
    ratio1 = V / gross
    c1 = share(complete & ratio1.ge(th1, axis=0))

    th2 = mkt["d_oi"].abs() / mkt["volume"]
    ratio2 = (dL + dS).abs() / V.where(V > 0)
    c2 = share(complete & ratio2.ge(th2, axis=0))

    w = CR["crowd3_window"]
    flow = (dL + dS).where(complete)
    vol = V.where(complete)
    appear = complete.astype(float).rolling(w, min_periods=w).sum()
    ratio3 = flow.rolling(w, min_periods=1).sum().abs() / vol.rolling(w, min_periods=1).sum()
    th3 = mkt["d_oi"].rolling(w, min_periods=w).sum().abs() / mkt["volume"].rolling(w, min_periods=w).sum()
    c3 = share(complete & appear.ge(CR["crowd3_min_appear"]) & ratio3.ge(th3, axis=0))
    return pd.DataFrame({"CROWD1": c1, "CROWD2": c2, "CROWD3": c3})


def bounds(x: pd.Series, window: int, q_high: float, q_low: float) -> tuple[pd.Series, pd.Series]:
    """滚动分位阈值，只用 t-N..t-1，不含当日。"""
    past = x.shift(1).rolling(window, min_periods=window)
    return past.quantile(q_high), past.quantile(q_low)


def crowding_signal(x: pd.Series, ret_cc: pd.Series, window: int) -> pd.Series:
    """商品采用反转：多头极端拥挤且连续收涨做空，空头极端拥挤且连续收跌做多。NaN 表示不可计算。"""
    hi, lo = bounds(x, window, CR["q_high"], CR["q_low"])
    k = CR["confirm_days"]
    up = (ret_cc > 0).astype(float).rolling(k, min_periods=k).sum().eq(k)
    down = (ret_cc < 0).astype(float).rolling(k, min_periods=k).sum().eq(k)
    upper, lower = x.ge(hi), x.le(lo)
    confirmed = ret_cc.notna().rolling(k, min_periods=k).sum().eq(k)
    sig = pd.Series(0.0, index=x.index)
    sig[upper & up] = -1.0
    sig[lower & down] = 1.0
    if CR["direction"] == "momentum":
        sig = -sig
    return sig.where(x.notna() & hi.notna() & lo.notna() & (~(upper | lower) | confirmed))


# ---------------------------------------------------------------- 兴业：一致性

def _mean_pair_corr(block: np.ndarray) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(block.T)
    iu = np.triu_indices(block.shape[1], 1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def _slope(series: np.ndarray) -> float:
    tau = np.arange(len(series), dtype=float)
    tau -= tau.mean()
    return float((tau * (series - series.mean())).sum() / (tau * tau).sum())


def imputed_net(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """会员净头寸。只在一侧上榜时，未上榜一侧取当日该榜末位持仓的一半（真实值介于 0 与末位之间）。

    商品前 20 名多空两榜同时在榜的会员每日仅约 10–13 家，要求两榜连续在榜会使样本池不足，
    故一致性指标改用此有界填补；两榜均未上榜的会员当日净头寸缺失。
    """
    L, S = t["L"], t["S"]
    fill_l = L.min(axis=1) / 2.0
    fill_s = S.min(axis=1) / 2.0
    any_side = L.notna() | S.notna()
    L_f = L.apply(lambda col: col.fillna(fill_l))
    S_f = S.apply(lambda col: col.fillna(fill_s))
    return (L_f - S_f).where(any_side)


def consistency_indicators(t: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    """窗口内每日至少一侧上榜的会员为样本池；按当日净头寸取空头、多头阵营各 N 家。"""
    return consistency_from_net(imputed_net(t), window)


def consistency_from_net(net_positions: pd.DataFrame, window: int) -> pd.DataFrame:
    """用窗口内净头寸完整的会员计算阵营相关性及方向斜率。"""
    net = net_positions.to_numpy(dtype=float)
    n = CS["camp_size"]
    rows = np.full((net.shape[0], 4), np.nan)
    for i in range(window - 1, net.shape[0]):
        block = net[i - window + 1:i + 1]
        pool = np.flatnonzero(~np.isnan(block).any(axis=0))
        if len(pool) < max(CS["min_pool"], 2 * n):
            continue
        order = pool[np.argsort(block[-1, pool], kind="mergesort")]
        short_camp, long_camp = order[:n], order[-n:]
        rows[i] = (_mean_pair_corr(block[:, long_camp]), _mean_pair_corr(block[:, short_camp]),
                   _slope(block[:, long_camp].sum(axis=1)), _slope(block[:, short_camp].sum(axis=1)))
    out = pd.DataFrame(rows, index=net_positions.index, columns=["C_LONG", "C_SHORT", "SLOPE_LONG", "SLOPE_SHORT"])
    out["POWER"] = out["C_LONG"] * out["SLOPE_LONG"] - out["C_SHORT"] * (-out["SLOPE_SHORT"])
    return out


# ---------------------------------------------------------------- 开源：蜘蛛网

def spider_signal(t: dict[str, pd.DataFrame], active_only: bool) -> pd.Series:
    """前 20 名多单增量 dB > 0 且空单增量 dS < 0 看多，反之看空；可只计成交榜也上榜的会员。"""
    dL, dS = t["dL"], t["dS"]
    if active_only:
        dL, dS = dL.where(t["V"].notna()), dS.where(t["V"].notna())
    db = dL.sum(axis=1, min_count=1)
    ds = dS.sum(axis=1, min_count=1)
    sig = pd.Series(0.0, index=db.index)
    sig[(db > 0) & (ds < 0)] = 1.0
    sig[(db < 0) & (ds > 0)] = -1.0
    return sig.where(t["ranked"] & db.notna() & ds.notna())

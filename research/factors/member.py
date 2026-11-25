"""会员持仓指标与研报信号：兴业证券拥挤度/一致性、开源证券蜘蛛网与持仓占比。

指标只使用 t 日收盘后已公布的会员排名。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data as D
from research.settings import CONSISTENCY as CS, CROWDING as CR

NET_WINDOWS = (10, 20, 60, 120)
NET_THRESHOLDS = (0, 0.01, 0.03, 0.05)
CONSISTENCY_WINDOWS = (20, 60)


# ---------------------------------------------------------------- 基础输入

def member_tables(symbol: str, dates: pd.DatetimeIndex, *, top20: bool = False,
                  partition: str = "research") -> dict[str, pd.DataFrame]:
    """日期 × 会员宽表：多、空持仓，成交量，多、空持仓变化（交易所披露的较前一日变化）。"""
    if partition == "research":
        ranks = D.load_member_rank(symbol, "research")
        C.assert_no_holdout_dates(ranks["trading_date"], f"{symbol} 会员排名")
    elif partition in ("oos", "combined"):
        ranks = D.load_member_rank_span(symbol)
    else:
        raise ValueError(f"未知分区 {partition}")
    if top20:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    members = sorted(ranks["member_name"].unique())
    out = {}
    for side, key in (("long", "L"), ("short", "S"), ("volume", "V")):
        sub = ranks[ranks["side"] == side]
        out[key] = sub.pivot(index="trading_date", columns="member_name", values="volume") \
            .reindex(index=dates, columns=members)
        if side != "volume":
            out[f"d{key}"] = sub.pivot(index="trading_date", columns="member_name", values="volume_change") \
                .reindex(index=dates, columns=members)
    out["ranked"] = pd.Series(dates.isin(ranks["trading_date"].unique()), index=dates)
    return out


def daily_position_flow(tables: dict[str, pd.DataFrame], oi: pd.Series) -> pd.Series:
    """品种榜披露的净持仓变化，以品种单边总持仓归一化。"""
    flow = tables["dL"].sum(axis=1, min_count=1) - tables["dS"].sum(axis=1, min_count=1)
    return (flow / oi.where(oi.gt(0))).rename("FLOW_CONTROL")


def market_inputs(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """因子所需的当日收盘收益、单边总成交量与总持仓。"""
    p = panel[panel["symbol"] == symbol].set_index("date").sort_index()
    return pd.DataFrame({
        "ret_cc": p["ret1"],
        "volume": p["volume_total"].astype(float),
        "oi": p["oi_total"].astype(float),
    }, index=p.index)


def bounds(x: pd.Series, window: int, q_high: float, q_low: float) -> tuple[pd.Series, pd.Series]:
    """滚动分位阈值，只用 t-N..t-1，不含当日。"""
    past = x.shift(1).rolling(window, min_periods=window)
    return past.quantile(q_high), past.quantile(q_low)


# ---------------------------------------------------------------- 兴业：拥挤度

def seat_turnover_groups(t: dict[str, pd.DataFrame], mkt: pd.DataFrame):
    """返回可见净持仓、总持仓及两组互斥的席位掩码。"""
    L = t["L"].fillna(0.0)
    S = t["S"].fillna(0.0)
    V = t["V"].fillna(0.0)
    has_day = t["ranked"].to_numpy(dtype=bool)[:, None]
    gross = L + S
    net = L - S
    oi = mkt["oi"].where(mkt["oi"] > 0)
    mkt_to = mkt["volume"] / oi
    high_to = t["V"].notna() & gross.gt(0) & (V / gross.where(gross.gt(0))).ge(mkt_to, axis=0)
    sticky = has_day & gross.gt(0) & (~high_to)
    spec = has_day & gross.gt(0) & high_to
    return net, gross, sticky, spec


def sticky_net_share(t: dict[str, pd.DataFrame], mkt: pd.DataFrame) -> pd.DataFrame:
    """按 CROWD1 换手门槛拆分粘性持仓席位（STICKY_NET_SHARE）与高换手活跃席位（SPEC_NET_SHARE）。"""
    net, gross, sticky, spec = seat_turnover_groups(t, mkt)

    def share(mask: pd.DataFrame) -> pd.Series:
        num = net.where(mask).sum(axis=1, min_count=1)
        den = gross.where(mask).sum(axis=1, min_count=1)
        return num / den.where(den.gt(0))

    return pd.DataFrame({
        "STICKY_NET_SHARE": share(sticky),
        "SPEC_NET_SHARE": share(spec),
    }, index=net.index)


def crowding(tables: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    """兴业原文拥挤度口径：逐侧绝对变动。市场成交量已统一为单边，三榜的缺失不视为零。"""
    long, short, volume = (tables[name] for name in ("L", "S", "V"))
    gross = long + short
    net = long - short
    complete = gross.gt(0) & volume.gt(0)
    seat_volume = volume.where(complete)
    market_volume = market["volume"].where(market["volume"] > 0)
    oi = market["oi"].where(market["oi"] > 0)
    delta_oi = oi.diff().abs()
    delta_seat = (long.diff().abs() + short.diff().abs()).where(complete)

    def share(ratio, threshold):
        selected = complete & ratio.ge(threshold, axis=0)
        return net.where(selected).sum(axis=1, min_count=1) / gross.where(selected).sum(axis=1, min_count=1)

    out = {
        "CROWD1": share(seat_volume / gross, market_volume / oi),
        "CROWD2": share(delta_seat / seat_volume, delta_oi / market_volume),
    }
    for window, name in ((3, "CROWD3"), (4, "CROWD3_4D")):
        seat_amount = delta_seat.rolling(window, min_periods=2).sum()
        seat_sum_volume = seat_volume.where(delta_seat.notna()).rolling(window, min_periods=2).sum()
        market_amount = delta_oi.rolling(window, min_periods=window).sum()
        market_sum_volume = market_volume.rolling(window, min_periods=window).sum()
        out[name] = share(seat_amount / seat_sum_volume, market_amount / market_sum_volume)
    return pd.DataFrame(out)


def crowding_signal(x: pd.Series, ret_cc: pd.Series, window: int) -> pd.Series:
    """商品采用反转：多头极端拥挤且连续收涨做空，空头极端拥挤且连续收跌做多。"""
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
    """会员净头寸。只在一侧上榜时，未上榜一侧取当日该榜末位持仓的一半。"""
    L, S = t["L"], t["S"]
    fill_l = L.min(axis=1) / 2.0
    fill_s = S.min(axis=1) / 2.0
    any_side = L.notna() | S.notna()
    L_f = L.apply(lambda col: col.fillna(fill_l))
    S_f = S.apply(lambda col: col.fillna(fill_s))
    return (L_f - S_f).where(any_side)


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


def _tail_signal(value: pd.Series) -> pd.Series:
    hi, lo = bounds(value, CR["reference_window"], CR["q_high"], CR["q_low"])
    result = value.gt(hi).astype(float) - value.lt(lo).astype(float)
    return result.where(value.notna() & hi.notna() & lo.notna())


def xingye_consistency(net: pd.DataFrame, suffix: str, window: int) -> pd.DataFrame:
    """三类基础指标及其六种时序信号。"""
    base = consistency_from_net(net, window)
    cl = _tail_signal(base["C_LONG"])
    cs = _tail_signal(base["C_SHORT"])
    up = _tail_signal(base["SLOPE_LONG"])
    down = _tail_signal(base["SLOPE_SHORT"])
    long_resonance = (cl.eq(1) & up.eq(1) & base["SLOPE_LONG"].gt(0)).astype(float)
    short_resonance = -(cs.eq(1) & down.eq(-1) & base["SLOPE_SHORT"].lt(0)).astype(float)
    long_resonance = long_resonance.where(cl.notna() & up.notna())
    short_resonance = short_resonance.where(cs.notna() & down.notna())
    signals = base.assign(
        LONG_SYNC=cl, SHORT_SYNC=-cs, LONG_RESONANCE=long_resonance,
        SHORT_RESONANCE=short_resonance, BOTH_RESONANCE=long_resonance + short_resonance,
        POWER_TAIL=_tail_signal(base["POWER"]),
    )
    return signals.add_suffix(f"_{window}{suffix}")


# ---------------------------------------------------------------- 开源：蜘蛛网与净多占比

def spider_signal(t: dict[str, pd.DataFrame], active_only: bool) -> pd.Series:
    """品种级蜘蛛网代理：前 20 名多单增量 dB > 0 且空单增量 dS < 0 看多，反之看空。"""
    dL, dS = t["dL"], t["dS"]
    if active_only:
        dL, dS = dL.where(t["V"].notna()), dS.where(t["V"].notna())
    db = dL.sum(axis=1, min_count=1)
    ds = dS.sum(axis=1, min_count=1)
    sig = pd.Series(0.0, index=db.index)
    sig[(db > 0) & (ds < 0)] = 1.0
    sig[(db < 0) & (ds > 0)] = -1.0
    return sig.where(t["ranked"] & db.notna() & ds.notna())


def kaiyuan(ranks: pd.DataFrame, contracts: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """按披露合约汇总原版蜘蛛网、多空增量及净多头占比。"""
    fields = ["volume", "volume_change"]
    if ranks.empty:
        totals = pd.DataFrame(np.nan, index=calendar, columns=pd.MultiIndex.from_product([fields, ["long", "short"]]))
        disclosed_oi = pd.Series(np.nan, index=calendar)
    else:
        keys = ["trading_date", "contract"]
        both = ranks.groupby(keys)["side"].transform("nunique").eq(2)
        visible = ranks.loc[both]
        totals = visible.groupby(["trading_date", "side"])[fields].sum(min_count=1).unstack("side")
        totals = totals.reindex(index=calendar, columns=pd.MultiIndex.from_product([fields, ["long", "short"]]))
        disclosed = visible[keys].drop_duplicates().merge(
            contracts[["date", "contract", "open_interest"]], left_on=keys,
            right_on=["date", "contract"], how="left", validate="one_to_one",
        )
        grouped = disclosed.groupby("trading_date")["open_interest"]
        disclosed_oi = grouped.sum(min_count=1).where(grouped.size().eq(grouped.count())).reindex(calendar)
    db, ds = totals[("volume_change", "long")], totals[("volume_change", "short")]
    net_share = (totals[("volume", "long")] - totals[("volume", "short")]) / disclosed_oi.where(disclosed_oi.gt(0))
    free = ((db.gt(0) & ds.lt(0)).astype(float) - (db.lt(0) & ds.gt(0)).astype(float)).where(db.notna() & ds.notna())
    weekdays = calendar.to_numpy(dtype="datetime64[D]")
    holiday = free.iloc[:-1].mask(np.busday_count(weekdays[:-1], weekdays[1:]) > 1, 0).reindex(calendar)
    holiday = holiday.where(free.notna())
    out = {"K_DB": db, "K_DS": ds, "SPIDER_HOLIDAY": holiday, "SPIDER_FREE": free, "K_NET_SHARE": net_share}
    for window in NET_WINDOWS:
        change = net_share.diff(window)
        out[f"K_NET_{window}"] = change
        for threshold in NET_THRESHOLDS:
            signal = change.gt(threshold).astype(float) - change.lt(-threshold).astype(float)
            out[f"K_NET_{window}_S{int(threshold * 100):02d}"] = signal.where(change.notna())
    return pd.DataFrame(out, index=calendar)


def kaiyuan_active(ranks: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """仅在同合约同日成交榜内的会员参与增量汇总，并保留同覆盖原版。"""
    matrix = pd.DataFrame(index=calendar)
    keys = ["trading_date", "contract"]
    if ranks.empty:
        return matrix.assign(SPIDER_ACTIVE=np.nan, SPIDER_SAME_COVERAGE=np.nan, contracts_covered=0)
    grouped = ranks.groupby(keys)["side"].nunique()
    valid = grouped[grouped.eq(3)].index.to_frame(index=False)
    holdings = ranks.loc[ranks.side.isin(["long", "short"])].merge(valid, on=keys)
    participants = ranks.loc[ranks.side.eq("volume"), [*keys, "member_name"]].drop_duplicates()
    holdings = holdings.merge(participants.assign(active=True), on=[*keys, "member_name"], how="left")
    covered_days = valid.groupby("trading_date").size().reindex(calendar, fill_value=0)
    for label, rows in (("SPIDER_ACTIVE", holdings[holdings["active"].eq(True)]),
                        ("SPIDER_SAME_COVERAGE", holdings)):
        changes = rows.groupby(["trading_date", "side"])["volume_change"]
        totals = changes.sum(min_count=1).where(changes.count().eq(changes.size())).unstack("side", fill_value=0)
        totals = totals.reindex(index=calendar, columns=["long", "short"], fill_value=0)
        long, short = totals["long"], totals["short"]
        value = (long.gt(0) & short.lt(0)).astype(float) - (long.lt(0) & short.gt(0)).astype(float)
        matrix[label] = value.where(covered_days.gt(0) & long.notna() & short.notna())
    return matrix.assign(contracts_covered=covered_days)


def symbol_factors(symbol: str, panel: pd.DataFrame, contracts: pd.DataFrame,
                   calendar: pd.DatetimeIndex, contract_ranks: pd.DataFrame) -> pd.DataFrame:
    """单品种全套研报指标与实验对照列。"""
    market = market_inputs(panel, symbol).reindex(calendar)
    tables = member_tables(symbol, calendar)
    crowd = crowding(tables, market)
    crowd_signals = {
        f"{name}_TAIL": crowding_signal(crowd[name], market["ret_cc"], CR["reference_window"])
        for name in ("CROWD1", "CROWD2", "CROWD3")
    }
    strict = tables["L"] - tables["S"]
    samples = [(strict, ""), (imputed_net(tables), "_FILLED")]
    consistency = [
        xingye_consistency(net, tag, window)
        for net, tag in samples
        for window in CONSISTENCY_WINDOWS
    ]
    parts = [crowd, pd.DataFrame(crowd_signals), *consistency, kaiyuan(contract_ranks, contracts, calendar)]
    return pd.concat([
        *parts,
        daily_position_flow(tables, market["oi"]),
        sticky_net_share(tables, market),
        spider_signal(tables, active_only=True).rename("SPIDER_ACTIVE_PROXY"),
    ], axis=1)

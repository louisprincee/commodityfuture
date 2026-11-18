"""会员持仓研报公式核对与商品样本上的构造。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import member_study as M
from research.settings import CROWDING as CR

NET_WINDOWS = (10, 20, 60, 120)
NET_THRESHOLDS = (0, 0.01, 0.03, 0.05)
CONSISTENCY_WINDOWS = (20, 60)


def crowding(tables: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    """兴业拥挤度。市场成交量已统一为单边，三榜的缺失不视为零。"""
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
        # 持仓变动沿用前一公式中的逐侧绝对值。分子和分母覆盖相同的可见席位日。
        seat_amount = delta_seat.rolling(window, min_periods=2).sum()
        seat_sum_volume = seat_volume.where(delta_seat.notna()).rolling(window, min_periods=2).sum()
        market_amount = delta_oi.rolling(window, min_periods=window).sum()
        market_sum_volume = market_volume.rolling(window, min_periods=window).sum()
        out[name] = share(seat_amount / seat_sum_volume, market_amount / market_sum_volume)
    return pd.DataFrame(out)


def _tail_signal(value: pd.Series) -> pd.Series:
    hi, lo = M.bounds(value, CR["reference_window"], CR["q_high"], CR["q_low"])
    result = value.gt(hi).astype(float) - value.lt(lo).astype(float)
    return result.where(value.notna() & hi.notna() & lo.notna())


def xingye_consistency(net: pd.DataFrame, suffix: str, window: int) -> pd.DataFrame:
    """三类基础指标及其六种时序信号。"""
    base = M.consistency_from_net(net, window)
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


def kaiyuan(ranks: pd.DataFrame, contracts: pd.DataFrame, calendar: pd.DatetimeIndex,
            *, include_disclosed_oi: bool = False) -> pd.DataFrame:
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
    next_day = calendar.to_series().shift(-1)
    holiday = free.mask(next_day.sub(calendar.to_series()).dt.days.gt(1), 0).where(free.notna() & next_day.notna())
    out = {"K_DB": db, "K_DS": ds, "SPIDER_HOLIDAY": holiday, "SPIDER_FREE": free, "K_NET_SHARE": net_share}
    if include_disclosed_oi:
        out["disclosed_oi"] = disclosed_oi
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


def symbol_factors(symbol, panel, contracts, calendar, contract_ranks, *, include_experiments=True):
    market = M.market_inputs(panel, contracts, symbol).reindex(calendar)
    tables = M.member_tables(symbol, calendar, top20=not include_experiments)
    crowd = crowding(tables, market)
    crowd_signals = {
        f"{name}_TAIL": M.crowding_signal(crowd[name], market["ret_cc"], CR["reference_window"])
        for name in ("CROWD1", "CROWD2", "CROWD3")
    }
    strict = tables["L"] - tables["S"]
    samples = [(strict, "")]
    if include_experiments:
        samples.append((M.imputed_net(tables), "_FILLED"))
    else:
        crowd = crowd.drop(columns="CROWD3_4D")
    consistency = [
        xingye_consistency(net, tag, window)
        for net, tag in samples
        for window in CONSISTENCY_WINDOWS
    ]
    parts = [crowd, pd.DataFrame(crowd_signals), *consistency, kaiyuan(contract_ranks, contracts, calendar)]
    if not include_experiments:
        return pd.concat(parts, axis=1)
    flow = tables["dL"].sum(axis=1, min_count=1) - tables["dS"].sum(axis=1, min_count=1)
    oi = market["oi"].where(market["oi"] > 0)
    return pd.concat([
        *parts,
        (flow / oi).rename("FLOW_CONTROL"),
        M.spider_signal(tables, active_only=True).rename("SPIDER_ACTIVE_PROXY"),
    ], axis=1)

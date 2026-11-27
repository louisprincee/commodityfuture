"""会员持仓指标与研报信号：兴业证券拥挤度/一致性、开源证券蜘蛛网与持仓占比。

指标只使用 t 日收盘后已公布的会员排名。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
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


def disclosed_change_sum(tables: dict[str, pd.DataFrame], side: str,
                         *, active_only: bool = False) -> pd.Series:
    """合计实际持仓榜记录的变动；任一应披露记录缺失则该侧合计未知。

    未上该侧持仓榜的席位不参与。活跃口径另要求同日上成交榜。
    """
    if side not in ("L", "S"):
        raise ValueError("side must be L or S")
    holdings = tables[side]
    expected = holdings.notna()
    if active_only:
        expected &= tables["V"].reindex_like(holdings).notna()
    changes = tables[f"d{side}"].reindex_like(holdings)
    changes = changes.where(np.isfinite(changes))
    complete = (~expected | changes.notna()).all(axis=1)
    return changes.where(expected).sum(axis=1, min_count=1).where(expected.any(axis=1) & complete)


def daily_position_flow(tables: dict[str, pd.DataFrame], oi: pd.Series) -> pd.Series:
    """品种榜披露的净持仓变化，以品种单边总持仓归一化。"""
    flow = disclosed_change_sum(tables, "L") - disclosed_change_sum(tables, "S")
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
    valid_market = np.isfinite(mkt["oi"]) & mkt["oi"].gt(0) & np.isfinite(mkt["volume"]) & mkt["volume"].gt(0)
    has_day = (t["ranked"] & valid_market).to_numpy(dtype=bool)[:, None]
    gross = L + S
    net = L - S
    oi = mkt["oi"].where(mkt["oi"] > 0)
    mkt_to = mkt["volume"] / oi
    high_to = t["V"].notna() & gross.gt(0) & (V / gross.where(gross.gt(0))).ge(mkt_to, axis=0)
    sticky = has_day & gross.gt(0) & (~high_to)
    spec = has_day & gross.gt(0) & high_to
    return net, gross, sticky, spec


def net_gross_share(net: pd.DataFrame, gross: pd.DataFrame, mask: pd.DataFrame,
                    *, require_complete: bool = False) -> pd.Series:
    """掩码内可见净头寸占总持仓比例；分母非正时返回缺失。"""
    num = net.where(mask).sum(axis=1, min_count=1)
    den = gross.where(mask).sum(axis=1, min_count=1)
    result = num / den.where(den.gt(0))
    if require_complete:
        result = result.where((~mask | (net.notna() & gross.notna())).all(axis=1))
    return result


def sticky_net_share(t: dict[str, pd.DataFrame], mkt: pd.DataFrame) -> pd.DataFrame:
    """按 CROWD1 换手门槛拆分粘性持仓席位（STICKY_NET_SHARE）与高换手活跃席位（SPEC_NET_SHARE）。"""
    net, gross, sticky, spec = seat_turnover_groups(t, mkt)
    return pd.DataFrame({
        "STICKY_NET_SHARE": net_gross_share(net, gross, sticky),
        "SPEC_NET_SHARE": net_gross_share(net, gross, spec),
    }, index=net.index)


def load_sticky_inputs(panel: pd.DataFrame, start: pd.Timestamp, partition: str,
                       jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """共用品种池内的粘性净多与主力映射；完整历史计算后再截取开户日。"""
    signals, dominant = _load_member_signals(panel, start, partition, jobs)
    return signals["STICKY_NET_SHARE"], dominant


def load_extension_inputs(panel: pd.DataFrame, start: pd.Timestamp, jobs: int, partition: str = "research",
                          names: tuple[str, ...] = ("STICKY_FLOW", "STICKY_IDENTITY"),
                          ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """基准粘性与所选扩展因子（见 extension_factors）；仅产出信号，不形成仓位。"""
    return _load_member_signals(panel, start, partition, jobs, extension=names)


_SIGNAL_CACHE: list[tuple[pd.DataFrame, tuple, dict[str, pd.DataFrame], pd.DataFrame]] = []
_SIGNAL_CACHE_SIZE = 4


def _load_member_signals(panel, start, partition, jobs, *, extension: tuple[str, ...] = ()):
    """同一进程内，同一面板对象、开户日、分区和因子列表的结果直接复用（返回副本）。"""
    key = (pd.Timestamp(start), partition, tuple(extension))
    for cached_panel, cached_key, signals, dominant in _SIGNAL_CACHE:
        if cached_panel is panel and cached_key == key:
            return {name: value.copy() for name, value in signals.items()}, dominant.copy()
    signals, dominant = _compute_member_signals(panel, start, partition, jobs, extension=tuple(extension))
    _SIGNAL_CACHE.append((panel, key, signals, dominant))
    del _SIGNAL_CACHE[:-_SIGNAL_CACHE_SIZE]
    return {name: value.copy() for name, value in signals.items()}, dominant.copy()


def _compute_member_signals(panel, start, partition, jobs, *, extension: tuple[str, ...] = ()):
    from joblib import Parallel, delayed

    start = pd.Timestamp(start)
    if partition == "research":
        C.assert_no_holdout_dates(panel["date"], "粘性净多日面板")
    elif partition == "oos":
        C.assert_oos_dates(panel.loc[panel["date"].ge(start), "date"], "粘性净多样本外日面板")
    elif partition != "combined":
        raise ValueError(f"未知分区 {partition}")
    eligible, dominant = D.member_universe(panel, start, partition)
    volume = panel.pivot(index="date", columns="symbol", values="volume_total")
    oi = panel.pivot(index="date", columns="symbol", values="oi_total")

    def one_symbol(symbol):
        tables = member_tables(symbol, eligible.index, top20=True, partition=partition)
        market = pd.DataFrame({"volume": volume[symbol], "oi": oi[symbol]}, index=eligible.index)
        result = sticky_net_share(tables, market)[["STICKY_NET_SHARE"]]
        if extension:
            ext = extension_factors(tables, market, names=extension)
            for col in extension:
                result[col] = ext[col]
        return result

    values = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(one_symbol)(symbol) for symbol in eligible.columns)
    names = ["STICKY_NET_SHARE", *extension]
    signals = {}
    for name in names:
        matrix = pd.DataFrame({symbol: value[name] for symbol, value in zip(eligible.columns, values)},
                              index=eligible.index, columns=eligible.columns)
        signals[name] = matrix.where(eligible).loc[start:]
    return signals, dominant


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
    db = disclosed_change_sum(t, "L", active_only=active_only)
    ds = disclosed_change_sum(t, "S", active_only=active_only)
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
        grouped = visible.groupby(["trading_date", "side"])
        complete = grouped[fields].count().eq(grouped.size(), axis=0)
        totals = grouped[fields].sum(min_count=1).where(complete).unstack("side")
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


# ---------------------------------------------------------------- 2011–2021 扩展会员分组与持仓流

FACTOR_META: dict[str, dict[str, str | int]] = {}


def _register(name: str, family: str, description: str, sign: int = 1) -> None:
    FACTOR_META[name] = {"family": family, "sign": sign, "description": description}


for _name, _description in (
    ("STICKY_NET_SHARE", "Visible net/gross of seats below market turnover, including undisclosed volume."),
    ("SPEC_NET_SHARE", "Visible net/gross of seats at or above market turnover."),
):
    _register(_name, "baseline", _description)

for _name, _description in (
    ("RUN_LONG", "Net/gross of currently held seats present for at least 20 consecutive ranked days."),
    ("DIR_STABLE", "Net/gross with absolute mean visible-net sign at least .8 over 20 consecutive holding days."),
):
    _register(_name, "profile", _description)

for _name, _description in (
    ("FLOW_CONTROL", "Disclosed aggregate long change minus short change / market open interest."),
    ("FLOW_CHANGE5", "FLOW_CONTROL minus its value five sessions earlier."),
    ("FLOW_MA5", "Five-day mean of FLOW_CONTROL with five valid observations."),
    ("NEG_SHORT_FLOW", "Negative disclosed aggregate short change / market open interest."),
):
    _register(_name, "flow", _description)

for _pct in ("00", "50", "80", "100"):
    for _group in ("STICKY", "SPEC"):
        _register(f"IMP{_pct}_{_group}", "imputation",
                  f"Net/gross after missing-side fill at {_pct}% of its daily disclosed minimum; original {_group.lower()} membership fixed.")


def _impute(long: pd.DataFrame, short: pd.DataFrame,
            fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fraction == 0:
        return long.fillna(0.0), short.fillna(0.0)
    long_fill = long.min(axis=1) * fraction
    short_fill = short.min(axis=1) * fraction
    return (long.mask(long.isna() & short.notna(), long_fill, axis=0),
            short.mask(short.isna() & long.notna(), short_fill, axis=0))


def profile_factors(tables: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    """在固定日期网格上计算连续上榜、方向稳定、四档插补与持仓流因子（纯因子计算，无外部读写）。"""
    index = market.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("market dates must be unique and increasing")
    columns = tables["L"].columns
    for key in ("S", "V", "dL", "dS"):
        columns = columns.union(tables[key].columns, sort=False)
    ranked = tables["ranked"].reindex(index).fillna(False).astype(bool)
    t = {key: tables[key].reindex(index=index, columns=columns).astype(float)
         for key in ("L", "S", "V", "dL", "dS")}
    t = {key: value.where(np.isfinite(value)).where(ranked, axis=0) for key, value in t.items()}
    t["ranked"] = ranked
    m = market[["oi", "volume"]].astype(float)
    m = m.where(np.isfinite(m))
    valid = ranked & m["oi"].gt(0) & m["volume"].gt(0)
    long, short = t["L"], t["S"]
    net, gross, sticky, spec = seat_turnover_groups(t, m)
    held = gross.gt(0) & ranked.to_numpy()[:, None]
    presence = (long.notna() | short.notna()) & ranked.to_numpy()[:, None]
    out = dict(sticky_net_share(t, m))

    streak = np.zeros(presence.shape, dtype=np.int64)
    present_array = presence.to_numpy()
    for row in range(len(index)):
        streak[row] = np.where(present_array[row], (streak[row - 1] if row else 0) + 1, 0)
    run = pd.DataFrame(streak, index=index, columns=columns)
    out["RUN_LONG"] = net_gross_share(net, gross, held & run.ge(20))
    direction = np.sign(net).where(presence).rolling(20, min_periods=20).mean().abs()
    out["DIR_STABLE"] = net_gross_share(net, gross, held & direction.ge(0.8))

    for pct in (0, 50, 80, 100):
        filled_long, filled_short = _impute(long, short, pct / 100)
        filled_net, filled_gross = filled_long - filled_short, filled_long + filled_short
        for group, mask in (("STICKY", sticky), ("SPEC", spec)):
            out[f"IMP{pct:02d}_{group}"] = net_gross_share(filled_net, filled_gross, mask, require_complete=True)

    flow = daily_position_flow(t, m["oi"]).where(valid)
    out["FLOW_CONTROL"] = flow
    out["FLOW_MA5"] = flow.rolling(5, min_periods=5).mean()
    out["FLOW_CHANGE5"] = flow.diff(5)
    oi = m["oi"].where(m["oi"].gt(0))
    out["NEG_SHORT_FLOW"] = -disclosed_change_sum(t, "S") / oi
    return pd.DataFrame(out, index=index).reindex(columns=FACTOR_META).where(valid, axis=0)


REDEFINITION_META: dict[str, dict[str, str | int]] = {
    "STICKY_NET_SHARE": {"family": "redefinition_baseline", "sign": 1,
                         "description": "Default V/(L+S) < VOL/OI low-turnover net-long share."},
    "STICKY_PERSIST": {"family": "redefinition", "sign": 1,
                       "description": "Held seats present >=15/20 days with 20d |mean sign(net)| >= 0.8 (turnover-free)."},
    "STICKY_LOW_ADJUST": {"family": "redefinition", "sign": 1,
                          "description": "Held seats whose 5d mean (|dL|+|dS|)/(L+S) is <= daily peer median."},
    "STICKY_UNILATERAL": {"family": "redefinition", "sign": 1,
                          "description": "Default sticky seats excluding two-sided balanced seats with |L-S|/(L+S) < 0.30."},
    "STICKY_PERSIST_PURE": {"family": "redefinition", "sign": 1,
                            "description": "Persistent directional seats excluding two-sided balanced seats (|L-S|/(L+S) < 0.30)."},
}


EXTENSION_META: dict[str, dict[str, str | int]] = {
    "STICKY_TOP20": {"family": "extension", "sign": 1,
                     "description": "Held seats below the daily median V/(L+S) of seats on the volume list (top-20 peer threshold)."},
    "STICKY_Z250": {"family": "extension", "sign": 1,
                    "description": "STICKY_NET_SHARE minus its prior 250-day mean over prior std (>=60 obs)."},
    "STICKY_CHG20": {"family": "extension", "sign": 1,
                     "description": "STICKY_NET_SHARE minus its value 20 sessions earlier."},
    "STICKY_FLOW": {"family": "extension", "sign": 1,
                    "description": "Disclosed (dL - dS) of today's sticky seats / market OI; missing if any sticky disclosure is missing."},
    "STICKY_IDENTITY": {"family": "extension", "sign": 1,
                        "description": "Held seats classified sticky on >=80% of their prior-242-day listed days (>=60 days)."},
    "SEAT_SKILL_NET": {"family": "extension", "sign": 1,
                       "description": "Held seats whose prior-242-day mean of sign(net[t-2]) * open-to-open return[t-1] is positive (>=60 obs)."},
}
CROWDING_META: dict[str, dict[str, str | int]] = {
    name: {"family": "crowding", "sign": -1, "description": f"{name} visible net/gross; reversal sign."}
    for name in ("CROWD1", "CROWD2", "CROWD3")
}


FOLLOWUP_META: dict[str, dict[str, str | int]] = {
    "STICKY_PERSIST_IDENTITY": {"family": "followup", "sign": 1,
                                "description": "Held seats passing both the STICKY_PERSIST and STICKY_IDENTITY criteria."},
    "STICKY_FLOW5": {"family": "followup", "sign": 1,
                     "description": "Five-session mean of STICKY_FLOW with five valid observations."},
}


def rank_composite(*scores: pd.DataFrame) -> pd.DataFrame:
    """所有输入同时有效的品种内，各因子截面百分位排名的等权平均。"""
    common = np.logical_and.reduce([np.isfinite(score) for score in scores])
    ranks = [score.where(common).rank(axis=1, pct=True) for score in scores]
    return sum(ranks) / len(ranks)


def _persist_mask(t: dict[str, pd.DataFrame], net: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """冻结的持续性筛选：方向均值需 20 个有效上榜观测，一致性至少 0.8。

    因此现有实现实际比名义上的“20 日上榜至少 15 日”更严格；保留已评估规则。
    """
    presence = (t["L"].notna() | t["S"].notna()) & t["ranked"].to_numpy()[:, None]
    present_20d = presence.astype(float).rolling(20, min_periods=20).sum()
    dir_cons_20d = np.sign(net).where(presence).rolling(20, min_periods=20).mean().abs()
    return held & present_20d.ge(15) & dir_cons_20d.ge(0.8)


def _aligned_tables(tables: dict[str, pd.DataFrame], market: pd.DataFrame):
    index = market.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("market dates must be unique and increasing")
    columns = tables["L"].columns
    for key in ("S", "V", "dL", "dS"):
        columns = columns.union(tables[key].columns, sort=False)
    ranked = tables["ranked"].reindex(index).fillna(False).astype(bool)
    t = {key: tables[key].reindex(index=index, columns=columns).astype(float)
         for key in ("L", "S", "V", "dL", "dS")}
    t = {key: value.where(np.isfinite(value)).where(ranked, axis=0) for key, value in t.items()}
    t["ranked"] = ranked
    m = market[["oi", "volume"]].astype(float)
    m = m.where(np.isfinite(m))
    valid = ranked & m["oi"].gt(0) & m["volume"].gt(0)
    return t, m, valid


def sticky_composition(tables: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    """粘性组构成诊断：未上成交榜席位在粘性持仓中的占比，以及粘性持仓占全部可见持仓的比例。"""
    t, m, valid = _aligned_tables(tables, market)
    _, gross, sticky, _ = seat_turnover_groups(t, m)
    held = gross.gt(0) & t["ranked"].to_numpy()[:, None]
    sticky_gross = gross.where(sticky).sum(axis=1, min_count=1)
    unlisted_gross = gross.where(sticky & t["V"].isna()).sum(axis=1)
    return pd.DataFrame({
        "UNLISTED_STICKY_SHARE": unlisted_gross / sticky_gross.where(sticky_gross.gt(0)),
        "STICKY_GROSS_SHARE": sticky_gross / gross.where(held).sum(axis=1, min_count=1),
    }, index=m.index).where(valid, axis=0)


def extension_factors(tables: dict[str, pd.DataFrame], market: pd.DataFrame,
                      returns: pd.Series | None = None, names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """事前冻结的粘性净多扩展因子。returns[d] 为 d 日开盘到 d+1 日开盘、按 d-1 收盘主力计算的收益；
    省略时 SEAT_SKILL_NET 为缺失。names 只决定计算哪些列，不改变任何一列的数值；省略时计算全部。"""
    from functools import cache

    t, m, valid = _aligned_tables(tables, market)
    index = m.index
    net, gross, sticky, _ = seat_turnover_groups(t, m)
    held = gross.gt(0) & t["ranked"].to_numpy()[:, None]

    @cache
    def base():
        return net_gross_share(net, gross, sticky).where(valid)

    def sticky_top20():
        seat_turnover = t["V"] / gross.where(gross.gt(0))
        listed = held & t["V"].notna()
        peer_median = seat_turnover.where(listed).median(axis=1)
        return net_gross_share(net, gross, held & ~(listed & seat_turnover.ge(peer_median, axis=0)))

    def z250():
        prior = base().shift(1).rolling(250, min_periods=60)
        return (base() - prior.mean()) / prior.std().where(lambda value: value.gt(0))

    @cache
    def flow():
        long_expected = sticky & t["L"].notna()
        short_expected = sticky & t["S"].notna()
        complete = ((~long_expected | t["dL"].notna()) & (~short_expected | t["dS"].notna())).all(axis=1)
        sticky_change = t["dL"].where(long_expected).sum(axis=1) - t["dS"].where(short_expected).sum(axis=1)
        return (sticky_change / m["oi"].where(m["oi"].gt(0))).where(complete & sticky.any(axis=1))

    @cache
    def identity():
        held_days = held.astype(float).shift(1).rolling(242, min_periods=120).sum()
        sticky_days = (sticky & held).astype(float).shift(1).rolling(242, min_periods=120).sum()
        return held & (sticky_days / held_days.where(held_days.ge(60))).ge(0.8)

    def skill_net():
        ret = pd.Series(np.nan, index=index) if returns is None else returns.reindex(index)
        hit = np.sign(net.where(held)).shift(1).mul(ret, axis=0)
        skill = hit.shift(1).rolling(242, min_periods=60).mean()
        return net_gross_share(net, gross, held & skill.gt(0))

    builders = {
        "STICKY_NET_SHARE": base,
        "STICKY_TOP20": sticky_top20,
        "STICKY_Z250": z250,
        "STICKY_CHG20": lambda: base() - base().shift(20),
        "STICKY_FLOW": flow,
        "STICKY_IDENTITY": lambda: net_gross_share(net, gross, identity()),
        "SEAT_SKILL_NET": skill_net,
        "STICKY_PERSIST_IDENTITY": lambda: net_gross_share(net, gross, identity() & _persist_mask(t, net, held)),
        "STICKY_FLOW5": lambda: flow().where(valid).rolling(5, min_periods=5).mean(),
        "STICKY_FLOW10": lambda: flow().where(valid).rolling(10, min_periods=10).mean(),
    }
    wanted = [*builders, *CROWDING_META] if names is None else list(names)
    unknown = sorted(set(wanted) - set(builders) - set(CROWDING_META))
    if unknown:
        raise KeyError(f"未知扩展因子: {unknown}")
    frame = pd.DataFrame({name: builders[name]() for name in wanted if name in builders}, index=index)
    crowd_names = [name for name in wanted if name in CROWDING_META]
    if crowd_names:
        crowd = crowding({key: t[key] for key in ("L", "S", "V")}, m)[crowd_names]
        frame = pd.concat([frame, crowd], axis=1)
    return frame[wanted].where(valid, axis=0)


def redefinition_factors(tables: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    """事前冻结的粘性席位重定义因子族（纯行为持续、低换仓强度、去双边对锁）。"""
    index = market.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("market dates must be unique and increasing")
    columns = tables["L"].columns
    for key in ("S", "V", "dL", "dS"):
        columns = columns.union(tables[key].columns, sort=False)
    ranked = tables["ranked"].reindex(index).fillna(False).astype(bool)
    t = {key: tables[key].reindex(index=index, columns=columns).astype(float)
         for key in ("L", "S", "V", "dL", "dS")}
    t = {key: value.where(np.isfinite(value)).where(ranked, axis=0) for key, value in t.items()}
    t["ranked"] = ranked
    m = market[["oi", "volume"]].astype(float)
    m = m.where(np.isfinite(m))
    valid = ranked & m["oi"].gt(0) & m["volume"].gt(0)

    net, gross, sticky, _ = seat_turnover_groups(t, m)
    held = gross.gt(0) & ranked.to_numpy()[:, None]
    both_sides = t["L"].notna() & t["S"].notna()
    purity = net.abs() / gross.where(gross.gt(0))
    balanced = both_sides & purity.lt(0.30)

    persist_mask = _persist_mask(t, net, held)

    # 低换仓强度：需该侧在榜持仓披露增减量（任一应披露侧缺失 dL/dS 则当日调仓强度未知）
    long_ok = t["L"].isna() | t["dL"].notna()
    short_ok = t["S"].isna() | t["dS"].notna()
    adj_num = t["dL"].fillna(0.0).abs() + t["dS"].fillna(0.0).abs()
    adj_ratio = (adj_num / gross.where(gross.gt(0))).where(held & long_ok & short_ok)
    adj_ma5 = adj_ratio.rolling(5, min_periods=3).mean()
    adj_med = adj_ma5.median(axis=1)
    low_adj_mask = held & adj_ma5.notna() & adj_ma5.le(adj_med, axis=0)

    out = {
        "STICKY_NET_SHARE": net_gross_share(net, gross, sticky),
        "STICKY_PERSIST": net_gross_share(net, gross, persist_mask),
        "STICKY_LOW_ADJUST": net_gross_share(net, gross, low_adj_mask),
        "STICKY_UNILATERAL": net_gross_share(net, gross, sticky & (~balanced)),
        "STICKY_PERSIST_PURE": net_gross_share(net, gross, persist_mask & (~balanced)),
    }
    return pd.DataFrame(out, index=index).reindex(columns=REDEFINITION_META).where(valid, axis=0)

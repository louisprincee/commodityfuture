"""读取米筐商品日频数据，构造主力 / 次主力日面板与时点流动性池。研究期与样本外分开读取。"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

def external_root(partition: str) -> Path:
    return C.daily_dir(partition)


def _guard(path: Path, partition: str) -> None:
    if partition == "research":
        C.assert_research_only(path)


def _check_dates(dates, partition: str, what: str) -> None:
    if partition == "research":
        C.assert_no_holdout_dates(dates, what)
    else:
        C.assert_oos_dates(dates, what)


def to_single_sided(frame: pd.DataFrame) -> pd.DataFrame:
    """2020-01-01 前商品期货成交量、成交额、持仓量按双边统计，统一折半为单边口径。"""
    out = frame.copy()
    before = out["date"] < pd.Timestamp(C.SINGLE_SIDED_FROM)
    for col in C.DOUBLE_SIDED_FIELDS:
        out[col] = out[col].astype(float)
        out.loc[before, col] = out.loc[before, col] / 2.0
    return out


def load_contracts(partition: str = "research", *, for_execution: bool = False) -> pd.DataFrame:
    """全部商品合约日行情，长表：contract、date、symbol 与价格/量/持仓字段。

    成交量、成交额与持仓量统一为单边口径。更名前合约保留到自身最后一条行情，
    品种标签改为后继代码；合约代码仍是交易所当时的代码。
    """
    root = external_root(partition) / "contracts"
    _guard(root, partition)
    frames = []
    for path in sorted(root.glob("*.pkl")):
        _guard(path, partition)
        frames.append(pd.read_pickle(path))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise FileNotFoundError(f"缺少逐合约日行情缓存: {root}")
    out = pd.concat(frames).reset_index().rename(columns={"order_book_id": "contract"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["contract"].str.extract(r"^([A-Z]+)\d{4}$", expand=False)
    if out["symbol"].isna().any():
        bad = out.loc[out["symbol"].isna(), "contract"].unique()[:5]
        raise ValueError(f"合约代码无法解析品种: {bad}")
    if out.duplicated(["contract", "date"]).any():
        raise ValueError("合约日行情有重复 (contract, date)")
    if partition == "research":
        out = apply_legacy_contracts(out, for_execution=for_execution)
    _check_dates(out["date"], partition, "合约日行情")
    out = to_single_sided(out)
    return out.sort_values(["symbol", "contract", "date"], kind="mergesort").reset_index(drop=True)


# 郑商所更名前代码。日期是后继代码开始交易的日子，不是旧合约的截止日。
# 旧月份继续留到各自最后一条行情；乘数不同的日子不按手数相加。
LEGACY_UNDERLYING = {
    "ME": ("MA", pd.Timestamp("2014-06-17")),
    "TC": ("ZC", pd.Timestamp("2015-05-18")),
    "RO": ("OI", pd.Timestamp("2012-07-16")),
    "WS": ("WH", pd.Timestamp("2012-07-24")),
    "ER": ("RI", pd.Timestamp("2012-07-24")),
}
DOMINANT_OI_SWITCH = 1.1
MEMBER_RANK_LIMIT = 20


def legacy_sources(symbol: str) -> list[tuple[str, pd.Timestamp]]:
    """后继品种对应的旧代码，以及后继代码开始交易的日期。"""
    return [(old, cut) for old, (new, cut) in LEGACY_UNDERLYING.items() if new == symbol]


def legacy_successor_symbols() -> list[str]:
    return sorted({new for new, _ in LEGACY_UNDERLYING.values()})


def apply_legacy_contracts(frame: pd.DataFrame, *, for_execution: bool = False) -> pd.DataFrame:
    """统一品种标签，并保留旧合约直到它自己的行情结束。

    研究面板和执行行情使用同一规则。``for_execution`` 只保留给既有调用。
    """
    del for_execution
    if frame.empty or not frame["symbol"].isin(LEGACY_UNDERLYING).any():
        return frame
    kept = [frame.loc[~frame["symbol"].isin(LEGACY_UNDERLYING)]]
    for old, (new, _cut) in LEGACY_UNDERLYING.items():
        part = frame.loc[frame["symbol"].eq(old)].copy()
        if part.empty:
            continue
        part["symbol"] = new
        kept.append(part)
    return pd.concat(kept, ignore_index=True)


def stitch_legacy_dominant(current: pd.Series, legacy: pd.Series, cutover: pd.Timestamp) -> pd.Series:
    """旧主力只补到后继品种上市日前，且不覆盖已有交易日。"""
    if legacy.empty:
        return current
    extra = legacy.loc[legacy.index < cutover]
    extra = extra.loc[~extra.index.isin(current.index)]
    if extra.empty:
        return current
    out = pd.concat([extra, current]).sort_index()
    out.name = current.name
    return out.astype(str)


def _contract_prefix(contracts: pd.Series) -> pd.Series:
    return contracts.astype(str).str.extract(r"^([A-Z]+)", expand=False)


def load_contract_multipliers(partition: str = "research") -> pd.Series:
    """合约静态乘数。只读取供应商写入的表，不补缺失值。"""
    path = C.execution_dir(partition) / "contract_multipliers.csv"
    if partition == "research":
        _guard(path, partition)
    if not path.is_file():
        return pd.Series(dtype=float)
    table = pd.read_csv(path)
    if table.empty or {"contract", "contract_multiplier"} - set(table.columns):
        raise ValueError(f"合约乘数表缺少字段: {path}")
    if table["contract"].duplicated().any():
        raise ValueError("合约乘数表有重复合约")
    values = table.set_index("contract")["contract_multiplier"].astype(float)
    if values.isna().any() or values.le(0).any():
        raise ValueError("合约乘数必须为正")
    return values.sort_index()


def prefix_multiplier(prefix: str, multipliers: pd.Series) -> float:
    """同一品种代码在表内只能有一个乘数。"""
    if multipliers.empty:
        raise ValueError(f"{prefix} 缺少合约乘数")
    matched = multipliers.loc[_contract_prefix(pd.Series(multipliers.index, index=multipliers.index)).eq(prefix)]
    unique = matched.dropna().unique()
    if len(unique) != 1:
        raise ValueError(f"{prefix} 合约乘数缺失或不唯一")
    return float(unique[0])


def _highest(scores: pd.Series) -> str:
    """得分最高者优先；得分相同则按合约代码。"""
    table = scores.rename("score").rename_axis("contract").reset_index()
    table = table.sort_values(["score", "contract"], ascending=[False, True], kind="mergesort")
    return str(table.iloc[0]["contract"])


def resolve_overlap_dominant(bars: pd.DataFrame, rank1: pd.Series, rank2: pd.Series,
                             old_prefix: str, multipliers: pd.Series,
                             switch: float = DOMINANT_OI_SWITCH) -> tuple[pd.Series, pd.Series]:
    """新旧代码同时交易时，用前一交易日的吨持仓决定当日主力。

    在任主力的吨持仓超过当前主力的 ``switch`` 倍才切换。重叠期以外沿用原序列。
    次主力是其余合约里前一日吨持仓最高的一张。
    """
    if rank1.empty or bars.empty:
        return rank1, rank2.reindex(rank1.index)
    work = bars.loc[:, ["contract", "date", "open_interest"]].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["prefix"] = _contract_prefix(work["contract"])
    old = work["prefix"].eq(old_prefix)
    overlap = {pd.Timestamp(day).normalize() for day in set(work.loc[old, "date"]) & set(work.loc[~old, "date"])}
    out1 = rank1.astype(object).copy()
    out1.index = pd.to_datetime(out1.index).normalize()
    out2 = rank2.reindex(out1.index).astype(object)
    if not overlap:
        return out1, out2
    needed = work.loc[work["date"].isin(overlap), "contract"].unique()
    missing = sorted(set(needed) - set(multipliers.index))
    if missing:
        raise ValueError(f"更名重叠日缺少合约乘数: {missing[:8]}")
    work["physical"] = work["open_interest"] * work["contract"].map(multipliers)
    physical = {
        date: part.groupby("contract")["physical"].sum(min_count=1)
        for date, part in work.groupby("date")
    }
    dates = list(out1.index)
    position = {date: i for i, date in enumerate(dates)}
    for date in dates:
        if date not in overlap:
            continue
        previous = dates[position[date] - 1] if position[date] else None
        prior = physical.get(previous, pd.Series(dtype=float))
        today = physical.get(date, pd.Series(dtype=float))
        eligible = prior.reindex(today.index).replace([np.inf, -np.inf], np.nan).dropna()
        if eligible.empty:
            out1.loc[date] = np.nan
            out2.loc[date] = np.nan
            continue
        incumbent = out1.loc[previous] if previous is not None else None
        if incumbent not in eligible.index:
            chosen = _highest(eligible)
        else:
            challengers = eligible.drop(labels=[incumbent])
            chosen = _highest(challengers) if not challengers.empty and challengers.max() > switch * eligible[incumbent] else incumbent
        out1.loc[date] = chosen
        rest = eligible.drop(labels=[chosen], errors="ignore")
        out2.loc[date] = _highest(rest) if not rest.empty else np.nan
    return out1, out2


def adjusted_dominant(symbol: str, partition: str, multipliers: pd.Series,
                      contracts: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """研究期对五个更名品种使用重叠期吨持仓规则，其余日子沿用米筐主力。"""
    rank1 = load_dominant(symbol, 1, partition)
    rank2 = load_dominant(symbol, 2, partition).reindex(rank1.index)
    rank2 = rank2.where(rank2 != rank1)
    sources = legacy_sources(symbol)
    if partition != "research" or not sources or multipliers.empty or rank1.empty:
        return rank1, rank2
    old, _cut = sources[0]
    bars = contracts.loc[contracts["symbol"].eq(symbol), ["contract", "date", "open_interest"]]
    return resolve_overlap_dominant(bars, rank1, rank2, old, multipliers)


def scale_mixed_multiplier_totals(contracts: pd.DataFrame, totals: pd.DataFrame,
                                  multipliers: pd.Series) -> pd.DataFrame:
    """成交额始终按金额相加。当日只有一个乘数时持仓和成交量仍按手相加，否则改为吨。"""
    successors = set(legacy_successor_symbols())
    frame = contracts.loc[contracts["symbol"].isin(successors),
                          ["symbol", "contract", "date", "volume", "open_interest"]].copy()
    if frame.empty:
        return totals
    frame["prefix"] = _contract_prefix(frame["contract"])
    old_prefixes = set(LEGACY_UNDERLYING)
    mixed = frame.groupby(["symbol", "date"], sort=False)["prefix"].agg(
        lambda prefixes: prefixes.isin(old_prefixes).any() and (~prefixes.isin(old_prefixes)).any()
    )
    mixed = mixed.loc[mixed]
    if mixed.empty:
        return totals
    part = frame.set_index(["symbol", "date"]).loc[mixed.index].copy()
    missing = sorted(set(part["contract"]) - set(multipliers.index))
    if missing:
        raise ValueError(f"更名重叠日缺少合约乘数: {missing[:8]}")
    part["multiplier"] = part["contract"].map(multipliers)
    if part[["volume", "open_interest", "multiplier"]].isna().any().any():
        raise ValueError("更名重叠日的成交量、持仓或乘数不完整，不能按吨相加")
    physical = pd.DataFrame({
        "volume_total": part["volume"] * part["multiplier"],
        "oi_total": part["open_interest"] * part["multiplier"],
    }).groupby(["symbol", "date"], sort=False).sum()
    out = totals.copy()
    out.loc[physical.index, ["volume_total", "oi_total"]] = physical[["volume_total", "oi_total"]].to_numpy()
    return out


def _sum_changes(values: pd.Series) -> float:
    if values.isna().any():
        return np.nan
    return float(values.sum())


def combine_legacy_member_ranks(current: pd.DataFrame, legacy: pd.DataFrame, successor: str,
                                cut: pd.Timestamp, new_multiplier: float,
                                old_multiplier: float) -> pd.DataFrame:
    """挂牌日前保留旧榜手数；重叠日两边换成吨后按会员相加，再取前 20 名。"""
    columns = _member_rank_columns()
    before = legacy.loc[legacy["trading_date"].lt(cut)].copy()
    before["symbol"] = successor
    overlap = legacy.loc[legacy["trading_date"].ge(cut)].copy()
    if overlap.empty:
        current = current.copy()
        if not current.empty:
            current["symbol"] = successor
        return pd.concat([current, before], ignore_index=True)[columns]
    dates = set(overlap["trading_date"])
    fresh = current.loc[current["trading_date"].isin(dates)].copy() if not current.empty else current.copy()
    rest = current.loc[~current["trading_date"].isin(dates)].copy() if not current.empty else current.iloc[0:0].copy()
    fresh["volume"] = fresh["volume"] * new_multiplier
    fresh["volume_change"] = fresh["volume_change"] * new_multiplier
    overlap["volume"] = overlap["volume"] * old_multiplier
    overlap["volume_change"] = overlap["volume_change"] * old_multiplier
    merged = pd.concat([fresh, overlap], ignore_index=True)
    merged = merged.dropna(subset=["volume"])
    summed = merged.groupby(["trading_date", "side", "member_name"], sort=False).agg(
        volume=("volume", "sum"), volume_change=("volume_change", _sum_changes),
    ).reset_index()
    summed = summed.sort_values(
        ["trading_date", "side", "volume", "member_name"],
        ascending=[True, True, False, True], kind="mergesort",
    )
    summed["rank"] = summed.groupby(["trading_date", "side"], sort=False).cumcount() + 1
    summed = summed.loc[summed["rank"].le(MEMBER_RANK_LIMIT)].copy()
    summed["symbol"] = successor
    return pd.concat([rest, before, summed], ignore_index=True)[columns]


def _read_dominant_file(symbol: str, rank: int, partition: str) -> pd.Series:
    path = external_root(partition) / "dominant" / f"{symbol}_rank{rank}.pkl"
    if not path.exists():
        return pd.Series(dtype=object)
    _guard(path, partition)
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0] if obj.shape[1] else pd.Series(dtype=object)
    obj = obj.dropna()
    obj.index = pd.to_datetime(obj.index).normalize()
    obj.name = f"rank{rank}"
    _check_dates(obj.index, partition, f"{symbol} rank{rank}")
    return obj.astype(str)


def load_dominant(symbol: str, rank: int, partition: str = "research") -> pd.Series:
    """米筐 get_dominant(rank) 的日序列；上市前为空序列。研究期把更名前主力接到后继品种。"""
    path = external_root(partition) / "dominant" / f"{symbol}_rank{rank}.pkl"
    _guard(path, partition)
    if not path.exists():
        raise FileNotFoundError(path)
    series = _read_dominant_file(symbol, rank, partition)
    if partition == "research":
        for old, cut in legacy_sources(symbol):
            series = stitch_legacy_dominant(series, _read_dominant_file(old, rank, partition), cut)
    return series


def load_roll_yield(symbol: str, partition: str = "research") -> pd.DataFrame:
    path = external_root(partition) / "roll_yield" / f"{symbol}_main_sub.pkl"
    _guard(path, partition)
    obj = pd.read_pickle(path)
    if obj is None or obj.empty:
        return pd.DataFrame()
    if isinstance(obj.index, pd.MultiIndex):
        obj = obj.droplevel("underlying_symbol")
    obj.index = pd.to_datetime(obj.index).normalize()
    _check_dates(obj.index, partition, f"{symbol} roll_yield")
    return obj


MEMBER_SIDES = ("long", "short", "volume")


def _read_rank(path: Path, partition: str) -> pd.DataFrame:
    _guard(path, partition)
    if not path.exists():
        return pd.DataFrame()
    obj = pd.read_pickle(path)
    if obj is None or obj.empty:
        return pd.DataFrame()
    out = obj.reset_index()
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.normalize()
    out["volume"] = out["volume"].astype(float)
    out["volume_change"] = out["volume_change"].astype(float)
    _check_dates(out["trading_date"], partition, f"会员排名 {path.name}")
    return out


def _member_rank_columns() -> list[str]:
    return ["trading_date", "symbol", "side", "rank", "member_name", "volume", "volume_change"]


def _member_rank_frame(symbol: str, partition: str) -> pd.DataFrame:
    root = external_root(partition) / "member_rank"
    parts = []
    for side in MEMBER_SIDES:
        frame = _read_rank(root / f"{symbol}_{side}.pkl", partition)
        if not frame.empty:
            parts.append(frame.assign(side=side))
    if not parts:
        return pd.DataFrame(columns=_member_rank_columns())
    out = pd.concat(parts, ignore_index=True)
    out["symbol"] = symbol
    return out[_member_rank_columns()]


def load_member_rank(symbol: str, partition: str = "research") -> pd.DataFrame:
    """品种级前 20 名会员排名长表：trading_date、side、rank、member_name、volume、volume_change。

    持仓排名为各会员单边持仓，2020 年前后口径一致，不随合约日线折半。
    研究期并入更名前代码。后继代码挂牌之后，两边先按各自乘数换成吨再按会员相加，
    然后重新保留前 20 名；挂牌之前仍用旧榜的原始手数。
    """
    out = _member_rank_frame(symbol, partition)
    if partition != "research":
        return out
    multipliers = load_contract_multipliers(partition)
    for old, cut in legacy_sources(symbol):
        extra = _member_rank_frame(old, partition)
        if extra.empty:
            continue
        if extra["trading_date"].ge(cut).any():
            out = combine_legacy_member_ranks(
                out, extra, symbol, cut,
                prefix_multiplier(symbol, multipliers), prefix_multiplier(old, multipliers),
            )
        else:
            extra = extra.copy()
            extra["symbol"] = symbol
            out = pd.concat([out, extra], ignore_index=True)
    return out


def load_member_rank_contracts(symbol: str, partition: str = "research",
                               sides: tuple[str, ...] = ("long", "short")) -> pd.DataFrame:
    """合约级前 20 名排名长表，附 contract 列。"""
    if not sides or not set(sides) <= set(MEMBER_SIDES):
        raise ValueError("合约级榜单类型必须是 long、short 或 volume")
    root = external_root(partition) / "member_rank_contracts"
    _guard(root, partition)
    if not root.exists():
        return pd.DataFrame()
    side_pattern = "|".join(sides)
    prefixes = [symbol]
    cuts: dict[str, pd.Timestamp] = {}
    if partition == "research":
        for old, cut in legacy_sources(symbol):
            prefixes.append(old)
            cuts[old] = cut
    parts = []
    names = sorted(os.listdir(root))
    for prefix in prefixes:
        pattern = re.compile(rf"^{re.escape(prefix)}\d{{4}}_({side_pattern})\.pkl$")
        for name in names:
            matched = pattern.match(name)
            if not matched:
                continue
            frame = _read_rank(root / name, partition)
            if frame.empty:
                continue
            if prefix in cuts:
                frame = frame.loc[frame["trading_date"].lt(cuts[prefix])]
            if not frame.empty:
                parts.append(frame.assign(contract=name.rsplit("_", 1)[0], side=matched.group(1)))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def symbols_with_data(partition: str = "research") -> list[str]:
    root = external_root(partition) / "dominant"
    _guard(root, partition)
    out = []
    for symbol in C.COMMODITY_SYMBOLS:
        if (root / f"{symbol}_rank1.pkl").exists() and len(load_dominant(symbol, 1, partition)):
            out.append(symbol)
    return out


def _leg(contracts: pd.DataFrame, dominant: pd.Series, prefix: str) -> pd.DataFrame:
    """某一档合约在每个日期的行情，以及同一合约前一交易日收盘（用于无跳空收益）。"""
    px = contracts.set_index(["contract", "date"])
    prev_close = contracts.groupby("contract")["close"].shift(1)
    prev_turnover = contracts.groupby("contract")["total_turnover"].shift(1)
    prev_close.index = px.index
    prev_turnover.index = px.index
    key = pd.MultiIndex.from_arrays([dominant.values, dominant.index])
    leg = pd.DataFrame({
        f"{prefix}contract": dominant.values,
        f"{prefix}close": px["close"].reindex(key).values,
        f"{prefix}prev_close": prev_close.reindex(key).values,
        f"{prefix}turnover": px["total_turnover"].reindex(key).values,
        f"{prefix}prev_turnover": prev_turnover.reindex(key).values,
        f"{prefix}oi": px["open_interest"].reindex(key).values,
        f"{prefix}volume": px["volume"].reindex(key).values,
    }, index=dominant.index)
    return leg


def build_daily_panel(partition: str = "research", symbols: list[str] | None = None) -> pd.DataFrame:
    """每个品种、每个交易日一行：主力 (c1) 与次主力 (c2) 收盘、同合约日收益、品种总持仓和成交额。

    收益 ret1_t = 当日主力合约 close_t / 同一合约 close_{t-1} - 1，换月不产生价格跳空，
    等价于论文的比例法后复权。米筐主力用前一交易日持仓决定。更名重叠期内改用前一日吨持仓，
    超过在任主力 1.1 倍才切换；这一天的持仓和成交量按吨合计，其余日子仍按手合计。
    """
    contracts = load_contracts(partition)
    leg_contracts = contracts if partition == "research" else load_contracts_span(partition)
    symbols = symbols or symbols_with_data(partition)
    totals = contracts.groupby(["symbol", "date"]).agg(
        oi_total=("open_interest", "sum"), turnover_total=("total_turnover", "sum"),
        volume_total=("volume", "sum"))
    multipliers = load_contract_multipliers(partition) if partition == "research" else pd.Series(dtype=float)
    if partition == "research":
        totals = scale_mixed_multiplier_totals(contracts, totals, multipliers)
    parts = []
    for symbol in symbols:
        r1, r2 = adjusted_dominant(symbol, partition, multipliers, leg_contracts)
        if r1.empty:
            continue
        sub = leg_contracts[leg_contracts["symbol"] == symbol]
        leg1 = _leg(sub, r1, "c1_")
        leg2 = _leg(sub, r2.dropna(), "c2_").reindex(r1.index)
        frame = leg1.join(leg2)
        ry = load_roll_yield(symbol, partition)
        if not ry.empty:
            frame["roll_yield_ann"] = ry["annualized_yield"].reindex(frame.index)
        else:
            frame["roll_yield_ann"] = np.nan
        frame["symbol"] = symbol
        frame.index.name = "date"
        parts.append(frame.reset_index())
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.join(totals, on=["symbol", "date"])
    panel["ret1"] = panel["c1_close"] / panel["c1_prev_close"] - 1.0
    panel["ret2"] = panel["c2_close"] / panel["c2_prev_close"] - 1.0
    panel["roll"] = panel.groupby("symbol")["c1_contract"].transform(lambda s: s.ne(s.shift()))
    first = panel.groupby("symbol").cumcount() == 0
    panel.loc[first, "roll"] = False
    panel["sector"] = panel["symbol"].map(C.SECTOR)
    _check_dates(panel["date"], partition, "日面板")
    return panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def symbols_with_member_rank(partition: str) -> set[str]:
    """有品种级多头榜文件的商品代码。完整区间同时查看研究期和样本外。"""
    parts = ("research", "oos") if partition == "combined" else (partition,)
    names: set[str] = set()
    for part in parts:
        root = C.daily_dir(part) / "member_rank"
        if part == "research":
            C.assert_research_only(root)
        names.update(path.stem.removesuffix("_long") for path in root.glob("*_long.pkl"))
    return names


def member_universe(panel: pd.DataFrame, start: pd.Timestamp, partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """流动性资格、主力映射和评价期内具有会员榜文件的商品品种。"""
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=eligibility(panel))
    eligible = panel.pivot(index="date", columns="symbol", values="eligible").eq(True)
    ranked = symbols_with_member_rank(partition)
    names = [name for name in eligible.columns[eligible.loc[start:].any()] if name in ranked]
    dominant = panel.pivot(index="date", columns="symbol", values="c1_contract")
    return eligible[names], dominant[names]


def eligibility(panel: pd.DataFrame) -> pd.Series:
    """时点品种池：过去 242 日成交额中位数、当日主力有成交。收盘时当日成交额已知。"""
    universe = C.UNIVERSE
    grouped = panel.groupby("symbol", sort=False)
    median = grouped["turnover_total"].transform(
        lambda s: s.rolling(
            universe["turnover_window_days"], min_periods=universe["turnover_window_days"]
        ).median()
    )
    ok = median >= universe["min_median_daily_turnover_cny"]
    if universe["require_dominant_volume_positive"]:
        ok &= panel["c1_volume"] > 0
    ok &= panel["ret1"].notna()
    ok &= ~panel["symbol"].isin(universe["excluded_symbols"])
    return ok.fillna(False)


PANEL_DIR = C.study_dir("panel")


def build_panel_cache() -> pd.DataFrame:
    """重建研究期日面板缓存与逐年品种池统计。"""
    import json
    import time

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel = build_daily_panel("research")
    C.assert_no_holdout_dates(panel["date"], "日面板")
    panel["eligible"] = eligibility(panel)
    panel.to_parquet(PANEL_DIR / "daily_panel.parquet", index=False)
    eligible = panel[panel["eligible"]]
    by_year = eligible.groupby(eligible["date"].dt.year)["symbol"].agg(["nunique"])
    by_year["avg_daily"] = eligible.groupby(
        [eligible["date"].dt.year, "date"]).size().groupby(level=0).mean().round(1)
    by_year.to_csv(PANEL_DIR / "universe_by_year.csv")
    members = eligible.groupby(eligible["date"].dt.year)["symbol"].unique().apply(sorted)
    (PANEL_DIR / "universe_members.json").write_text(
        json.dumps({int(year): list(symbols) for year, symbols in members.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(by_year.to_string())
    print(f"日面板 {panel.shape}，用时 {time.time() - started:.0f}s → {PANEL_DIR}")
    return panel


def load_panel() -> pd.DataFrame:
    """读取研究期日面板缓存；缺失时重建。"""
    path = PANEL_DIR / "daily_panel.parquet"
    if not path.exists():
        return build_panel_cache()
    C.assert_research_only(path)
    panel = pd.read_parquet(path)
    C.assert_no_holdout_dates(panel["date"], "日面板")
    return panel


def load_contracts_span(partition: str = "research", *, for_execution: bool = False) -> pd.DataFrame:
    """研究期只读研究分区。样本外评价把研究期合约接到样本外之前，供滚动窗口预热。"""
    if partition == "research":
        return load_contracts("research", for_execution=for_execution)
    if partition != "oos":
        raise ValueError(f"未知分区 {partition}")
    research = load_contracts("research", for_execution=for_execution)
    oos = load_contracts("oos", for_execution=for_execution)
    out = pd.concat([research, oos], ignore_index=True)
    if out.duplicated(["contract", "date"]).any():
        raise ValueError("合约日行情在研究期与样本外分区重复")
    return out.sort_values(["symbol", "contract", "date"], kind="mergesort").reset_index(drop=True)


def _concat_rank_frames(research: pd.DataFrame, oos: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if research.empty:
        return oos
    if oos.empty:
        return research
    if not research.merge(oos, on=keys, how="inner").empty:
        raise ValueError("会员排名在研究期与样本外分区重复")
    return pd.concat([research, oos], ignore_index=True)


def load_member_rank_span(symbol: str) -> pd.DataFrame:
    """样本外和完整区间共用：研究期排名接在样本外排名之前。"""
    return _concat_rank_frames(
        load_member_rank(symbol, "research"),
        load_member_rank(symbol, "oos"),
        ["trading_date", "side", "rank", "member_name"],
    )


def load_evaluation_panel() -> pd.DataFrame:
    """研究期接样本外的日面板，供样本外评价（研究期只作预热）和完整区间连续回测使用，不写入研究缓存。"""
    research = load_panel().drop(columns=["eligible"], errors="ignore")
    oos = build_daily_panel("oos")
    missing = [column for column in oos.columns if column not in research.columns]
    if missing:
        raise ValueError(f"研究期日面板缺少字段: {missing}")
    panel = pd.concat([research[list(oos.columns)], oos], ignore_index=True)
    if panel.duplicated(["symbol", "date"]).any():
        raise ValueError("日面板在研究期与样本外分区重复")
    panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    panel["roll"] = panel.groupby("symbol")["c1_contract"].transform(lambda s: s.ne(s.shift()))
    panel.loc[panel.groupby("symbol").cumcount() == 0, "roll"] = False
    panel["eligible"] = eligibility(panel)
    evaluation = panel.loc[panel["date"].ge(pd.Timestamp(C.OOS_START)), "date"]
    C.assert_oos_dates(evaluation, "样本外日面板")
    return panel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建研究期日面板缓存")
    parser.parse_args()
    build_panel_cache()

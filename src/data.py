"""读取米筐商品日频数据，构造主力 / 次主力日面板与时点流动性池。研究期与样本外分开读取。"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

PARTITIONS = C.PARTITIONS


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


def load_contracts(partition: str = "research", raw_units: bool = False) -> pd.DataFrame:
    """全部商品合约日行情，长表：contract、date、symbol 与价格/量/持仓字段。

    默认返回单边口径的成交量、成交额与持仓量；raw_units=True 只供数据检查核对原始口径。
    """
    root = external_root(partition) / "contracts"
    _guard(root, partition)
    frames = []
    for path in sorted(root.glob("*.pkl")):
        _guard(path, partition)
        frames.append(pd.read_pickle(path))
    frames = [f for f in frames if f is not None and not f.empty]
    out = pd.concat(frames).reset_index().rename(columns={"order_book_id": "contract"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["contract"].str.extract(r"^([A-Z]+)\d{4}$", expand=False)
    if out["symbol"].isna().any():
        bad = out.loc[out["symbol"].isna(), "contract"].unique()[:5]
        raise ValueError(f"合约代码无法解析品种: {bad}")
    if out.duplicated(["contract", "date"]).any():
        raise ValueError("合约日行情有重复 (contract, date)")
    if partition == "research":
        out = apply_legacy_contracts(out)
    _check_dates(out["date"], partition, "合约日行情")
    if not raw_units:
        out = to_single_sided(out)
    return out.sort_values(["symbol", "contract", "date"], kind="mergesort").reset_index(drop=True)


# 郑商所更名前代码。只保留后继品种上市日之前的交易日，切换日当天起用新代码。
LEGACY_UNDERLYING = {
    "ME": ("MA", pd.Timestamp("2014-06-17")),
    "TC": ("ZC", pd.Timestamp("2015-05-18")),
    "RO": ("OI", pd.Timestamp("2012-07-16")),
    "WS": ("WH", pd.Timestamp("2012-07-24")),
    "ER": ("RI", pd.Timestamp("2012-07-24")),
}


def legacy_sources(symbol: str) -> list[tuple[str, pd.Timestamp]]:
    """后继品种在上市日前对应的旧代码。"""
    return [(old, cut) for old, (new, cut) in LEGACY_UNDERLYING.items() if new == symbol]


def apply_legacy_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    """把更名前合约并入后继品种；切换日及之后的旧代码行不保留。"""
    if frame.empty or not frame["symbol"].isin(LEGACY_UNDERLYING).any():
        return frame
    kept = [frame.loc[~frame["symbol"].isin(LEGACY_UNDERLYING)]]
    for old, (new, cut) in LEGACY_UNDERLYING.items():
        part = frame.loc[frame["symbol"].eq(old) & frame["date"].lt(cut)].copy()
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
    研究期在文件存在时并入更名前代码、且只保留后继品种上市日之前的排名。
    """
    out = _member_rank_frame(symbol, partition)
    if partition != "research":
        return out
    extras = []
    for old, cut in legacy_sources(symbol):
        extra = _member_rank_frame(old, partition)
        extra = extra.loc[extra["trading_date"].lt(cut)].copy()
        if extra.empty:
            continue
        extra["symbol"] = symbol
        extras.append(extra)
    if not extras:
        return out
    return pd.concat([out, *extras], ignore_index=True)


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


def member_rank_daily(symbol: str, contracts: pd.DataFrame, partition: str = "research") -> pd.DataFrame:
    """品种日度汇总：前 20 名多、空持仓合计，入榜合约数与入榜合约单边持仓合计（排名占比的分母）。"""
    ranks = load_member_rank(symbol, partition)
    if ranks.empty:
        return pd.DataFrame()
    held = ranks[ranks["side"].isin(["long", "short"])]
    top = held.pivot_table(index="trading_date", columns="side", values="volume", aggfunc="sum")
    top = top.reindex(columns=["long", "short"]).add_prefix("top20_")
    top["n_long"] = held[held.side == "long"].groupby("trading_date").size()
    top["n_short"] = held[held.side == "short"].groupby("trading_date").size()
    by_contract = load_member_rank_contracts(symbol, partition)
    if not by_contract.empty:
        ranked = by_contract[["trading_date", "contract"]].drop_duplicates()
        oi = contracts.loc[contracts["symbol"] == symbol, ["date", "contract", "open_interest"]]
        ranked = ranked.merge(oi, left_on=["trading_date", "contract"], right_on=["date", "contract"], how="left")
        agg = ranked.groupby("trading_date").agg(n_ranked_contracts=("contract", "nunique"),
                                                 ranked_oi=("open_interest", "sum"),
                                                 ranked_oi_missing=("open_interest", lambda s: int(s.isna().sum())))
        top = top.join(agg)
    total = contracts[contracts["symbol"] == symbol].groupby("date")["open_interest"].sum()
    top["total_oi"] = total.reindex(top.index).values
    top["symbol"] = symbol
    top.index.name = "trading_date"
    return top.reset_index()


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
    等价于论文的比例法后复权；主力 rank1 由米筐按前一交易日数据确定。
    """
    contracts = load_contracts(partition)
    leg_contracts = load_contracts_span(partition)
    symbols = symbols or symbols_with_data(partition)
    totals = contracts.groupby(["symbol", "date"]).agg(
        oi_total=("open_interest", "sum"), turnover_total=("total_turnover", "sum"),
        volume_total=("volume", "sum"))
    parts = []
    for symbol in symbols:
        r1 = load_dominant(symbol, 1, partition)
        if r1.empty:
            continue
        r2 = load_dominant(symbol, 2, partition).reindex(r1.index)
        r2 = r2.where(r2 != r1)  # 米筐偶有次主力等于主力的日子，置空
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


def load_contracts_span(partition: str = "research") -> pd.DataFrame:
    """研究期只读研究分区。样本外评价把研究期合约接到样本外之前，供滚动窗口预热。"""
    if partition == "research":
        return load_contracts("research")
    if partition != "oos":
        raise ValueError(f"未知分区 {partition}")
    research = load_contracts("research")
    oos = load_contracts("oos")
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
    """样本外因子预热：研究期排名接在样本外排名之前。"""
    return _concat_rank_frames(
        load_member_rank(symbol, "research"),
        load_member_rank(symbol, "oos"),
        ["trading_date", "side", "rank", "member_name"],
    )


def load_evaluation_panel() -> pd.DataFrame:
    """样本外评价面板。研究期行情只作预热，不写入研究缓存。"""
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

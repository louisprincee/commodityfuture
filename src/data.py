"""读取米筐商品日频数据并构造主力 / 次主力日面板。研究期与样本外分开读取。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

PARTITIONS = ("research", "oos")


def external_root(partition: str) -> Path:
    if partition == "research":
        return C.EXTERNAL_DIR
    if partition == "oos":
        return C.EXTERNAL_OOS_DIR
    raise C.HoldoutViolation(f"未知分区 {partition}")


def _guard(path: Path, partition: str) -> None:
    if partition == "research":
        C.assert_research_only(path)


def _check_dates(dates, partition: str, what: str) -> None:
    if partition == "research":
        C.assert_no_holdout_dates(dates, what)
    else:
        C.assert_oos_dates(dates, what)


def load_contracts(partition: str = "research") -> pd.DataFrame:
    """全部商品合约日行情，长表：contract、date、symbol 与价格/量/持仓字段。"""
    root = external_root(partition) / "contracts"
    _guard(root, partition)
    frames = [pd.read_pickle(root / f) for f in sorted(os.listdir(root)) if f.endswith(".pkl")]
    frames = [f for f in frames if f is not None and not f.empty]
    out = pd.concat(frames).reset_index().rename(columns={"order_book_id": "contract"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["contract"].str.extract(r"^([A-Z]+)\d{4}$", expand=False)
    if out["symbol"].isna().any():
        bad = out.loc[out["symbol"].isna(), "contract"].unique()[:5]
        raise ValueError(f"合约代码无法解析品种: {bad}")
    if out.duplicated(["contract", "date"]).any():
        raise ValueError("合约日行情有重复 (contract, date)")
    _check_dates(out["date"], partition, "合约日行情")
    return out.sort_values(["symbol", "contract", "date"], kind="mergesort").reset_index(drop=True)


def load_dominant(symbol: str, rank: int, partition: str = "research") -> pd.Series:
    """米筐 get_dominant(rank) 的日序列；上市前为空序列。"""
    path = external_root(partition) / "dominant" / f"{symbol}_rank{rank}.pkl"
    _guard(path, partition)
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0] if obj.shape[1] else pd.Series(dtype=object)
    obj = obj.dropna()
    obj.index = pd.to_datetime(obj.index).normalize()
    obj.name = f"rank{rank}"
    _check_dates(obj.index, partition, f"{symbol} rank{rank}")
    return obj.astype(str)


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
    prev_close.index = px.index
    key = pd.MultiIndex.from_arrays([dominant.values, dominant.index])
    leg = pd.DataFrame({
        f"{prefix}contract": dominant.values,
        f"{prefix}close": px["close"].reindex(key).values,
        f"{prefix}prev_close": prev_close.reindex(key).values,
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
        sub = contracts[contracts["symbol"] == symbol]
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

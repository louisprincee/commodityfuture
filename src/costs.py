"""合约手续费、保证金、成交容量及历史交易参数。"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

BROKER_MARGIN_ADDON = 0.03
VOLUME_FRACTION = 0.01
PARAMETER_CACHE = C.RESEARCH_EXECUTION_PARAMETERS / "contracts"


@dataclass(frozen=True)
class Quote:
    open: float
    settle: float
    multiplier: float
    tick: float
    long_margin: float
    short_margin: float
    by_lot: bool
    open_fee: float
    close_fee: float
    min_order: int
    max_order: int
    limit: int
    open_cap: int = 10**9
    fee_fallback: bool = False
    exit_cap: int = 10**9
    limit_up: float = np.nan
    limit_down: float = np.nan

    @property
    def can_trade(self):
        return (self.open > 0 and self.multiplier > 0 and self.tick > 0 and
                self.max_order > 0 and isfinite(self.open_fee) and isfinite(self.close_fee))

    def limit_blocks(self, delta):
        return ((delta > 0 and self.limit_up > 0 and self.open >= self.limit_up - 1e-8)
                or (delta < 0 and self.limit_down > 0 and self.open <= self.limit_down + 1e-8))


def _positive_int(value, default):
    return max(1, floor(value)) if np.isfinite(value) and value > 0 else default


def load_market(bars: pd.DataFrame, fees: pd.DataFrame, ids, start, root: Path = PARAMETER_CACHE,
                flat_margin: float | None = None, partition: str = "research"):
    """交易参数与合约日线对齐，成交量容量仅使用截至前日的数据。"""
    if flat_margin is not None and not (0 < flat_margin <= 1):
        raise ValueError("flat_margin 必须在 (0, 1] 内")
    if partition == "research":
        C.assert_research_only(root)
        check_dates = C.assert_no_holdout_dates
    elif partition == "oos":
        check_dates = C.assert_oos_dates
    else:
        raise ValueError(f"未知分区 {partition}")
    names = set(ids)
    bars = bars.loc[bars.contract.isin(names)].sort_values(["contract", "date"]).copy()
    bars["date"] = pd.to_datetime(bars["date"]).astype("datetime64[ns]")
    bars["prior_volume"] = bars.groupby("contract")["volume"].transform(
        lambda volume: volume.shift(1).rolling(20, min_periods=5).median()
    )
    bars = bars.loc[bars.date.ge(start)]
    tables = []
    price_limits = []
    for contract in sorted(names):
        path = root / f"{contract}.pkl"
        if partition == "research":
            C.assert_research_only(path)
        if path.is_file():
            table = pd.read_pickle(path)
            check_dates(table.index, f"{contract} 交易参数")
            if not table.empty:
                tables.append(table.rename_axis("date").reset_index().assign(contract=contract))
        path = root.parent / "price_limits" / f"{contract}.pkl"
        if partition == "research":
            C.assert_research_only(path)
        if path.is_file():
            table = pd.read_pickle(path)
            check_dates(table.index, f"{contract} 涨跌停价")
            if not table.empty:
                price_limits.append(table.rename_axis("date").reset_index().assign(contract=contract))
    if not tables:
        raise RuntimeError("历史执行参数为空")
    actual = pd.concat(tables, ignore_index=True)
    bars = bars.merge(actual, on=["contract", "date"], how="left", validate="one_to_one")
    if not price_limits:
        raise RuntimeError("历史涨跌停价为空")
    bars = bars.merge(pd.concat(price_limits, ignore_index=True), on=["contract", "date"],
                      how="left", validate="one_to_one")
    fallback = fees.rename(columns={"trading_date": "date"}).copy()
    fallback["date"] = pd.to_datetime(fallback["date"]).astype("datetime64[ns]")
    fallback = fallback.sort_values("date")
    cols = ["commission_type", "open_commission", "close_commission"]
    specs = ["contract_multiplier", "tick_size"]
    fallback = pd.merge_asof(
        bars[["contract", "date"]].sort_values("date"),
        fallback[["contract", "date", *cols, *specs]], on="date", by="contract", direction="backward",
    ).set_index(["contract", "date"])
    bars = bars.set_index(["contract", "date"])
    actual_fee = bars[cols].notna().all(axis=1)
    bars[cols] = bars[cols].where(actual_fee, fallback[cols].reindex(bars.index), axis=0)
    bars[specs] = fallback[specs].reindex(bars.index)
    bars["fee_fallback"] = ~actual_fee
    bars = bars.reset_index()
    market = {}
    for row in bars.itertuples(index=False):
        ratio_long = row.long_margin_ratio if 0 < row.long_margin_ratio <= 1 else np.nan
        ratio_short = row.short_margin_ratio if 0 < row.short_margin_ratio <= 1 else np.nan
        known_margin = isfinite(ratio_long) and isfinite(ratio_short)
        volume_cap = floor(row.prior_volume * VOLUME_FRACTION) if np.isfinite(row.prior_volume) else 0
        opening_cap = volume_cap
        if row.commission_type not in ("by_volume", "by_money") or not known_margin:
            opening_cap = 0
        max_order = _positive_int(row.max_order_quantity, 10**9)
        min_order = _positive_int(np.ceil(row.min_order_quantity), 1)
        long_margin = flat_margin if flat_margin is not None else (
            min(1.0, ratio_long + BROKER_MARGIN_ADDON) if isfinite(ratio_long) else 1.0
        )
        short_margin = flat_margin if flat_margin is not None else (
            min(1.0, ratio_short + BROKER_MARGIN_ADDON) if isfinite(ratio_short) else 1.0
        )
        market.setdefault(row.date, {})[row.contract] = Quote(
            row.open, row.settlement, row.contract_multiplier, row.tick_size,
            long_margin,
            short_margin,
            row.commission_type == "by_volume",
            row.open_commission, row.close_commission,
            min_order, max_order, max(0, floor(row.client_limit)) if isfinite(row.client_limit) else 10**9,
            opening_cap, row.fee_fallback, max(1, volume_cap), row.limit_up, row.limit_down,
        )
    return market

"""合约手续费、保证金、成交容量及历史交易参数。"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

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
    fee_source_date: pd.Timestamp | None = None
    fee_carried: bool = False
    fee_source_contract: str | None = None
    fee_cross_contract: bool = False
    last_trading_date: pd.Timestamp | None = None
    start_delivery_date: pd.Timestamp | None = None

    @property
    def can_trade(self):
        return (self.open > 0 and self.multiplier > 0 and self.tick > 0 and
                self.max_order > 0 and isfinite(self.open_fee) and isfinite(self.close_fee))

    def limit_blocks(self, delta):
        return ((delta > 0 and self.limit_up > 0 and self.open >= self.limit_up - 1e-8)
                or (delta < 0 and self.limit_down > 0 and self.open <= self.limit_down + 1e-8))


def _positive_int(value, default):
    return max(1, floor(value)) if np.isfinite(value) and value > 0 else default


def fill_historical_fees(bars: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """仅在有来源的有效区间内补缺；不改已知费率、不向区间外外推。"""
    result = bars.copy()
    columns = ["commission_type", "open_commission", "close_commission"]
    dates = pd.to_datetime(result.index.get_level_values("date"))
    symbols = result.index.get_level_values("contract").str.extract(r"^([A-Z]+)", expand=False)
    for rule in rules.itertuples(index=False):
        selected = (symbols == rule.symbol) & (dates >= pd.Timestamp(rule.start)) & (dates <= pd.Timestamp(rule.end))
        for column in columns:
            missing = selected & result[column].isna().to_numpy()
            result.loc[missing, column] = getattr(rule, column)
    return result


def carry_forward_fees(bars: pd.DataFrame) -> pd.DataFrame:
    """同合约前值优先，再取同品种此前无冲突的费率；不倒填或混合收费单位。"""
    result = bars.sort_index().copy()
    cols = ["commission_type", "open_commission", "close_commission"]
    for column in cols[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    known = result[cols].notna().all(axis=1)
    known &= result.commission_type.isin(("by_volume", "by_money"))
    known &= np.isfinite(result.open_commission) & np.isfinite(result.close_commission)
    dates = pd.Series(result.index.get_level_values("date"), index=result.index)
    contracts = pd.Series(result.index.get_level_values("contract"), index=result.index)
    source = result.get("fee_source_date", dates).where(known)
    pool = result.loc[known, cols].copy()
    pool["fee_source_date"] = source.loc[known]
    pool["fee_source_contract"] = contracts.loc[known]
    pool["symbol"] = contracts.loc[known].str.extract(r"^([A-Z]+)", expand=False)
    # Several contracts can disclose different rates on one day. Such a day
    # supplies no common schedule for an unknown contract; do not pick one.
    pool = pool.drop_duplicates(["symbol", "fee_source_date", *cols])
    pool = pool.loc[~pool.duplicated(["symbol", "fee_source_date"], keep=False)]
    result[cols] = result[cols].where(known).groupby(level="contract").ffill()
    result["fee_source_date"] = source.groupby(level="contract").ffill()
    result["fee_source_contract"] = contracts.where(known).groupby(level="contract").ffill()
    missing = result[cols].isna().any(axis=1)
    if missing.any() and not pool.empty:
        query = pd.DataFrame({"contract": contracts.loc[missing].to_numpy(),
                              "date": dates.loc[missing].to_numpy(),
                              "symbol": contracts.loc[missing].str.extract(r"^([A-Z]+)", expand=False).to_numpy()})
        matched = pd.merge_asof(query.sort_values("date"),
                                pool.reset_index(drop=True).sort_values("fee_source_date"),
                                left_on="date", right_on="fee_source_date", by="symbol",
                                direction="backward", allow_exact_matches=False)
        matched = matched.set_index(["contract", "date"])
        fields = [*cols, "fee_source_date", "fee_source_contract"]
        result.loc[matched.index, fields] = matched[fields]
    result["fee_carried"] = result.fee_source_date.lt(dates)
    result["fee_cross_contract"] = result.fee_source_contract.notna() & result.fee_source_contract.ne(contracts)
    return result


_CONTRACT_TABLES: dict[tuple, pd.DataFrame | None] = {}


def _contract_table(path: Path, contract: str, partition: str, check_dates, what: str) -> pd.DataFrame | None:
    """读取单个合约的执行参数表。同一进程内按路径、修改时间和大小缓存；文件一变即重读。"""
    try:
        stat = path.stat()
        version = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        version = None
    key = (str(path), partition, version)
    if key not in _CONTRACT_TABLES:
        if partition == "research":
            C.assert_research_only(path)
        table = None
        if path.is_file():
            raw = pd.read_pickle(path)
            check_dates(raw.index, f"{contract} {what}")
            if not raw.empty:
                table = raw.rename_axis("date").reset_index().assign(contract=contract)
        _CONTRACT_TABLES[key] = table
    return _CONTRACT_TABLES[key]


def _quote_fields(bars: pd.DataFrame, flat_margin: float | None) -> dict[str, list]:
    """逐行规则的向量化版本：保证金、成交容量、单笔上下限与持仓限额。"""
    ratio_long = bars.long_margin_ratio.to_numpy(float)
    ratio_short = bars.short_margin_ratio.to_numpy(float)
    ratio_long = np.where((ratio_long > 0) & (ratio_long <= 1), ratio_long, np.nan)
    ratio_short = np.where((ratio_short > 0) & (ratio_short <= 1), ratio_short, np.nan)
    known_margin = np.isfinite(ratio_long) & np.isfinite(ratio_short)
    up, down = bars.limit_up.to_numpy(float), bars.limit_down.to_numpy(float)
    known_limits = np.isfinite(up) & (up > 0) & np.isfinite(down) & (down > 0)
    prior = bars.prior_volume.to_numpy(float) * VOLUME_FRACTION
    volume_cap = np.where(np.isfinite(prior), np.floor(np.where(np.isfinite(prior), prior, 0)), 0).astype(np.int64)
    fee_known = bars.commission_type.isin(("by_volume", "by_money")).to_numpy()
    opening = fee_known & known_margin & known_limits & bars.last_trading_date.notna().to_numpy()

    def positive(values, default):
        good = np.isfinite(values) & (values > 0)
        whole = np.floor(np.where(good, values, 1))
        return np.where(good, np.maximum(1, whole), default).astype(np.int64)

    def margin(ratio):
        if flat_margin is not None:
            return np.full(len(ratio), flat_margin, dtype=float)
        return np.where(np.isfinite(ratio), np.minimum(1.0, ratio + BROKER_MARGIN_ADDON), 1.0)

    client = bars.client_limit.to_numpy(float)
    client_ok = np.isfinite(client)
    limit = np.where(client_ok, np.maximum(0, np.floor(np.where(client_ok, client, 0))), 10**9).astype(np.int64)
    return {
        "long_margin": margin(ratio_long).tolist(),
        "short_margin": margin(ratio_short).tolist(),
        "by_lot": bars.commission_type.eq("by_volume").fillna(False).to_numpy(bool).tolist(),
        "min_order": positive(np.ceil(bars.min_order_quantity.to_numpy(float)), 1).tolist(),
        "max_order": positive(bars.max_order_quantity.to_numpy(float), 10**9).tolist(),
        "limit": limit.tolist(),
        "open_cap": np.where(opening, volume_cap, 0).tolist(),
        "exit_cap": np.maximum(0, volume_cap).tolist(),
    }


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
    tables = []
    price_limits = []
    specifications = []
    for contract in sorted(names):
        for folder, what, out in ((root, "交易参数", tables), (root.parent / "price_limits", "涨跌停价", price_limits),
                                  (root.parent / "specifications", "合约规格", specifications)):
            table = _contract_table(folder / f"{contract}.pkl", contract, partition, check_dates, what)
            if table is not None:
                out.append(table)
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
    fallback["fee_source_date"] = fallback["date"]
    fallback = fallback.sort_values("date")
    cols = ["commission_type", "open_commission", "close_commission"]
    specs = ["contract_multiplier", "tick_size"]
    fallback = pd.merge_asof(
        bars[["contract", "date"]].sort_values("date"),
        fallback[["contract", "date", *cols, *specs, "fee_source_date"]], on="date", by="contract", direction="backward",
    ).set_index(["contract", "date"])
    bars = bars.set_index(["contract", "date"])
    actual_fee = bars[cols].notna().all(axis=1)
    bars[cols] = bars[cols].where(actual_fee, fallback[cols].reindex(bars.index), axis=0)
    dates = pd.Series(bars.index.get_level_values("date"), index=bars.index)
    bars["fee_source_date"] = dates.where(actual_fee, fallback.fee_source_date.reindex(bars.index))
    bars[specs] = fallback[specs].reindex(bars.index)
    bars["last_trading_date"] = pd.NaT
    bars["start_delivery_date"] = pd.NaT
    if specifications:
        # Contract metadata is independent of commission coverage. Align exact
        # observations only, never backfill a later contract specification.
        independent = pd.concat(specifications, ignore_index=True).set_index(["contract", "date"])
        bars[specs] = independent[specs].reindex(bars.index).combine_first(bars[specs])
        for column in ("last_trading_date", "start_delivery_date"):
            if column in independent:
                bars[column] = pd.to_datetime(independent[column]).reindex(bars.index)
    bars["fee_fallback"] = ~actual_fee
    rules_path = C.CONFIG_DIR / "historical_fee_rules.csv"
    if partition == "research" and rules_path.is_file():
        known = bars[cols].notna().all(axis=1)
        bars = fill_historical_fees(bars, pd.read_csv(rules_path))
        supplemented = ~known & bars[cols].notna().all(axis=1)
        bars.loc[supplemented, "fee_source_date"] = dates.loc[supplemented]
    bars = carry_forward_fees(bars)
    bars = bars.reset_index()
    bars = bars.loc[bars.date.ge(start)]
    market = {}
    derived = _quote_fields(bars, flat_margin)
    columns = (
        bars.open, bars.settlement, bars.contract_multiplier, bars.tick_size,
        derived["long_margin"], derived["short_margin"], derived["by_lot"],
        bars.open_commission, bars.close_commission,
        derived["min_order"], derived["max_order"], derived["limit"],
        derived["open_cap"], bars.fee_fallback, derived["exit_cap"], bars.limit_up, bars.limit_down,
        bars.fee_source_date, bars.fee_carried, bars.fee_source_contract, bars.fee_cross_contract,
        bars.last_trading_date, bars.start_delivery_date,
    )
    for day, contract, *fields in zip(bars.date, bars.contract, *columns):
        market.setdefault(day, {})[contract] = Quote(*fields)
    return market

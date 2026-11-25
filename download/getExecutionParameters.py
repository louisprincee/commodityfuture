"""下载商品主力合约存续期交易参数与涨跌停价，覆盖换月后的旧合约。

    python -m download.getExecutionParameters --partition research
    python -m download.getExecutionParameters --partition oos
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from download.getExternal import C, CoverageCache, rqdata_credentials
from src import data as D  # noqa: E402

FIELDS = [
    "long_margin_ratio", "short_margin_ratio", "commission_type", "open_commission",
    "close_commission", "close_commission_today", "client_limit", "min_order_quantity",
    "max_order_quantity",
]
PARTITION_CACHE = {
    "research": C.RESEARCH_EXECUTION_PARAMETERS,
    "oos": C.OOS_EXECUTION_PARAMETERS,
}
PARTITION_START = {
    "research": pd.Timestamp("2016-01-04"),
    "oos": pd.Timestamp(C.OOS_START),
}


def _check_dates(values, partition: str, what: str) -> None:
    if partition == "oos":
        C.assert_oos_dates(values, what)
    else:
        C.assert_no_holdout_dates(values, what)


def _parameters(rq, contract, left, right, partition: str = "research"):
    out = rq.futures.get_trading_parameters(contract, left, right, fields=FIELDS)
    if out is None or out.empty:
        return pd.DataFrame(columns=FIELDS, index=pd.DatetimeIndex([], name="trading_date"))
    out = out.reset_index()
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.normalize()
    _check_dates(out["trading_date"], partition, f"{contract} 交易参数")
    return out.set_index("trading_date")[FIELDS].sort_index()


def _price_limits(rq, contract, left, right, partition: str = "research"):
    fields = ["limit_up", "limit_down"]
    out = rq.get_price(contract, start_date=left, end_date=right,
                       frequency="1d", fields=fields, adjust_type="none")
    if out is None or out.empty:
        return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], name="date"))
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    _check_dates(out["date"], partition, f"{contract} 涨跌停价")
    return out.set_index("date")[fields].sort_index()


def contract_spans(partition: str) -> pd.DataFrame:
    start = PARTITION_START[partition]
    wanted = set()
    for symbol in D.symbols_with_data(partition):
        dom = D.load_dominant(symbol, 1, partition)
        wanted.update(dom.loc[dom.index >= start].dropna().unique())
    bars = D.load_contracts(partition)[["contract", "date"]]
    return (
        bars.loc[bars["contract"].isin(wanted) & bars["date"].ge(start)]
        .groupby("contract")["date"]
        .agg(["min", "max"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=list(PARTITION_CACHE), default="research")
    parser.add_argument("--price-limits-only", action="store_true")
    args = parser.parse_args()
    credentials = rqdata_credentials()
    if not credentials:
        print("交易参数缺少米筐配置", file=sys.stderr)
        return 1
    import rqdatac as rq

    rq.init(*credentials)
    spans = contract_spans(args.partition)
    cache_dir = PARTITION_CACHE[args.partition]
    if args.partition == "research":
        C.assert_research_only(cache_dir)
    cache = CoverageCache(cache_dir)
    try:
        for contract, (first, last) in spans.iterrows():
            if not args.price_limits_only:
                cache.fetch("contracts", contract, first.date(), last.date(),
                            lambda left, right, contract=contract: _parameters(
                                rq, contract, left, right, args.partition))
            cache.fetch("price_limits", contract, first.date(), last.date(),
                        lambda left, right, contract=contract: _price_limits(
                            rq, contract, left, right, args.partition))
    finally:
        cache.flush()
    print(f"{args.partition} 交易参数合约 {len(spans)}；接口失败 {len(cache.failures)}")
    return int(bool(cache.failures))


if __name__ == "__main__":
    raise SystemExit(main())

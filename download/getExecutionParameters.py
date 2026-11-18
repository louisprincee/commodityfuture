"""下载 2016–2021 商品主力合约存续期交易参数，覆盖换月后的旧合约。"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from download.getExternal import C, CoverageCache, rqdata_credentials
from research.__main__ import D, load_panel

FIELDS = [
    "long_margin_ratio", "short_margin_ratio", "commission_type", "open_commission",
    "close_commission", "close_commission_today", "client_limit", "min_order_quantity",
    "max_order_quantity",
]
CACHE = C.RESEARCH_EXECUTION_PARAMETERS


def _parameters(rq, contract, left, right):
    out = rq.futures.get_trading_parameters(contract, left, right, fields=FIELDS)
    if out is None or out.empty:
        return pd.DataFrame(columns=FIELDS, index=pd.DatetimeIndex([], name="trading_date"))
    out = out.reset_index()
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.normalize()
    C.assert_no_holdout_dates(out["trading_date"], f"{contract} 交易参数")
    return out.set_index("trading_date")[FIELDS].sort_index()


def _price_limits(rq, contract, left, right):
    fields = ["limit_up", "limit_down"]
    out = rq.get_price(contract, start_date=left, end_date=right,
                       frequency="1d", fields=fields, adjust_type="none")
    if out is None or out.empty:
        return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], name="date"))
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    C.assert_no_holdout_dates(out["date"], f"{contract} 涨跌停价")
    return out.set_index("date")[fields].sort_index()


def main():
    import rqdatac as rq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-limits-only", action="store_true")
    args = parser.parse_args()
    credentials = rqdata_credentials()
    if not credentials:
        print("交易参数缺少米筐配置", file=sys.stderr)
        return 1
    rq.init(*credentials)
    panel = load_panel()
    start = pd.Timestamp("2016-01-04")
    candidates = panel.loc[panel["date"].ge(start) & panel["symbol"].isin(C.COMMODITY_SYMBOLS), "c1_contract"]
    wanted = candidates.dropna().unique()
    bars = D.load_contracts("research")[["contract", "date"]]
    spans = bars.loc[bars["contract"].isin(wanted) & bars["date"].ge(start)].groupby("contract")["date"].agg(["min", "max"])
    C.assert_research_only(CACHE)
    cache = CoverageCache(CACHE)
    try:
        for contract, (first, last) in spans.iterrows():
            if not args.price_limits_only:
                cache.fetch("contracts", contract, first.date(), last.date(),
                            lambda left, right, contract=contract: _parameters(rq, contract, left, right))
            cache.fetch("price_limits", contract, first.date(), last.date(),
                        lambda left, right, contract=contract: _price_limits(rq, contract, left, right))
    finally:
        cache.flush()
    print(f"交易参数合约 {len(spans)}；接口失败 {len(cache.failures)}")
    return int(bool(cache.failures))


if __name__ == "__main__":
    raise SystemExit(main())

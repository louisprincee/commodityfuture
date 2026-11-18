"""按主力合约逐日下载商品期货手续费与成本换算所需规格，写成 fee_history.csv。

    python -m download.getFeeHistory --partition research
    python -m download.getFeeHistory --partition oos

写入 data/{research,oos}/execution/fee_history.csv。
保存历史手续费、合约乘数与合约最小变动价位。米筐部分交易所早期手续费缺失，
读取方须显式处理；tick 情景不等同于盘口价差或冲击成本。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config as C  # noqa: E402
from rq_auth import rqdata_credentials  # noqa: E402

FIELDS = ["commission_type", "open_commission", "close_commission", "close_commission_today"]
WINDOWS = {partition: tuple(day.strftime("%Y%m%d") for day in bounds)
           for partition, bounds in C.PARTITION_WINDOWS.items()}
PATHS = {"research": C.RESEARCH_FEE_PATH, "oos": C.OOS_FEE_PATH}


def one_symbol(rq, symbol: str, start: str, end: str) -> pd.DataFrame | None:
    dominant = rq.futures.get_dominant(symbol, start, end)
    if dominant is None or dominant.empty:
        return None
    multiplier = rq.futures.get_contract_multiplier(symbol, start, end)
    if multiplier is None or multiplier.empty:
        raise ValueError(f"{symbol} 缺少历史合约乘数")
    multiplier = multiplier.reset_index()
    multiplier["date"] = pd.to_datetime(multiplier["date"]).dt.normalize()
    multiplier = multiplier.set_index("date")["contract_multiplier"].sort_index()
    multiplier = multiplier[~multiplier.index.duplicated(keep="last")]

    parts = []
    for contract, days in dominant.groupby(dominant):
        days = pd.to_datetime(days.index)
        table = rq.futures.get_trading_parameters(contract, days.min(), days.max(), fields=FIELDS)
        if table is None or table.empty:
            continue
        table = table.reset_index()
        table["trading_date"] = pd.to_datetime(table.trading_date)
        parts.append(table[table.trading_date.isin(days)])
    if not parts:
        return None
    out = pd.concat(parts).rename(columns={"order_book_id": "contract"})
    out = out.dropna(subset=["commission_type", "open_commission", "close_commission"])
    if out.empty:
        return None
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.normalize()
    multiplier = multiplier.reindex(
        multiplier.index.union(out["trading_date"])
    ).sort_index().ffill()
    out["contract_multiplier"] = multiplier.reindex(out["trading_date"]).to_numpy()
    if out["contract_multiplier"].isna().any():
        raise ValueError(f"{symbol} 手续费日期缺少历史合约乘数")
    contracts = out["contract"].drop_duplicates().tolist()
    instruments = rq.instruments(contracts)
    if not isinstance(instruments, (list, tuple)):
        instruments = [instruments]
    tick_sizes = {}
    for instrument in instruments:
        contract = instrument.order_book_id
        tick_size = instrument.tick_size()
        if tick_size is None or tick_size <= 0:
            raise ValueError(f"{contract} 缺少有效最小变动价位")
        tick_sizes[contract] = float(tick_size)
    if set(tick_sizes) != set(contracts):
        raise ValueError(f"{symbol} 期货合约规格返回不完整")
    out["tick_size"] = out["contract"].map(tick_sizes)
    out.insert(0, "symbol", symbol)
    return out.sort_values("trading_date")[
        ["symbol", "trading_date", "contract", *FIELDS, "contract_multiplier", "tick_size"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=list(WINDOWS), default="research")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="只下载这些品种；已有文件中其他品种保留")
    parser.add_argument("--start-date", default=None, help="YYYYMMDD，不早于分区起点")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD，不晚于分区终点")
    args = parser.parse_args()
    names = sorted({s.upper() for s in args.symbols}) if args.symbols else sorted(C.COMMODITY_SYMBOLS)
    financial = sorted(set(names) & set(C.FINANCIAL_SYMBOLS))
    if financial:
        parser.error(f"本仓库只下载商品期货，拒绝金融期货: {financial}")
    start, end = WINDOWS[args.partition]
    start = max(start, args.start_date or start)
    end = min(end, args.end_date or end)
    if start > end:
        parser.error(f"日期窗口落在分区 {WINDOWS[args.partition]} 之外")

    credentials = rqdata_credentials()
    if credentials is None:
        parser.error("请在 config/rqdata.env 填写米筐凭证")
    import rqdatac as rq

    rq.init(*credentials)
    parts = []
    for symbol in names:
        table = one_symbol(rq, symbol, start, end)
        print(symbol, 0 if table is None else len(table), flush=True)
        if table is not None:
            parts.append(table)
    path = PATHS[args.partition]
    if not parts:
        print("窗口内没有主力合约，未改文件", path)
        return 0
    table = pd.concat(parts, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["trading_date"])
        replaced = old.symbol.isin(set(table.symbol)) & old.trading_date.between(
            pd.Timestamp(start), pd.Timestamp(end))
        table = pd.concat([old[~replaced], table], ignore_index=True)
    table = table.sort_values(["symbol", "trading_date"])
    if args.partition == "oos":
        C.assert_oos_dates(table.trading_date, "手续费下载")
    else:
        C.assert_no_holdout_dates(table.trading_date, "手续费下载")
    if table.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("手续费有重复品种/日期")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(path, table.shape, table.symbol.nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

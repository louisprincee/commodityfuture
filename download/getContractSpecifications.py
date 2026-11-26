"""Download expired-contract specifications independently of commission coverage.

    python -m download.getContractSpecifications --start-date 20100104 --end-date 20211231

Each dated row describes the actual expired contract, not its renamed modern
underlying. This matters when old and new contract sizes overlap (TC versus ZC,
ME versus MA, RO versus OI). No current-contract substitute is permitted.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os

import numpy as np
import pandas as pd

from download.getExecutionParameters import PARTITION_CACHE, contract_spans
from download.getExternal import parse_day, rqdata_credentials
from src import config as C
from src import data as D


def specification_rows(instrument, days):
    """Metadata is specific to this expired contract's contractual terms."""
    multiplier = float(instrument.contract_multiplier)
    tick = float(instrument.tick_size())
    if not np.isfinite(multiplier) or multiplier <= 0 or not np.isfinite(tick) or tick <= 0:
        raise ValueError("nonpositive contract specification")
    def term(name):
        value = getattr(instrument, name, None)
        return pd.NaT if value in (None, "0000-00-00", "") else pd.Timestamp(value).normalize()

    final_dates = [value for value in (term("maturity_date"), term("de_listed_date")) if pd.notna(value)]
    if not final_dates:
        raise ValueError("missing contractual maturity")
    return pd.DataFrame({"contract_multiplier": multiplier, "tick_size": tick,
                         "last_trading_date": min(final_dates),
                         "start_delivery_date": term("start_delivery_date"),
                         "source": f"rqdatac.instruments:{instrument.order_book_id}",
                         "underlying_symbol": instrument.underlying_symbol},
                        index=pd.DatetimeIndex(days, name="date"))


def write_multipliers(rq, contract_ids, path):
    """把供应商返回的静态合约乘数写入一张表。缺一条或非正就失败，不补数。"""
    rows = []
    identifiers = sorted(set(contract_ids))
    for offset in range(0, len(identifiers), 100):
        batch = identifiers[offset:offset + 100]
        instruments = rq.instruments(batch)
        if not isinstance(instruments, (list, tuple)):
            instruments = [instruments]
        for instrument in instruments:
            if instrument is None:
                continue
            multiplier = float(instrument.contract_multiplier)
            if not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError(f"{instrument.order_book_id} 乘数无效")
            rows.append({
                "contract": instrument.order_book_id,
                "contract_multiplier": multiplier,
                "source": f"rqdatac.instruments:{instrument.order_book_id}",
            })
    table = pd.DataFrame(rows).drop_duplicates("contract")
    missing = sorted(set(identifiers) - set(table["contract"]))
    if missing:
        raise ValueError(f"缺少合约乘数: {missing[:8]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    table.sort_values("contract").to_csv(temporary, index=False)
    os.replace(temporary, path)
    return len(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=list(PARTITION_CACHE), default="research")
    parser.add_argument("--start-date", type=parse_day, default=parse_day("20100104"))
    parser.add_argument("--end-date", type=parse_day, default=parse_day("20211231"))
    parser.add_argument("--multipliers", action="store_true",
                        help="只把研究分区里更名品种及其后继合约的静态乘数写入 contract_multipliers.csv")
    args = parser.parse_args()
    if args.multipliers:
        if args.partition != "research":
            raise SystemExit("合约乘数表只写入研究分区")
        bars = D.load_contracts(args.partition)
        code = bars["contract"].str.extract(r"^([A-Z]+)", expand=False)
        prefixes = set(D.LEGACY_UNDERLYING) | set(D.legacy_successor_symbols())
        identifiers = bars.loc[code.isin(prefixes), "contract"].unique()
        credentials = rqdata_credentials()
        if not credentials:
            raise RuntimeError("missing RiceQuant configuration")
        import rqdatac as rq
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rq.init(*credentials)
        path = C.execution_dir(args.partition) / "contract_multipliers.csv"
        C.assert_research_only(path)
        count = write_multipliers(rq, identifiers, path)
        print(f"合约乘数 {count} 条写入 {path}")
        return 0
    spans = contract_spans(args.partition, args.start_date, args.end_date)
    bars = D.load_contracts(args.partition, for_execution=True)[["contract", "date"]]
    bars = bars.loc[bars.date.between(pd.Timestamp(args.start_date), pd.Timestamp(args.end_date))]
    days_by_contract = bars.groupby("contract").date
    credentials = rqdata_credentials()
    if not credentials:
        raise RuntimeError("missing RiceQuant configuration")
    import rqdatac as rq
    # Vendor authentication diagnostics may contain account details.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rq.init(*credentials)
    root = PARTITION_CACHE[args.partition] / "specifications"
    if args.partition == "research":
        C.assert_research_only(root)
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(spans.index)
    failures = []
    records = 0
    for offset in range(0, len(names), 100):
        batch = names[offset:offset + 100]
        try:
            instruments = rq.instruments(batch)
            if not isinstance(instruments, (list, tuple)):
                instruments = [instruments]
            lookup = {x.order_book_id: x for x in instruments if x is not None}
        except Exception as exc:
            print(f"contract metadata batch failed: {type(exc).__name__}", flush=True)
            failures.extend(batch)
            continue
        for contract in batch:
            if contract not in lookup:
                failures.append(contract)
                continue
            try:
                table = specification_rows(lookup[contract], days_by_contract.get_group(contract))
                check = C.assert_no_holdout_dates if args.partition == "research" else C.assert_oos_dates
                check(table.index, "expired contract specifications")
                path = root / f"{contract}.pkl"
                if path.exists():
                    previous = pd.read_pickle(path)
                    check(previous.index, "cached expired contract specifications")
                    table = pd.concat([previous.loc[~previous.index.isin(table.index)], table]).sort_index()
                temporary = path.with_suffix(".pkl.tmp")
                table.to_pickle(temporary)
                os.replace(temporary, path)
                records += len(table)
            except Exception as exc:
                print(f"{contract}: {type(exc).__name__}", flush=True)
                failures.append(contract)
        print(f"specifications {min(offset + 100, len(names))}/{len(names)}; failures {len(failures)}", flush=True)
    print(f"specification records {records}; missing contracts {failures}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

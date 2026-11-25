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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=list(PARTITION_CACHE), default="research")
    parser.add_argument("--start-date", type=parse_day, default=parse_day("20100104"))
    parser.add_argument("--end-date", type=parse_day, default=parse_day("20211231"))
    args = parser.parse_args()
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

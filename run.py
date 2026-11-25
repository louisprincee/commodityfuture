"""Run the four existing accounts, 2011–2021 with full 2010 warm-up.

    python run.py --jobs 4

The legacy engine entry point and its 2016 start remain unchanged. This entry
reads only research files and writes a distinct strategy and benchmark folder.
Missing historical execution records are retained as missing, preventing entry.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from math import isfinite

import pandas as pd

from backtest import engine as E
from src import config as C
from src import data as D
from strategies import sticky_only
from strategies.common import NAME_CAP, TARGET_MULTIPLIER

WARMUP_START = pd.Timestamp("2010-01-04")
START = pd.Timestamp("2011-01-04")
END = pd.Timestamp("2021-12-31")
OUTPUT_NAME = "research_2011_2021_execution_v1"
FLOW_EXPERIMENT = "member_sticky_flow_v1"
PERIOD_LABEL = "探索研究期 2011–2021 · 2011–2015 执行数据不完整"


def plot_figures(equity, out, book=sticky_only.MAIN_BOOK, data_status="available"):
    frame = equity[[book, E.BENCHMARK_BOOK]].rename(columns={E.BENCHMARK_BOOK: E.BENCHMARK_LABEL})
    suffix = (" · 估值未解决，仅供执行诊断" if data_status == "unresolved_valuation" else
              " · 执行数据仍有缺口" if data_status != "available" else "")
    title = f"{book} · 研究期 2011–2021{suffix}"
    E._plot_performance(frame / E.CAPITAL_CNY, out / "performance.png", title=title)
    E._plot_yearly_performance(frame, out / "yearly", title)


def _known_margins(ids: set[str]) -> set[tuple[pd.Timestamp, str]]:
    known = set()
    for contract in sorted(ids):
        path = C.RESEARCH_EXECUTION_PARAMETERS / "contracts" / f"{contract}.pkl"
        C.assert_research_only(path)
        if not path.is_file():
            continue
        table = pd.read_pickle(path)
        C.assert_no_holdout_dates(table.index, "历史保证金覆盖")
        if table.empty:
            continue
        good = table.long_margin_ratio.between(0, 1, inclusive="right")
        good &= table.short_margin_ratio.between(0, 1, inclusive="right")
        known.update((day, contract) for day in table.index[good])
    return known


def execution_coverage(weights, contracts, market, known_margins) -> pd.DataFrame:
    """Inspect each intended next-open target using the same timing as simulate."""
    shifted = weights.reindex_like(contracts).fillna(0).shift(1).fillna(0)
    rows = []
    last = contracts.index[-1]
    for index, (day, values) in enumerate(shifted.iterrows()):
        for symbol, value in values.items():
            contract = contracts.at[day, symbol]
            quote = market.get(day, {}).get(contract)
            deadline = E.exit_deadline(contract, quote, last) if quote else None
            in_exit_window = (deadline is not None and
                              contracts.index.searchsorted(deadline, side="right") - index <= E.LIQUIDATION_DAYS)
            rows.append({
                "date": day, "symbol": symbol, "contract": contract,
                "desired_weight": value,
                "targeted": bool(value != 0 and day != last),
                "quote_available": quote is not None,
                "fees_available": bool(quote and isfinite(quote.open_fee) and isfinite(quote.close_fee)),
                "historical_margin_available": (day, contract) in known_margins,
                "specifications_available": bool(quote and quote.multiplier > 0 and quote.tick > 0),
                "price_limits_available": bool(quote and quote.limit_up > 0 and quote.limit_down > 0),
                "can_open": bool(quote and quote.can_trade and quote.open_cap > 0 and quote.limit > 0
                                 and deadline is not None and not in_exit_window),
                "planned_exit_window": in_exit_window,
                "fee_fallback": bool(quote and quote.fee_fallback),
                "fee_carried": bool(quote and quote.fee_carried),
                "fee_source_date": quote.fee_source_date if quote else pd.NaT,
                "fee_source_contract": quote.fee_source_contract if quote else None,
                "fee_cross_contract": bool(quote and quote.fee_cross_contract),
                "expiry_available": bool(quote and pd.notna(quote.last_trading_date)),
            })
    return pd.DataFrame(rows)


def coverage_by_year(coverage, ledger, fills, selection=None):
    rows = []
    for year, daily in ledger.groupby(ledger.index.year):
        records = coverage.loc[coverage.date.dt.year.eq(year)]
        targets = records.loc[records.targeted]
        trades = fills.loc[fills.date.dt.year.eq(year)] if not fills.empty else fills
        missing_margin = int((~targets.historical_margin_available).sum())
        missing_fee = int((~targets.fees_available).sum())
        missing_spec = int((~targets.specifications_available).sum())
        missing_quote = int((~targets.quote_available).sum())
        missing_limits = int((~targets.price_limits_available).sum())
        missing_expiry = int((~targets.expiry_available).sum())
        row = {
            "year": year, "days": len(daily), "contract_days": len(records),
            "target_contract_days": len(targets),
            "target_tradeable_days": int(targets.can_open.sum()),
            "target_missing_quote_days": missing_quote,
            "target_missing_fee_days": missing_fee,
            "target_missing_margin_days": missing_margin,
            "target_missing_specification_days": missing_spec,
            "target_missing_price_limit_days": missing_limits,
            "target_missing_expiry_days": missing_expiry,
            "target_exit_window_days": int(targets.planned_exit_window.sum()),
            "target_fee_fallback_days": int(targets.fee_fallback.sum()),
            "target_fee_carried_days": int(targets.fee_carried.sum()),
            "target_fee_cross_contract_days": int(targets.fee_cross_contract.sum()),
            "fills": len(trades), "days_in_market": int(daily.open_lots.gt(0).sum()),
            "unfilled_orders": int(daily.unfilled_orders.sum()),
            "stale_marks": int(daily.stale_marks.sum()),
            "gross_exposure": daily.gross_notional.div(daily.equity).mean(),
            "status": ("unresolved_valuation" if daily.stale_marks.gt(0).any() else
                       "insufficient_execution_data" if len(targets) and trades.empty else
                       "partial_execution_data" if any((missing_margin, missing_fee, missing_spec,
                                                         missing_quote, missing_limits, missing_expiry)) else "available"),
        }
        if selection is not None:
            choices = selection.loc[selection.date.dt.year.eq(year)]
            row.update(avg_candidates=choices.candidates.mean(),
                       avg_long_slots=choices.long_slots.mean(), avg_short_slots=choices.short_slots.mean(),
                       avg_target_gross=choices.target_gross.mean() * TARGET_MULTIPLIER)
        rows.append(row)
    return pd.DataFrame(rows)


def _flow_comparisons(equity, ledgers):
    from strategies import sticky_flow as S
    from research.member.review import evaluate_net_return_gains

    books = [S.STICKY_BOOK, S.FLOW_BOOK, S.COMBINED_BOOK]
    returns = equity[books].pct_change(fill_method=None)
    returns.iloc[0] = equity[books].iloc[0] / E.CAPITAL_CNY - 1
    if returns.isna().any().any():
        raise ValueError("共同样本账户的日期不一致")
    parts = {}
    for book in books:
        daily = ledgers[book]
        prior = daily.equity.shift(1).fillna(E.CAPITAL_CNY)
        parts[book] = pd.DataFrame({"gross": daily.gross_pnl / prior,
                                    "cost": (daily.fees + daily.slippage) / prior})
    return evaluate_net_return_gains(returns, parts, S.STICKY_BOOK, books[1:])


def run(jobs: int = 4, selected=None, experiment=None) -> pd.DataFrame:
    """运行已冻结策略；因子计算在 research，账户执行在 backtest.engine。"""
    if experiment and selected is not None:
        raise ValueError("固定增量试验不能同时筛选原策略")
    names = list(E.MODULES) if selected is None else list(selected)
    output_name = FLOW_EXPERIMENT if experiment else OUTPUT_NAME
    comparison = C.strategy_dir("") / "comparison" / output_name
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel.date, "2011–2021 回测预热面板")
    panel = panel.loc[panel.date.between(WARMUP_START, END)]
    if panel.empty or panel.date.min() > WARMUP_START:
        raise ValueError("缺少完整 2010 年预热面板")
    bars = D.load_contracts("research", for_execution=True)
    C.assert_no_holdout_dates(bars.date, "研究期合约行情")
    C.assert_research_only(C.RESEARCH_FEE_PATH)
    fees = pd.read_csv(C.RESEARCH_FEE_PATH, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees.trading_date, "历史手续费")
    configured = []
    if experiment:
        from strategies import sticky_flow as S
        print("构造共同样本 Sticky / Flow / Sticky+Flow 及完整样本参考", flush=True)
        targets = S.build_books(panel, START, jobs)
        names = list(targets)
        root = C.strategy_dir("member") / "sticky_flow" / output_name
        for book, target in targets.items():
            configured.append((book, target, root / book.lower(), True, E.HORIZON))
        samples = [targets[b].selection[["date", "sample_hash", "candidates"]]
                   for b in (S.STICKY_BOOK, S.FLOW_BOOK, S.COMBINED_BOOK)]
        for sample in samples[1:]:
            pd.testing.assert_frame_equal(samples[0], sample)
    else:
        for name in names:
            module = E.MODULES[name]
            print(f"构造 {module.MAIN_BOOK} 的既定信号", flush=True)
            target = module.build_targets(panel, bars, START, "research", jobs)
            configured.append((module.MAIN_BOOK, target, E.book_dir(module.STRATEGY_ID) / output_name,
                               True, getattr(module, "HOLD", E.HORIZON)))
    benchmark = E._benchmark_targets(panel, START, "research")
    configured.append((E.BENCHMARK_BOOK, benchmark, C.strategy_dir("benchmark") / output_name, False, 1))
    calls = {book: target.dominant.shift(1).reindex_like(target.weights)
             for book, target, _, _, _ in configured}
    ids = {value for matrix in calls.values() for value in matrix.to_numpy().ravel() if isinstance(value, str)}
    market = E.load_market(bars, fees, ids, START, root=C.RESEARCH_EXECUTION_PARAMETERS / "contracts",
                           flat_margin=E.FLAT_MARGIN, partition="research")
    margins = _known_margins(ids)
    comparison.mkdir(parents=True, exist_ok=True)
    rules_path = C.CONFIG_DIR / "historical_fee_rules.csv"
    if rules_path.is_file():
        pd.read_csv(rules_path).to_csv(comparison / "historical_fee_rules.csv", index=False)
    manifest = {"start":str(START.date()), "end":str(END.date()), "warmup":str(WARMUP_START.date()),
                "books":names, "ticks_saved":[1,2], "reported_ticks":1,
                "flat_margin":E.FLAT_MARGIN, "unknown_historical_margin":"entry blocked",
                "price_source":"actual contract bars retained until actual record end; panel splice unchanged",
                "unresolved_positions":"retained with stale_marks; curves are diagnostics when stale marks exist",
                "fee_priority":"vendor exact contract-date, prior same-contract fee history, sourced intervals, last complete same-contract fee tuple, then prior unambiguous same-underlying schedule",
                "fee_policy_approved":"2026-11-25: missing fees use previous known period; never backfill from future",
                "account_type":"non-delivery; fail if unable to exit by contractual/account deadline",
                "liquidation_days":E.LIQUIDATION_DAYS,
                "inventory_cap":"liquidation_days * min(prior-volume exit capacity, maximum order size)",
                "exit_deadline":"earliest of maturity, day before delivery starts, day before delivery month, account end",
                "contract_specification_files":len(list((C.RESEARCH_EXECUTION_PARAMETERS/"specifications").glob("*.pkl"))),
                "fee_rules_sha256":hashlib.sha256(rules_path.read_bytes()).hexdigest() if rules_path.is_file() else None}
    manifest["execution_spec"] = OUTPUT_NAME
    manifest["benchmark"] = E.BENCHMARK_BOOK
    if experiment:
        manifest.update(
            experiment=FLOW_EXPERIMENT, min_names=8, hold=E.HORIZON,
            target_multiplier=TARGET_MULTIPLIER, name_cap=NAME_CAP,
            liquidity_universe=C.UNIVERSE,
            common_sample="finite sticky, raw flow and same-day rank residual; new cohorts share the mask",
            common_sample_sha256=hashlib.sha256(samples[0].to_csv(index=False).encode()).hexdigest(),
            residual="cross-sectional flow ranks on sticky ranks, centered with intercept; no return fit",
            slot_rule="k=ceil(0.10*N); each sticky tail supplies 2*k, choose k by residual; fixed symbol-order ties",
            primary_pairs=[[S.FLOW_BOOK,S.STICKY_BOOK],[S.COMBINED_BOOK,S.STICKY_BOOK]],
            primary_family="two full-period one-tick daily net-return differences; HAC lag 20, two-sided, BH",
            full_family="six one-tick comparisons: two pairs x full/early/late periods",
            reference="full-sample Sticky-only is descriptive, outside the signal-gain family",
            two_tick="saved cost sensitivity; no additional return hypothesis tests",
            exploratory=True)
    (comparison/"protocol.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    summaries, curves, statuses, ledgers = [], {}, {}, {}
    for book, targets, out, neutral, hold in configured:
        contracts = calls[book]
        C.assert_no_holdout_dates(contracts.index, "研究期策略日期")
        out.mkdir(parents=True, exist_ok=True)
        weights = targets.weights * (TARGET_MULTIPLIER if neutral else 1)
        coverage = execution_coverage(weights, contracts, market, margins)
        coverage.to_csv(out / "execution_coverage.csv", index=False)
        if experiment and targets.selection is not None:
            targets.selection.to_csv(out / "selection_daily.csv", index=False)
        rows = []
        for ticks in (1, 2):
            ledger, fills = E.simulate(weights, contracts, market, START, neutral=neutral,
                                      slippage_ticks=ticks, name_cap=NAME_CAP if neutral else E.BENCHMARK_NAME_CAP)
            row = E._book_summary(book, ticks, ledger)
            row.update(start=START.date(), end=END.date(), warmup_start=WARMUP_START.date(), hold=hold,
                       target_multiplier=TARGET_MULTIPLIER if neutral else 1,
                       fee_policy="previous_known_contract_then_underlying",
                       execution_spec=OUTPUT_NAME, liquidation_days=E.LIQUIDATION_DAYS,
                       first_fill=None if fills.empty else fills.date.min().date())
            by_year = coverage_by_year(coverage, ledger, fills, None)
            status = ("unresolved_valuation" if ledger.stale_marks.gt(0).any() else
                      "partial_execution_data" if by_year.status.ne("available").any() else "available")
            if ticks == 1:
                ledger.to_csv(out / "daily_ledger.csv", index_label="date")
                fills.to_csv(out / "fills.csv", index=False)
                by_year.to_csv(out / "coverage_by_year.csv", index=False)
                statuses[book] = status
                curves[book] = ledger.equity
                if experiment:
                    ledgers[book] = ledger
                if not fills.empty:
                    lots = fills.groupby(["symbol", "contract"]).trade_lots.sum()
                    terminal = lots.loc[lots.ne(0)].rename("open_lots").reset_index().assign(
                        last_quote_date=pd.NaT, last_tradeable_quote_date=pd.NaT,
                        last_open=float("nan"), last_limit_down=float("nan"),
                        last_exit_cap=0, last_open_blocks_exit=False)
                    for contract in terminal.contract:
                        quotes = [(day, values[contract]) for day, values in market.items()
                                  if day <= END and contract in values]
                        known_fees = [day for day, quote in quotes if quote.can_trade]
                        selected = terminal.contract.eq(contract)
                        terminal.loc[selected, "last_quote_date"] = max((day for day, _ in quotes), default=pd.NaT)
                        terminal.loc[selected, "last_tradeable_quote_date"] = max(known_fees, default=pd.NaT)
                        if quotes:
                            _, final_quote = max(quotes, key=lambda value: value[0])
                            terminal.loc[selected, "last_open"] = final_quote.open
                            terminal.loc[selected, "last_limit_down"] = final_quote.limit_down
                            terminal.loc[selected, "last_exit_cap"] = final_quote.exit_cap
                            open_lots = int(terminal.loc[selected, "open_lots"].iloc[0])
                            terminal.loc[selected, "last_open_blocks_exit"] = final_quote.limit_blocks(-open_lots)
                    terminal.to_csv(out / "terminal_positions.csv", index=False)
            row["data_status"] = status
            row["complete_execution_data"] = status == "available"
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        equity = pd.DataFrame({book: curves[book]})
        equity.to_csv(out / "equity_curves.csv", index_label="date")
        E.yearly_returns(equity).to_csv(out / "yearly_returns.csv", index=False)
        summaries.extend(rows)
        print(f"{book} 已保存至 {out}", flush=True)
    equity = pd.DataFrame(curves)
    for book, _, out, neutral, _ in configured:
        if neutral: plot_figures(equity, out, book, statuses[book])
    equity.to_csv(comparison / "equity_curves.csv", index_label="date")
    result = pd.DataFrame(summaries)
    result.to_csv(comparison / "summary.csv", index=False)
    if experiment:
        _flow_comparisons(equity, ledgers).to_csv(comparison / "paired_return_gains.csv", index=False)
    E.yearly_returns(equity).to_csv(comparison / "yearly_returns.csv", index=False)
    frame = equity.rename(columns={E.BENCHMARK_BOOK: E.BENCHMARK_LABEL}) / E.CAPITAL_CNY
    if experiment:
        labels = {S.STICKY_BOOK: "Sticky-only（共同样本）", S.FLOW_BOOK: "Flow-only（共同样本）",
                  S.COMBINED_BOOK: "Sticky + Flow（共同样本）", S.REFERENCE_BOOK: "Sticky-only（完整样本参考）"}
        if frame[S.REFERENCE_BOOK].equals(frame[S.STICKY_BOOK]):
            frame = frame.drop(columns=S.REFERENCE_BOOK)
            labels[S.STICKY_BOOK] = "Sticky-only（共同/完整样本重合）"
        frame = frame.rename(columns=labels)
    title = "Sticky + Flow 固定增量试验" if experiment else "四套策略"
    E._plot_performance(frame, comparison / "overview.png", title=f"{title} · 2011–2021 · 统一非交割执行（早期数据有缺口）")
    print(result.loc[result.ticks.eq(1), ["book", "net_ann", "sharpe", "max_drawdown", "first_fill", "data_status"]].to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strategies", nargs="+", choices=list(E.MODULES))
    mode.add_argument("--experiment", choices=["sticky-flow"])
    args = parser.parse_args()
    run(args.jobs, args.strategies, args.experiment)


if __name__ == "__main__":
    main()

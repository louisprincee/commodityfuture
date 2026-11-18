"""1000 万元整数手账户与合约级成交榜核验。"""
from __future__ import annotations

from itertools import product

import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from . import execution as E
from research import member_study as M
from research import paper_factors as F
from . import paper_ic as I
from research import portfolio as P
from research.settings import XS_POSITION as XS


def _signal_frames(symbol, contracts, dates):
    ranks = D.load_member_rank_contracts(symbol, "research", D.MEMBER_SIDES)
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
        position_ranks = ranks.loc[ranks.side.ne("volume")]
        counts = position_ranks.groupby(["trading_date", "contract"])["side"].nunique()
        two = counts.eq(2).groupby(level="trading_date").sum().reindex(dates, fill_value=0)
    else:
        position_ranks = ranks
        two = pd.Series(0, index=dates)
    active = F.kaiyuan_active(ranks, dates)
    book = F.kaiyuan(position_ranks, contracts, dates)
    members = M.member_tables(symbol, dates)
    strict = M.consistency_from_net(members["L"] - members["S"], 20)["POWER"]
    return active.assign(two_sides=two, K_DS=-book["K_DS"], POWER_20=strict)


def run(panel, fee_path, out, jobs):
    C.assert_no_holdout_dates(panel["date"], "整数手日面板")
    C.assert_research_only(fee_path)
    fee = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fee["trading_date"], "整数手手续费")
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    eligible = P.eligibility(panel)
    root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(root)
    names = sorted(s for s in panel.loc[eligible, "symbol"].unique() if (root / f"{s}_long.pkl").is_file())
    start = pd.Timestamp(XS["test_start"])
    days = pd.DatetimeIndex(panel.date.unique()).sort_values()
    sample = days[days >= start]
    bars = D.load_contracts("research")
    series = dict(tuple(bars.loc[bars.symbol.isin(names), ["symbol", "contract", "date", "open_interest"]].groupby("symbol")))
    parts = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_signal_frames)(name, series[name], days) for name in names
    )
    factors = pd.concat(parts, axis=1, keys=names).swaplevel(axis=1)
    use = panel.assign(eligible=eligible).pivot(index="date", columns="symbol", values="eligible")
    use = use.reindex(index=days, columns=names).astype("boolean").fillna(False)
    rolls = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(index=days, columns=names)
    return_matrix = I._return_matrix(rolls, bars.set_index(["contract", "date"])["open"])
    n3 = factors["contracts_covered"].where(use, 0).loc[sample]
    n2 = factors["two_sides"].where(use, 0).loc[sample]
    coverage = pd.DataFrame({
        "two_side_contract_days": n2.sum(), "three_side_contract_days": n3.sum(),
        "symbol_days_with_two_sides": n2.gt(0).sum(), "symbol_days_with_three_sides": n3.gt(0).sum(),
    })
    coverage.to_csv(out / "execution_spider_coverage.csv", index_label="symbol")
    rows = []
    for name in ("SPIDER_SAME_COVERAGE", "SPIDER_ACTIVE"):
        score = factors[name].where(use).loc[sample]
        for delay in (0, 1):
            outcome = return_matrix.shift(-1 - delay).loc[sample]
            ic, count = I._rank_corr(score, outcome, XS["min_names"])
            rows.append(dict(signal=name, delay=delay, **I._stats(ic, count)))
    pd.DataFrame(rows).to_csv(out / "execution_spider_ic.csv", index=False)

    score = factors["K_DS"]
    prior = score.shift(1).rolling(XS["zscore_window"], min_periods=XS["zscore_min_periods"])
    delta = (score - prior.mean()).where(use).loc[sample]
    z = delta / prior.std().loc[sample].where(lambda frame: frame.gt(0))
    power = factors["POWER_20"]
    prior_power = power.shift(1).rolling(XS["zscore_window"], min_periods=XS["zscore_min_periods"])
    power_z = ((power - prior_power.mean()) / prior_power.std().where(lambda frame: frame.gt(0))).where(use).loc[sample]
    active = factors["SPIDER_ACTIVE"].where(use).loc[sample]
    active_count = active.count(axis=1)
    active_weight = active.div(active_count.where(active_count.ge(XS["min_names"])), axis=0).fillna(0)
    books = {
        "K_DS_DAILY": I.buffered_tails(delta, XS["min_names"], False),
        "K_DS_BUFFER": I.buffered_tails(delta, XS["min_names"], True),
        "K_DS_BUFFER_GATE": I.buffered_tails(delta, XS["min_names"], True, entry_gate=z),
        "POWER_20_STRICT": I.buffered_tails(power_z, XS["min_names"], False),
        "SPIDER_ACTIVE": active_weight,
    }
    ids = pd.unique(rolls.loc[sample].to_numpy().ravel())
    ids = [contract for contract in ids if isinstance(contract, str)]
    market = E.load_market(bars[["date", "contract", "open", "settlement", "volume"]], fee, ids, start)
    summaries, ledgers, orders = [], [], []
    for name, weights in books.items():
        tick_scenarios = (1, 2) if name == "K_DS_BUFFER_GATE" else (1,)
        for delay, ticks in product((0, 1), tick_scenarios):
            daily, fills = E.simulate(weights, rolls, market, start, delay, neutral=name != "SPIDER_ACTIVE",
                                      slippage_ticks=ticks)
            stats = P.metrics(daily["return"], daily["turnover"])
            ending = daily["equity"].iloc[-1]
            peak = daily["equity"].cummax().clip(lower=E.CAPITAL_CNY)
            summaries.append({
                "book": name, "delay": delay, "slippage_ticks": ticks,
                "start_equity": E.CAPITAL_CNY, "end_equity": ending,
                "annualized_arithmetic": stats["ann_return"], "net_sharpe": stats["sharpe"],
                "max_drawdown": (daily["equity"] / peak - 1).min(), "turnover": stats["daily_turnover"],
                "gross_pnl_cny": daily.gross_pnl.sum(), "fees_cny": daily.fees.sum(),
                "slippage_cny": daily.slippage.sum(), "peak_margin_fraction": daily.margin_fraction.max(),
                "avg_gross_notional": daily.gross_notional.mean(), "active_days": daily.open_contracts.gt(0).sum(),
                "avg_abs_net_fraction": daily.net_notional.abs().div(daily.equity).mean(),
                "unfilled_orders": daily.unfilled_orders.sum(), "blocked_rolls": daily.blocked_rolls.sum(),
                "limit_rejections": daily.limit_rejections.sum(),
                "stale_marks": daily.stale_marks.sum(), "terminal_contracts": daily.open_contracts.iloc[-1],
                "fee_fallback_orders": fills["fee_fallback"].sum() if len(fills) else 0,
                "missing_limit_orders": fills["missing_limit"].sum() if len(fills) else 0,
            })
            ledgers.append(daily.assign(book=name, delay=delay, slippage_ticks=ticks))
            orders.append(fills.assign(book=name, delay=delay, slippage_ticks=ticks))
    pd.concat(ledgers).to_csv(out / "execution_daily.csv", index_label="date")
    pd.concat(orders, ignore_index=True).to_csv(out / "execution_fills.csv", index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(out / "execution_summary.csv", index=False)
    show = ["book", "delay", "slippage_ticks", "end_equity", "net_sharpe", "turnover", "peak_margin_fraction"]
    print(result[show].round(4).to_string(index=False))

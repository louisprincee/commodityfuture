"""会员持仓异常事件的开盘收益、交易成本与整数手账户。"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research import member_study as M, paper_factors as F, portfolio as P
from research.settings import CROWDING as CR, XS_POSITION as XS
from . import execution as E

WINDOW = 250
MINIMUM = 60
EVENTS = ("short_shock", "crowding_reversal")
TRADE_COLUMNS = ("symbol", "contract", "side", "entry_date", "exit_date", "lots", "entry_notional",
                 "gross_pnl", "fees", "slippage", "net_pnl")


def short_shock(values, window=WINDOW, minimum=MINIMUM):
    """按此前窗口的分位开仓，并用上下分位间距衡量异常强度。"""
    past = values.shift(1).rolling(window, min_periods=minimum)
    hi, lo = past.quantile(CR["q_high"]), past.quantile(CR["q_low"])
    spread = hi - lo
    direction = values.gt(hi).astype(float) - values.lt(lo).astype(float)
    distance = values - (hi + lo) / 2
    strength = distance.div(spread.where(spread.gt(0)))
    flat = spread.eq(0) & distance.notna()
    strength = strength.mask(flat & distance.eq(0), 0)
    strength = strength.mask(flat & distance.gt(0), np.inf).mask(flat & distance.lt(0), -np.inf)
    return direction, strength


def session_prices(minutes, daily):
    """用未复权主连首尾价格核对日线合约，取准确的日盘首根分钟开盘。"""
    minutes = minutes.sort_index()
    days = pd.to_datetime(minutes.trading_date)
    C.assert_no_holdout_dates(days, "分钟交易日")
    C.assert_no_holdout_dates(minutes.index.normalize(), "分钟自然日")
    elapsed = minutes.index.hour * 60 + minutes.index.minute
    session_open = minutes.groupby(days).open.first()
    session_close = minutes.groupby(days).close.last()
    first_timestamp = minutes.index.to_series().groupby(days).first()
    day_start = minutes.loc[(minutes.index.normalize() == days.to_numpy()) & (elapsed >= 540) & (elapsed <= 541)]
    day_open = day_start.groupby("trading_date").open.first()
    verified = daily.open.eq(session_open) & daily.close.eq(session_close)
    out = daily[["contract"]].copy()
    out["verified"] = verified
    out["night"] = first_timestamp.reindex(daily.index).dt.normalize().lt(daily.index)
    out["day_open"] = day_open.reindex(daily.index).where(verified & day_open.reindex(daily.index).gt(0))
    return out


def _symbol_inputs(name, panel, bars, dates):
    ranks = D.load_member_rank_contracts(name, "research")
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    kaiyuan = F.kaiyuan(ranks, bars, dates, include_disclosed_oi=True)
    oi = kaiyuan.disclosed_oi
    shock = -kaiyuan.K_DS.div(oi.where(oi.gt(0)))
    short, short_strength = short_shock(shock)
    tables = M.member_tables(name, dates, top20=True)
    market = M.market_inputs(panel, bars, name).reindex(dates)
    crowd = F.crowding(tables, market).CROWD1
    reversal = M.crowding_signal(crowd, market.ret_cc, CR["reference_window"])
    _, reversal_strength = short_shock(-crowd, minimum=WINDOW)
    strict = M.consistency_from_net(tables["L"] - tables["S"], 20).POWER
    filled = M.consistency_from_net(M.imputed_net(tables), 20).POWER
    scores = pd.DataFrame({EVENTS[0]: short, EVENTS[1]: reversal,
                           "short_strength": short_strength, "reversal_strength": reversal_strength,
                           "power_strict": strict, "power_filled": filled}, index=dates)

    path = C.minute_dir("research") / f"{name}.parquet"
    C.assert_research_only(path)
    if path.is_file():
        minutes = pd.read_parquet(path, columns=["trading_date", "open", "close"])
        dominant = panel[["date", "c1_contract"]].rename(columns={"c1_contract": "contract"})
        daily = dominant.merge(bars[["date", "contract", "open", "close"]], how="left",
                               on=["date", "contract"], validate="one_to_one").set_index("date")
        day = session_prices(minutes, daily)
        day = day.loc[day.index >= XS["test_start"]].reset_index().assign(symbol=name)
    else:
        day = pd.DataFrame(columns=["date", "symbol", "contract", "verified", "night", "day_open"])
    return scores, day


def _event_weights(signal, strength, eligible, cross_section):
    """每侧预算取该侧全部触发品种的份额。截面对照集中于较强的一半。"""
    count = eligible.sum(axis=1)
    if not cross_section:
        return signal.div(count.where(count.gt(0)), axis=0).fillna(0)
    weight = pd.DataFrame(0., index=signal.index, columns=signal.columns)
    for direction in (1, -1):
        triggered = signal.eq(direction)
        n = triggered.sum(axis=1)
        slots = (n + 1) // 2
        ranks = strength.abs().where(triggered).rank(axis=1, method="first", ascending=False)
        strongest = ranks.le(slots, axis=0)
        scale = n.div(count.where(count.gt(0))).div(slots.where(slots.gt(0)))
        weight += strongest.mul(direction * scale.fillna(0), axis=0)
    return weight.clip(-E.NAME_CAP, E.NAME_CAP)


def roundtrips(fills):
    """将同合约成交按 FIFO 配对，持有、减仓、反手和换月使用实际整数手。"""
    opened = defaultdict(deque)
    rows = []
    for fill in fills.itertuples(index=False):
        size = abs(fill.trade_lots)
        side = int(np.sign(fill.trade_lots))
        if fill.action == "open":
            opened[fill.contract].append([size, fill, side])
            continue
        while size:
            lot = opened[fill.contract][0]
            amount = min(size, lot[0])
            _, entry, held_side = lot
            if side == held_side:
                raise RuntimeError(f"{fill.contract} 平仓方向与持仓相同")
            prior_share = amount / abs(entry.trade_lots)
            exit_share = amount / abs(fill.trade_lots)
            entry_value = entry.notional * prior_share
            exit_value = fill.notional * exit_share
            gross = held_side * (exit_value - entry_value)
            fees = entry.fee * prior_share + fill.fee * exit_share
            slip = entry.slippage * prior_share + fill.slippage * exit_share
            rows.append((fill.symbol, fill.contract, held_side, entry.date, fill.date, amount, entry_value,
                         gross, fees, slip, gross - fees - slip))
            lot[0] -= amount
            size -= amount
            if lot[0] == 0:
                opened[fill.contract].popleft()
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _returns_and_costs(entry, exit, side, ticks=1):
    """单事件按同一合约、单手往返计算基点，标记涨跌停或容量限制。"""
    if not entry or not exit or not (entry.open > 0 and exit.open > 0):
        return np.nan, np.nan, np.nan, False
    gross = side * (exit.open / entry.open - 1)
    if not entry.can_trade or not exit.can_trade:
        return gross * 1e4, np.nan, np.nan, False
    notional = entry.open * entry.multiplier
    open_fee = entry.open_fee * (1 if entry.by_lot else notional)
    close_fee = exit.close_fee * (1 if exit.by_lot else exit.open * exit.multiplier)
    per_tick = (entry.tick * entry.multiplier + exit.tick * exit.multiplier) / notional
    executable = (
        min(entry.open_cap, entry.max_order, entry.limit) >= entry.min_order and
        min(exit.exit_cap, exit.max_order) >= entry.min_order and
        not entry.limit_blocks(side) and not exit.limit_blocks(-side)
    )
    return gross * 1e4, (open_fee + close_fee) / notional * 1e4, per_tick * ticks * 1e4, executable


def event_diagnostics(signals, contracts, market, day_prices):
    """固定入场时可知的合约。分期毛收益独立于是否能在计划开盘成交。"""
    dates = contracts.index
    day_quotes = day_prices.set_index(["date", "contract"])["day_open"].dropna().to_dict()
    rows = []
    for factor, signal in signals.items():
        for i, j in zip(*np.nonzero(signal.to_numpy())):
            for delay in (0, 1):
                first = i + 1 + delay
                if first >= len(dates) - 1:
                    continue
                symbol = signal.columns[j]
                side = int(signal.iat[i, j])
                contract = contracts.iat[first, j]
                opened, closed = dates[first:first + 2]
                quote0 = market.get(opened, {}).get(contract)
                quote1 = market.get(closed, {}).get(contract)
                fifth = market.get(dates[first + 5], {}).get(contract) if first + 5 < len(dates) else None
                gross, fee, slip, executable = _returns_and_costs(quote0, quote1, side)
                rest = np.nan
                total = np.nan
                if fifth and fifth.open > 0 and quote0 and quote0.open > 0 and quote1 and quote1.open > 0:
                    total = side * (fifth.open / quote0.open - 1) * 1e4
                    rest = side * (fifth.open - quote1.open) / quote0.open * 1e4
                open_day, close_day = day_quotes.get((opened, contract)), day_quotes.get((closed, contract))
                if open_day and close_day and quote0 and quote1:
                    day_result = _returns_and_costs(replace(quote0, open=open_day), replace(quote1, open=close_day), side)
                else:
                    day_result = (np.nan, np.nan, np.nan, False)
                holder = (factor, dates[i], symbol, side, delay, contract, opened, closed,
                          gross, fee, slip, executable, total, rest, *day_result)
                rows.append(holder)
    cols = ["factor", "signal_date", "symbol", "side", "delay", "contract", "entry_date", "exit_date",
            "gross_bps", "fee_bps", "slip_bps", "executable", "gross_5d_bps", "gross_days_2_to_5_bps",
            "day_gross_bps", "day_fee_bps", "day_slip_bps", "day_executable"]
    diagnostics = pd.DataFrame(rows, columns=cols)
    diagnostics["net_1tick_bps"] = diagnostics.gross_bps - diagnostics.fee_bps - diagnostics.slip_bps
    diagnostics["net_2tick_bps"] = diagnostics.net_1tick_bps - diagnostics.slip_bps
    diagnostics["matched_first_gross_bps"] = diagnostics.gross_bps.where(diagnostics.day_gross_bps.notna())
    matched = diagnostics.executable & diagnostics.day_executable
    diagnostics["matched_first_net_bps"] = diagnostics.net_1tick_bps.where(matched)
    diagnostics["matched_day_net_bps"] = (
        diagnostics.day_gross_bps - diagnostics.day_fee_bps - diagnostics.day_slip_bps
    ).where(matched)
    return diagnostics


def _subgroups(rows, dated):
    copy = rows.assign(sector=rows.symbol.map(C.SECTOR),
                       half=np.where(rows[dated].dt.year <= 2018, "2016-18", "2019-21"))
    yield "all", "all", copy
    yield "ex_RB", "ex_RB", copy.loc[copy.symbol.ne("RB")]
    for column in ("side", "half", "sector", "symbol"):
        for name, sub in copy.groupby(column):
            yield column, name, sub


def _event_summary(events):
    for (factor, delay), group in events.groupby(["factor", "delay"]):
        for scope, category, sub in _subgroups(group, "signal_date"):
            valid = sub.loc[sub.executable]
            net = valid.net_1tick_bps
            ticks = valid.slip_bps.sum()
            yield dict(factor=factor, delay=delay, scope=scope, category=category,
                       signals=len(sub), first_pairs=sub.gross_bps.count(), tradeable=len(net),
                       gross_bps=valid.gross_bps.mean(), fee_bps=valid.fee_bps.mean(),
                       slip_bps=valid.slip_bps.mean(), net_bps=net.mean(), median_net_bps=net.median(),
                       net_2tick_bps=valid.net_2tick_bps.mean(),
                       break_even_ticks=(valid.gross_bps - valid.fee_bps).sum() / ticks if ticks > 0 else np.nan,
                       five_day_pairs=sub.gross_5d_bps.count(), gross_5d_bps=sub.gross_5d_bps.mean(),
                       gross_days_2_to_5_bps=sub.gross_days_2_to_5_bps.mean(),
                       day_pairs=sub.day_gross_bps.count(), day_gross_bps=sub.day_gross_bps.mean(),
                       matched_first_gross_bps=sub.matched_first_gross_bps.mean(),
                       day_tradeable_pairs=sub.matched_day_net_bps.count(),
                       matched_first_net_bps=sub.matched_first_net_bps.mean(),
                       matched_day_net_bps=sub.matched_day_net_bps.mean())


def direction_controls(events, principal, eligible, bar_open):
    """同品种同年与同日同板块的未触发价格对照，统一按事件方向作差。"""
    days = principal.index.to_series()
    principal = principal.reindex_like(eligible)
    prices = {}
    for step in (1, 2, 6):
        forward_date = days.shift(-step).reindex(eligible.index)
        labels = pd.MultiIndex.from_arrays([principal.to_numpy().ravel(),
                                            np.repeat(forward_date.to_numpy(), principal.shape[1])])
        quote = bar_open.reindex(labels).to_numpy()
        prices[step] = np.where(quote > 0, quote, np.nan)
    pool = pd.DataFrame({"signal_date": np.repeat(eligible.index.to_numpy(), eligible.shape[1]),
                         "symbol": np.tile(eligible.columns, len(eligible)),
                         "gross_d1": (prices[2] / prices[1] - 1) * 1e4,
                         "gross_d2_to_d5": (prices[6] - prices[2]) / prices[1] * 1e4})
    pool = pool.loc[eligible.to_numpy().ravel()]
    pool = pool.assign(year=pool.signal_date.dt.year, sector=pool.symbol.map(C.SECTOR))
    pool["half"] = np.where(pool.year.le(2018), "2016-18", "2019-21")
    windows = ["gross_d1", "gross_d2_to_d5"]
    results = []
    for factor, group in events.loc[events.delay.eq(0)].groupby("factor"):
        sample = pool.merge(group[["signal_date", "symbol", "side"]], how="left",
                            on=["signal_date", "symbol"], validate="one_to_one")
        baseline = sample.loc[sample.side.isna()]
        triggered = sample.loc[sample.side.notna()].copy()
        for label, keys in (("symbol_year", ["symbol", "year"]), ("sector_date", ["sector", "signal_date"])):
            means = baseline.groupby(keys)[windows].mean().add_suffix("_baseline")
            triggered = triggered.join(means, on=keys)
            for window in windows:
                triggered[f"{window}_vs_{label}"] = triggered[window].sub(triggered.pop(f"{window}_baseline"))
        fields = [col for col in triggered if col.startswith("gross_")]
        triggered[fields] = triggered[fields].mul(triggered.side, axis=0)
        stats = triggered.groupby(["side", "half"])[fields].mean().reset_index()
        results.append(stats.assign(factor=factor))
    if not results:
        return pd.DataFrame(columns=["factor", "side", "half", *windows])
    return pd.concat(results, ignore_index=True)


def _trade_summary(trips, ticks):
    for scope, category, sub in _subgroups(trips, "entry_date"):
        traded = sub.entry_notional.sum()
        costs = sub.slippage.sum()
        yield dict(scope=scope, category=category, matched_lot_groups=len(sub), traded_entry_notional=traded,
                   **{col: sub[col].sum() for col in ("gross_pnl", "fees", "slippage", "net_pnl")},
                   mean_gross_bps=(sub.gross_pnl / sub.entry_notional).mean() * 1e4,
                   mean_fee_bps=(sub.fees / sub.entry_notional).mean() * 1e4,
                   mean_slippage_bps=(sub.slippage / sub.entry_notional).mean() * 1e4,
                   mean_net_bps=(sub.net_pnl / sub.entry_notional).mean() * 1e4,
                   break_even_ticks=ticks * (sub.gross_pnl.sum() - sub.fees.sum()) / costs if costs else np.nan)


def run(panel, fee_path, out, jobs):
    C.assert_no_holdout_dates(panel.date, "事件研究日面板")
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees.trading_date, "事件研究手续费")
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=P.eligibility(panel))
    dates = pd.DatetimeIndex(panel.date.unique()).sort_values()
    start = pd.Timestamp(XS["test_start"])
    names = sorted(panel.loc[panel.eligible & panel.date.ge(start), "symbol"].unique())
    if not names:
        raise RuntimeError("事件研究区间没有满足流动性条件的商品")
    bars = D.load_contracts("research")
    panels = dict(tuple(panel.groupby("symbol")))
    by_symbol = dict(tuple(bars.loc[bars.symbol.isin(names)].groupby("symbol")))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_symbol_inputs)(name, panels[name], by_symbol[name], dates) for name in names
    )
    scores = pd.concat([result[0] for result in pieces], axis=1, keys=names).swaplevel(axis=1).sort_index(axis=1)
    day = pd.concat([result[1] for result in pieces], ignore_index=True)
    eligible = panel.pivot(index="date", columns="symbol", values="eligible").reindex(columns=names)
    eligible = eligible.astype("boolean").fillna(False).astype(bool).loc[start:]
    # 决策时已经公布的主力，延至下一交易日才执行。
    principal = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(columns=names)
    contracts = principal.shift(1).loc[start:]
    ids = {name for name in contracts.to_numpy().ravel() if isinstance(name, str)}
    market = E.load_market(bars[["date", "contract", "open", "settlement", "volume"]], fees, ids, start)

    out = out / "event_timing"
    out.mkdir(parents=True, exist_ok=True)
    signals = {factor: scores[factor].loc[start:].where(eligible, 0).fillna(0) for factor in EVENTS}
    strengths = {EVENTS[0]: scores.short_strength.loc[start:], EVENTS[1]: scores.reversal_strength.loc[start:]}
    event_results = event_diagnostics(signals, contracts, market, day)
    event_results.to_csv(out / "events.csv", index=False)
    pd.DataFrame(_event_summary(event_results)).to_csv(out / "event_summary.csv", index=False)
    control = direction_controls(event_results, principal, eligible, bars.set_index(["contract", "date"]).open)
    control.to_csv(out / "direction_controls.csv", index=False)
    day.to_csv(out / "session_audit.csv", index=False)
    print(f"事件与分钟核对完成：{len(event_results)} 个事件时点，{len(names)} 个流动性品种", flush=True)

    def books():
        for factor in EVENTS:
            for mode in ("time_series", "cross_section"):
                weights = _event_weights(signals[factor], strengths[factor], eligible, mode == "cross_section")
                for delay in (0, 1):
                    for ticks in (1, 2):
                        yield factor, mode, 1, 0, delay, ticks, True, weights
            weights = _event_weights(signals[factor], strengths[factor], eligible, False)
            yield factor, "resize", 1, 0, 0, 1, False, weights
        from .timing_study import timing_directions
        for factor in ("power_strict", "power_filled"):
            raw = scores[factor]
            past = raw.shift(1).rolling(WINDOW, min_periods=MINIMUM)
            z = (raw - past.mean()).div(past.std().where(lambda values: values.gt(0)))
            for phase in range(5):
                score = z.loc[start:].where(eligible).iloc[phase::5]
                direction = timing_directions(score, raw.reindex(score.index).where(eligible))
                weights = _event_weights(direction, score, eligible.reindex(score.index), False)
                for delay in (0, 1):
                    for ticks in (1, 2):
                        yield factor, "time_series", 5, phase, delay, ticks, False, weights

    accounts, fills_all, ledger_all, trade_all, summaries = [], [], [], [], []
    for factor, mode, interval, phase, delay, ticks, keep, weights in books():
        daily, fills = E.simulate(weights, contracts, market, start, delay, neutral=False, slippage_ticks=ticks,
                                  rebalance_every=interval, phase=phase, keep_same_lots=keep)
        metadata = dict(factor=factor, mode=mode, phase=phase, delay=delay, ticks=ticks, interval=interval)
        stats = P.metrics(daily["return"], daily.turnover)
        trips = roundtrips(fills)
        gross, fee, slip = (daily[col].sum() for col in ("gross_pnl", "fees", "slippage"))
        terminal = daily.open_contracts.iloc[-1]
        if not terminal:
            np.testing.assert_allclose([trips[col].sum() for col in ("gross_pnl", "fees", "slippage")],
                                       [gross, fee, slip], rtol=1e-9, atol=0.05)
        np.testing.assert_allclose(daily.equity.iloc[-1], E.CAPITAL_CNY + gross - fee - slip, atol=0.05)
        roll = fills.groupby(["date", "symbol"]).contract.transform("nunique").gt(1) if len(fills) else []
        accounts.append(dict(**metadata, end_equity=daily.equity.iloc[-1], sharpe=stats["sharpe"],
                             net_ann=stats["ann_return"], turnover=stats["daily_turnover"],
                             max_drawdown=daily.equity.div(daily.equity.cummax().clip(lower=E.CAPITAL_CNY)).min() - 1,
                             gross_pnl=gross, fees=fee, slippage=slip, trade_groups=len(trips),
                             fills=len(fills), roll_fills=int(np.sum(roll)), terminal_contracts=terminal,
                             stale_marks=daily.stale_marks.sum(), limit_rejections=daily.limit_rejections.sum(),
                             gross_exposure=daily.gross_notional.div(daily.equity).mean(),
                             abs_net_exposure=daily.net_notional.abs().div(daily.equity).mean(),
                             net_ann_2016_2018=daily.loc[:"2018", "return"].mean() * P.TD,
                             net_ann_2019_2021=daily.loc["2019":, "return"].mean() * P.TD))
        if len(trips):
            summaries.extend(dict(**metadata, **row) for row in _trade_summary(trips, ticks))
        ledger_all.append(daily.reset_index().assign(**metadata))
        if len(fills):
            fills_all.append(fills.assign(**metadata))
        if len(trips):
            trade_all.append(trips.assign(**metadata))
    pd.DataFrame(accounts).to_csv(out / "accounts.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "trade_summary.csv", index=False)
    pd.concat(ledger_all, ignore_index=True).to_csv(out / "daily_accounts.csv", index=False)
    (pd.concat(fills_all, ignore_index=True) if fills_all else pd.DataFrame()).to_csv(out / "fills.csv", index=False)
    (pd.concat(trade_all, ignore_index=True) if trade_all else pd.DataFrame(columns=TRADE_COLUMNS)).to_csv(
        out / "trades.csv", index=False)
    print(f"整数手账户完成：{len(accounts)} 个规格 → {out}", flush=True)

"""回测引擎：按合约、整数手与日结算价核算 1000 万元账户，权重层面的成本、换手与绩效统计，及策略命令行入口。

    python -m backtest.engine buffer        # 两端入场与中位数缓冲的截面多空对照
    python -m backtest.engine hold-five     # 五日预测候选的固定周期整数手账户
    python -m backtest.engine risk-budget   # 低频底仓 + 事件卫星的风险预算组合

持仓规则在 strategies，因子在 research。只读取 2016–2021 研究期；结果写入 runs/strategies/<策略>/。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config as C  # noqa: E402
import data as D  # noqa: E402

CAPITAL_CNY = 10_000_000
MARGIN_CAP = 0.50
BROKER_MARGIN_ADDON = 0.03
VOLUME_FRACTION = 0.01
NAME_CAP = 0.10
SLIPPAGE_TICKS = 1
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


def lot_targets(weights, notionals, margins, limits, equity, margin_cap=MARGIN_CAP):
    """按一手名义金额向零取整，先等比例压缩以满足组合保证金预算。"""
    ideal = np.divide(weights * max(equity, 0), notionals, out=np.zeros_like(weights, dtype=float),
                      where=np.isfinite(notionals) & (notionals > 0))
    initial_margin = (abs(ideal) * np.nan_to_num(notionals * margins, nan=0)).sum()
    scale = min(1.0, margin_cap * max(equity, 0) / initial_margin) if initial_margin else 1.0
    return np.trunc(np.clip(ideal * scale, -limits, limits)).astype(np.int64)


def _positive_int(value, default):
    return max(1, floor(value)) if np.isfinite(value) and value > 0 else default


def load_market(bars: pd.DataFrame, fees: pd.DataFrame, ids, start, root: Path = PARAMETER_CACHE,
                flat_margin: float | None = None):
    """交易参数与合约日线对齐，成交量容量仅使用截至前日的数据。"""
    if flat_margin is not None and not (0 < flat_margin <= 1):
        raise ValueError("flat_margin 必须在 (0, 1] 内")
    C.assert_research_only(root)
    names = set(ids)
    bars = bars.loc[bars.contract.isin(names)].sort_values(["contract", "date"]).copy()
    bars["prior_volume"] = bars.groupby("contract")["volume"].transform(
        lambda volume: volume.shift(1).rolling(20, min_periods=5).median()
    )
    bars = bars.loc[bars.date.ge(start)]
    tables = []
    price_limits = []
    for contract in sorted(names):
        path = root / f"{contract}.pkl"
        C.assert_research_only(path)
        if path.is_file():
            table = pd.read_pickle(path)
            C.assert_no_holdout_dates(table.index, f"{contract} 交易参数")
            if not table.empty:
                tables.append(table.rename_axis("date").reset_index().assign(contract=contract))
        path = root.parent / "price_limits" / f"{contract}.pkl"
        C.assert_research_only(path)
        if path.is_file():
            table = pd.read_pickle(path)
            C.assert_no_holdout_dates(table.index, f"{contract} 涨跌停价")
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
    fallback = fees.rename(columns={"trading_date": "date"}).sort_values("date")
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


def simulate(weights, contracts, market, start, delay=0, capital=CAPITAL_CNY, neutral=True,
             slippage_ticks=SLIPPAGE_TICKS, rebalance_every=1, phase=0, keep_same_lots=False,
             name_cap=NAME_CAP):
    """按开盘标价和滑点成交，按结算价盯市。先平仓，之后才允许开仓。"""
    if capital <= 0 or slippage_ticks < 0 or not (0 < name_cap <= 1):
        raise ValueError("初始本金必须为正，滑点不得为负，单品种上限必须在 (0, 1] 内")
    if rebalance_every < 1 or not 0 <= phase < rebalance_every:
        raise ValueError("调仓周期和起始偏移不匹配")
    dates = contracts.loc[start:].index
    targets = weights.reindex_like(contracts).fillna(0).shift(1 + delay).loc[dates].fillna(0)
    targets = targets.clip(-name_cap, name_cap)
    if neutral:
        long, short = targets.clip(lower=0), targets.clip(upper=0)
        long_total, short_total = long.sum(axis=1), -short.sum(axis=1)
        gross_per_side = long_total.clip(upper=short_total)
        long_scale = gross_per_side.div(long_total.where(long_total.gt(0))).fillna(0)
        short_scale = gross_per_side.div(short_total.where(short_total.gt(0))).fillna(0)
        targets = long.mul(long_scale, axis=0) + short.mul(short_scale, axis=0)
    equity = float(capital)
    positions = {}
    ledger, fills = [], []
    for i, day in enumerate(dates):
        quotes = market.get(day, {})
        prior_equity = equity
        day_pnl = daily_fee = slippage = turnover = 0.0
        cancelled = zero_rounded = blocked_rolls = stale = limit_rejections = 0
        for contract, (lots, basis, multiplier, symbol) in list(positions.items()):
            quote = quotes.get(contract)
            if quote and quote.open > 0:
                if isfinite(quote.multiplier) and quote.multiplier != multiplier:
                    raise RuntimeError(f"{day.date()} {contract} 持仓时合约乘数变更，缺少换算规则")
                day_pnl += lots * multiplier * (quote.open - basis)
                positions[contract] = (lots, quote.open, multiplier, symbol)
        equity += day_pnl
        row = targets.loc[day]
        signals = []
        keep_pending = {}
        target = {}
        signal_index = i - 1 - delay - phase
        rebalance = rebalance_every == 1 or (signal_index >= 0 and signal_index % rebalance_every == 0)
        if i < len(dates) - 1 and rebalance:
            for symbol, value in row[row.ne(0)].items():
                contract = contracts.at[day, symbol]
                quote = quotes.get(contract)
                if quote and quote.can_trade and quote.open_cap > 0:
                    signals.append((contract, symbol, value, quote))
                else:
                    current = positions.get(contract, (0,))[0]
                    if value * current > 0:
                        keep_pending[contract] = current
                    cancelled += 1
            new = lot_targets(
                np.array([value for _, _, value, _ in signals]),
                np.array([quote.open * quote.multiplier for _, _, _, quote in signals]),
                np.array([quote.long_margin if value > 0 else quote.short_margin for _, _, value, quote in signals]),
                np.array([quote.limit for _, _, _, quote in signals]),
                equity, MARGIN_CAP,
            )
            target = {**keep_pending, **{contract: int(lots) for lots, (contract, _, _, _) in zip(new, signals)}}
            if keep_same_lots:
                retained = target.copy()
                for contract, _, value, quote in signals:
                    held = positions.get(contract, (0,))[0]
                    name_limit = name_cap * equity / (quote.open * quote.multiplier)
                    if held * value > 0 and abs(held) <= min(quote.limit, name_limit):
                        retained[contract] = held
                gross_at_open = 0.0
                margin_at_open = 0.0
                for contract, lots in retained.items():
                    quote = quotes.get(contract)
                    if not quote or not quote.can_trade:
                        _, basis, multiplier, _ = positions[contract]
                        gross_at_open += abs(lots) * basis * multiplier
                        margin_at_open += abs(lots) * basis * multiplier
                        continue
                    notional_at_open = abs(lots) * quote.open * quote.multiplier
                    gross_at_open += notional_at_open
                    margin_at_open += notional_at_open * (quote.long_margin if lots > 0 else quote.short_margin)
                if gross_at_open <= max(equity, 0) and margin_at_open <= MARGIN_CAP * max(equity, 0):
                    target = retained
            zero_rounded = sum(target[contract] == 0 for contract, _, _, _ in signals)
        elif i < len(dates) - 1:
            # 非调仓日保留原整数手。主力切换时仅将旧仓的当日名义金额折算至新合约。
            for old, (lots, _, multiplier, symbol) in positions.items():
                contract = contracts.at[day, symbol]
                prior, quote = quotes.get(old), quotes.get(contract)
                if (contract == old or not prior or not prior.can_trade or not quote or not quote.can_trade
                        or quote.open_cap <= 0 or prior.limit_blocks(-lots) or quote.limit_blocks(lots)
                        or abs(lots) > min(prior.exit_cap, prior.max_order)):
                    target[old] = lots
                    blocked_rolls += int(isinstance(contract, str) and contract != old)
                    continue
                notional = lots * prior.open * multiplier
                size = min(quote.limit, floor(abs(notional) / (quote.open * quote.multiplier) + 1e-10))
                target[contract] = size if lots > 0 else -size
                zero_rounded += int(size == 0)
                signals.append((contract, symbol, notional, quote))

        def order(contract, symbol, delta, opening):
            nonlocal equity, daily_fee, slippage, turnover, cancelled, limit_rejections
            quote = quotes.get(contract)
            if not quote or not quote.can_trade:
                cancelled += 1
                return 0
            if quote.limit_blocks(delta):
                cancelled += 1
                limit_rejections += 1
                return 0
            capacity = quote.open_cap if opening else quote.exit_cap
            size = min(abs(delta), quote.max_order, capacity)
            if opening:
                ratio = quote.long_margin if delta > 0 else quote.short_margin
                unit_fee = quote.open_fee * (1 if quote.by_lot else quote.open * quote.multiplier)
                unit_cost = quote.tick * slippage_ticks * quote.multiplier + unit_fee
                margin_room = max(0, MARGIN_CAP * max(equity, 0) - margin_used())
                size = min(size, floor(margin_room / (quote.open * quote.multiplier * ratio + MARGIN_CAP * unit_cost)))
            if (opening and size < quote.min_order) or size <= 0:
                cancelled += 1
                return 0
            if size < abs(delta):
                cancelled += 1
            signed = size if delta > 0 else -size
            effective = quote.open + (quote.tick if signed > 0 else -quote.tick) * slippage_ticks
            rate = quote.open_fee if opening else quote.close_fee
            fee = size * rate * (1 if quote.by_lot else quote.open * quote.multiplier)
            slip = size * abs(effective - quote.open) * quote.multiplier
            amount = size * quote.open * quote.multiplier
            equity -= fee + slip
            daily_fee += fee
            slippage += slip
            turnover += amount
            fills.append({"date": day, "symbol": symbol, "contract": contract, "trade_lots": signed,
                          "action": "open" if opening else "close", "fill": quote.open, "fee": fee,
                          "effective_price": effective, "slippage": slip, "notional": amount,
                          "fee_fallback": quote.fee_fallback,
                          "missing_limit": not (quote.limit_up > 0 and quote.limit_down > 0)})
            return signed

        def margin_used():
            total = 0.0
            for contract, (lots, basis, multiplier, _) in positions.items():
                quote = quotes.get(contract)
                ratio = (quote.long_margin if lots > 0 else quote.short_margin) if quote else 1.0
                total += abs(lots) * basis * multiplier * ratio
            return total

        for contract, (lots, basis, multiplier, symbol) in list(positions.items()):
            desired = target.get(contract, 0)
            keep = (1 if lots > 0 else -1) * min(abs(lots), abs(desired)) if lots * desired > 0 else 0
            if keep != lots:
                remaining = lots + order(contract, symbol, keep - lots, False)
                if remaining:
                    positions[contract] = (remaining, basis, multiplier, symbol)
                else:
                    del positions[contract]

        open_symbol = {value[3]: contract for contract, value in positions.items()}
        for contract, symbol, value, quote in sorted(signals, key=lambda item: (-abs(item[2]), item[0])):
            desired = target[contract]
            have = positions.get(contract, (0,))[0]
            if desired and open_symbol.get(symbol, contract) != contract:
                blocked_rolls += 1
                continue
            if (have == 0 or have * desired > 0) and abs(desired) > abs(have):
                filled = order(contract, symbol, desired - have, True)
                if filled:
                    positions[contract] = (have + filled, quote.open, quote.multiplier, symbol)

        for contract, (lots, basis, multiplier, symbol) in list(positions.items()):
            quote = quotes.get(contract)
            if quote and quote.settle > 0:
                pnl = lots * multiplier * (quote.settle - basis)
                day_pnl += pnl
                equity += pnl
                positions[contract] = (lots, quote.settle, multiplier, symbol)
            else:
                stale += 1
        notional = sum(lots * basis * mult for lots, basis, mult, _ in positions.values())
        gross = sum(abs(lots) * basis * mult for lots, basis, mult, _ in positions.values())
        margin = margin_used()
        ledger.append({
            "date": day, "equity": equity, "return": equity / prior_equity - 1 if prior_equity > 0 else np.nan,
            "gross_pnl": day_pnl, "fees": daily_fee, "slippage": slippage, "traded_notional": turnover,
            "turnover": turnover / prior_equity if prior_equity > 0 else np.nan, "margin": margin,
            "gross_notional": gross, "net_notional": notional, "margin_fraction": margin / equity if equity > 0 else np.nan,
            "open_contracts": len(positions), "open_lots": sum(abs(value[0]) for value in positions.values()),
            "unfilled_orders": cancelled, "zero_rounded_targets": zero_rounded,
            "limit_rejections": limit_rejections, "blocked_rolls": blocked_rolls, "stale_marks": stale,
        })
    return pd.DataFrame(ledger).set_index("date"), pd.DataFrame(fills)

COST_BPS = 5.0
TD = C.TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------- 权重层面的成本、换手与绩效

def cost_rate_matrix(rows: pd.DataFrame, fees: pd.DataFrame, slippage_ticks: float) -> pd.DataFrame:
    """按品种与主力合约匹配费用；只向过去匹配已知手续费，不回填未来费率。"""
    fee_fields = {
        "symbol", "trading_date", "contract", "commission_type", "open_commission",
        "close_commission", "contract_multiplier", "tick_size",
    }
    missing = fee_fields - set(fees.columns)
    if missing:
        raise ValueError(
            f"手续费文件缺少成本估计字段 {sorted(missing)}；请重新运行 "
            "python -m download.getFeeHistory --partition research"
        )
    if "c1_prev_close" not in rows or "c1_contract" not in rows:
        raise ValueError("日面板缺少主力合约代码或决策时可用的前收盘价")
    left = rows[["date", "symbol", "c1_contract", "c1_prev_close"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    left = left.sort_values("date", kind="mergesort")
    right = fees[list(fee_fields)].copy()
    right["trading_date"] = pd.to_datetime(right["trading_date"])
    right = right.rename(columns={"trading_date": "date", "contract": "c1_contract"})
    right = right.sort_values("date", kind="mergesort")
    if right.duplicated(["date", "symbol", "c1_contract"]).any():
        raise ValueError("手续费文件存在重复的日期/品种/合约记录")
    matched = pd.merge_asof(
        left, right, on="date", by=["symbol", "c1_contract"], direction="backward"
    )
    components = one_way_cost_components(matched, slippage_ticks)
    matched["cost_bps"] = components.sum(axis=1, min_count=2)
    return matched.pivot(index="date", columns="symbol", values="cost_bps").sort_index()


def one_way_cost_components(rows: pd.DataFrame, slippage_ticks: float) -> pd.DataFrame:
    """按每边手续费与 tick 数换算 bp；按手手续费使用合约名义价值折算。"""
    if slippage_ticks < 0:
        raise ValueError("slippage_ticks 不得为负")
    required = {
        "commission_type", "open_commission", "close_commission",
        "contract_multiplier", "tick_size", "c1_prev_close",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"成本输入缺少字段: {sorted(missing)}")

    price = pd.to_numeric(rows["c1_prev_close"], errors="coerce")
    multiplier = pd.to_numeric(rows["contract_multiplier"], errors="coerce")
    tick = pd.to_numeric(rows["tick_size"], errors="coerce")
    commission = pd.concat(
        [pd.to_numeric(rows["open_commission"], errors="coerce"),
         pd.to_numeric(rows["close_commission"], errors="coerce")],
        axis=1,
    ).max(axis=1)
    kind = rows["commission_type"]
    by_money = kind.eq("by_money")
    by_volume = kind.eq("by_volume")
    unknown = kind.notna() & ~(by_money | by_volume)
    if unknown.any():
        raise ValueError(f"未知手续费类型: {sorted(kind[unknown].astype(str).unique())}")

    commission_bps = pd.Series(np.nan, index=rows.index, dtype=float)
    commission_bps.loc[by_money] = commission.loc[by_money] * 1e4
    valid_volume = by_volume & price.gt(0) & multiplier.gt(0)
    commission_bps.loc[valid_volume] = (
        commission.loc[valid_volume] /
        (price.loc[valid_volume] * multiplier.loc[valid_volume]) * 1e4
    )
    slippage_bps = slippage_ticks * tick / price.where(price > 0) * 1e4
    return pd.DataFrame({
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
    })


def turnover_by_symbol(weights: pd.DataFrame, rolls: pd.DataFrame,
                       returns: pd.DataFrame | None = None, delay: int = 0) -> pd.DataFrame:
    """返回每个交易日、品种的单边组合换手，展期按平旧开新计两次。"""
    dates = returns.index if returns is not None else rolls.index
    columns = returns.columns if returns is not None else weights.columns
    w = weights.reindex(index=dates, columns=columns).fillna(0.0)
    held = w.shift(1 + delay).fillna(0.0)
    trade = held.diff().abs()
    if len(trade):
        trade.iloc[0] = held.iloc[0].abs()
    roll = (held.abs() * rolls.reindex(index=dates, columns=columns)
            .fillna(False).astype(float)) * 2
    return trade + roll


def backtest(weights: pd.DataFrame, returns: pd.DataFrame, rolls: pd.DataFrame,
             delay: int = 0, cost_bps: float | pd.DataFrame = COST_BPS) -> pd.DataFrame:
    """weights 为 t 日收盘目标权重，持有到 t+1+delay 日收盘取得 returns 的该日收益。

    换手 = Σ|Δw| + 换月日 2|w|（平旧主力、开新主力）；成本 = cost_bps × 换手。
    """
    dates = returns.index
    w = weights.reindex(index=dates, columns=returns.columns).fillna(0.0)
    held = w.shift(1 + delay).fillna(0.0)          # 当日收益对应的持仓
    r = returns.fillna(0.0)
    gross = (held * r).sum(axis=1)
    trade = held.diff().abs().sum(axis=1)
    trade.iloc[0] = held.iloc[0].abs().sum()
    roll = (held.abs() * rolls.reindex(index=dates, columns=returns.columns).fillna(False).astype(float)).sum(axis=1) * 2
    turnover = trade + roll
    if isinstance(cost_bps, pd.DataFrame):
        rates = cost_bps.reindex(index=dates, columns=returns.columns)
        symbol_turnover = (held.diff().abs().fillna(0.0)
                           + held.abs() * rolls.reindex(
                               index=dates, columns=returns.columns
                           ).fillna(False).astype(float) * 2)
        if len(symbol_turnover):
            symbol_turnover.iloc[0] = held.iloc[0].abs()
        missing = symbol_turnover.gt(0) & rates.isna()
        if missing.any().any():
            raise ValueError(f"活跃交易缺少成本数据: {int(missing.sum().sum())} 个品种日")
        active_rates = rates.where(symbol_turnover.gt(0), 0.0)
        invalid = active_rates.lt(0) | ~np.isfinite(active_rates)
        if invalid.any().any():
            raise ValueError("成本 bp 必须为有限非负数")
        cost = (symbol_turnover * active_rates).sum(axis=1) / 1e4
    else:
        if cost_bps < 0 or not np.isfinite(cost_bps):
            raise ValueError("cost_bps 必须为有限非负数")
        cost = turnover * cost_bps / 1e4
    net = gross - cost
    active = held.abs().sum(axis=1) > 0
    out = pd.DataFrame({"gross": gross, "net": net, "cost": cost,
                        "turnover": turnover, "roll_turnover": roll,
                        "gmv": held.abs().sum(axis=1), "nmv": held.sum(axis=1)})
    first = active.idxmax() if active.any() else dates[-1]
    return out.loc[first:]


def metrics(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Exhibit 5 口径：年化收益为日均 × 242，夏普不扣无风险利率（期货多空无资金占用）。"""
    r = r.dropna()
    ann = r.mean() * TD
    vol = r.std() * np.sqrt(TD)
    down = np.sqrt((r.clip(upper=0) ** 2).mean()) * np.sqrt(TD)
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    out = {
        "ann_return": ann, "ann_vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
        "sortino": ann / down if down > 0 else np.nan, "downside_vol": down,
        "var95": -np.quantile(r, 0.05), "success_rate": (r > 0).mean(), "max_drawdown": dd,
        "t_stat": r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan,
        "days": len(r),
    }
    if turnover is not None:
        out["daily_turnover"] = turnover.reindex(r.index).mean()
    return out


# ---------------------------------------------------------------- 命令行入口

STRATEGIES = {
    "buffer": "strategies.buffer",
    "hold-five": "strategies.hold_five",
    "risk-budget": "strategies.risk_budget",
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m backtest.engine")
    parser.add_argument("strategy", choices=list(STRATEGIES))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    from importlib import import_module

    module = import_module(STRATEGIES[args.strategy])
    module.run(D.load_panel(), C.RESEARCH_FEE_PATH, C.strategy_dir(module.STRATEGY_ID), jobs=args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""商品期货整数手账本、绩效统计和策略命令行入口。

    python -m backtest.engine --research
    python -m backtest.engine --oos
    python -m backtest.engine --start 2016 --end 2025
    python -m backtest.engine --family member --start 2016 --end 2025

默认运行 2016–2021 研究期。两类策略共用同一整数手引擎：member 只用会员持仓，
member_price 加入价格因子。输出按类别存放在 runs/strategies/<类别>/ 下。
"""
from __future__ import annotations

import argparse
import sys
from math import floor, isfinite
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config as C  # noqa: E402
import data as D  # noqa: E402
from costs import load_market  # noqa: E402
from research.settings import XS_POSITION  # noqa: E402
from strategies import Targets, sticky_chain, sticky_core, sticky_core_h20, sticky_only  # noqa: E402
from strategies.common import HORIZON, NAME_CAP, TARGET_MULTIPLIER  # noqa: E402

CAPITAL_CNY = 10_000_000
BENCHMARK_BOOK = "LONG_ONLY_BMK"
BENCHMARK_LABEL = "流动性池等权多头"
MARGIN_CAP = 0.50
FLAT_MARGIN = 0.20
BENCHMARK_NAME_CAP = 0.10
SLIPPAGE_TICKS = 1
BACKTEST_START = pd.Timestamp(XS_POSITION["test_start"])


def lot_targets(weights, notionals, margins, limits, equity, margin_cap=MARGIN_CAP):
    """按一手名义金额向零取整，先等比例压缩以满足组合保证金预算。"""
    ideal = np.divide(weights * max(equity, 0), notionals, out=np.zeros_like(weights, dtype=float),
                      where=np.isfinite(notionals) & (notionals > 0))
    initial_margin = (abs(ideal) * np.nan_to_num(notionals * margins, nan=0)).sum()
    scale = min(1.0, margin_cap * max(equity, 0) / initial_margin) if initial_margin else 1.0
    return np.trunc(np.clip(ideal * scale, -limits, limits)).astype(np.int64)


def simulate(weights, contracts, market, start, *, capital=CAPITAL_CNY, neutral=True,
             slippage_ticks=SLIPPAGE_TICKS, name_cap=BENCHMARK_NAME_CAP, margin_cap=MARGIN_CAP):
    """按开盘标价和滑点成交，按结算价盯市。先平仓，之后才允许开仓。"""
    if capital <= 0 or slippage_ticks < 0 or not (0 < name_cap <= 1) or not (0 < margin_cap <= 1):
        raise ValueError("初始本金必须为正，滑点不得为负，单品种上限与保证金上限必须在 (0, 1] 内")
    dates = contracts.loc[start:].index
    targets = weights.reindex_like(contracts).fillna(0).shift(1).loc[dates].fillna(0)
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
    last_quote: dict[str, pd.Timestamp] = {}
    for quoted_day, book in market.items():
        for contract, quote in book.items():
            if quote.settle > 0 and (contract not in last_quote or quoted_day > last_quote[contract]):
                last_quote[contract] = quoted_day
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
        if i < len(dates) - 1:
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
                equity, margin_cap,
            )
            target = {**keep_pending, **{contract: int(lots) for lots, (contract, _, _, _) in zip(new, signals)}}
            zero_rounded = sum(target[contract] == 0 for contract, _, _, _ in signals)

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
            # A zero client limit forbids carry, so fully liquidate instead of applying the entry-volume cap.
            capacity = quote.open_cap if opening else (
                abs(delta) if quote.limit == 0 else quote.exit_cap
            )
            size = min(abs(delta), quote.max_order, capacity)
            if opening:
                ratio = quote.long_margin if delta > 0 else quote.short_margin
                unit_fee = quote.open_fee * (1 if quote.by_lot else quote.open * quote.multiplier)
                unit_cost = quote.tick * slippage_ticks * quote.multiplier + unit_fee
                margin_room = max(0, margin_cap * max(equity, 0) - margin_used())
                size = min(size, floor(margin_room / (quote.open * quote.multiplier * ratio + margin_cap * unit_cost)))
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
                quote = quotes.get(contract)
                if quote and quote.limit == 0:
                    remaining = lots
                    while remaining != keep:
                        filled = order(contract, symbol, keep - remaining, False)
                        if filled == 0:
                            break
                        remaining += filled
                else:
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
                # 合约已无后续行情时按最近结算价结束；中间缺一天则保留持仓。
                if last_quote.get(contract, day) <= day:
                    del positions[contract]
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

TD = C.TRADING_DAYS_PER_YEAR


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


def _book_summary(book: str, ticks: int, daily: pd.DataFrame) -> dict:
    stats = metrics(daily["return"], daily["turnover"])
    peak = daily["equity"].cummax().clip(lower=CAPITAL_CNY)
    return {
        "book": book,
        "ticks": ticks,
        "end_equity": float(daily["equity"].iloc[-1]),
        "net_ann": float(stats["ann_return"]),
        "ann_vol": float(stats["ann_vol"]),
        "sharpe": float(stats["sharpe"]),
        "sortino": float(stats["sortino"]),
        "max_drawdown": float((daily["equity"] / peak - 1).min()),
        "turnover": float(stats["daily_turnover"]),
        "gross_pnl": float(daily["gross_pnl"].sum()),
        "fees": float(daily["fees"].sum()),
        "slippage": float(daily["slippage"].sum()),
        "gross_exposure": float(daily["gross_notional"].div(daily["equity"]).mean()),
        "net_exposure": float(daily["net_notional"].div(daily["equity"]).mean()),
        "margin_mean": float(daily["margin_fraction"].mean()),
        "margin_max": float(daily["margin_fraction"].max()),
        "stale_marks": int(daily["stale_marks"].sum()),
        "terminal_contracts": int(daily["open_contracts"].iloc[-1]),
    }


_CHART_FONTS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
}
_OVERVIEW_STYLE = {
    **_CHART_FONTS,
    "font.size": 10.5,
    "text.color": "#293e4c",
    "axes.labelcolor": "#5c6d79",
    "xtick.color": "#657682",
    "ytick.color": "#657682",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
}
_OOS_BACKGROUND = "#f3eeea"


def _chart_frame(equity: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(equity.columns)
    if missing:
        raise KeyError(f"权益曲线缺少绘图序列: {sorted(missing)}")
    frame = equity.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if (
        len(frame) < 2
        or not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
        or not np.isfinite(frame.to_numpy(dtype=float)).all()
        or (frame <= 0).any().any()
    ):
        raise ValueError("绘图权益曲线必须包含至少两日、日期有序且为正的完整数据")
    return frame


def _calendar_year_nav(equity: pd.DataFrame):
    opening = CAPITAL_CNY
    for year, frame in equity.groupby(equity.index.year, sort=True):
        yield year, frame / opening
        opening = frame.iloc[-1]


def yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    annual = pd.DataFrame({year: nav.iloc[-1] - 1 for year, nav in _calendar_year_nav(equity)}).T
    rows = annual.reset_index(names="year").melt(id_vars="year", var_name="book", value_name="return")
    return rows[["book", "year", "return"]]


def _plot_performance(nav: pd.DataFrame, out_path: Path, *, title: str) -> Path:
    """沿用 time_factor_cta 的双面板布局绘制收益和回撤。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    drawdown = nav.div(nav.cummax().clip(lower=1.0)) - 1.0
    with plt.rc_context(_CHART_FONTS):
        figure, (return_axis, drawdown_axis) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, height_ratios=(3, 1),
        )
        for column in nav:
            return_axis.plot(nav.index, nav[column] - 1, label=column, lw=1.2)
            drawdown_axis.plot(drawdown.index, drawdown[column], lw=1.0)
        drawdown_axis.fill_between(drawdown.index, drawdown.min(axis=1), 0, color="C0", alpha=0.2)
        return_axis.set_title(title)
        return_axis.set_ylabel("累计净收益")
        return_axis.legend(loc="upper left", fontsize=8)
        drawdown_axis.set_ylabel("回撤")
        for axis in (return_axis, drawdown_axis):
            axis.grid(alpha=0.3)
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        drawdown_axis.xaxis.set_major_locator(locator)
        drawdown_axis.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        figure.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=130)
        plt.close(figure)
    return out_path


def _plot_yearly_performance(equity: pd.DataFrame, out: Path, title: str) -> list[Path]:
    return [
        _plot_performance(nav, out / f"{year}.png", title=f"{title} · {year} 年")
        for year, nav in _calendar_year_nav(equity)
    ]


def plot_yearly_returns_drawdowns(
    equity: pd.DataFrame, out: Path, *, strategy: str, benchmark: str | None = None
) -> list[Path]:
    """每年一张图，以年初权益为起点，回撤峰值按年重置。"""
    columns = [strategy] if benchmark is None else [strategy, benchmark]
    frame = _chart_frame(equity, columns).rename(columns={benchmark: BENCHMARK_LABEL})
    return _plot_yearly_performance(frame, Path(out), strategy)


def _plot_book_figures(equity: pd.DataFrame, out: Path, book: str, period_label: str) -> None:
    frame = equity[[book, BENCHMARK_BOOK]].rename(columns={BENCHMARK_BOOK: BENCHMARK_LABEL})
    _plot_performance(frame / CAPITAL_CNY, out / "performance.png", title=f"{book} · {period_label}")
    plot_yearly_returns_drawdowns(equity, out / "yearly", strategy=book, benchmark=BENCHMARK_BOOK)


def _period_sharpe(equity: pd.DataFrame) -> pd.Series:
    returns = equity.pct_change(fill_method=None)
    volatility = returns.std()
    return returns.mean() * np.sqrt(TD) / volatility.where(volatility > 0)


def plot_equity_drawdown(
    equity: pd.DataFrame, out_path: Path, *, strategies: list[str], benchmark: str,
    period_label: str = "研究期", title: str = "会员持仓因子 · 商品期货截面策略",
    research_sharpe: pd.Series | None = None, oos_sharpe: pd.Series | None = None,
) -> Path:
    """绘制对数累计净值、回撤及逐年收益总览。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    frame = _chart_frame(equity, [*strategies, benchmark])
    nav = frame / CAPITAL_CNY
    drawdown = nav.div(nav.cummax().clip(lower=1.0)) - 1.0
    palette = ("#5B8FA8", "#C28A5A", "#6E9C7B", "#B76569")
    colors = {book: palette[i % len(palette)] for i, book in enumerate(strategies)}
    colors[benchmark] = "#8D83B5"
    split = pd.Timestamp(C.OOS_START)
    if research_sharpe is None:
        research_sharpe = _period_sharpe(frame.loc[frame.index < split])
    if oos_sharpe is None:
        oos_sharpe = _period_sharpe(frame.loc[frame.index >= split])
    research_sharpe = research_sharpe.reindex(frame.columns)
    oos_sharpe = oos_sharpe.reindex(frame.columns)
    annualized = nav.pct_change(fill_method=None).mean() * TD
    annual = pd.DataFrame({year: yearly.iloc[-1] - 1 for year, yearly in _calendar_year_nav(frame)}).T
    divider_y = min(0.752, 0.843 - (len(nav.columns) - 1) * 0.026 - 0.017)

    with plt.rc_context(_OVERVIEW_STYLE):
        figure = plt.figure(figsize=(20, 11.5))
        grid = figure.add_gridspec(
            2, 2, height_ratios=(2.8, 1), width_ratios=(2.05, 1),
            left=0.055, right=0.955, top=min(0.71, divider_y - 0.04), bottom=0.07, hspace=0.19, wspace=0.14,
        )
        nav_axis = figure.add_subplot(grid[0, 0])
        drawdown_axis = figure.add_subplot(grid[1, 0], sharex=nav_axis)
        year_axis = figure.add_subplot(grid[:, 1])
        for column in nav.columns:
            nav_axis.plot(
                nav.index, nav[column], color=colors[column],
                linewidth=1.8, solid_capstyle="round",
            )
            drawdown_axis.plot(
                drawdown.index, drawdown[column], color=colors[column],
                linewidth=1.25, solid_capstyle="round",
            )

        drawdown_axis.axhline(0, color="#aab8c4", linewidth=0.7)
        for axis in (nav_axis, drawdown_axis, year_axis):
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#d7e0e6")
            axis.tick_params(length=0, pad=7)
        for axis in (nav_axis, drawdown_axis):
            if nav.index.max() >= split:
                axis.axvspan(max(nav.index.min(), split), nav.index.max(),
                             color=_OOS_BACKGROUND, alpha=0.65, zorder=0)
                if nav.index.min() < split:
                    axis.axvline(split, color="#d8c8bc", linewidth=0.8, linestyle=(0, (3, 3)))
            axis.grid(axis="y", color="#e6ecf0", linewidth=0.7)
            axis.grid(axis="x", color="#eff3f6", linewidth=0.6)

        nav_axis.set_yscale("log")
        nav_min, nav_max = float(nav.min().min()), float(nav.max().max())
        nav_ticks = np.array([0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        nav_ticks = nav_ticks[(nav_ticks >= nav_min * 0.88) & (nav_ticks <= nav_max * 1.12)]
        if len(nav_ticks) >= 3:
            nav_axis.yaxis.set_major_locator(FixedLocator(nav_ticks))
        else:
            nav_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        nav_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        nav_axis.yaxis.set_minor_formatter(NullFormatter())
        nav_axis.set_title("累计净值", loc="left", fontsize=13, fontweight="bold", pad=13)
        plt.setp(nav_axis.get_xticklabels(), visible=False)
        nav_axis.tick_params(axis="y", labelsize=9.5)
        drawdown_axis.set_title("回撤", loc="left", fontsize=12, fontweight="bold", pad=8)
        drawdown_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        first, last = nav.index[[0, -1]]
        year_starts = pd.date_range(f"{first.year + 2}-01-01", f"{last.year - 1}-01-01", freq="2YS")
        tick_dates = [first, *year_starts, last] if first.year != last.year else [first]
        drawdown_axis.set_xticks(tick_dates)
        drawdown_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        drawdown_axis.set_xlim(nav.index.min(), nav.index.max())

        years = annual.index.to_numpy()
        positions = np.arange(len(years))
        bar_height = 0.72 / len(annual.columns)
        for y, year in zip(positions, years):
            if year >= split.year:
                year_axis.axhspan(y - 0.5, y + 0.5, color=_OOS_BACKGROUND, alpha=0.65, linewidth=0)
        for i, column in enumerate(annual.columns):
            values = annual[column].to_numpy(dtype=float) * 100
            offsets = positions + (i - (len(annual.columns) - 1) / 2) * bar_height
            year_axis.barh(offsets, values, height=bar_height * 0.82, color=colors[column], linewidth=0)
            for y, value in zip(offsets, values):
                year_axis.annotate(
                    f"{value:+.2f}%", (value, y),
                    xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center",
                    fontsize=8.2, color="#526571",
                )
        year_axis.set_yticks(positions, [str(year) for year in years])
        year_axis.tick_params(axis="y", labelsize=9.5)
        year_axis.set_ylim(len(years) - 0.5, -0.5)
        year_axis.axvline(0, color="#aab8c4", linewidth=0.8)
        year_axis.grid(axis="x", color="#e6ecf0", linewidth=0.7)
        low, high = float(annual.min().min() * 100), float(annual.max().max() * 100)
        span = max(high - low, 10.0)
        year_axis.set_xlim(
            min(0.0, low) - (max(span * 0.24, 8.0) if low < 0 else 4.0),
            max(0.0, high) + (max(span * 0.24, 8.0) if high > 0 else 4.0),
        )
        year_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        year_axis.set_title("逐年净收益", loc="left", fontsize=13, fontweight="bold", pad=13)

        figure.text(0.055, 0.955, title, fontsize=23, fontweight="bold")
        figure.text(
            0.055, 0.918, f"{period_label}   |   {first:%Y.%m.%d} — {last:%Y.%m.%d}   |   单边 1 tick",
            fontsize=11.5, color="#697c89",
        )
        figure.text(0.075, 0.872, "策略与基准", fontsize=10, color="#7b8b96")
        metric_columns = (
            (0.57, "研究期 Sharpe", research_sharpe, ".3f"),
            (0.69, "样本外 Sharpe", oos_sharpe, ".3f"),
            (0.81, "年化收益", annualized, ".2%"),
            (0.935, "全区间最大回撤", drawdown.min(), ".2%"),
        )
        for x, label, _, _ in metric_columns:
            figure.text(x, 0.872, label, ha="right", fontsize=10, color="#7b8b96")
        for i, column in enumerate(nav.columns):
            y = 0.843 - i * 0.026
            figure.add_artist(Line2D(
                [0.055, 0.068], [y + 0.004, y + 0.004],
                transform=figure.transFigure, color=colors[column], linewidth=2.2,
            ))
            figure.text(0.075, y, BENCHMARK_LABEL if column == benchmark else column, fontsize=10.8)
            for x, _, values, fmt in metric_columns:
                value = values[column]
                figure.text(
                    x, y, format(value, fmt) if pd.notna(value) else "—",
                    ha="right", fontsize=10.8, color=colors[column],
                )
        figure.add_artist(Line2D(
            [0.055, 0.955], [divider_y, divider_y], transform=figure.transFigure,
            color="#e2e8ed", linewidth=0.8,
        ))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=180, facecolor=figure.get_facecolor())
        plt.close(figure)
    return out_path


def _write_strategy_comparison(equity: pd.DataFrame, partition: str, family: str | None, *,
                              research_sharpe: pd.Series | None = None,
                              oos_sharpe: pd.Series | None = None) -> None:
    full_sample = partition == "combined"
    first, last = equity.index[[0, -1]]
    folder = f"{first.year}-{last.year}" if full_sample else partition
    out = comparison_dir(family) / folder
    out.mkdir(parents=True, exist_ok=True)
    daily_returns = equity.div(equity.shift(fill_value=CAPITAL_CNY)) - 1
    daily_returns.to_csv(out / "daily_returns.csv", index_label="date")
    plots = out / "plots"
    title = FAMILY_TITLES[family] if family is not None else "商品期货截面策略总览"
    if family is not None:
        frame = equity.rename(columns={BENCHMARK_BOOK: BENCHMARK_LABEL})
        _plot_performance(frame / CAPITAL_CNY, plots / "performance.png", title=f"{title} · {first.year}–{last.year}")
        _plot_yearly_performance(frame, plots, title)
    if full_sample:
        books = equity.columns.drop(BENCHMARK_BOOK).tolist()
        plot_equity_drawdown(equity, plots / "overview.png", strategies=books, benchmark=BENCHMARK_BOOK,
                             period_label="研究期 + 样本外", title=title,
                             research_sharpe=research_sharpe, oos_sharpe=oos_sharpe)
    print(f"策略对比图写入 {plots}")


# ---------------------------------------------------------------- 命令行入口

# 两类策略：只用会员持仓（member），会员持仓加价格因子（member_price）。输出目录按类别对称：
#   runs/strategies/<类别>/<策略>/{research,oos,combined}/
#   runs/strategies/<类别>/comparison/{research,oos,2016-2025}/
# 除 STICKY_CORE_H20 为 20 日外，所有策略 10 日重叠持有。
FAMILIES = {
    "member": (sticky_only, sticky_chain),
    "member_price": (sticky_core, sticky_core_h20),
}
FAMILY_TITLES = {
    "member": "会员持仓因子 · 商品期货截面策略",
    "member_price": "会员持仓因子 · 价格因子 · 商品期货截面策略",
}
MODULES = {module.STRATEGY_ID: module for books in FAMILIES.values() for module in books}
FAMILY_OF = {module.STRATEGY_ID: family for family, books in FAMILIES.items() for module in books}
PERIOD_LABELS = {"research": "研究期", "oos": "样本外"}


def book_dir(name: str) -> Path:
    return C.strategy_dir(FAMILY_OF[name]) / name


def comparison_dir(family: str | None) -> Path:
    return C.strategy_dir(family or "") / "comparison"


def _benchmark_targets(panel: pd.DataFrame, start: pd.Timestamp, partition: str) -> Targets:
    eligible, dominant = D.member_universe(panel, start, partition)
    eligible = eligible.loc[start:]
    weights = eligible.div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return Targets(weights, dominant)


def _assert_continuous_dates(values, what: str) -> None:
    """完整区间账本必须从研究期交易到样本外，且不越出 2025 年末。"""
    ts = pd.to_datetime(pd.Index(values))
    if len(ts) < 2 or ts.min() < BACKTEST_START or ts.max() > pd.Timestamp(C.DEFAULT_OOS_END):
        raise ValueError(f"{what} 日期越出 2016–2025")
    split = pd.Timestamp(C.OOS_START)
    if not ((ts < split).any() and (ts >= split).any()):
        raise ValueError(f"{what} 没有同时覆盖研究期和样本外")


def _span_fees() -> tuple[pd.DataFrame, pd.DataFrame]:
    research = pd.read_csv(C.RESEARCH_FEE_PATH, parse_dates=["trading_date"])
    oos = pd.read_csv(C.OOS_FEE_PATH, parse_dates=["trading_date"])
    C.assert_research_only(C.RESEARCH_FEE_PATH)
    C.assert_no_holdout_dates(research["trading_date"], "历史手续费")
    C.assert_oos_dates(oos["trading_date"], "历史手续费")
    both = pd.concat([research, oos], ignore_index=True)
    if both.duplicated(["contract", "trading_date"]).any():
        raise ValueError("手续费在研究期与样本外分区重复")
    return research, oos


def _oos_fee_fallback(research: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    """样本外段的手续费后备表。

    下载只覆盖样本外主力，2021 年末主力在 2022 年首日仍按前一日主力成交时没有样本外记录。
    研究期记录放在前面，按合约向后取最近一条，只把旧数据带入新分区。
    """
    return pd.concat([research, oos], ignore_index=True)


def _span_market(bars: pd.DataFrame, research_fees: pd.DataFrame, oos_fees: pd.DataFrame,
                 ids: set[str], start: pd.Timestamp) -> dict:
    """研究期和样本外各自用本分区的手续费与执行参数，再按交易日合成一个行情表。

    样本外一段与单独的样本外回测相同：传入全部合约日线，成交量容量用跨年的前 20 日中位数，
    load_market 再按 oos_start 截取。
    """
    research_end, oos_start = pd.Timestamp(C.RESEARCH_END), pd.Timestamp(C.OOS_START)
    research = load_market(
        bars.loc[bars["date"].le(research_end)], research_fees, ids, start,
        root=C.RESEARCH_EXECUTION_PARAMETERS / "contracts", flat_margin=FLAT_MARGIN, partition="research",
    )
    oos = load_market(
        bars, _oos_fee_fallback(research_fees, oos_fees), ids, oos_start,
        root=C.OOS_EXECUTION_PARAMETERS / "contracts", flat_margin=FLAT_MARGIN, partition="oos",
    )
    overlap = set(research) & set(oos)
    if overlap:
        raise ValueError(f"执行参数日期在两个分区重复: {min(overlap).date()}")
    return {**research, **oos}


def _run_partition(module, partition: str, base_out: Path, jobs: int) -> None:
    if partition == "combined":
        panel = D.load_evaluation_panel()
        research_fees, oos_fees = _span_fees()
        fees = pd.concat([research_fees, oos_fees], ignore_index=True)
        start, root = BACKTEST_START, None
        check_dates = _assert_continuous_dates
    elif partition == "oos":
        panel = D.load_evaluation_panel()
        start, root = pd.Timestamp(C.OOS_START), C.OOS_EXECUTION_PARAMETERS
        check_dates = C.assert_oos_dates
        fees = _oos_fee_fallback(*_span_fees())
    else:
        panel, fee_path = D.load_panel(), C.RESEARCH_FEE_PATH
        start, root = BACKTEST_START, C.RESEARCH_EXECUTION_PARAMETERS
        check_dates = C.assert_no_holdout_dates
        C.assert_research_only(fee_path)
        check_dates(panel["date"], "研究期日面板")
        fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
        check_dates(fees["trading_date"], "历史手续费")

    bars = D.load_contracts_span("oos" if partition == "combined" else partition)
    benchmark = module is None
    if benchmark:
        book, curve_file = BENCHMARK_BOOK, "equity_curves.csv"
        targets = _benchmark_targets(panel, start, partition)
    else:
        book, curve_file = module.MAIN_BOOK, module.EQUITY_CURVE_FILE
        targets = module.build_targets(panel, bars, start, partition, jobs)
    contracts = targets.dominant.shift(1).reindex_like(targets.weights)
    check_dates(contracts.index, "策略净值交易日")
    ids = {contract for contract in contracts.to_numpy().ravel() if isinstance(contract, str)}
    if partition == "combined":
        market = _span_market(bars, research_fees, oos_fees, ids, start)
    else:
        market = load_market(bars, fees, ids, start, root=root / "contracts",
                             flat_margin=FLAT_MARGIN, partition=partition)
    weights = targets.weights if benchmark else targets.weights * TARGET_MULTIPLIER
    name_cap = BENCHMARK_NAME_CAP if benchmark else NAME_CAP

    rows, curves = [], {}
    for ticks in (SLIPPAGE_TICKS, 2):
        ledger, _ = simulate(
            weights, contracts, market, start, neutral=not benchmark, slippage_ticks=ticks, name_cap=name_cap,
        )
        row = _book_summary(book, ticks, ledger)
        if not benchmark:
            row.update(hold=getattr(module, "HOLD", HORIZON), target_multiplier=TARGET_MULTIPLIER)
        rows.append(row)
        if ticks == SLIPPAGE_TICKS:
            curves[book] = ledger["equity"]

    out = base_out / partition
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    equity = pd.DataFrame(curves)
    equity.to_csv(out / curve_file, index_label="date")
    yearly_returns(equity).to_csv(out / "yearly_returns.csv", index=False)
    if targets.selection is not None:
        targets.selection.to_csv(out / "selection_summary.csv", index=False)
    fields = ["book", "end_equity", "net_ann", "sharpe", "max_drawdown"]
    print(summary.loc[summary.ticks.eq(SLIPPAGE_TICKS), fields].round(4).to_string(index=False))
    print(f"回测结果写入 {out}")


def _read_curve(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date") if path.is_file() else pd.DataFrame()


def _benchmark_curve(partition: str, jobs: int, *, reuse: bool) -> pd.Series:
    base = C.strategy_dir("benchmark")
    path = base / partition / "equity_curves.csv"
    cached = _read_curve(path) if reuse else pd.DataFrame()
    if BENCHMARK_BOOK not in cached:
        _run_partition(None, partition, base, jobs)
        cached = _read_curve(path)
    return _chart_frame(cached, [BENCHMARK_BOOK])[BENCHMARK_BOOK]


def _evaluate_partition(selected: list[str], partition: str, jobs: int, *, reuse: bool,
                        benchmark: pd.Series | None = None) -> pd.DataFrame:
    """独立读取或生成基准和所选策略的分区账本。"""
    if benchmark is None:
        benchmark = _benchmark_curve(partition, jobs, reuse=reuse)
    check_dates = C.assert_oos_dates if partition == "oos" else C.assert_no_holdout_dates
    check_dates(benchmark.index, f"{partition} 等权基准净值")
    curves = []
    for name in selected:
        module, base_out = MODULES[name], book_dir(name)
        path = base_out / partition / module.EQUITY_CURVE_FILE
        cached = _read_curve(path) if reuse else pd.DataFrame()
        if module.MAIN_BOOK not in cached:
            _run_partition(module, partition, base_out, jobs)
            cached = _read_curve(path)
        curve = _chart_frame(cached, [module.MAIN_BOOK])[module.MAIN_BOOK]
        check_dates(curve.index, f"{name} {partition} 净值")
        if not curve.index.equals(benchmark.index):
            raise ValueError(f"{partition} {name} 与等权基准的交易日未对齐")
        curves.append(curve)
        _plot_book_figures(pd.concat([curve, benchmark], axis=1), base_out / partition,
                           module.MAIN_BOOK, PERIOD_LABELS[partition])
    return pd.concat([*curves, benchmark], axis=1)


def _evaluate_continuous(selected: list[str], jobs: int) -> pd.DataFrame:
    """2016–2025 用同一账户连续回测。仓位、保证金和重叠持有都跨过 2022 年。"""
    benchmark = _benchmark_curve("combined", jobs, reuse=False)
    _assert_continuous_dates(benchmark.index, "完整区间等权基准净值")
    curves = []
    for name in selected:
        module, base_out = MODULES[name], book_dir(name)
        _run_partition(module, "combined", base_out, jobs)
        path = base_out / "combined" / module.EQUITY_CURVE_FILE
        curve = _chart_frame(_read_curve(path), [module.MAIN_BOOK])[module.MAIN_BOOK]
        _assert_continuous_dates(curve.index, f"{name} 完整区间净值")
        if not curve.index.equals(benchmark.index):
            raise ValueError(f"完整区间 {name} 与等权基准的交易日未对齐")
        curves.append(curve)
    return pd.concat([*curves, benchmark], axis=1)


def _run_combined(family: str, selected: list[str], jobs: int) -> dict[str, pd.DataFrame]:
    """研究期和样本外各自开户。完整区间另跑一条连续账本，图上的夏普仍取两个独立分区。"""
    partitions = {split: _evaluate_partition(selected, split, jobs, reuse=True) for split in C.PARTITIONS}
    combined = _evaluate_continuous(selected, jobs)
    for name in selected:
        # 连续账本的 summary.csv（含 2 跳）已由 _run_partition 写出，这里只补画净值图。
        _plot_book_figures(combined, book_dir(name) / "combined", MODULES[name].MAIN_BOOK, "研究期 + 样本外")
    partitions["combined"] = combined
    if len(selected) > 1:
        research_sharpe = _period_sharpe(partitions["research"])
        oos_sharpe = _period_sharpe(partitions["oos"])
        for split, equity in partitions.items():
            _write_strategy_comparison(
                equity, split, family,
                research_sharpe=research_sharpe if split == "combined" else None,
                oos_sharpe=oos_sharpe if split == "combined" else None,
            )
    return partitions


def _partition_frame(families: list[dict[str, pd.DataFrame]], phase: str) -> pd.DataFrame:
    """把各类策略在同一分区的净值拼到一条基准旁边。"""
    books = [family[phase].drop(columns=BENCHMARK_BOOK) for family in families]
    return pd.concat([*books, families[0][phase][BENCHMARK_BOOK]], axis=1)


def _write_all_comparison(families: list[dict[str, pd.DataFrame]]) -> None:
    rows = []
    for phase in (*C.PARTITIONS, "combined"):
        frames = [family[phase] for family in families]
        benchmark = frames[0][BENCHMARK_BOOK]
        if any(not frame[BENCHMARK_BOOK].equals(benchmark) for frame in frames[1:]):
            raise ValueError(f"{phase} 各策略类别的等权基准未对齐")
        equity = pd.concat([*(frame.drop(columns=BENCHMARK_BOOK) for frame in frames), benchmark], axis=1)
        if phase == "combined":
            research = _partition_frame(families, "research")
            oos = _partition_frame(families, "oos")
            _write_strategy_comparison(equity, phase, None, research_sharpe=_period_sharpe(research),
                                       oos_sharpe=_period_sharpe(oos))
        stats_nav = equity.drop(columns=BENCHMARK_BOOK)
        for book, nav in stats_nav.items():
            stats = metrics(nav.div(nav.shift(fill_value=CAPITAL_CNY)) - 1)
            rows.append(dict(book=book, phase=phase, net_ann=stats["ann_return"], sharpe=stats["sharpe"],
                             max_drawdown=(nav / nav.cummax().clip(lower=CAPITAL_CNY) - 1).min()))
    pd.DataFrame(rows).to_csv(C.strategy_dir("") / "summary.csv", index=False)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m backtest.engine")
    parser.add_argument("strategy", nargs="?", choices=list(MODULES), help="省略时运行所选类别的全部策略")
    parser.add_argument("--family", choices=list(FAMILIES), help="策略类别；省略时两类都运行")
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--research", dest="partition", action="store_const", const="research")
    period.add_argument("--oos", dest="partition", action="store_const", const="oos")
    period.add_argument("--start", type=int, help="回测起始年，搭配 --end")
    period.add_argument("--partition", choices=("research", "oos", "combined"))
    parser.add_argument("--end", type=int, help="回测结束年，搭配 --start")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start 和 --end 须一起使用")
    partition = args.partition or "research"
    if args.start is not None:
        windows = {
            (BACKTEST_START.year, C.RESEARCH_END.year): "research",
            (C.OOS_START.year, C.DEFAULT_OOS_END.year): "oos",
            (BACKTEST_START.year, C.DEFAULT_OOS_END.year): "combined",
        }
        partition = windows.get((args.start, args.end))
        if partition is None:
            allowed = "、".join(f"{start}–{end}" for start, end in windows)
            parser.error(f"可用回测区间为 {allowed}")
    if args.strategy and args.family and FAMILY_OF[args.strategy] != args.family:
        parser.error(f"{args.strategy} 不属于 {args.family}")

    if args.strategy:
        families = [FAMILY_OF[args.strategy]]
    else:
        families = [args.family] if args.family else list(FAMILIES)
    combined_families = []
    benchmark = None if partition == "combined" else _benchmark_curve(partition, args.jobs, reuse=False)
    for family in families:
        selected = [args.strategy] if args.strategy else [module.STRATEGY_ID for module in FAMILIES[family]]
        if partition == "combined":
            combined_families.append(_run_combined(family, selected, args.jobs))
            continue
        equity = _evaluate_partition(selected, partition, args.jobs, reuse=False, benchmark=benchmark)
        if len(selected) > 1:
            _write_strategy_comparison(equity, partition, family)
    if len(combined_families) > 1:
        _write_all_comparison(combined_families)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

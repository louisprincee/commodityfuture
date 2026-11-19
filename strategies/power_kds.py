"""唯一主策略 POWER_KDS_3X：填补一致性底仓 + 空仓冲击多头卫星，统一 5 日重叠、固定 50/50 合并与 3 倍目标敞口。

    python -m backtest.engine power_kds --jobs 8
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from backtest import engine as E
from research import factors as F
from research.settings import XS_POSITION as XS
from strategies import rules as R

STRATEGY_ID = "power_kds"
HORIZON = 5
LEVERAGE = 3.0
ACTIVE_SLOT_FLOOR = 8
SLOT_FLOORS = (1, 2, 3, 5, 8, 10, 15, 20)
MAIN_BOOK = "POWER_KDS_3X"
COMBO_NAME_CAP = 0.20
FLAT_MARGIN = 0.20
MARGIN_CAP = 0.50
SIGNAL_WINDOW = 250
SIGNAL_MIN = 60


def calendar_year_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """各自然年收益：当年末权益相对上年末权益。首年以该列首个权益为基数。"""
    frame = equity.sort_index().apply(pd.to_numeric, errors="coerce")
    rows = []
    for column in frame.columns:
        series = frame[column].dropna()
        for year, part in series.groupby(series.index.year):
            earlier = series.loc[series.index.year < year]
            base = float(earlier.iloc[-1] if len(earlier) else part.iloc[0])
            rows.append({"book": column, "year": int(year), "return": float(part.iloc[-1] / base - 1.0)})
    return pd.DataFrame(rows)


def _symbol_scores(name: str, panel: pd.DataFrame, bars: pd.DataFrame,
                   dates: pd.DatetimeIndex, partition: str = "research") -> pd.DataFrame:
    if partition == "research":
        ranks = D.load_member_rank_contracts(name, "research")
    elif partition == "oos":
        ranks = D.load_member_rank_contracts_span(name)
    else:
        raise ValueError(f"未知分区 {partition}")
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    kaiyuan = F.kaiyuan(ranks, bars, dates, include_disclosed_oi=True)
    oi = kaiyuan["disclosed_oi"]
    shock = -kaiyuan["K_DS"].div(oi.where(oi.gt(0)))
    shock_dir, _ = R.short_shock(shock, window=SIGNAL_WINDOW, minimum=SIGNAL_MIN)
    tables = F.member_tables(name, dates, top20=True, partition=partition)
    power = F.consistency_from_net(F.imputed_net(tables), 20)["POWER"]
    flow_tables = F.member_tables(name, dates, partition=partition)
    flow = flow_tables["dL"].sum(axis=1, min_count=1) - flow_tables["dS"].sum(axis=1, min_count=1)
    market_oi = F.market_inputs(panel, bars, name)["oi"].where(lambda value: value.gt(0))
    flow_control = flow.div(market_oi)
    return pd.DataFrame({
        "power_filled": power, "short_shock": shock_dir, "flow_control": flow_control,
    }, index=dates)


def _eval_book(name: str, sleeve: str, delay: int, ticks: int,
               daily: pd.DataFrame, bmk_ret: pd.Series) -> dict:
    stats = E.metrics(daily["return"], daily["turnover"])
    peak = daily["equity"].cummax().clip(lower=E.CAPITAL_CNY)
    max_drawdown = float((daily["equity"] / peak - 1).min())
    net_ann = float(stats["ann_return"])
    ret = daily["return"].fillna(0.0)
    bmk = bmk_ret.reindex(ret.index).fillna(0.0)
    bmk_var = float(bmk.var())
    beta = float(ret.cov(bmk) / bmk_var) if bmk_var > 0 else np.nan
    residual = ret - beta * bmk if np.isfinite(beta) else ret
    years = len(ret) / E.TD
    terminal_growth = float(daily["equity"].iloc[-1] / E.CAPITAL_CNY)
    cagr = terminal_growth ** (1 / years) - 1 if years > 0 and terminal_growth > 0 else np.nan
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "book": name, "sleeve": sleeve, "delay": delay, "ticks": ticks,
        "end_equity": float(daily["equity"].iloc[-1]),
        "net_ann": net_ann, "ann_vol": float(stats["ann_vol"]),
        "sharpe": float(stats["sharpe"]), "sortino": float(stats["sortino"]),
        "calmar": calmar, "max_drawdown": max_drawdown,
        "turnover": float(stats["daily_turnover"]),
        "gross_pnl": float(daily["gross_pnl"].sum()), "fees": float(daily["fees"].sum()),
        "slippage": float(daily["slippage"].sum()),
        "gross_exposure": float(daily["gross_notional"].div(daily["equity"]).mean()),
        "signed_net_exposure": float(daily["net_notional"].div(daily["equity"]).mean()),
        "abs_net_exposure": float(daily["net_notional"].abs().div(daily["equity"]).mean()),
        "margin_mean": float(daily["margin_fraction"].mean()),
        "margin_max": float(daily["margin_fraction"].max()),
        "beta_to_bmk": beta,
        "market_adj_net_ann": float(residual.mean() * E.TD),
        "market_adj_sharpe": float(E.metrics(residual)["sharpe"]),
        "net_ann_2016_2018": float(daily.loc[:"2018-12-31", "return"].mean() * E.TD),
        "net_ann_2019_2021": float(daily.loc["2019-01-01":, "return"].mean() * E.TD),
        "terminal_contracts": int(daily["open_contracts"].iloc[-1]),
        "stale_marks": int(daily["stale_marks"].sum()),
    }


def prepare_signals(panel: pd.DataFrame, fee_path: Path, jobs: int, partition: str = "research") -> dict:
    """准备流动性池、主力合约行情、20% 保证金市场及底仓/卫星原始信号。

    研究期只读研究分区。样本外评价使用研究期至信号日的历史做窗口预热，成交与费用只取样本外分区。
    """
    if partition not in C.PARTITIONS:
        raise ValueError(f"未知分区 {partition}")
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    if partition == "research":
        C.assert_no_holdout_dates(panel["date"], "策略日面板")
        C.assert_research_only(fee_path)
        C.assert_no_holdout_dates(fees["trading_date"], "策略手续费")
        start = pd.Timestamp(XS["test_start"])
        bars = D.load_contracts("research")
        market_root = C.RESEARCH_EXECUTION_PARAMETERS / "contracts"
    else:
        start = pd.Timestamp(C.OOS_START)
        if not panel["date"].lt(start).any():
            raise RuntimeError("样本外回测缺少研究期预热")
        C.assert_oos_dates(panel.loc[panel["date"].ge(start), "date"], "样本外策略日面板")
        C.assert_oos_dates(fees["trading_date"], "策略手续费")
        bars = D.load_contracts_span("oos")
        market_root = C.OOS_EXECUTION_PARAMETERS / "contracts"

    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    names = sorted(panel.loc[panel["eligible"] & panel["date"].ge(start), "symbol"].unique())
    if len(names) < XS["min_names"]:
        raise RuntimeError("评价区间没有满足流动性条件的商品")

    by_symbol = dict(tuple(bars.loc[bars["symbol"].isin(names)].groupby("symbol")))
    panel_by_symbol = dict(tuple(panel.groupby("symbol", sort=False)))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_symbol_scores)(name, panel_by_symbol[name], by_symbol[name], dates, partition)
        for name in names
    )
    scores = pd.concat(pieces, axis=1, keys=names).swaplevel(axis=1).sort_index(axis=1)

    eligible = (
        panel.pivot(index="date", columns="symbol", values="eligible")
        .reindex(index=dates, columns=names)
        .astype("boolean").fillna(False).astype(bool).loc[start:]
    )
    principal = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(columns=names)
    contracts = principal.shift(1).loc[start:]
    ids = {name for name in contracts.to_numpy().ravel() if isinstance(name, str)}
    market = E.load_market(
        bars[["date", "contract", "open", "settlement", "volume"]],
        fees, ids, start, root=market_root, flat_margin=FLAT_MARGIN, partition=partition,
    )

    raw_power = scores["power_filled"]
    past = raw_power.shift(1).rolling(SIGNAL_WINDOW, min_periods=SIGNAL_MIN)
    z_power = (raw_power - past.mean()).div(past.std().where(lambda s: s.gt(0)))
    raw_flow = scores["flow_control"]
    flow_past = raw_flow.shift(1).rolling(SIGNAL_WINDOW, min_periods=SIGNAL_MIN)
    z_flow = (raw_flow - flow_past.mean()).div(flow_past.std().where(lambda s: s.gt(0)))
    shock_sig = scores["short_shock"].loc[start:].where(eligible, 0.0).fillna(0.0)
    w_bmk = eligible.astype(float).div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {
        "start": start, "eligible": eligible, "contracts": contracts, "market": market,
        "raw_power": raw_power, "z_power": z_power, "shock_sig": shock_sig,
        "raw_flow": raw_flow, "z_flow": z_flow, "w_bmk": w_bmk,
    }


def power_kds_weights(z_power: pd.DataFrame, raw_power: pd.DataFrame, shock_sig: pd.DataFrame,
                      eligible: pd.DataFrame, horizon: int = HORIZON,
                      min_slots: int = R.MIN_SLOTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """统一持有期 H：底仓合力顺势 + 卫星空仓冲击多头，固定 50/50 合成。"""
    core_dir = R.timing_directions(z_power.reindex(eligible.index).where(eligible),
                                   raw_power.reindex(eligible.index).where(eligible))
    w_core = R.hold_cohorts(R.active_weights(core_dir, min_slots=min_slots), horizon)
    w_sat = R.hold_cohorts(R.active_weights(shock_sig.clip(lower=0.0), min_slots=min_slots), horizon)
    return w_core, w_sat, 0.5 * w_core + 0.5 * w_sat


def flow_ablation_weights(z_power: pd.DataFrame, raw_power: pd.DataFrame, shock_sig: pd.DataFrame,
                          z_flow: pd.DataFrame, raw_flow: pd.DataFrame, eligible: pd.DataFrame,
                          horizon: int = HORIZON) -> dict[str, pd.DataFrame]:
    """构造固定预算的 FLOW_CONTROL 替代与第三腿对照，不改变主策略权重。"""
    w_core, w_sat, _ = power_kds_weights(z_power, raw_power, shock_sig, eligible, horizon)
    flow_dir = R.timing_directions(
        z_flow.reindex(eligible.index).where(eligible),
        raw_flow.reindex(eligible.index).where(eligible),
    )
    w_flow = R.hold_cohorts(R.active_weights(flow_dir), horizon)
    return {
        "POWER_KDS_FLOW_REPLACES_SAT_2X": 0.5 * w_core + 0.5 * w_flow,
        "POWER_KDS_FLOW_502525_2X": 0.5 * w_core + 0.25 * w_sat + 0.25 * w_flow,
    }


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int, partition: str = "research") -> None:
    inputs = prepare_signals(panel, fee_path, jobs, partition)
    start, contracts, market, w_bmk = inputs["start"], inputs["contracts"], inputs["market"], inputs["w_bmk"]
    w_core, w_sat, w50 = power_kds_weights(
        inputs["z_power"], inputs["raw_power"], inputs["shock_sig"], inputs["eligible"],
        HORIZON, min_slots=ACTIVE_SLOT_FLOOR,
    )
    books = {
        "POWER_20_CORE_1X": ("sleeve", w_core, 1.0, E.NAME_CAP),
        "KDS_SHORT_SHOCK_1X": ("sleeve", w_sat, 1.0, E.NAME_CAP),
        MAIN_BOOK: ("combo", w50, LEVERAGE, COMBO_NAME_CAP),
    }

    out.mkdir(parents=True, exist_ok=True)
    rows, curves, benchmark_returns = [], {}, {}
    for ticks in (1, 2):
        bmk_daily, _ = E.simulate(
            w_bmk, contracts, market, start, delay=0, neutral=False,
            slippage_ticks=ticks, margin_cap=MARGIN_CAP,
        )
        bmk_ret = bmk_daily["return"]
        benchmark_returns[ticks] = bmk_ret
        rows.append(_eval_book("LONG_ONLY_BMK", "benchmark", 0, ticks, bmk_daily, bmk_ret))
        if ticks == 1:
            curves["LONG_ONLY_BMK"] = bmk_daily["equity"]

        for name, (sleeve, w_base, lev, cap) in books.items():
            daily, _ = E.simulate(
                w_base * lev, contracts, market, start, delay=0, neutral=False,
                slippage_ticks=ticks, name_cap=cap, margin_cap=MARGIN_CAP,
            )
            rows.append(_eval_book(name, sleeve, 0, ticks, daily, bmk_ret))
            if ticks == 1:
                curves[name] = daily["equity"]

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    equity_curves = pd.DataFrame(curves)
    equity_curves.to_csv(out / "equity_curves_d0_t1.csv", index_label="date")
    calendar_year_returns(equity_curves).to_csv(out / "yearly_returns.csv", index=False)
    figure = E.plot_equity_drawdown(
        equity_curves,
        out / "performance.png",
        strategy=MAIN_BOOK,
        benchmark="LONG_ONLY_BMK",
        period_label="样本外" if partition == "oos" else "研究期",
    )

    if partition == "research":
        _write_slot_grid(inputs, benchmark_returns, curves, out)
    view = summary.loc[:, [
        "book", "ticks", "end_equity", "net_ann", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
        "turnover", "gross_exposure", "margin_mean", "margin_max", "beta_to_bmk", "market_adj_net_ann",
    ]]
    print(view.round(4).to_string(index=False))
    print(f"净值与回撤图写入 {figure}")


def _write_slot_grid(inputs: dict, benchmark_returns: dict, curves: dict, out: Path) -> None:
    """研究期活动槽位网格。样本外评价不重搜该参数。"""
    start, contracts, market = inputs["start"], inputs["contracts"], inputs["market"]
    grid_rows, grid_curves = [], {"LONG_ONLY_BMK": curves["LONG_ONLY_BMK"]}
    for slot_floor in SLOT_FLOORS:
        _, _, weights = power_kds_weights(
            inputs["z_power"], inputs["raw_power"], inputs["shock_sig"],
            inputs["eligible"], HORIZON, min_slots=slot_floor,
        )
        grid_book = f"POWER_KDS_3X_K{slot_floor}"
        for ticks in (1, 2):
            daily, _ = E.simulate(
                weights * LEVERAGE, contracts, market, start, delay=0, neutral=False,
                slippage_ticks=ticks, name_cap=COMBO_NAME_CAP, margin_cap=MARGIN_CAP,
            )
            result = _eval_book(
                grid_book, f"slot_floor_{slot_floor}", 0, ticks, daily, benchmark_returns[ticks]
            )
            result["min_slots"] = slot_floor
            grid_rows.append(result)
            if ticks == 1:
                grid_curves[grid_book] = daily["equity"]

    grid_out = out / "slot_floor_grid"
    grid_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).to_csv(grid_out / "summary.csv", index=False)
    pd.DataFrame(grid_curves).to_csv(grid_out / "equity_curves_t1.csv", index_label="date")
    grid_view = pd.DataFrame(grid_rows)[[
        "min_slots", "ticks", "net_ann", "sharpe", "sortino", "calmar", "max_drawdown",
        "turnover", "gross_exposure", "margin_max",
    ]]
    print("\nmin_slots 参数网格：")
    print(grid_view.round(4).to_string(index=False))
    print(f"\n参数网格结果写入 {grid_out}")

"""多周期子策略风险预算组合：低频一致性底仓 + 空仓冲击多头卫星（统一 20% 保证金）。

    python -m backtest.engine risk-budget
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
from strategies import rules as R
from research.settings import XS_POSITION as XS

FLAT_MARGIN = 0.20
STRATEGY_ID = "risk_budget"
SIGNAL_WINDOW = 250
SIGNAL_MIN = 60
LEVERAGE_SPECS = ((1.0, E.NAME_CAP), (2.0, 0.15), (3.0, 0.20), (4.0, 0.20))


def _symbol_scores(name: str, panel: pd.DataFrame, bars: pd.DataFrame,
                   dates: pd.DatetimeIndex) -> pd.DataFrame:
    ranks = D.load_member_rank_contracts(name, "research")
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    kaiyuan = F.kaiyuan(ranks, bars, dates, include_disclosed_oi=True)
    oi = kaiyuan["disclosed_oi"]
    shock = -kaiyuan["K_DS"].div(oi.where(oi.gt(0)))
    shock_dir, _ = R.short_shock(shock)
    tables = F.member_tables(name, dates, top20=True)
    power = F.consistency_from_net(F.imputed_net(tables), 20)["POWER"]
    return pd.DataFrame({"power_filled": power, "short_shock": shock_dir}, index=dates)


def _eval_book(name: str, sleeve: str, delay: int, ticks: int, phase: int,
               daily: pd.DataFrame, bmk_ret: pd.Series) -> dict:
    stats = E.metrics(daily["return"], daily["turnover"])
    peak = daily["equity"].cummax().clip(lower=E.CAPITAL_CNY)
    ret = daily["return"].fillna(0.0)
    bmk = bmk_ret.reindex(ret.index).fillna(0.0)
    bmk_var = float(bmk.var())
    beta = float(ret.cov(bmk) / bmk_var) if bmk_var > 0 else np.nan
    residual = ret - beta * bmk if np.isfinite(beta) else ret
    return {
        "book": name, "sleeve": sleeve, "phase": phase, "delay": delay, "ticks": ticks,
        "end_equity": float(daily["equity"].iloc[-1]),
        "net_ann": float(stats["ann_return"]), "ann_vol": float(stats["ann_vol"]),
        "sharpe": float(stats["sharpe"]), "sortino": float(stats["sortino"]),
        "max_drawdown": float((daily["equity"] / peak - 1).min()),
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


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int) -> None:
    C.assert_no_holdout_dates(panel["date"], "风险预算日面板")
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees["trading_date"], "风险预算手续费")

    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    start = pd.Timestamp(XS["test_start"])
    names = sorted(panel.loc[panel["eligible"] & panel["date"].ge(start), "symbol"].unique())
    if len(names) < XS["min_names"]:
        raise RuntimeError("风险预算区间没有满足流动性条件的商品")

    bars = D.load_contracts("research")
    panels = dict(tuple(panel.groupby("symbol")))
    by_symbol = dict(tuple(bars.loc[bars["symbol"].isin(names)].groupby("symbol")))
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_symbol_scores)(name, panels[name], by_symbol[name], dates) for name in names
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
        fees, ids, start, flat_margin=FLAT_MARGIN,
    )

    raw_power = scores["power_filled"]
    past = raw_power.shift(1).rolling(SIGNAL_WINDOW, min_periods=SIGNAL_MIN)
    z_power = (raw_power - past.mean()).div(past.std().where(lambda s: s.gt(0)))
    daily_core_dir = R.timing_directions(z_power.loc[start:].where(eligible), raw_power.loc[start:].where(eligible))
    w_core_10d = R.hold_cohorts(R.active_weights(daily_core_dir), 10)
    w_core_5d_phases = {}
    for phase in range(5):
        obs = z_power.loc[start:].where(eligible).iloc[phase::5]
        w_core_5d_phases[phase] = R.active_weights(R.timing_directions(obs, raw_power.reindex(obs.index).where(eligible)))
    w_core_5d_step0 = w_core_5d_phases[0].reindex(eligible.index).ffill().fillna(0.0)

    shock_sig = scores["short_shock"].loc[start:].where(eligible, 0.0).fillna(0.0)
    w_sat_1d = R.active_weights(shock_sig.clip(lower=0.0))
    w_sat_1d_hedged = R.hedge_against_benchmark(w_sat_1d, eligible)
    w_sat_5d = R.hold_cohorts(w_sat_1d, 5)
    w_sat_5d_hedged = R.hedge_against_benchmark(w_sat_5d, eligible)
    w_sat_5d_bi = R.hold_cohorts(R.active_weights(shock_sig), 5)
    w_bmk = eligible.astype(float).div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    out_dir = out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, ledger_curves, corr_rows = [], {}, []

    for delay in (0, 1):
        for ticks in (1, 2):
            bmk_daily, _ = E.simulate(w_bmk, contracts, market, start, delay=delay, neutral=False, slippage_ticks=ticks)
            bmk_ret = bmk_daily["return"]
            rows.append(_eval_book("LONG_ONLY_BMK", "benchmark", delay, ticks, 0, bmk_daily, bmk_ret))

            standalone = {
                "CORE_POWER_10D": (w_core_10d, 1, 0, E.NAME_CAP),
                "CORE_POWER_10D_2X": (w_core_10d * 2.0, 1, 0, 0.20),
                "SAT_SS_LONG_1D": (w_sat_1d, 1, 0, E.NAME_CAP),
                "SAT_SS_LONG_1D_HEDGED": (w_sat_1d_hedged, 1, 0, E.NAME_CAP),
                "SAT_SS_LONG_5D": (w_sat_5d, 1, 0, E.NAME_CAP),
                "SAT_SS_LONG_5D_HEDGED": (w_sat_5d_hedged, 1, 0, E.NAME_CAP),
                "SAT_SS_BI_5D": (w_sat_5d_bi, 1, 0, E.NAME_CAP),
            }
            sim_res = {}
            for name, (w, itv, ph, cap) in standalone.items():
                daily, _ = E.simulate(w, contracts, market, start, delay=delay, neutral=False,
                                      slippage_ticks=ticks, rebalance_every=itv, phase=ph, name_cap=cap)
                sim_res[name] = daily
                rows.append(_eval_book(name, "sleeve", delay, ticks, ph, daily, bmk_ret))

            for ph in range(5):
                daily, _ = E.simulate(w_core_5d_phases[ph], contracts, market, start, delay=delay,
                                      neutral=False, slippage_ticks=ticks, rebalance_every=5, phase=ph)
                rows.append(_eval_book("CORE_POWER_5D", "sleeve", delay, ticks, ph, daily, bmk_ret))
                if ph == 0:
                    sim_res["CORE_POWER_5D_P0"] = daily

            for core_key, sat_key in (
                ("CORE_POWER_10D", "SAT_SS_LONG_5D_HEDGED"),
                ("CORE_POWER_10D", "SAT_SS_LONG_5D"),
                ("CORE_POWER_5D_P0", "SAT_SS_LONG_5D_HEDGED"),
                ("CORE_POWER_5D_P0", "SAT_SS_LONG_1D_HEDGED"),
            ):
                rc, rs = sim_res[core_key]["return"], sim_res[sat_key]["return"]
                corr_rows.append({
                    "core": core_key, "satellite": sat_key, "delay": delay, "ticks": ticks,
                    "daily_corr": float(rc.corr(rs)),
                    "corr_2016_2018": float(rc.loc[:"2018-12-31"].corr(rs.loc[:"2018-12-31"])),
                    "corr_2019_2021": float(rc.loc["2019-01-01":].corr(rs.loc["2019-01-01":])),
                })

            combos = [
                ("COMBO_10D_5DH", w_core_10d, sim_res["CORE_POWER_10D"]["return"],
                 w_sat_5d_hedged, sim_res["SAT_SS_LONG_5D_HEDGED"]["return"], ((1.0, E.NAME_CAP), (2.0, 0.15))),
                ("COMBO_5D_5DH", w_core_5d_step0, sim_res["CORE_POWER_5D_P0"]["return"],
                 w_sat_5d_hedged, sim_res["SAT_SS_LONG_5D_HEDGED"]["return"], ((1.0, E.NAME_CAP), (2.0, 0.15))),
                ("COMBO_10D_5D_UNHEDGED", w_core_10d, sim_res["CORE_POWER_10D"]["return"],
                 w_sat_5d, sim_res["SAT_SS_LONG_5D"]["return"], LEVERAGE_SPECS),
                ("COMBO_5D_1DH", w_core_5d_step0, sim_res["CORE_POWER_5D_P0"]["return"],
                 w_sat_1d_hedged, sim_res["SAT_SS_LONG_1D_HEDGED"]["return"], ((1.0, E.NAME_CAP), (2.0, 0.15))),
            ]
            for cname, wc, rc, ws, rs, lev_specs in combos:
                alpha_c, alpha_s = R.inverse_vol_weights(rc, rs)
                w_base = wc.mul(alpha_c, axis=0) + ws.mul(alpha_s, axis=0)
                for lev, cap in lev_specs:
                    book_name = f"{cname}_{int(lev)}X"
                    daily, _ = E.simulate(w_base * lev, contracts, market, start, delay=delay, neutral=False,
                                          slippage_ticks=ticks, name_cap=cap)
                    rows.append(_eval_book(book_name, "combo", delay, ticks, 0, daily, bmk_ret))
                    if delay == 0 and ticks == 1:
                        ledger_curves[book_name] = daily["equity"]

            if delay == 0 and ticks == 1:
                ledger_curves["CORE_POWER_10D"] = sim_res["CORE_POWER_10D"]["equity"]
                ledger_curves["SAT_SS_LONG_5D_HEDGED"] = sim_res["SAT_SS_LONG_5D_HEDGED"]["equity"]
                ledger_curves["LONG_ONLY_BMK"] = bmk_daily["equity"]

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out_dir / "sleeve_correlations.csv", index=False)
    pd.DataFrame(ledger_curves).to_csv(out_dir / "equity_curves_d0_t1.csv", index_label="date")

    view = summary.loc[summary["ticks"].eq(1)].groupby(["book", "delay"])[
        ["end_equity", "net_ann", "ann_vol", "sharpe", "max_drawdown", "turnover",
         "gross_exposure", "signed_net_exposure", "margin_mean", "margin_max", "market_adj_net_ann"]
    ].median()
    print(view.round(4).to_string())

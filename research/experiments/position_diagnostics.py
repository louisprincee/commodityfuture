"""复算历史截面账户的成本归因与时序对照。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research import portfolio as P
from . import xs_position as X

FACTORS = ("CROWD1", "POWER_20", "NET_CHG_60", "FLOW")
HORIZONS = (1, 10)
TD = C.TRADING_DAYS_PER_YEAR
BPS = 10_000


def hac_t(series, lags=20):
    values = series.to_numpy()
    resid = values - values.mean()
    n = len(values)
    variance = np.dot(resid, resid) / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * np.dot(resid[lag:], resid[:-lag]) / n
    return values.mean() / np.sqrt(variance / n) if variance > 0 else np.nan


def run(panel: pd.DataFrame, fee_path: Path, output: Path, jobs: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    saved_path = output / "performance.csv"
    C.assert_research_only(saved_path)
    if not saved_path.is_file():
        raise FileNotFoundError("缺少截面账户结果，请先运行 python -m research xs")
    saved = pd.read_csv(saved_path).set_index(["book", "delay"])
    C.assert_no_holdout_dates(panel["date"], "成本诊断日面板")
    panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    eligible = P.eligibility(panel)
    ranks = C.daily_dir("research") / "member_rank"
    C.assert_research_only(ranks)
    symbols = sorted(s for s in panel.loc[eligible, "symbol"].unique() if (ranks / f"{s}_long.pkl").exists())
    contracts = D.load_contracts("research")
    slim = panel.loc[panel["symbol"].isin(symbols), ["date", "symbol", "c1_contract", "ret1", "volume_total", "oi_total"]]
    opens = contracts.loc[contracts["symbol"].isin(symbols), ["symbol", "contract", "date", "open"]]
    panels = dict(tuple(slim.groupby("symbol")))
    bars = dict(tuple(opens.groupby("symbol")))
    long = pd.concat(Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(X._symbol_frame)(symbol, panels[symbol], bars[symbol]) for symbol in symbols
    ), ignore_index=True).sort_values(["symbol", "date"], kind="mergesort")
    C.assert_no_holdout_dates(long["date"], "member diagnostics")
    flags = panel[["date", "symbol"]].assign(eligible=eligible.to_numpy())
    long = long.merge(flags, on=["date", "symbol"], how="left")
    tradable = long["eligible"].fillna(False) & long["ranked"].fillna(False)
    returns = X._pivot(long, "ret_oo")
    rolls = X._pivot(long, "roll").reindex_like(returns).astype("boolean").fillna(False).astype(bool)
    start = pd.Timestamp(X.XS["test_start"])
    calendar = returns.index[(returns.index >= start) & (returns.index <= pd.Timestamp(C.RESEARCH_END))]
    assert len(calendar) == 1461

    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees["trading_date"], "diagnostic fees")
    fee = P.cost_rate_matrix(panel, fees, 0).reindex_like(returns)
    combined = P.cost_rate_matrix(panel, fees, 1).reindex_like(returns)
    tick = combined - fee
    assert (~(tick < 0)).to_numpy().all()

    def measure(weights, delay):
        target = X._align_weights(weights, returns, delay).where(combined.shift(-(1 + delay)).notna(), 0.0)
        target.loc[target.index < start] = 0
        held = target.shift(1 + delay).fillna(0.0)
        previous = held.shift().fillna(0.0)
        turnover = P.turnover_by_symbol(target, rolls, returns, delay=delay)
        true_turnover = turnover.where(~rolls, previous.abs() + held.abs())
        gross = (held * returns.fillna(0)).sum(axis=1)
        commission = (turnover * fee).sum(axis=1) / BPS
        slip = (turnover * tick).sum(axis=1) / BPS
        roll_turnover = (held.abs() * rolls * 2).sum(axis=1)
        legacy = pd.DataFrame({
            "gross": gross, "net": gross - commission - slip, "fee": commission, "tick": slip,
            "turnover": turnover.sum(axis=1), "roll": roll_turnover,
            "turnover_on_switch": turnover.where(rolls, 0).sum(axis=1),
            "roll_cost": ((held.abs() * rolls * 2) * combined).sum(axis=1) / BPS,
            "gmv": held.abs().sum(axis=1),
            "nmv": held.sum(axis=1),
            "contract_turnover": true_turnover.sum(axis=1),
            "contract_fee_proxy": (true_turnover * fee).sum(axis=1) / BPS,
            "contract_tick_proxy": (true_turnover * tick).sum(axis=1) / BPS,
        }).loc[calendar]
        return legacy

    eligible_wide = X._pivot(long, "eligible").reindex_like(returns).astype("boolean").fillna(False).astype(bool)
    count = eligible_wide.sum(axis=1)
    benchmark = eligible_wide.div(count.replace(0, np.nan), axis=0).where(count >= X.XS["min_names"], 0).fillna(0)
    cost_rows = []
    match_rows = []

    def summarize(result):
        net = P.metrics(result["net"], result["turnover"])
        gross = P.metrics(result["gross"])
        breakeven = (result["gross"].mean() - result["fee"].mean()) / result["tick"].mean()
        return {
            "gross_ann": gross["ann_return"], "fee_ann": result["fee"].mean() * TD,
            "tick_ann": result["tick"].mean() * TD, "net_ann": net["ann_return"],
            "gross_sharpe": gross["sharpe"], "net_sharpe": net["sharpe"],
            "net_sharpe_half_tick": P.metrics(result["net"] + 0.5 * result["tick"])["sharpe"],
            "net_sharpe_two_ticks": P.metrics(result["net"] - result["tick"])["sharpe"],
            "break_even_ticks": breakeven, "daily_turnover": net["daily_turnover"],
            "roll_addon_turnover": result["roll"].mean(),
            "roll_addon_cost_ann": result["roll_cost"].mean() * TD,
            "turnover_on_switch": result["turnover_on_switch"].mean(),
            "avg_gmv": result["gmv"].mean(), "avg_nmv": result["nmv"].mean(),
            "avg_abs_nmv": result["nmv"].abs().mean(), "net_t_hac20": hac_t(result["net"]),
            "net_ann_2016_2018": result.loc[:"2018-12-31", "net"].mean() * TD,
            "net_ann_2019_2021": result.loc["2019-01-01":, "net"].mean() * TD,
            "n_days": net["days"],
        }

    raw, z = {}, {}
    for factor in FACTORS:
        values = long[factor]
        sign = X.XS["signs"][factor]
        raw[factor] = X._pivot(long.assign(score=sign * values), "score", tradable & values.notna())
        zvalues = long.groupby("symbol", sort=False)[factor].transform(
            lambda s: X.past_zscore(s, X.XS["zscore_window"], X.XS["zscore_min_periods"])
        ) * sign
        z[factor] = X._pivot(long.assign(score=zvalues), "score", tradable & zvalues.notna())

    legacy_books = {"LONG_ONLY": benchmark}
    min_names = X.XS["min_names"]
    for factor in FACTORS:
        w, _ = X._rank_weights(raw[factor], min_names)
        legacy_books[factor] = w
        for h in (5, 10, 60):
            legacy_books[f"{factor}_H{h}"] = P.hold_cohorts(w.reindex_like(returns).fillna(0), h)
    for name, weights in legacy_books.items():
        for delay in (0, 1):
            result = measure(weights, delay)
            row = summarize(result)
            old = saved.loc[(name, delay)]
            for key, old_key in (("net_ann", "ann_return"), ("net_sharpe", "sharpe"),
                                 ("gross_sharpe", "gross_sharpe"), ("daily_turnover", "daily_turnover")):
                np.testing.assert_allclose(row[key], old[old_key], atol=1e-10)
            row.update({
                "contract_turnover": result["contract_turnover"].mean(),
                "contract_proxy_net_ann": (result["gross"] - result["contract_fee_proxy"] -
                                           result["contract_tick_proxy"]).mean() * TD,
            })
            cost_rows.append({"book": name, "delay": delay, **row})

    for factor, score in z.items():
        count = score.notna().sum(axis=1)
        xs, _ = X._rank_weights(score, min_names)
        ts = score.clip(-2, 2).div(count.where(count >= min_names), axis=0).fillna(0)
        for mode, daily in (("cross_section", xs), ("time_series", ts)):
            for h in HORIZONS:
                weights = P.hold_cohorts(daily.reindex_like(returns).fillna(0), h)
                for delay in (0, 1):
                    result = measure(weights, delay)
                    # 本对照使用按新旧合约分别计算的换手；旧合约费率暂以当日主力费率代理。
                    for base, proxy in (("fee", "contract_fee_proxy"), ("tick", "contract_tick_proxy"),
                                        ("turnover", "contract_turnover")):
                        result[base] = result[proxy]
                    result["net"] = result["gross"] - result["fee"] - result["tick"]
                    row = summarize(result)
                    for key in ("roll_addon_turnover", "roll_addon_cost_ann", "turnover_on_switch"):
                        del row[key]
                    bmk = measure(benchmark, delay)["gross"]
                    beta = result["net"].cov(bmk) / bmk.var()
                    residual = result["net"] - beta * bmk
                    row["beta_to_commodity_long"] = beta
                    row["market_adjusted_net_ann"] = residual.mean() * TD
                    row["market_adjusted_hac20_t"] = hac_t(residual)
                    match_rows.append({"factor": factor, "mode": mode, "hold": h, "delay": delay, **row})

    costs = pd.DataFrame(cost_rows)
    matched = pd.DataFrame(match_rows)
    assert len(costs) == 34 and len(matched) == 32
    costs.to_csv(output / "cost_diagnostics.csv", index=False)
    matched.to_csv(output / "xs_ts_comparison.csv", index=False)
    print("Matched 34 legacy rows against four saved metrics each.")
    fields = ["gross_ann", "fee_ann", "tick_ann", "net_ann", "net_sharpe", "daily_turnover",
              "roll_addon_turnover", "contract_turnover", "break_even_ticks", "net_t_hac20"]
    print(costs[costs.delay == 0].set_index("book")[fields].round(4).to_string())
    fields = ["net_sharpe", "net_ann", "avg_gmv", "daily_turnover", "beta_to_commodity_long",
              "market_adjusted_net_ann", "net_t_hac20"]
    print(matched.set_index(["factor", "hold", "mode", "delay"])[fields].round(4).to_string())

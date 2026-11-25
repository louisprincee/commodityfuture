"""研究期席位分组与五日均值预设实验。

原值与五日均值使用同一家族内共同有效的品种日，均取正向。
净多贡献族以全榜可见多空总持仓统一归一化，检查 1/5/10/20 日。
持仓流族检查 1/5/10 日及剔除首段后的 2–5、6–10 日。窗口必须连续完整。
每族的 IC 检验单独做 BH。前后半期的两端价差仅作描述。
"""
from __future__ import annotations

import argparse
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config as C
from src import data as D
from research.factors.member import daily_position_flow, member_tables, seat_turnover_groups
from research.ic.labels import forward_returns, return_matrix
from research.ic.robustness import MIN_NAMES, START, leg_returns, membership_turnover
from research.ic.statistics import bh as _bh, hac_t, rank_corr as _rank_corr

LEVEL_WINDOWS = {"d1": (1, 1), "d5": (1, 5), "d10": (1, 10), "d20": (1, 20)}
RESIDUAL_WINDOWS = {"d2_to_d5": (2, 5), "d6_to_d10": (6, 10)}
FLOW_WINDOWS = {**{k: LEVEL_WINDOWS[k] for k in ("d1", "d5", "d10")}, **RESIDUAL_WINDOWS}


def common_seat_shares(tables: dict, market: pd.DataFrame) -> pd.DataFrame:
    net, gross, sticky, spec = seat_turnover_groups(tables, market)
    valid_market = market["oi"].gt(0) & market["volume"].gt(0)
    full_gross = gross.where(sticky | spec).sum(axis=1, min_count=1).where(valid_market)
    sticky_share = net.where(sticky, 0).sum(axis=1) / full_gross
    spec_share = net.where(spec, 0).sum(axis=1) / full_gross
    return pd.DataFrame({
        "BOOK_NET_TOTAL": sticky_share + spec_share,
        "BOOK_NET_STICKY": sticky_share,
        "BOOK_NET_HIGH_TURNOVER": spec_share,
    })


def matched_raw_and_ma5(raw: dict[str, pd.DataFrame], eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    versions = {}
    common = eligible.copy()
    for name, original in raw.items():
        score = original.where(np.isfinite(original))
        mean = score.rolling(5, min_periods=5).mean()
        versions[f"{name}_RAW"] = score
        versions[f"{name}_MA5"] = mean
        common &= mean.notna()
    return {name: score.where(common) for name, score in versions.items()}


def window_returns(returns: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    return forward_returns(returns, last - first + 1).shift(1 - first)


def _evaluate(scores, returns, windows, family):
    rows = []
    for name, score in scores.items():
        turnover = membership_turnover(score)
        for window, (first, last) in windows.items():
            label = window_returns(returns, first, last).reindex_like(score)
            ic, names = _rank_corr(score, label, MIN_NAMES)
            legs = leg_returns(score, label)
            long = legs["top"] - legs["pool"]
            short = legs["pool"] - legs["bottom"]
            spread = long + short
            t = hac_t(ic, max(20, 2 * last))
            rows.append(dict(
                family=family, factor=name, window=window, first_day=first, last_day=last,
                mean_ic=ic.mean(), hac_t=t, p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
                days=ic.count(), avg_names=names.mean(), early_ic=ic.loc[:"2018"].mean(),
                late_ic=ic.loc["2019":].mean(), long_bp=1e4 * long.mean(),
                short_bp=1e4 * short.mean(), spread_bp=1e4 * spread.mean(),
                early_spread_bp=1e4 * spread.loc[:"2018"].mean(),
                late_spread_bp=1e4 * spread.loc["2019":].mean(), tail_turnover=turnover,
            ))
    result = pd.DataFrame(rows)
    result["q"] = _bh(result["p"])
    return result


def _seat_symbol(symbol, dates, market):
    tables = member_tables(symbol, dates, top20=True, partition="research")
    return common_seat_shares(tables, market)


def load_seat_levels(panel: pd.DataFrame, eligible: pd.DataFrame, jobs: int) -> dict[str, pd.DataFrame]:
    """在研究期品种日历上计算同分母席位贡献。"""
    C.assert_no_holdout_dates(panel["date"], "席位平滑日面板")
    rank_root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(rank_root)
    symbols = [s for s in eligible.columns[eligible.loc[START:].any()] if (rank_root / f"{s}_long.pkl").is_file()]
    dates = eligible.index
    volumes = panel.pivot(index="date", columns="symbol", values="volume_total")
    oi = panel.pivot(index="date", columns="symbol", values="oi_total")
    groups = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_seat_symbol)(s, dates, pd.DataFrame({"volume": volumes[s], "oi": oi[s]})) for s in symbols
    )
    parts = pd.concat(groups, axis=1, keys=symbols).swaplevel(axis=1)
    return {name: parts[name] for name in parts.columns.get_level_values(0).unique()}


def load_daily_flow(universe: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    """由研究分区榜单直接计算每个交易日的持仓流，不先按流动性池遮掩。

    五日均值因此用完整的 t-4..t 历史，流动性池只在 t 日筛选。
    """
    dates, symbols = universe.index, universe.columns
    flow = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for s in symbols:
        tables = member_tables(s, dates, partition="research")
        flow[s] = daily_position_flow(tables, oi[s].reindex(dates))
    return flow


def run(out: Path, jobs: int) -> pd.DataFrame:
    panel = D.load_panel()
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))

    def wide(name):
        return panel.pivot(index="date", columns="symbol", values=name)

    eligible = wide("eligible").eq(True)
    level_inputs = load_seat_levels(panel, eligible, jobs)
    symbols = level_inputs["BOOK_NET_TOTAL"].columns
    eligible = eligible[symbols]
    flow = load_daily_flow(eligible, wide("oi_total"))
    bars = D.load_contracts("research")
    returns = return_matrix(wide("c1_contract")[symbols], bars.set_index(["contract", "date"])["open"])
    levels = {name: score.loc[START:] for name, score in matched_raw_and_ma5(level_inputs, eligible).items()}
    flows = {name: score.loc[START:] for name, score in matched_raw_and_ma5({"FLOW": flow}, eligible).items()}
    result = pd.concat([
        _evaluate(levels, returns, LEVEL_WINDOWS, "net_share"),
        _evaluate(flows, returns, FLOW_WINDOWS, "flow"),
    ], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "factor_ic.csv", index=False)
    pd.concat(flows, axis=1).stack(level=1, future_stack=True).dropna(how="all").rename_axis(
        index=["signal_date", "symbol"]
    ).to_csv(out / "flow_signals.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="运行会员席位平滑研究")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    out = C.study_dir("member_smoothing")
    result = run(out, args.jobs)
    print(f"{len(result)} 项 IC 写入 {out}")


if __name__ == "__main__":
    main()

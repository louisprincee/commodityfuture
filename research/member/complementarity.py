"""五日条件检验，样本固定为 2016–2021 年。

仅使用两项条件与 STICKY_NET_SHARE 均有效的品种，日截面至少 20 个。
先按粘性净多排序，两端各 ceil(20% N)，中间均分三组。在每组内
按 FLOW_MA5 或粘性席位占全榜持仓的份额排序，取高、低各一半；
奇数时留出中间一名，并按固定品种顺序处理同分。
两个主检验为流量确认价差及规模放大价差，单独做 BH。两组各五个
组内价差为诊断项，十项单独做 BH。全部均值用 20 阶 Newey–West。
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
from research.ic.study import forward_returns, return_matrix
from research.ic.statistics import bh as _bh, hac_t
from research.settings import XS_POSITION

HOLD = 5
MIN_NAMES = 20
START = pd.Timestamp(XS_POSITION["test_start"])
PRIMARY = ("flow_confirmation", "size_amplification")


def sticky_quintiles(sticky: pd.DataFrame) -> pd.DataFrame:
    n = sticky.notna().sum(axis=1)
    edge = np.ceil(n / 5)
    rank = sticky.rank(axis=1, method="first")
    inner = n - 2 * edge
    buckets = np.floor(rank.sub(edge + 1, axis=0).mul(3).div(inner.where(inner.gt(0)), axis=0)) + 2
    buckets = buckets.mask(rank.le(edge, axis=0), 1).mask(rank.gt(n - edge, axis=0), 5)
    return buckets.where(n.ge(MIN_NAMES), axis=0)


def ranked_halves(factor: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在组内返回因子较高和较低的各一半，奇数时跳过中间一名。"""
    rank = factor.where(mask).rank(axis=1, method="first")
    count = rank.notna().sum(axis=1)
    half = count // 2
    high = rank.gt(count - half, axis=0)
    low = rank.le(half, axis=0)
    return high, low


def daily_contrasts(buckets, flow, size, labels) -> pd.DataFrame:
    spreads = {}
    for name, factor in (("flow", flow), ("size", size)):
        for q in range(1, 6):
            high, low = ranked_halves(factor, buckets.eq(q))
            spreads[f"{name}_q{q}"] = labels.where(high).mean(axis=1) - labels.where(low).mean(axis=1)
    spreads["flow_confirmation"] = spreads["flow_q5"] + spreads["flow_q1"]
    spreads["size_amplification"] = spreads["size_q5"] - spreads["size_q1"]
    return pd.DataFrame(spreads)


def _seat_state(symbol, dates, market, partition):
    tables = member_tables(symbol, dates, top20=True, partition=partition)
    net, gross, sticky, active = seat_turnover_groups(tables, market)
    valid = market["oi"].gt(0) & market["volume"].gt(0)
    sticky_gross = gross.where(sticky).sum(axis=1, min_count=1).where(valid)
    visible_gross = gross.where(sticky | active).sum(axis=1, min_count=1).where(valid)
    sticky_net = net.where(sticky).sum(axis=1, min_count=1)
    return pd.DataFrame({"sticky": sticky_net / sticky_gross, "size": sticky_gross / visible_gross})


def load_seat_inputs(panel: pd.DataFrame, jobs: int, *, partition: str, start=None):
    """从会员榜加载流动性池内的粘性净多、席位份额和主力映射。"""
    if partition == "research":
        start = START if start is None else pd.Timestamp(start)
        C.assert_no_holdout_dates(panel["date"], "持仓互补研究日面板")
    elif partition == "oos":
        start = pd.Timestamp(C.OOS_START) if start is None else pd.Timestamp(start)
        C.assert_oos_dates(panel.loc[panel["date"].ge(start), "date"], "持仓互补样本外日面板")
    elif partition == "combined":
        start = START if start is None else pd.Timestamp(start)
    else:
        raise ValueError(f"未知分区 {partition}")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))

    def wide(name):
        return panel.pivot(index="date", columns="symbol", values=name)

    eligible = wide("eligible").eq(True)
    ranked = D.symbols_with_member_rank(partition)
    names = [s for s in eligible.columns[eligible.loc[start:].any()] if s in ranked]
    eligible = eligible[names]
    volume, oi = wide("volume_total"), wide("oi_total")
    states = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_seat_state)(s, eligible.index, pd.DataFrame({"volume": volume[s], "oi": oi[s]}), partition)
        for s in names
    )
    states = pd.concat(states, axis=1, keys=names).swaplevel(axis=1)
    return (states["sticky"].where(eligible).loc[start:],
            states["size"].where(eligible).loc[start:], wide("c1_contract")[names])


def load_daily_flow(universe: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    """由研究分区榜单直接计算每个交易日的持仓流，不先按流动性池遮掩。"""
    dates, symbols = universe.index, universe.columns
    flow = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for symbol in symbols:
        tables = member_tables(symbol, dates, partition="research")
        flow[symbol] = daily_position_flow(tables, oi[symbol].reindex(dates))
    return flow


def run(out: Path, jobs: int) -> pd.DataFrame:
    panel = D.load_panel()
    sticky, size, contracts = load_seat_inputs(panel, jobs, partition="research")
    oi = panel.pivot(index="date", columns="symbol", values="oi_total").reindex_like(contracts)
    flow = load_daily_flow(contracts, oi).rolling(5, min_periods=5).mean().reindex_like(sticky)
    bars = D.load_contracts("research")
    returns = return_matrix(contracts, bars.set_index(["contract", "date"])["open"])
    labels = forward_returns(returns, HOLD).reindex_like(sticky)
    sticky = sticky.where(flow.notna() & size.notna() & labels.notna())
    daily = daily_contrasts(sticky_quintiles(sticky), flow, size, labels)
    per_day = sticky.notna().sum(axis=1)
    coverage = per_day.where(per_day.ge(MIN_NAMES))

    rows = []
    for name, values in daily.items():
        t = hac_t(values, 20)
        rows.append(dict(
            contrast=name, family="primary" if name in PRIMARY else "diagnostic",
            spread_bp=values.mean() * 1e4, hac_t=t, p=erfc(abs(t) / sqrt(2)),
            early_bp=values.loc[:"2018"].mean() * 1e4, late_bp=values.loc["2019":].mean() * 1e4,
            days=values.count(), avg_names=coverage.where(values.notna()).mean(),
        ))
    result = pd.DataFrame(rows)
    result["q"] = result.groupby("family")["p"].transform(_bh)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "conditional_tests.csv", index=False)
    daily.rename_axis(index="date").to_csv(out / "daily_contrasts.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="运行会员持仓互补研究")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    out = C.study_dir("member_complementarity")
    result = run(out, args.jobs)
    print(f"{len(result)} 项持仓条件价差写入 {out}")


if __name__ == "__main__":
    main()

"""跨品种一致性净多及其相对普通粘性净多的残差。"""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research.factors import member_tables, seat_turnover_groups
from research.ic_study import _rank_corr, forward_returns, return_matrix
from research.xs_robustness import MIN_NAMES, START, hac_t


def masked_net_share(net: pd.DataFrame, gross: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    num = net.where(mask).sum(axis=1, min_count=1)
    den = gross.where(mask).sum(axis=1, min_count=1)
    return num / den.where(den.gt(0))


def chain_consistent_share(net: dict[str, pd.DataFrame], gross: dict[str, pd.DataFrame],
                           sticky: dict[str, pd.DataFrame], oi: pd.DataFrame,
                           symbols: list[str]) -> dict[str, pd.Series]:
    """同一板块内，对每个品种按其他品种上的方向筛出一致的粘性席位，再算可见净多占比。"""
    members = sorted(set().union(*(net[s].columns for s in symbols)))
    dates = next(iter(net.values())).index
    scaled, present = {}, {}
    for s in symbols:
        held = net[s].reindex(columns=members).where(sticky[s].reindex(columns=members, fill_value=False))
        held = held.where(gross[s].reindex(columns=members).gt(0))
        scaled[s] = held.div(oi[s].where(oi[s].gt(0)), axis=0)
        present[s] = scaled[s].notna()
    total = sum(frame.fillna(0.0) for frame in scaled.values())
    count = sum(mask.astype(int) for mask in present.values())
    out = {}
    for s in symbols:
        others = total - scaled[s].fillna(0.0)
        others_count = count - present[s].astype(int)
        direction = np.sign(others).where(others_count.ge(1))
        own = np.sign(scaled[s])
        consistent = own.ne(0) & own.eq(direction) & present[s]
        mask = consistent.reindex(columns=net[s].columns, fill_value=False)
        out[s] = masked_net_share(net[s], gross[s], mask).reindex(dates)
    return out


def residual_scores(score: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """同一批品种上，因子截面秩对控制变量截面秩回归后的残差。"""
    common = score.notna() & control.notna()
    x = score.where(common).rank(axis=1, pct=True)
    z = control.where(common).rank(axis=1, pct=True)
    x = x.sub(x.mean(axis=1), axis=0)
    z = z.sub(z.mean(axis=1), axis=0)
    beta = (x * z).sum(axis=1) / z.pow(2).sum(axis=1).where(lambda value: value.gt(0))
    return x.sub(z.mul(beta, axis=0)).where(common.sum(axis=1).ge(MIN_NAMES), axis=0)


# ---------------------------------------------------------------- 分区输入


def _symbol_state(symbol: str, dates: pd.DatetimeIndex, market: pd.DataFrame, partition: str):
    tables = member_tables(symbol, dates, top20=True, partition=partition)
    net, gross, sticky, _ = seat_turnover_groups(tables, market)
    return net, gross, sticky


def load_chain_inputs(panel: pd.DataFrame, jobs: int, *, partition: str = "research"
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从同一份席位持仓计算普通粘性净多和一致性净多，并返回主力合约映射。"""
    if partition == "research":
        start = START
        C.assert_no_holdout_dates(panel["date"], "跨品种一致性日面板")
    elif partition == "oos":
        start = pd.Timestamp(C.OOS_START)
        C.assert_oos_dates(panel.loc[panel["date"].ge(start), "date"], "跨品种一致性样本外日面板")
    else:
        raise ValueError(f"未知分区 {partition}")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))

    def wide(name: str) -> pd.DataFrame:
        return panel.pivot(index="date", columns="symbol", values=name)

    root = C.daily_dir(partition) / "member_rank"
    if partition == "research":
        C.assert_research_only(root)
    eligible = wide("eligible").eq(True)
    ranked = [symbol for symbol in eligible if (root / f"{symbol}_long.pkl").is_file()]
    dates = eligible.index
    volume, oi = wide("volume_total"), wide("oi_total")
    states = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_symbol_state)(symbol, dates, pd.DataFrame({"volume": volume[symbol], "oi": oi[symbol]}), partition)
        for symbol in ranked
    )
    net, gross, sticky = (
        {symbol: state[i] for symbol, state in zip(ranked, states)} for i in range(3)
    )

    sectors = panel.drop_duplicates("symbol").set_index("symbol")["sector"].loc[ranked]
    chain = {}
    for _, group in sectors.groupby(sectors):
        chain.update(chain_consistent_share(net, gross, sticky, oi, sorted(group.index)))
    symbols = [symbol for symbol in ranked if eligible.loc[start:, symbol].any()]
    eligible = eligible[symbols]
    valid = eligible & oi[symbols].gt(0) & volume[symbols].gt(0)
    net_share = pd.DataFrame({
        symbol: masked_net_share(net[symbol], gross[symbol], sticky[symbol]) for symbol in symbols
    }, index=dates)
    chain_share = pd.DataFrame({symbol: chain[symbol] for symbol in symbols}, index=dates)
    return (net_share.where(valid).loc[start:],
            chain_share.replace([np.inf, -np.inf], np.nan).where(eligible).loc[start:],
            wide("c1_contract")[symbols])


def run_chain_ic(out: Path, jobs: int) -> pd.DataFrame:
    """复现研究期 10 日残差 IC，报告 20 阶 Newey–West 双侧原始 p 值。"""
    sticky, chain, dominant = load_chain_inputs(D.load_panel(), jobs)
    bars = D.load_contracts("research")
    returns = return_matrix(dominant, bars.set_index(["contract", "date"])["open"])
    labels = forward_returns(returns, 10).reindex_like(sticky)
    ic, names = _rank_corr(residual_scores(chain, sticky), labels, MIN_NAMES)
    rows = []
    for phase, values in (("2016–2021", ic), ("2016–2018", ic.loc[:"2018"]), ("2019–2021", ic.loc["2019":])):
        t_stat = hac_t(values, 20)
        rows.append(dict(phase=phase, horizon=10, mean_ic=values.mean(), hac_t=t_stat,
                         p=erfc(abs(t_stat) / sqrt(2)), days=values.count(),
                         avg_names=names.reindex(values.index).mean()))
    result = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "chain_ic.csv", index=False)
    return result

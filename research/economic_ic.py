"""经济先验商品因子：期限结构、12-1 月动量与仓单变化的研究期 IC 检验。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import config as C
import data as D
from research.ic_study import (
    MIN_DAYS,
    _bh,
    _rank_corr,
    _stats,
    forward_returns,
    return_matrix,
)
from research.settings import XS_POSITION as XS

FACTORS = ("ROLL_YIELD", "MOM_12_1", "WAREHOUSE_DRAW_20")


def momentum_12_1(returns: pd.Series) -> pd.Series:
    """过去 242 个交易日复合收益，跳过最近 21 个交易日。"""
    gross = 1.0 + returns.replace([np.inf, -np.inf], np.nan)
    return gross.shift(21).rolling(242, min_periods=242).apply(np.prod, raw=True) - 1.0


def warehouse_draw_20(stock: pd.Series, open_interest: pd.Series) -> pd.Series:
    """滞后一个交易日的 20 日仓单下降，以同日市场持仓量标准化。"""
    lagged_stock = stock.shift(1)
    prior_stock = stock.shift(21)
    lagged_oi = open_interest.shift(1)
    return (prior_stock - lagged_stock).div(lagged_oi.where(lagged_oi.gt(0)))


def candidate_factors(panel: pd.DataFrame, warehouse_dir: Path | None = None) -> pd.DataFrame:
    """构造预先指定方向的研究期候选因子，不读取 OOS 数据。"""
    C.assert_no_holdout_dates(panel["date"], "经济因子研究日面板")
    required = {"date", "symbol", "ret1", "oi_total", "roll_yield_ann"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"经济因子日面板缺少字段: {sorted(missing)}")

    daily = panel.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.loc[daily["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    dates = pd.DatetimeIndex(daily["date"].unique()).sort_values()
    symbols = sorted(daily["symbol"].unique())
    roll_yield = daily.pivot(index="date", columns="symbol", values="roll_yield_ann")

    momentum = {}
    open_interest = {}
    for symbol, rows in daily.groupby("symbol", sort=False):
        indexed = rows.set_index("date").reindex(dates)
        momentum[symbol] = momentum_12_1(indexed["ret1"])
        open_interest[symbol] = indexed["oi_total"]
    momentum = pd.DataFrame(momentum).reindex(index=dates, columns=symbols)
    open_interest = pd.DataFrame(open_interest).reindex(index=dates, columns=symbols)

    warehouse_root = warehouse_dir or C.daily_dir("research") / "warehouse"
    C.assert_research_only(warehouse_root)
    inventory = {}
    for symbol in symbols:
        path = warehouse_root / f"{symbol}.pkl"
        if not path.is_file():
            continue
        C.assert_research_only(path)
        stock = pd.read_pickle(path)
        if isinstance(stock.index, pd.MultiIndex):
            if "underlying_symbol" not in stock.index.names:
                raise ValueError(f"{path.name} 仓单索引缺少 underlying_symbol")
            stock = stock.droplevel("underlying_symbol")
        if "on_warrant" not in stock:
            raise ValueError(f"{path.name} 仓单数据缺少 on_warrant")
        stock.index = pd.to_datetime(stock.index).normalize()
        C.assert_no_holdout_dates(stock.index, f"{path.name} 仓单")
        if stock.index.has_duplicates:
            raise ValueError(f"{path.name} 仓单日期重复")
        stock = stock["on_warrant"].reindex(dates).astype(float)
        inventory[symbol] = warehouse_draw_20(stock, open_interest[symbol])
    inventory = pd.DataFrame(inventory).reindex(index=dates, columns=symbols)

    return pd.concat(
        {
            "ROLL_YIELD": roll_yield.reindex(index=dates, columns=symbols),
            "MOM_12_1": momentum,
            "WAREHOUSE_DRAW_20": inventory,
        },
        axis=1,
    ).sort_index(axis=1)


def _time_series_summary(scores: dict[str, pd.DataFrame], outcomes: pd.DataFrame) -> list[dict]:
    rows = []
    for factor in FACTORS:
        score = scores[factor]
        correlations = []
        for symbol in score.columns:
            paired = pd.concat([score[symbol], outcomes[symbol]], axis=1).dropna()
            if len(paired) < MIN_DAYS or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
                continue
            correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
            if np.isfinite(correlation):
                correlations.append(float(correlation))
        count = len(correlations)
        positive = sum(value > 0 for value in correlations)
        rows.append({
            "study": "time_series",
            "factor": factor,
            "horizon": "d1",
            "mean_ic": float(np.mean(correlations)) if count else np.nan,
            "median_ic": float(np.median(correlations)) if count else np.nan,
            "p": float(binomtest(positive, count, 0.5, alternative="greater").pvalue) if count else np.nan,
            "n_symbols": count,
            "positive_symbols": positive,
        })
    adjusted = _bh(pd.Series({row["factor"]: row["p"] for row in rows}))
    for row in rows:
        row["q_ts"] = adjusted[row["factor"]]
    return rows


def run(panel: pd.DataFrame, out: Path) -> None:
    """研究期内的候选因子检验；主族为次日截面，其余规格另作全检验族。"""
    C.assert_no_holdout_dates(panel["date"], "经济因子研究日面板")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    sample = dates[dates >= pd.Timestamp(XS["test_start"])]
    symbols = sorted(panel.loc[panel["eligible"] & panel["date"].ge(XS["test_start"]), "symbol"].unique())
    if not symbols:
        raise RuntimeError("经济因子研究区间没有满足流动性条件的商品")

    factors = candidate_factors(panel).reindex(index=sample, columns=pd.MultiIndex.from_product([FACTORS, symbols]))
    scores = {
        factor: factors[factor].where(
            panel.pivot(index="date", columns="symbol", values="eligible").reindex(
                index=sample, columns=symbols
            ).fillna(False)
        )
        for factor in FACTORS
    }

    contract_map = panel.pivot(index="date", columns="symbol", values="c1_contract").reindex(
        index=dates, columns=symbols
    )
    bars = D.load_contracts("research")[["contract", "date", "open"]]
    returns = return_matrix(contract_map, bars.set_index(["contract", "date"])["open"])
    eligible = panel.pivot(index="date", columns="symbol", values="eligible").reindex(
        index=dates, columns=symbols
    ).fillna(False)
    outcomes = {
        "d1": returns.shift(-1).loc[sample].where(eligible.loc[sample]),
        "d5": forward_returns(returns, 5).loc[sample].where(eligible.loc[sample]),
    }

    rows = []
    for factor, score in scores.items():
        for horizon, outcome in outcomes.items():
            ic, count = _rank_corr(score, outcome, XS["min_names"])
            rows.append({
                "study": "cross_section",
                "factor": factor,
                "horizon": horizon,
                **_stats(ic, count),
            })
    cross = pd.DataFrame(rows)
    primary = cross["horizon"].eq("d1")
    cross["q_primary"] = np.nan
    cross.loc[primary, "q_primary"] = _bh(cross.loc[primary, "p"]).to_numpy()
    cross["q_full"] = _bh(cross["p"])

    time_series = _time_series_summary(scores, outcomes["d1"])
    result = pd.concat([cross, pd.DataFrame(time_series)], ignore_index=True, sort=False)
    result = result.sort_values(["study", "factor", "horizon"], kind="mergesort")
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "candidate_ic.csv", index=False)
    print(result.round(5).to_string(index=False))
    print(f"候选因子检验结果写入 {out / 'candidate_ic.csv'}")

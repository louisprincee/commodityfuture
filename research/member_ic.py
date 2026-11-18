"""逐品种研报会员指标与下一交易日开盘至再下一交易日开盘的秩 IC。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research import paper_factors as F
from research.settings import XS_POSITION as XS

LABEL_FIELDS = ("entry_date", "exit_date", "entry_contract", "entry_open", "exit_open", "forward_return")


def forward_open_returns(dominant: pd.Series, opens: pd.Series) -> pd.DataFrame:
    """全市场交易日索引上，入场与退出都使用 t+1 的主力合约。"""
    dates = dominant.index.to_series()
    entry_day = dates.shift(-1)
    exit_day = dates.shift(-2)
    contract = dominant.shift(-1)
    entry = opens.reindex(pd.MultiIndex.from_arrays([contract, entry_day])).to_numpy(dtype=float)
    exit_price = opens.reindex(pd.MultiIndex.from_arrays([contract, exit_day])).to_numpy(dtype=float)
    valid = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_price) & (exit_price > 0)
    returns = np.divide(exit_price, entry, out=np.full(len(dates), np.nan), where=valid) - 1
    return pd.DataFrame({
        "entry_date": entry_day, "exit_date": exit_day, "entry_contract": contract,
        "entry_open": entry, "exit_open": exit_price, "forward_return": returns,
    }, index=dominant.index)


def _direction(factor):
    return -1 if factor in {"CROWD1", "CROWD2", "CROWD3", "K_DS"} or factor.startswith("C_SHORT_") else 1


def summarize_symbol(symbol, daily, factors):
    """逐因子配对，缺失不填零，离散的 0 信号保留。"""
    rows = []
    for name in factors:
        values = daily[name]
        outcome = daily["forward_return"]
        paired = np.isfinite(values) & np.isfinite(outcome)
        score, returns = values.loc[paired], outcome.loc[paired]
        n = len(score)
        coefficient = (score.rank().corr(returns.rank())
                       if n >= 3 and score.nunique() > 1 and returns.nunique() > 1 else np.nan)
        direction = _direction(name)
        rows.append({
            "symbol": symbol, "factor": name, "ic": coefficient, "direction": direction,
            "signed_ic": direction * coefficient, "n_pairs": n, "n_nonzero": int(score.ne(0).sum()),
            "n_factor_days": int(np.isfinite(values).sum()), "n_return_days": int(np.isfinite(outcome).sum()),
        })
    return pd.DataFrame(rows)


def _one_symbol(symbol, panel, bars, calendar, start):
    ranks = D.load_member_rank_contracts(symbol, "research", D.MEMBER_SIDES)
    if not ranks.empty:
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    holds = ranks.loc[ranks.side.ne("volume")] if not ranks.empty else ranks
    features = F.symbol_factors(symbol, panel, bars, calendar, holds, include_experiments=False)
    features["SPIDER_ACTIVE"] = F.kaiyuan_active(ranks, calendar)["SPIDER_ACTIVE"]
    dominant = panel.set_index("date")["c1_contract"].reindex(calendar)
    labels = forward_open_returns(dominant, bars.set_index(["contract", "date"])["open"])
    sample = panel.loc[panel.date.ge(start), "date"]
    daily = labels.join(features).loc[sample].rename_axis("signal_date")
    daily.insert(0, "symbol", symbol)
    return daily.reset_index(), summarize_symbol(symbol, daily, features.columns)


def run(panel: pd.DataFrame, out: Path, jobs: int):
    C.assert_no_holdout_dates(panel.date, "逐品种 IC 日面板")
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    calendar = pd.DatetimeIndex(panel.date.unique()).sort_values()
    start = pd.Timestamp(XS["test_start"])
    symbols = sorted(panel.loc[panel.date.ge(start), "symbol"].unique())
    if not symbols:
        raise RuntimeError("评价区间没有商品期货")
    prices = D.load_contracts("research")[["date", "contract", "symbol", "open", "open_interest"]]
    panels = {name: part for name, part in panel.groupby("symbol")}
    bars = {name: part for name, part in prices.groupby("symbol")}
    pieces = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_one_symbol)(symbol, panels[symbol], bars[symbol], calendar, start)
        for symbol in symbols
    )
    daily = pd.concat([rows for rows, _ in pieces], ignore_index=True)
    summary = pd.concat([rows for _, rows in pieces], ignore_index=True)
    factors = [name for name in daily.columns if name not in {"signal_date", "symbol", *LABEL_FIELDS}]
    matrix = summary.pivot(index="symbol", columns="factor", values="ic").reindex(columns=factors)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily_factors_and_returns.csv", index=False)
    summary.to_csv(out / "ic_by_symbol.csv", index=False)
    matrix.to_csv(out / "ic_matrix.csv")
    print(f"{len(symbols)} 个品种，{len(factors)} 个指标与信号，{len(daily)} 个品种日")
    print(f"可估 IC {summary.ic.notna().sum()} / {len(summary)}，结果写入 {out}")

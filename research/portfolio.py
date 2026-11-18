"""截面得分 → 多空权重 → 日收益、换手与绩效指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from research.settings import UNIVERSE

COST_BPS = 5.0
TD = C.TRADING_DAYS_PER_YEAR


def eligibility(panel: pd.DataFrame) -> pd.Series:
    """时点品种池：过去 242 日成交额中位数、上市天数、当日主力有成交。收盘时当日成交额已知。"""
    grouped = panel.groupby("symbol", sort=False)
    median = grouped["turnover_total"].transform(
        lambda s: s.rolling(
            UNIVERSE["turnover_window_days"], min_periods=UNIVERSE["turnover_min_periods"]
        ).median()
    )
    age = grouped.cumcount() + 1
    ok = (median >= UNIVERSE["min_median_daily_turnover_cny"]) & (age >= UNIVERSE["min_history_days"])
    if UNIVERSE["require_dominant_volume_positive"]:
        ok &= panel["c1_volume"] > 0
    ok &= panel["ret1"].notna()
    ok &= ~panel["symbol"].isin(UNIVERSE["excluded_symbols"])
    return ok.fillna(False)


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


def score_to_weights(score: pd.DataFrame) -> pd.DataFrame:
    """score: 日期 × 品种（缺失 = 不在池内）。中位数以上做多、以下做空，腿内按 |得分-中位| 加权，每腿总敞口 1。"""
    q = score.rank(axis=1, pct=True)
    s = q.sub(q.mean(axis=1), axis=0)
    long = s.clip(lower=0)
    short = s.clip(upper=0)
    w = long.div(long.sum(axis=1).replace(0, np.nan), axis=0) \
        + short.div(short.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return w.fillna(0.0)


def hold_cohorts(weights: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """重叠持有：t 日持仓为最近 H 个日度组合之和 / H；首个预测日之前的组合视为现金。"""
    return weights.fillna(0.0).rolling(horizon, min_periods=1).sum() / horizon


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

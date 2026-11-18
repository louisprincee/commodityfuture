"""截面会员持仓因子。

方向在看到收益之前按两篇报告固定：商品拥挤度反转，一致性合力、净多头占比变化和
会员持仓流顺势。t 日收盘信号于 t+1 日开盘成交，赚取主力合约开盘到下一开盘的收益；
延后一日的结果一并报告。不在研究期收益上挑选因子、窗口或符号。
5 日重叠持有是第一设定看到日频换手之后追加的第二设定，持有期本身没有再搜索。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as D
from research import member_study as M
from research import portfolio as P
from research.settings import CROWDING, XS_POSITION as XS


def past_pct(series: pd.Series, window: int) -> pd.Series:
    """当前值在此前 window 个观测中的分位。窗口不含当日，也不使用未来值。"""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if window <= 0 or len(values) <= window:
        return pd.Series(out, index=series.index)
    from numpy.lib.stride_tricks import sliding_window_view

    history = sliding_window_view(values, window)[:-1]
    current = values[window:]
    finite = np.isfinite(history)
    complete = finite.all(axis=1) & np.isfinite(current)
    below = ((history <= current[:, None]) & finite).sum(axis=1)
    out[window:] = np.where(complete, below / window, np.nan)
    return pd.Series(out, index=series.index)


def past_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """相对该品种自身过去的均值和标准差。均值与标准差不含当日，也不使用未来值。"""
    if window <= 1 or min_periods <= 1:
        raise ValueError("标准化窗口至少为 2")
    past = series.shift(1)
    mean = past.rolling(window, min_periods=min_periods).mean()
    scale = past.rolling(window, min_periods=min_periods).std()
    return (series - mean) / scale.where(scale > 0)


def tail_weights(score: pd.DataFrame, tail: float, min_names: int) -> tuple[pd.DataFrame, pd.Series]:
    """只做截面两端。最高 tail 等权做多，最低 tail 等权做空，中间权重为 0。"""
    if not 0 < tail < 0.5:
        raise ValueError("tail 必须在 (0, 0.5)")
    rank = score.rank(axis=1, pct=True)
    long = rank.ge(1.0 - tail)
    short = rank.le(tail)
    count = score.notna().sum(axis=1)
    long_n = long.sum(axis=1).replace(0, np.nan)
    short_n = short.sum(axis=1).replace(0, np.nan)
    weights = long.astype(float).div(long_n, axis=0).fillna(0.0)
    weights = weights - short.astype(float).div(short_n, axis=0).fillna(0.0)
    return weights.where(count >= min_names, 0.0), count


def band_weights(score: pd.DataFrame, band: float, min_names: int) -> tuple[pd.DataFrame, pd.Series]:
    """分位要越过中位数两侧的带宽才改变方向。带宽内维持原方向，同侧等权。"""
    if not 0 < band < 0.5:
        raise ValueError("rank_band 必须在 (0, 0.5)")
    rank = score.rank(axis=1, pct=True).to_numpy(dtype=float)
    state = np.zeros(rank.shape, dtype=float)
    previous = np.zeros(rank.shape[1], dtype=float)
    high, low = 0.5 + band, 0.5 - band
    for i in range(rank.shape[0]):
        row = rank[i]
        valid = np.isfinite(row)
        current = previous.copy()
        current[~valid] = 0.0
        current[valid & (row >= high)] = 1.0
        current[valid & (row <= low)] = -1.0
        state[i] = current
        previous = current
    side = pd.DataFrame(state, index=score.index, columns=score.columns)
    long = side.eq(1.0)
    short = side.eq(-1.0)
    count = score.notna().sum(axis=1)
    weights = long.astype(float).div(long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    weights = weights - short.astype(float).div(short.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return weights.where(count >= min_names, 0.0), count


def signal_book(signal: pd.DataFrame, rule: str, min_names: int) -> pd.DataFrame:
    """原文离散信号的等权组合。信号尚未可计算的品种不进入分母；0 为空仓并稀释权重。"""
    signal = signal.sort_index()
    if rule == "reset":
        position = signal.fillna(0.0)
    elif rule == "hold":
        position = signal.replace(0.0, np.nan).ffill().fillna(0.0)
    else:
        raise ValueError(rule)
    live = signal.notna().astype(float).cummax().astype(bool)
    count = live.sum(axis=1)
    weights = position.where(live, 0.0).div(count.replace(0, np.nan), axis=0).fillna(0.0)
    return weights.where(count >= min_names, 0.0)


def _symbol_frame(symbol: str, panel: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """单品种连续因子与原文离散信号。全部由当日及以前的会员排名和行情计算。"""
    market = M.market_inputs(panel, contracts, symbol)
    if market.empty:
        return pd.DataFrame()
    tables = M.member_tables(symbol, market.index)
    if not bool(tables["ranked"].fillna(False).any()):
        return pd.DataFrame()
    crowd = M.crowding_indicators(tables, market)
    out = crowd.copy()
    for window in XS["consistency_windows"]:
        out[f"POWER_{window}"] = M.consistency_indicators(tables, window)["POWER"]
    oi = market["oi"].where(market["oi"] > 0)
    net = (tables["L"].sum(axis=1, min_count=1) - tables["S"].sum(axis=1, min_count=1)) / oi
    net = net.where(tables["ranked"])
    for window in XS["net_windows"]:
        out[f"NET_CHG_{window}"] = net - net.shift(window)
    flow = tables["dL"].sum(axis=1, min_count=1) - tables["dS"].sum(axis=1, min_count=1)
    active_long = tables["dL"].where(tables["V"].notna())
    active_short = tables["dS"].where(tables["V"].notna())
    active = active_long.sum(axis=1, min_count=1) - active_short.sum(axis=1, min_count=1)
    out["FLOW"] = (flow / oi).where(tables["ranked"])
    out["FLOW_ACTIVE"] = (active / oi).where(tables["ranked"])
    out["MOM20"] = np.log1p(market["ret_cc"]).rolling(XS["mom_window"], min_periods=XS["mom_window"]).sum()
    out["ret_oo"] = market["ret_oo"]
    out["roll"] = market["roll"].astype(bool)
    out["ranked"] = tables["ranked"].fillna(False).to_numpy()
    out["SIG_CROWD"] = M.crowding_signal(crowd["CROWD1"], market["ret_cc"], CROWDING["reference_window"])
    out["SIG_SPIDER"] = M.spider_signal(tables, active_only=True)
    out["symbol"] = symbol
    out.index.name = "date"
    return out.reset_index()


def _pivot(frame: pd.DataFrame, column: str, mask: pd.Series | None = None) -> pd.DataFrame:
    use = frame if mask is None else frame.loc[mask]
    if use.empty:
        return pd.DataFrame()
    wide = use.pivot(index="date", columns="symbol", values=column)
    return wide.sort_index()


def _rank_weights(score: pd.DataFrame, min_names: int) -> tuple[pd.DataFrame, pd.Series]:
    count = score.notna().sum(axis=1)
    weights = P.score_to_weights(score)
    return weights.where(count >= min_names, 0.0), count


def _align_weights(weights: pd.DataFrame, returns: pd.DataFrame, delay: int) -> pd.DataFrame:
    """收益缺失的那一天不持仓。权重日期 t 对应收益日期 t+1+delay。"""
    weights = weights.reindex(index=returns.index, columns=returns.columns)
    earned = returns.shift(-(1 + delay))
    return weights.where(earned.notna(), 0.0).fillna(0.0)


def _composite(parts: list[pd.DataFrame], columns: pd.Index, min_parts: int) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    index = parts[0].index
    mats = [part.reindex(index=index, columns=columns) for part in parts]
    count = sum(mat.notna().astype(int) for mat in mats)
    total = sum(mat.fillna(0.0) for mat in mats)
    average = total.div(count.where(count > 0))
    return average.where(count >= min_parts)


def _metrics_row(name: str, family: str, norm: str, delay: int, bt: pd.DataFrame,
                 names: pd.Series) -> dict:
    net = P.metrics(bt["net"], bt["turnover"])
    gross = P.metrics(bt["gross"])
    drawdown = net["max_drawdown"]
    return {
        "book": name, "family": family, "norm": norm, "delay": delay,
        "ann_return": net["ann_return"], "ann_vol": net["ann_vol"], "sharpe": net["sharpe"],
        "gross_sharpe": gross["sharpe"], "max_drawdown": drawdown,
        "calmar": net["ann_return"] / -drawdown if drawdown < 0 else np.nan,
        "success_rate": net["success_rate"], "t_stat": net["t_stat"],
        "daily_turnover": net.get("daily_turnover", np.nan),
        "avg_gmv": float(bt["gmv"].mean()), "avg_names": float(names.reindex(bt.index).mean()),
        "days": net["days"],
    }


def _rank_ic(score: pd.DataFrame, forward: pd.DataFrame, min_names: int) -> pd.Series:
    columns = score.columns.intersection(forward.columns)
    left = score[columns]
    right = forward[columns].reindex(left.index)
    values = {}
    for day, row in left.iterrows():
        outcome = right.loc[day]
        keep = row.notna() & outcome.notna()
        if int(keep.sum()) < min_names:
            continue
        values[day] = float(row[keep].corr(outcome[keep], method="spearman"))
    return pd.Series(values, dtype=float)


def _figure(path: Path, curves: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for column in curves.columns:
        ax.plot(curves.index, (1 + curves[column].fillna(0)).cumprod(), linewidth=1.4, label=column)
    ax.set_ylabel("net value")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Cross-sectional member-position factors, 2016–2021", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(panel: pd.DataFrame, fee_path: Path, out: Path, jobs: int = 8) -> pd.DataFrame:
    from joblib import Parallel, delayed

    out.mkdir(parents=True, exist_ok=True)
    C.assert_no_holdout_dates(panel["date"], "截面研究日面板")
    C.assert_research_only(fee_path)
    fees = pd.read_csv(fee_path, parse_dates=["trading_date"])
    C.assert_no_holdout_dates(fees["trading_date"], "截面研究手续费")
    panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])
    eligible = P.eligibility(panel)
    rank_root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(rank_root)
    symbols = sorted(
        s for s in panel.loc[eligible, "symbol"].unique()
        if (rank_root / f"{s}_long.pkl").exists()
    )
    if len(symbols) < XS["min_names"]:
        raise RuntimeError(f"有会员排名的流动性品种只有 {len(symbols)} 个")

    contracts = D.load_contracts("research")
    started = time.time()
    panel_slim = panel.loc[panel["symbol"].isin(symbols), ["date", "symbol", "c1_contract", "ret1", "volume_total", "oi_total"]]
    contract_slim = contracts.loc[contracts["symbol"].isin(symbols), ["symbol", "contract", "date", "open"]]
    panels = dict(tuple(panel_slim.groupby("symbol")))
    bars = dict(tuple(contract_slim.groupby("symbol")))
    frames = Parallel(n_jobs=jobs, max_nbytes=None)(
        delayed(_symbol_frame)(symbol, panels[symbol], bars[symbol]) for symbol in symbols
    )
    long = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if long.empty:
        raise RuntimeError("会员因子为空")
    long["date"] = pd.to_datetime(long["date"])
    C.assert_no_holdout_dates(long["date"], "会员因子")
    long = long.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    for name in XS["signs"]:
        if name not in long.columns:
            raise RuntimeError(f"缺少因子 {name}")
        grouped = long.groupby("symbol", sort=False)[name]
        long[f"{name}_TS"] = grouped.transform(lambda s: past_pct(s, XS["ts_window"]))
        long[f"{name}_Z"] = grouped.transform(
            lambda s: past_zscore(s, XS["zscore_window"], XS["zscore_min_periods"])
        )
    flag = panel[["date", "symbol"]].copy()
    flag["eligible"] = eligible.to_numpy()
    long = long.merge(flag, on=["date", "symbol"], how="left")
    long["eligible"] = long["eligible"].fillna(False).astype(bool)
    long["ranked"] = long["ranked"].fillna(False).astype(bool)
    tradable = long["eligible"] & long["ranked"]

    returns = _pivot(long, "ret_oo")
    rolls = _pivot(long, "roll").reindex_like(returns).astype("boolean").fillna(False).astype(bool)
    cost = P.cost_rate_matrix(panel, fees, XS["slippage_ticks"]).reindex(index=returns.index, columns=returns.columns)
    test_start = pd.Timestamp(XS["test_start"])
    calendar = returns.index[(returns.index >= test_start) & (returns.index <= pd.Timestamp(C.RESEARCH_END))]
    if len(calendar) == 0:
        raise RuntimeError("研究期交易日为空")

    scores: dict[str, pd.DataFrame] = {}
    books: dict[str, tuple[pd.DataFrame, pd.Series, str, str]] = {}

    def add_rank_book(name: str, family: str, norm: str, score: pd.DataFrame, min_names: int) -> None:
        weights, count = _rank_weights(score, min_names)
        scores[name] = score
        books[name] = (weights, count, family, norm)

    def _add_standardized(name: str, family: str, score: pd.DataFrame, min_count: int) -> None:
        scores[f"{name}_Z"] = score
        weights, count = _rank_weights(score, min_count)
        books[f"{name}_Z"] = (weights, count, family, "z")
        weights, count = tail_weights(score, XS["tail"], min_count)
        books[f"{name}_Z_TAIL"] = (weights, count, family, "z_tail")
        weights, count = band_weights(score, XS["rank_band"], min_count)
        books[f"{name}_Z_BAND"] = (weights, count, family, "z_band")

    min_names = XS["min_names"]
    for name, sign in XS["signs"].items():
        if sign not in (-1, 1):
            raise ValueError(f"{name} 的符号必须为 ±1")
        family = "momentum_baseline" if name == "MOM20" else name.split("_")[0].lower()
        if name.startswith("CROWD"):
            family = "crowding"
        elif name.startswith("POWER"):
            family = "consistency"
        elif name.startswith("NET_CHG"):
            family = "net_share"
        elif name.startswith("FLOW"):
            family = "spider_flow"
        raw = pd.Series(sign * long[name], index=long.index)
        ts = pd.Series(sign * long[f"{name}_TS"], index=long.index)
        raw_score = _pivot(long.assign(_score=raw), "_score", tradable & raw.notna())
        ts_score = _pivot(long.assign(_score=ts), "_score", tradable & ts.notna())
        add_rank_book(name, family, "raw", raw_score, min_names)
        add_rank_book(f"{name}_TS", family, "ts", ts_score, min_names)
        z = pd.Series(sign * long[f"{name}_Z"], index=long.index)
        z_score = _pivot(long.assign(_score=z), "_score", tradable & z.notna())
        _add_standardized(name, family, z_score, min_names)

    member_names = [name for name in XS["signs"] if name != "MOM20"]
    for label, names, least in (
        ("COMPOSITE_CORE", XS["core"], XS["composite_min"]["core"]),
        ("COMPOSITE_ALL", member_names, XS["composite_min"]["all"]),
    ):
        parts = [scores[f"{name}_TS"] for name in names]
        combined = _composite(parts, returns.columns, least)
        add_rank_book(label, "composite", "ts", combined, min_names)
        z_parts = [scores[f"{name}_Z"] for name in names]
        _add_standardized(label, "composite", _composite(z_parts, returns.columns, least), min_names)

    eligible_wide = (
        _pivot(long, "eligible").reindex(index=returns.index, columns=returns.columns)
        .astype("boolean").fillna(False).astype(bool)
    )
    for column, family in (("SIG_CROWD", "paper_crowding"), ("SIG_SPIDER", "paper_spider")):
        signal = _pivot(long, column).where(eligible_wide)
        for rule in ("reset", "hold") if column == "SIG_CROWD" else ("reset",):
            book_name = f"PAPER_{'CROWD' if column == 'SIG_CROWD' else 'SPIDER'}_{rule.upper()}"
            weights = signal_book(signal, rule, min_names)
            count = signal.notna().astype(float).cummax().sum(axis=1)
            books[book_name] = (weights, count, family, "signal")
            paper_columns = [s for s in XS["paper_symbols"] if s in signal.columns]
            if paper_columns:
                books[book_name + "_8"] = (
                    signal_book(signal[paper_columns], rule, XS["min_names_paper"]),
                    signal[paper_columns].notna().astype(float).cummax().sum(axis=1),
                    family, "signal8",
                )

    long_count = eligible_wide.sum(axis=1)
    long_only = eligible_wide.astype(float).div(long_count.replace(0, np.nan), axis=0).fillna(0.0)
    books["LONG_ONLY"] = (long_only.where(long_count >= min_names, 0.0), long_count, "benchmark", "long")

    # 5 日只作用于原始截面和时序分位，与此前结果保持同一规则。
    # 10 日和 60 日再覆盖 z-score 全截面；两端和带宽不与持有期交叉。
    horizons = [(int(XS["later_hold_days"]), ("raw", "ts"))]
    horizons += [(int(h), ("raw", "ts", "z")) for h in XS["hold_horizons"]]
    slowed = {}
    for hold_days, norms in horizons:
        if hold_days <= 1:
            continue
        for name, (weights, count, family, norm) in books.items():
            if norm not in norms:
                continue
            aligned = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
            slowed[f"{name}_H{hold_days}"] = (
                P.hold_cohorts(aligned, hold_days), count, family, f"{norm}_hold{hold_days}",
            )
    books.update(slowed)

    rows = []
    daily = {}
    annual = {}
    for name, (weights, count, family, norm) in books.items():
        for delay in (0, 1):
            held = _align_weights(weights, returns, delay)
            earned_cost = cost.shift(-(1 + delay))
            # 研究期之前不建仓，避免用缺失的早期手续费成交；成本未知的交易日也不持仓。
            held = held.where(earned_cost.notna(), 0.0)
            held.loc[held.index < test_start] = 0.0
            result = P.backtest(held, returns.fillna(0.0), rolls, delay=delay, cost_bps=cost)
            result = result.reindex(calendar)
            for column in ("gross", "net", "cost", "turnover", "gmv", "nmv"):
                result[column] = result[column].fillna(0.0)
            rows.append(_metrics_row(name, family, norm, delay, result, count))
            daily[f"{name}_d{delay}"] = result["net"]
            if delay == 0:
                annual[name] = result["net"].groupby(result.index.year).sum()
    performance = pd.DataFrame(rows)
    performance.to_csv(out / "performance.csv", index=False)
    pd.DataFrame(daily).to_csv(out / "daily_net.csv")
    pd.DataFrame(annual).to_csv(out / "annual_net.csv")

    forward = returns.shift(-1)
    ic_rows = []
    ic_books = [(name, "raw") for name in XS["signs"]]
    ic_books += [(f"{name}_TS", "ts") for name in XS["signs"]]
    ic_books += [(f"{name}_Z", "z") for name in XS["signs"]]
    ic_books += [("COMPOSITE_CORE", "ts"), ("COMPOSITE_ALL", "ts")]
    ic_books += [("COMPOSITE_CORE_Z", "z"), ("COMPOSITE_ALL_Z", "z")]
    for score_name, norm in ic_books:
        if score_name not in scores:
            continue
        ic = _rank_ic(scores[score_name].reindex(calendar), forward.reindex(calendar), min_names)
        ic_rows.append({
            "book": score_name, "norm": norm, "mean_ic": float(ic.mean()) if len(ic) else np.nan,
            "ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else np.nan,
            "hit_rate": float((ic > 0).mean()) if len(ic) else np.nan, "days": int(len(ic)),
        })
    ic_table = pd.DataFrame(ic_rows)
    ic_table.to_csv(out / "rank_ic.csv", index=False)

    show = ["FLOW", "FLOW_H5", "FLOW_H10", "FLOW_H60", "NET_CHG_60_H60", "LONG_ONLY"]
    curves = pd.DataFrame({name: daily[f"{name}_d0"] for name in show if f"{name}_d0" in daily})
    _figure(out / "figures" / "nav.png", curves)

    summary = {
        "symbols": symbols,
        "n_symbols": len(symbols),
        "test_start": XS["test_start"],
        "signs": XS["signs"],
        "core": XS["core"],
        "slippage_ticks": XS["slippage_ticks"],
        "later_hold_days": XS["later_hold_days"],
        "hold_horizons": XS["hold_horizons"],
        "zscore_window": XS["zscore_window"],
        "zscore_min_periods": XS["zscore_min_periods"],
        "tail": XS["tail"],
        "rank_band": XS["rank_band"],
        "seconds": round(time.time() - started),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    view = performance[performance.delay == 0].set_index("book")[
        ["sharpe", "gross_sharpe", "ann_return", "ann_vol", "max_drawdown", "daily_turnover", "t_stat", "avg_names"]
    ]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 240)
    print(view.round(3).to_string())
    print(f"品种 {len(symbols)}，用时 {summary['seconds']}s → {out}")
    return performance

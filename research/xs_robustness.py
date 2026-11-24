"""截面因子稳健性检验（研究期 2016–2021，不读取封存分区）。

本检验的因子、方向、窗口、预测期、状态划分与通过标准在计算任何结果之前写定如下，
结果出来后不改：

因子（方向）
  经济族
    ROLL_YIELD       +  米筐主力/次主力年化展期收益，当日（Koijen et al. 2018；Szymanowska et al. 2014）
    MOM_12_1         +  过去 242 日复合收益，跳过最近 21 日（Asness, Moskowitz & Pedersen 2013）
    BASIS_MOM_242    +  过去 242 日主力复合收益减次主力复合收益，各用同合约日收益（Boons & Prado 2019）
    SKEW_242         -  过去 242 日（至少 200 日）主力日收益偏度，低偏度做多（Fernandez-Perez et al. 2018）
  会员持仓族（开源/兴业原文方向，不改号）
    K_DB_OI          +  前 20 名多头持仓变化 / 当日商品总持仓
    NEG_K_DS_OI      +  −前 20 名空头持仓变化 / 当日商品总持仓
    FLOW_CONTROL     +  (ΣΔL − ΣΔS) / 当日商品总持仓
    K_NET_SHARE      +  前 20 名净多头 / 披露合约持仓

标签：信号日 t 之后连续 h 段主力开盘到开盘收益连乘，h ∈ {5, 10, 20}。
截面 IC：每日 Spearman，至少 8 个品种；HAC 滞后 max(20, 2h)。
主族：8 个因子 × 3 个预测期 = 24 项，一次 BH。

稳健性（每项都按方向调整后的 IC 判断，h = 10 为判定口径）
  R1  主族 BH q < 0.05，三个预测期中至少一个
  R2  趋势状态：t 时已知的过去 60 日流动性池等权收益 > 0 与 ≤ 0 两组，IC 均值同为正
  R3  波动状态：过去 60 日等权收益波动率高于 / 不高于其截至 t 的扩展中位数，两组 IC 均值同为正
  R4  2016–2021 六个年度中至少 5 个年度 IC 均值为正
  R5  多头腿与空头腿都赚钱：前 20% 等权收益减池均值 > 0，且池均值减后 20% 等权收益 > 0
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as D
from research import factors as F
from research.economic_ic import candidate_factors
from research.ic_study import _bh, _rank_corr, forward_returns, return_matrix

HORIZONS = (5, 10, 20)
JUDGE_H = 10
MIN_NAMES = 8
TAIL = 0.20
STATE_WINDOW = 60
START = pd.Timestamp("2016-01-04")
SIGNS = {
    "ROLL_YIELD": 1, "MOM_12_1": 1, "BASIS_MOM_242": 1, "SKEW_242": -1,
    "K_DB_OI": 1, "NEG_K_DS_OI": 1, "FLOW_CONTROL": 1, "K_NET_SHARE": 1,
}
FAMILY = {name: ("economic" if i < 4 else "member") for i, name in enumerate(SIGNS)}


def basis_momentum(ret1: pd.DataFrame, ret2: pd.DataFrame, window: int = 242) -> pd.DataFrame:
    """主力与次主力各自同合约日收益的 window 日复合收益之差；任一侧不满窗为缺失。"""
    def compound(r: pd.DataFrame) -> pd.DataFrame:
        return np.expm1(np.log1p(r).rolling(window, min_periods=window).sum())
    return compound(ret1) - compound(ret2)


def rolling_skew(ret1: pd.DataFrame, window: int = 242, min_periods: int = 200) -> pd.DataFrame:
    return ret1.rolling(window, min_periods=min_periods).skew()


def load_price_factors(index: pd.Index, columns: pd.Index, *, partition: str = "research",
                       panel: pd.DataFrame | None = None):
    """在完整预热日历上计算固定的基差动量和负偏度，再对齐到评价品种日。"""
    if partition == "research":
        panel = D.load_panel() if panel is None else panel
        C.assert_no_holdout_dates(panel["date"], "价格因子研究期日面板")
    elif partition == "oos":
        C.assert_oos_dates(index, "价格因子样本外日历")
        panel = D.load_evaluation_panel() if panel is None else panel
    else:
        raise ValueError(f"未知分区 {partition}")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)]
    ret1 = panel.pivot(index="date", columns="symbol", values="ret1")
    ret2 = panel.pivot(index="date", columns="symbol", values="ret2")
    basis = basis_momentum(ret1, ret2).reindex(index=index, columns=columns)
    neg_skew = -rolling_skew(ret1).reindex(index=index, columns=columns)
    return basis, neg_skew


def hac_t(series: pd.Series, lags: int) -> float:
    x = series.dropna()
    n = len(x)
    if n < 120:
        return np.nan
    resid = (series - x.mean()).fillna(0).to_numpy()
    var = resid @ resid
    for lag in range(1, lags + 1):
        var += 2 * (1 - lag / (lags + 1)) * (resid[lag:] @ resid[:-lag])
    return x.mean() * n / math.sqrt(var) if var > 0 else np.nan


def leg_returns(score: pd.DataFrame, label: pd.DataFrame, tail: float = TAIL) -> pd.DataFrame:
    """每日前、后 tail 的等权标签收益与池均值；只用因子和标签都有效的品种。"""
    paired = score.notna() & label.notna()
    s, y = score.where(paired), label.where(paired)
    n = paired.sum(axis=1)
    k = np.ceil(n * tail)
    rank_hi = s.rank(axis=1, ascending=False, method="first")
    rank_lo = s.rank(axis=1, ascending=True, method="first")
    top = y.where(rank_hi.le(k, axis=0)).mean(axis=1)
    bottom = y.where(rank_lo.le(k, axis=0)).mean(axis=1)
    pool = y.mean(axis=1)
    ok = n.ge(MIN_NAMES)
    return pd.DataFrame({"top": top, "bottom": bottom, "pool": pool}).where(ok, axis=0)


def membership_turnover(score: pd.DataFrame, tail: float = TAIL) -> float:
    """前、后 tail 名单的日均替换比例（两端平均）。"""
    n = score.notna().sum(axis=1)
    k = np.ceil(n * tail)
    hi = score.rank(axis=1, ascending=False, method="first").le(k, axis=0)
    lo = score.rank(axis=1, ascending=True, method="first").le(k, axis=0)
    out = []
    for book in (hi, lo):
        stay = (book & book.shift(1, fill_value=False)).sum(axis=1)
        size = book.sum(axis=1)
        frac = 1 - stay / size.where(size.gt(0))
        out.append(frac.where(n.ge(MIN_NAMES) & n.shift(1).ge(MIN_NAMES)).mean())
    return float(np.nanmean(out))


def _member_factors(path: Path, oi: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["signal_date", "symbol", "K_DB", "K_DS", "FLOW_CONTROL", "K_NET_SHARE"]
    daily = pd.read_csv(path, usecols=cols, parse_dates=["signal_date"])
    C.assert_no_holdout_dates(daily["signal_date"], "会员因子文件")
    wide = {c: daily.pivot(index="signal_date", columns="symbol", values=c) for c in cols[2:]}
    denom = oi.reindex_like(wide["K_DB"])
    denom = denom.where(denom.gt(0))
    return {
        "K_DB_OI": wide["K_DB"] / denom,
        "NEG_K_DS_OI": -wide["K_DS"] / denom,
        "FLOW_CONTROL": wide["FLOW_CONTROL"],
        "K_NET_SHARE": wide["K_NET_SHARE"],
    }


def _evaluate_family(scores: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame],
                     states: pd.DataFrame, family_map: dict[str, str],
                     sign_map: dict[str, int], q_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """按给定因子集计算 5/10/20 日截面 IC、BH 校正及 R1–R5 稳健性表。"""
    rows, yearly, legs_yearly, daily_ic = [], [], [], {}
    for name, score in scores.items():
        turnover = membership_turnover(score)
        for h in HORIZONS:
            ic, count = _rank_corr(score, labels[h], MIN_NAMES)
            lags = max(20, 2 * h)
            t = hac_t(ic, lags)
            legs = leg_returns(score, labels[h])
            long_x, short_x = legs["top"] - legs["pool"], legs["pool"] - legs["bottom"]
            row = dict(
                factor=name, family=family_map[name], sign=sign_map[name], horizon=h,
                mean_ic=ic.mean(), hac_t=t, p=math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan,
                days=int(ic.count()), avg_names=count.mean(), ic_ir=ic.mean() / ic.std(),
                ic_2016_2018=ic.loc[:"2018"].mean(), ic_2019_2021=ic.loc["2019":].mean(),
                long_excess_bp=1e4 * long_x.mean(), long_t=hac_t(long_x, lags),
                short_excess_bp=1e4 * short_x.mean(), short_t=hac_t(short_x, lags),
                spread_bp=1e4 * (legs["top"] - legs["bottom"]).mean(),
                tail_turnover=turnover,
            )
            for col in states:
                flag = states[col]
                row[f"ic_{col}"] = ic.where(flag.eq(True)).mean()
                row[f"ic_not_{col}"] = ic.where(flag.eq(False)).mean()
            rows.append(row)
            by_year = ic.groupby(ic.index.year).mean()
            yearly.extend(dict(factor=name, horizon=h, year=int(y), mean_ic=v) for y, v in by_year.items())
            ly = pd.DataFrame({"long": long_x, "short": short_x}).groupby(long_x.index.year).mean() * 1e4
            legs_yearly.extend(dict(factor=name, horizon=h, year=int(y), long_bp=r.long, short_bp=r.short)
                               for y, r in ly.iterrows())
            daily_ic[(name, h)] = ic

    result = pd.DataFrame(rows)
    result[q_col] = _bh(result["p"])
    yearly_df = pd.DataFrame(yearly)
    legs_yearly_df = pd.DataFrame(legs_yearly)
    judge = result.loc[result.horizon.eq(JUDGE_H)].set_index("factor")
    pos_years = yearly_df.loc[yearly_df.horizon.eq(JUDGE_H)].groupby("factor")["mean_ic"].apply(lambda s: int(s.gt(0).sum()))
    checks = pd.DataFrame({
        "R1_q_any": (result[q_col].lt(0.05) & result["mean_ic"].gt(0)).groupby(result["factor"]).any(),
        "R2_trend": judge["ic_trend_up"].gt(0) & judge["ic_not_trend_up"].gt(0),
        "R3_vol": judge["ic_vol_high"].gt(0) & judge["ic_not_vol_high"].gt(0),
        "R4_years": pos_years.ge(5),
        "R5_legs": judge["long_excess_bp"].gt(0) & judge["short_excess_bp"].gt(0),
        "positive_years": pos_years,
    }).reindex(list(sign_map))
    checks["robust"] = checks[["R1_q_any", "R2_trend", "R3_vol", "R4_years", "R5_legs"]].all(axis=1)
    return result, yearly_df, legs_yearly_df, checks, daily_ic


def seat_purified_scores(panel: pd.DataFrame,
                         dates: pd.DatetimeIndex, symbols: list[str],
                         eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """独立事后检验族：按 CROWD1 换手门槛拆分的 STICKY_NET_SHARE 与 SPEC_NET_SHARE。"""
    sticky = pd.DataFrame(np.nan, index=dates, columns=symbols)
    spec = pd.DataFrame(np.nan, index=dates, columns=symbols)
    rank_root = C.daily_dir("research") / "member_rank"
    for sym in symbols:
        if not (rank_root / f"{sym}_long.pkl").is_file():
            continue
        mkt = F.market_inputs(panel, sym).reindex(dates)
        tables = F.member_tables(sym, dates, top20=True, partition="research")
        split = F.sticky_net_share(tables, mkt)
        sticky[sym] = split["STICKY_NET_SHARE"]
        spec[sym] = split["SPEC_NET_SHARE"]
    sample = dates[dates >= START]
    return {
        "STICKY_NET_SHARE": sticky.replace([np.inf, -np.inf], np.nan).where(eligible).loc[sample],
        "SPEC_NET_SHARE": spec.replace([np.inf, -np.inf], np.nan).where(eligible).loc[sample],
    }


def build_inputs(member_path: Path):
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel["date"], "截面稳健性日面板")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    wide = lambda col: panel.pivot(index="date", columns="symbol", values=col).reindex(dates)
    eligible = wide("eligible").astype("boolean").fillna(False).astype(bool)
    symbols = sorted(eligible.columns[eligible.loc[START:].any()])
    eligible = eligible[symbols]
    ret1, ret2, oi = wide("ret1")[symbols], wide("ret2")[symbols], wide("oi_total")[symbols]

    econ = candidate_factors(panel)
    raw = {
        "ROLL_YIELD": econ["ROLL_YIELD"].reindex(index=dates, columns=symbols),
        "MOM_12_1": econ["MOM_12_1"].reindex(index=dates, columns=symbols),
        "BASIS_MOM_242": basis_momentum(ret1, ret2),
        "SKEW_242": rolling_skew(ret1),
    }
    raw.update({k: v.reindex(index=dates, columns=symbols) for k, v in _member_factors(member_path, oi).items()})
    sample = dates[dates >= START]
    scores = {k: (v * SIGNS[k]).replace([np.inf, -np.inf], np.nan).where(eligible).loc[sample] for k, v in raw.items()}

    contracts = D.load_contracts("research")
    bars = contracts[["contract", "date", "open"]]
    returns = return_matrix(wide("c1_contract")[symbols], bars.set_index(["contract", "date"])["open"])
    labels = {h: forward_returns(returns, h).loc[sample] for h in HORIZONS}

    market = ret1.where(eligible).mean(axis=1)
    trend = market.rolling(STATE_WINDOW, min_periods=STATE_WINDOW).sum()
    vol = market.rolling(STATE_WINDOW, min_periods=STATE_WINDOW).std()
    vol_med = vol.expanding(min_periods=STATE_WINDOW).median()
    states = pd.DataFrame({
        "trend_up": trend.gt(0).where(trend.notna()),
        "vol_high": vol.gt(vol_med).where(vol_med.notna()),
    }).loc[sample]
    seat_scores = seat_purified_scores(panel, dates, symbols, eligible)
    return scores, labels, states, seat_scores


def run(out: Path, member_path: Path | None = None) -> pd.DataFrame:
    member_path = member_path or C.study_dir("member_ic") / "daily_factors_and_returns.csv"
    C.assert_research_only(member_path)
    scores, labels, states, seat_scores = build_inputs(member_path)

    result, yearly, legs_yearly, checks, daily_ic = _evaluate_family(
        scores, labels, states, FAMILY, SIGNS, "q_primary",
    )
    ranked = {k: v.rank(axis=1, pct=True) for k, v in scores.items()}
    corr = pd.DataFrame({a: {b: _rank_corr(ranked[a], ranked[b], MIN_NAMES)[0].mean() for b in ranked} for a in ranked})

    composite_rows = []
    chosen = list(checks.index[checks["robust"]])
    if chosen:
        stack = pd.concat([ranked[k] for k in chosen], keys=chosen)
        comp = stack.groupby(level=1).mean().where(stack.groupby(level=1).count().ge(len(chosen)))
        comp = comp.reindex(next(iter(scores.values())).index)
        for h in HORIZONS:
            ic, count = _rank_corr(comp, labels[h], MIN_NAMES)
            legs = leg_returns(comp, labels[h])
            composite_rows.append(dict(
                members="+".join(chosen), horizon=h, mean_ic=ic.mean(), hac_t=hac_t(ic, max(20, 2 * h)),
                ic_2016_2018=ic.loc[:"2018"].mean(), ic_2019_2021=ic.loc["2019":].mean(),
                long_excess_bp=1e4 * (legs.top - legs.pool).mean(), short_excess_bp=1e4 * (legs.pool - legs.bottom).mean(),
                tail_turnover=membership_turnover(comp), positive_years=int(ic.groupby(ic.index.year).mean().gt(0).sum()),
            ))

    seat_signs = {"STICKY_NET_SHARE": 1, "SPEC_NET_SHARE": 1}
    seat_family = {k: "seat_purified" for k in seat_signs}
    seat_res, _, _, seat_checks, _ = _evaluate_family(
        seat_scores, labels, states, seat_family, seat_signs, "q_seat",
    )
    seat_out = seat_res.merge(
        seat_checks.add_prefix("chk_"), left_on="factor", right_index=True, how="left",
    )

    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "factor_horizons.csv", index=False)
    yearly.to_csv(out / "ic_by_year.csv", index=False)
    legs_yearly.to_csv(out / "legs_by_year.csv", index=False)
    checks.to_csv(out / "robustness_checks.csv")
    corr.to_csv(out / "factor_rank_correlation.csv")
    pd.DataFrame(composite_rows).to_csv(out / "composite.csv", index=False)
    seat_out.to_csv(out / "seat_purified.csv", index=False)
    pd.DataFrame({f"{k}_d{h}": v for (k, h), v in daily_ic.items()}).to_csv(out / "daily_ic.csv", index_label="date")
    return result

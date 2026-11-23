"""粘性净多的 4 个持仓类补充因子：研究期截面 IC、与粘性净多的相关和控制后的增量。

规格在读取任何结果前固定（2026-11-23），只用 2016-01-04 至 2021-12-31，2010–2015 只做滚动窗口预热。

因子（均用前 20 名榜单，t 日收盘可见，方向事先按经济假说取正）
  TRACK_NET_SHARE   会员历史表现。会员 j 在 t 日的得分为信号日 τ ∈ [t−260, t−11] 内、
                    所有入池品种上披露净持仓变化的符号乘以之后 10 日主力开盘收益的均值，
                    至少 100 个非零观测。τ ≤ t−11 保证标签在 t 日开盘前已实现。
                    只保留得分为正的会员，算它们在品种上的可见净多占比。会员名不做合并映射。
  STICKY_TERM_SPREAD 粘性席位在非主力合约上的净多占比减去在主力合约上的净多占比。
                    粘性席位按品种榜换手规则判定，与 STICKY_NET_SHARE 相同。两段都要有可见持仓。
  STICKY_RANGE_60   粘性净多在最近 60 个有效日区间中的位置，映射到 [−1, 1]。
                    接近 +1 表示粘性席位正处于 60 日内最偏多的新建仓状态。
  TOP5_CONC_SPREAD  多头榜前 5 名持仓占前 20 名之比减空头榜对应比例。每侧至少 10 名会员。
                    正号对应"集中的一方有信息"；负的显著 IC 则支持"集中一方被挤仓"。

检验
  品种池为流动性池；标签为信号日后 5、10、20 日主力开盘到开盘复利收益；每天至少 8 个品种。
  IC 为日截面 Spearman，Newey–West 滞后 max(20, 2h)，双侧 p。
  主检验族：4 个因子的 10 日 IC，族内 BH。全规格族：4 × 3 个预测期，另做 BH。
  增量族：在因子、粘性净多、标签同时有效的品种上，先把因子的截面秩对粘性净多的截面秩做回归，
  取残差的 10 日 IC，4 个因子族内 BH。
  另报 2016–2018 与 2019–2021 两段的 IC、与粘性净多的日均截面秩相关、两端价差。
"""
from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data as D
from research.factors import member_tables, seat_turnover_groups
from research.ic_study import _bh, _rank_corr, forward_returns, return_matrix
from research.xs_robustness import MIN_NAMES, START, hac_t, leg_returns

END = pd.Timestamp(C.RESEARCH_END)
FACTORS = ("TRACK_NET_SHARE", "STICKY_TERM_SPREAD", "STICKY_RANGE_60", "TOP5_CONC_SPREAD")
HORIZONS = (5, 10, 20)
PRIMARY_HORIZON = 10
SKILL_HORIZON = 10
SKILL_WINDOW = 250
SKILL_LAG = SKILL_HORIZON + 1
SKILL_MIN_OBS = 100
RANGE_WINDOW = 60
TOP_K = 5
CONC_MIN_NAMES = 10


# ---------------------------------------------------------------- 因子构造

def masked_net_share(net: pd.DataFrame, gross: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    num = net.where(mask).sum(axis=1, min_count=1)
    den = gross.where(mask).sum(axis=1, min_count=1)
    return num / den.where(den.gt(0))


def top_share(holdings: pd.DataFrame, k: int = TOP_K, min_names: int = CONC_MIN_NAMES) -> pd.Series:
    """每日前 k 名持仓之和占全部上榜持仓之和的比例。"""
    values = holdings.to_numpy(dtype=float)
    filled = np.where(np.isfinite(values), values, -np.inf)
    ordered = -np.sort(-filled, axis=1)
    top = np.where(np.isfinite(ordered[:, :k]), ordered[:, :k], 0.0).sum(axis=1)
    total = np.nansum(values, axis=1)
    count = np.isfinite(values).sum(axis=1)
    share = np.divide(top, total, out=np.full(len(total), np.nan), where=total > 0)
    return pd.Series(np.where(count >= min_names, share, np.nan), index=holdings.index)


def concentration_spread(tables: dict[str, pd.DataFrame]) -> pd.Series:
    return (top_share(tables["L"]) - top_share(tables["S"])).where(tables["ranked"])


def range_position(sticky: pd.Series, window: int = RANGE_WINDOW) -> pd.Series:
    """最近 window 个有效观测（含当日）区间内的位置，映射到 [−1, 1]。"""
    valid = sticky.dropna()
    high = valid.rolling(window, min_periods=window).max()
    low = valid.rolling(window, min_periods=window).min()
    span = high - low
    position = 2 * (valid - low) / span.where(span.gt(0)) - 1
    return position.reindex(sticky.index).where(sticky.notna())


def position_signs(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """会员当日披露的净持仓变化符号；在榜一侧的变化缺失时记为缺失。"""
    on_long, on_short = tables["L"].notna(), tables["S"].notna()
    d_long = tables["dL"].where(on_long, 0.0)
    d_short = tables["dS"].where(on_short, 0.0)
    valid = (on_long | on_short) & d_long.notna() & d_short.notna()
    valid &= tables["ranked"].to_numpy(dtype=bool)[:, None]
    return np.sign(d_long - d_short).where(valid)


def member_skill(signs: dict[str, pd.DataFrame], label: pd.DataFrame, eligible: pd.DataFrame,
                 window: int = SKILL_WINDOW, lag: int = SKILL_LAG,
                 min_obs: int = SKILL_MIN_OBS) -> pd.DataFrame:
    """日期 × 会员得分：窗口内各品种 sign(ΔN) × 后续标签的均值，整体后移 lag 日。"""
    score_parts, count_parts = [], []
    for symbol, sign in signs.items():
        y = label[symbol].where(eligible[symbol])
        used = sign.ne(0) & sign.notna() & y.notna().to_numpy()[:, None]
        score_parts.append(sign.mul(y, axis=0).where(used))
        count_parts.append(used.astype(float))
    score = pd.concat(score_parts, axis=1).T.groupby(level=0).sum(min_count=1).T
    count = pd.concat(count_parts, axis=1).T.groupby(level=0).sum().T
    score_sum = score.fillna(0.0).rolling(window, min_periods=1).sum().shift(lag)
    count_sum = count.rolling(window, min_periods=1).sum().shift(lag)
    return (score_sum / count_sum).where(count_sum.ge(min_obs))


def term_spread(ranks: pd.DataFrame, sticky: pd.DataFrame, dominant: pd.Series) -> pd.Series:
    """粘性席位在非主力合约上的净多占比减去在主力合约上的净多占比。"""
    if ranks.empty:
        return pd.Series(np.nan, index=dominant.index)
    held = ranks.loc[ranks["side"].isin(["long", "short"])
                     & ranks["volume"].gt(0), ["trading_date", "member_name", "contract", "side", "volume"]]
    flags = sticky.rename_axis(index="trading_date", columns="member_name").reset_index().melt(
        id_vars="trading_date", var_name="member_name", value_name="sticky")
    held = held.merge(flags.loc[flags["sticky"]], on=["trading_date", "member_name"], how="inner")
    front = dominant.rename("dominant").rename_axis("trading_date").reset_index()
    held = held.merge(front, on="trading_date", how="inner")
    held["front"] = held["contract"].eq(held["dominant"])
    held["signed"] = held["volume"].where(held["side"].eq("long"), -held["volume"])
    grouped = held.groupby(["trading_date", "front"])[["signed", "volume"]].sum().unstack("front")
    grouped = grouped.reindex(columns=pd.MultiIndex.from_product([["signed", "volume"], [False, True]]))
    share = grouped["signed"] / grouped["volume"].where(grouped["volume"].gt(0))
    return (share[False] - share[True]).reindex(dominant.index)


def residual_scores(score: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """同一批品种上，因子截面秩对控制变量截面秩回归后的残差。"""
    common = score.notna() & control.notna()
    x = score.where(common).rank(axis=1, pct=True)
    z = control.where(common).rank(axis=1, pct=True)
    x = x.sub(x.mean(axis=1), axis=0)
    z = z.sub(z.mean(axis=1), axis=0)
    beta = (x * z).sum(axis=1) / z.pow(2).sum(axis=1).where(lambda value: value.gt(0))
    return x.sub(z.mul(beta, axis=0)).where(common.sum(axis=1).ge(MIN_NAMES), axis=0)


# ---------------------------------------------------------------- 研究期输入

def _symbol_state(symbol: str, dates: pd.DatetimeIndex, market: pd.DataFrame, dominant: pd.Series) -> dict:
    tables = member_tables(symbol, dates, top20=True, partition="research")
    net, gross, sticky, _ = seat_turnover_groups(tables, market)
    sticky_share = masked_net_share(net, gross, sticky)
    ranks = D.load_member_rank_contracts(symbol, "research")
    if not ranks.empty:
        C.assert_no_holdout_dates(ranks["trading_date"], f"{symbol} 合约级会员排名")
        ranks = ranks.loc[ranks["rank"].between(1, 20)]
    return {
        "net": net, "gross": gross, "has_day": tables["ranked"],
        "signs": position_signs(tables),
        "STICKY_NET_SHARE": sticky_share,
        "STICKY_TERM_SPREAD": term_spread(ranks, sticky, dominant),
        "STICKY_RANGE_60": range_position(sticky_share),
        "TOP5_CONC_SPREAD": concentration_spread(tables),
    }


def load_inputs(jobs: int):
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel["date"], "持仓补充因子日面板")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))

    def wide(name):
        return panel.pivot(index="date", columns="symbol", values=name)

    eligible = wide("eligible").eq(True)
    root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(root)
    symbols = [s for s in eligible.columns[eligible.loc[START:].any()] if (root / f"{s}_long.pkl").is_file()]
    eligible = eligible[symbols]
    dates = eligible.index
    volume, oi, dominant = wide("volume_total"), wide("oi_total"), wide("c1_contract")[symbols]
    states = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_symbol_state)(s, dates, pd.DataFrame({"volume": volume[s], "oi": oi[s]}), dominant[s])
        for s in symbols
    )
    states = dict(zip(symbols, states))
    bars = D.load_contracts("research")
    returns = return_matrix(dominant, bars.set_index(["contract", "date"])["open"])
    labels = {h: forward_returns(returns, h) for h in HORIZONS}

    skill = member_skill({s: st["signs"] for s, st in states.items()}, labels[SKILL_HORIZON], eligible)
    track = {}
    for s, st in states.items():
        positive = skill.reindex(index=dates, columns=st["net"].columns).gt(0)
        mask = positive & st["gross"].gt(0) & st["has_day"].to_numpy(dtype=bool)[:, None]
        track[s] = masked_net_share(st["net"], st["gross"], mask)

    factors = {"TRACK_NET_SHARE": pd.DataFrame(track)}
    for name in ("STICKY_TERM_SPREAD", "STICKY_RANGE_60", "TOP5_CONC_SPREAD", "STICKY_NET_SHARE"):
        factors[name] = pd.DataFrame({s: st[name] for s, st in states.items()})
    factors = {
        name: frame.reindex(index=dates, columns=symbols).replace([np.inf, -np.inf], np.nan)
        .where(eligible).loc[START:END]
        for name, frame in factors.items()
    }
    labels = {h: label.loc[START:END] for h, label in labels.items()}
    skilled = skill.loc[START:END].gt(0).sum(axis=1)
    return factors, labels, skilled


# ---------------------------------------------------------------- 检验

def _ic_row(score: pd.DataFrame, label: pd.DataFrame, horizon: int) -> dict:
    ic, names = _rank_corr(score, label, MIN_NAMES)
    t = hac_t(ic, max(20, 2 * horizon))
    legs = leg_returns(score, label)
    spread = legs["top"] - legs["bottom"]
    early, late = ic.loc[:"2018"], ic.loc["2019":]
    lags = max(20, 2 * horizon)
    return dict(
        mean_ic=ic.mean(), hac_t=t, p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
        days=int(ic.count()), avg_names=names.mean(), positive_fraction=ic.dropna().gt(0).mean(),
        ic_2016_2018=early.mean(), t_2016_2018=hac_t(early, lags),
        ic_2019_2021=late.mean(), t_2019_2021=hac_t(late, lags),
        spread_bp=1e4 * spread.mean(),
    )


def evaluate(factors: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    sticky = factors["STICKY_NET_SHARE"]
    rows = []
    for name in (*FACTORS, "STICKY_NET_SHARE"):
        score = factors[name]
        corr, _ = _rank_corr(score, sticky, MIN_NAMES)
        for horizon in HORIZONS:
            rows.append(dict(test="ic", factor=name, horizon=horizon, corr_with_sticky=corr.mean(),
                             **_ic_row(score, labels[horizon], horizon)))
        if name == "STICKY_NET_SHARE":
            continue
        common = score.notna() & sticky.notna()
        label = labels[PRIMARY_HORIZON]
        residual = residual_scores(score, sticky)
        rows.append(dict(test="residual", factor=name, horizon=PRIMARY_HORIZON,
                         **_ic_row(residual, label, PRIMARY_HORIZON)))
        rows.append(dict(test="common_raw", factor=name, horizon=PRIMARY_HORIZON,
                         **_ic_row(score.where(common), label, PRIMARY_HORIZON)))
        rows.append(dict(test="common_sticky", factor=name, horizon=PRIMARY_HORIZON,
                         **_ic_row(sticky.where(common), label, PRIMARY_HORIZON)))
    result = pd.DataFrame(rows)
    in_family = result["factor"].isin(FACTORS)
    families = {
        "q_primary": in_family & result["test"].eq("ic") & result["horizon"].eq(PRIMARY_HORIZON),
        "q_full": in_family & result["test"].eq("ic"),
        "q_increment": result["test"].eq("residual"),
    }
    for column, members in families.items():
        result[column] = _bh(result["p"].where(members))
    return result


def run(out: Path, jobs: int) -> pd.DataFrame:
    factors, labels, skilled = load_inputs(jobs)
    result = evaluate(factors, labels)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "factor_ic.csv", index=False)
    coverage = pd.DataFrame({name: frame.notna().sum(axis=1) for name, frame in factors.items()})
    coverage.assign(skilled_members=skilled).describe().to_csv(out / "coverage.csv")
    return result

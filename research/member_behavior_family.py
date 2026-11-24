"""粘性净多的第 2 组持仓补充：逆势加仓、跨品种一致性、榜单成员稳定性。研究期截面检验。

规格在读取任何结果前固定（2026-11-23），只用 2016-01-04 至 2021-12-31，2010–2015 只做滚动窗口预热。
粘性席位的判定与 STICKY_NET_SHARE 相同（品种榜前 20 名，会员换手低于市场换手）。

CONTRA_FLOW_20（方向 +）
  粘性席位当日披露的净持仓变化 f_t = Σ_{j∈粘性} (ΔL_j − ΔS_j)，只上一边榜时另一边记 0，
  用粘性组可见总持仓 G_t 归一化。只计与当日主力收盘收益 r_t 反向的那部分：
  c_t = (f_t / G_t) · 1[f_t · r_t < 0]，同向或任一为 0 时 c_t = 0。
  因子为最近 20 个交易日 c_t 之和，至少 15 个有效日。
  假说：粘性资金在下跌时加多、上涨时加空，是在承接流动性，之后价格向它们的方向回归。

CHAIN_CONSISTENT_NET（方向 +）
  产业链即 config 中事先定义的板块。会员 j 在品种 s 上的粘性净头寸按品种单边总持仓归一化，
  n_{j,s} = (L − S)_{j,s} / OI_s，只在 j 是 s 上的粘性席位时有值。
  j 在板块内除 s 以外品种上的方向 d_{j,−s} = sign(Σ_{s'≠s} n_{j,s'})，要求 j 当日至少还在 1 个
  同板块品种上是粘性席位。sign(n_{j,s}) = d_{j,−s} ≠ 0 的会员算作方向一致。
  因子为品种 s 上方向一致的粘性席位的可见净多占比。板块品种不限于流动性池。
  假说：在整条产业链上同向持仓的低换手席位，更可能是基于产业判断。

BOARD_STABILITY_20（条件变量，不单独定方向）
  品种当日多头榜或空头榜上的会员集合，与 20 个交易日前集合的 Jaccard 重合度，两日都要有榜单。
  条件检验：每天在粘性净多、稳定性、标签都有效且至少 16 个品种时，按稳定性截面排序分为高、低两半，
  奇数时去掉中间一个，同分按品种代码顺序。分别算两半内粘性净多与标签的 Spearman IC，
  检验 IC_高 − IC_低 > 0 的假说：榜单稳定的品种上，粘性净多更有信息。
  稳定性自身的 IC 只作描述，不进任何检验族。

检验
  品种池为流动性池；标签为信号日后 5、10、20 日主力开盘到开盘复利收益；每天至少 8 个品种。
  Newey–West 滞后 max(20, 2h)，双侧 p。
  主检验族（BH）：CONTRA_FLOW_20 与 CHAIN_CONSISTENT_NET 的 10 日 IC，及稳定性条件差的 10 日结果，共 3 项。
  全规格族（BH）：上述 3 项各在 5、10、20 日，共 9 项。
  增量族（BH）：2 个因子对粘性净多截面秩回归后的残差 10 日 IC，共 2 项。

本文件按上述规格完整运行这一族。规格文字取自 2026-11-23 提交 8f19d46 的 research/member_behavior.py，
未改动。跨品种一致性的构造复用现行 research.member_behavior，与当时的定义相同。
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
from research.member_behavior import chain_consistent_share, masked_net_share, residual_scores
from research.xs_robustness import MIN_NAMES, START, hac_t, leg_returns

END = pd.Timestamp(C.RESEARCH_END)
FACTORS = ("CONTRA_FLOW_20", "CHAIN_CONSISTENT_NET")
CONDITION = "BOARD_STABILITY_20"
HORIZONS = (5, 10, 20)
PRIMARY_HORIZON = 10
CONTRA_WINDOW = 20
CONTRA_MIN_DAYS = 15
STABILITY_LAG = 20
HALF_MIN_NAMES = 8


# ---------------------------------------------------------------- 因子构造

def sticky_flow(tables: dict[str, pd.DataFrame], sticky: pd.DataFrame) -> pd.Series:
    """粘性席位披露的净持仓变化之和；没有可用变化时为缺失。"""
    on_long, on_short = tables["L"].notna(), tables["S"].notna()
    d_long = tables["dL"].where(on_long, 0.0)
    d_short = tables["dS"].where(on_short, 0.0)
    change = (d_long - d_short).where(sticky & d_long.notna() & d_short.notna())
    return change.sum(axis=1, min_count=1)


def contra_flow(flow: pd.Series, gross: pd.Series, ret: pd.Series,
                window: int = CONTRA_WINDOW, min_days: int = CONTRA_MIN_DAYS) -> pd.Series:
    scaled = flow / gross.where(gross.gt(0))
    valid = scaled.notna() & ret.notna()
    against = (scaled * ret).lt(0)
    daily = scaled.where(against, 0.0).where(valid)
    return daily.rolling(window, min_periods=min_days).sum()


def board_stability(tables: dict[str, pd.DataFrame], lag: int = STABILITY_LAG) -> pd.Series:
    """当日多空榜会员集合与 lag 个交易日前集合的 Jaccard 重合度。"""
    on_board = (tables["L"].notna() | tables["S"].notna()) & tables["ranked"].to_numpy(dtype=bool)[:, None]
    past = on_board.shift(lag, fill_value=False)
    inter = (on_board & past).sum(axis=1)
    union = (on_board | past).sum(axis=1)
    both = tables["ranked"] & tables["ranked"].shift(lag, fill_value=False)
    return (inter / union.where(union.gt(0))).where(both)


def split_halves(condition: pd.DataFrame, valid: pd.DataFrame,
                 min_half: int = HALF_MIN_NAMES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按条件变量的截面排序分出高、低两半；奇数时去掉中间一个。"""
    score = condition.where(valid)
    n = score.notna().sum(axis=1)
    half = (n // 2).where(n.ge(2 * min_half))
    rank = score.rank(axis=1, method="first")
    low = rank.le(half, axis=0)
    high = rank.gt(n - half, axis=0)
    return high, low


def conditional_ic_gap(sticky: pd.DataFrame, condition: pd.DataFrame,
                       label: pd.DataFrame) -> pd.DataFrame:
    valid = sticky.notna() & condition.notna() & label.notna()
    high, low = split_halves(condition, valid)
    ic_high, _ = _rank_corr(sticky.where(high), label.where(high), HALF_MIN_NAMES)
    ic_low, _ = _rank_corr(sticky.where(low), label.where(low), HALF_MIN_NAMES)
    return pd.DataFrame({"high": ic_high, "low": ic_low, "gap": ic_high - ic_low})


# ---------------------------------------------------------------- 研究期输入

def _symbol_state(symbol: str, dates: pd.DatetimeIndex, market: pd.DataFrame) -> dict:
    tables = member_tables(symbol, dates, top20=True, partition="research")
    net, gross, sticky, _ = seat_turnover_groups(tables, market)
    sticky_gross = gross.where(sticky).sum(axis=1, min_count=1)
    return {
        "net": net, "gross": gross, "sticky": sticky,
        "STICKY_NET_SHARE": masked_net_share(net, gross, sticky),
        "CONTRA_FLOW_20": contra_flow(sticky_flow(tables, sticky), sticky_gross, market["ret"]),
        CONDITION: board_stability(tables),
    }


def load_inputs(jobs: int):
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel["date"], "持仓行为因子日面板")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    panel = panel.assign(eligible=D.eligibility(panel))

    def wide(name):
        return panel.pivot(index="date", columns="symbol", values=name)

    root = C.daily_dir("research") / "member_rank"
    C.assert_research_only(root)
    eligible = wide("eligible").eq(True)
    ranked = [s for s in eligible.columns if (root / f"{s}_long.pkl").is_file()]
    dates = eligible.index
    volume, oi, ret = wide("volume_total"), wide("oi_total"), wide("ret1")
    states = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_symbol_state)(s, dates, pd.DataFrame({"volume": volume[s], "oi": oi[s], "ret": ret[s]}))
        for s in ranked
    )
    states = dict(zip(ranked, states))

    sectors = panel.drop_duplicates("symbol").set_index("symbol")["sector"]
    chain = {}
    for sector, group in sectors.loc[ranked].groupby(sectors.loc[ranked]):
        names = sorted(group.index)
        chain.update(chain_consistent_share(
            {s: states[s]["net"] for s in names}, {s: states[s]["gross"] for s in names},
            {s: states[s]["sticky"] for s in names}, oi, names,
        ))

    symbols = [s for s in ranked if eligible.loc[START:, s].any()]
    eligible = eligible[symbols]
    factors = {"CHAIN_CONSISTENT_NET": pd.DataFrame({s: chain[s] for s in symbols})}
    for name in ("STICKY_NET_SHARE", "CONTRA_FLOW_20", CONDITION):
        factors[name] = pd.DataFrame({s: states[s][name] for s in symbols})
    factors = {
        name: frame.reindex(index=dates, columns=symbols).replace([np.inf, -np.inf], np.nan)
        .where(eligible).loc[START:END]
        for name, frame in factors.items()
    }
    bars = D.load_contracts("research")
    returns = return_matrix(wide("c1_contract")[symbols], bars.set_index(["contract", "date"])["open"])
    labels = {h: forward_returns(returns, h).loc[START:END] for h in HORIZONS}
    return factors, labels


# ---------------------------------------------------------------- 检验

def _ic_row(score: pd.DataFrame, label: pd.DataFrame, horizon: int) -> dict:
    ic, names = _rank_corr(score, label, MIN_NAMES)
    lags = max(20, 2 * horizon)
    t = hac_t(ic, lags)
    legs = leg_returns(score, label)
    spread = legs["top"] - legs["bottom"]
    early, late = ic.loc[:"2018"], ic.loc["2019":]
    return dict(
        mean_ic=ic.mean(), hac_t=t, p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
        days=int(ic.count()), avg_names=names.mean(), positive_fraction=ic.dropna().gt(0).mean(),
        ic_2016_2018=early.mean(), t_2016_2018=hac_t(early, lags),
        ic_2019_2021=late.mean(), t_2019_2021=hac_t(late, lags),
        spread_bp=1e4 * spread.mean(),
    )


def _gap_row(gap: pd.DataFrame, horizon: int) -> dict:
    lags = max(20, 2 * horizon)
    t = hac_t(gap["gap"], lags)
    early, late = gap["gap"].loc[:"2018"], gap["gap"].loc["2019":]
    return dict(
        mean_ic=gap["gap"].mean(), hac_t=t, p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
        days=int(gap["gap"].count()), ic_high=gap["high"].mean(), ic_low=gap["low"].mean(),
        ic_2016_2018=early.mean(), t_2016_2018=hac_t(early, lags),
        ic_2019_2021=late.mean(), t_2019_2021=hac_t(late, lags),
    )


def evaluate(factors: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    sticky = factors["STICKY_NET_SHARE"]
    rows = []
    for name in (*FACTORS, CONDITION, "STICKY_NET_SHARE"):
        score = factors[name]
        corr, _ = _rank_corr(score, sticky, MIN_NAMES)
        for horizon in HORIZONS:
            rows.append(dict(test="ic", factor=name, horizon=horizon, corr_with_sticky=corr.mean(),
                             **_ic_row(score, labels[horizon], horizon)))
    for horizon in HORIZONS:
        gap = conditional_ic_gap(sticky, factors[CONDITION], labels[horizon])
        rows.append(dict(test="condition_gap", factor=CONDITION, horizon=horizon, **_gap_row(gap, horizon)))
    for name in FACTORS:
        residual = residual_scores(factors[name], sticky)
        rows.append(dict(test="residual", factor=name, horizon=PRIMARY_HORIZON,
                         **_ic_row(residual, labels[PRIMARY_HORIZON], PRIMARY_HORIZON)))
    result = pd.DataFrame(rows)
    tested = (result["test"].eq("ic") & result["factor"].isin(FACTORS)) | result["test"].eq("condition_gap")
    families = {
        "q_primary": tested & result["horizon"].eq(PRIMARY_HORIZON),
        "q_full": tested,
        "q_increment": result["test"].eq("residual"),
    }
    for column, members in families.items():
        result[column] = _bh(result["p"].where(members))
    return result


def run(out: Path, jobs: int) -> pd.DataFrame:
    factors, labels = load_inputs(jobs)
    result = evaluate(factors, labels)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "declared_family_ic.csv", index=False)
    pd.DataFrame({name: frame.notna().sum(axis=1) for name, frame in factors.items()}).describe().to_csv(
        out / "declared_family_coverage.csv")
    return result

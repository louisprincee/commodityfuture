"""粘性净多扩展研究（2011–2021，2010 预热）：榜内换手门槛、品种内标准化与变化、粘性席位持仓流、
席位身份与历史记录，以及拥挤度三因子的截面与时序 IC。

规格、符号与检验族在看到结果之前写入 PROTOCOL。只计算因子与 IC，不构造仓位。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config as C
from src import data as D
from research.factors.member import (
    CROWDING_META, EXTENSION_META, FOLLOWUP_META, extension_factors, market_inputs, member_tables, rank_composite,
)
from research.ic.statistics import fixed_bh, mean_inference, rank_corr, rank_residual
from research.ic.study import forward_returns, return_matrix
from research.member.review import (
    END, HORIZONS, MIN_DAYS, MIN_NAMES, PERIODS, START, _FONTS, _GRID, _INK, _MUTED, _SURFACE,
    evaluate_factors, plot_redefinition_ic,
)

OUT_NAME = "member_extensions"
SMOOTHING_PROTOCOL = {
    "declared": "2026-11-26, before the newly requested 10-day smoothing results",
    "window": "2011-01-04..2021-12-31; 2010 warm-up; research only; no OOS access",
    "sample": "Same finite STICKY_PERSIST_IDENTITY, STICKY_FLOW5 and STICKY_FLOW10 sample for all comparisons",
    "factors": "FLOW5/FLOW10: trailing arithmetic means, all 5/10 observations valid; identity unchanged; signs +1",
    "composites": "Each flow percentile rank plus identity percentile rank, fixed equal 1:1 weights",
    "ic_families": "Separate flow and composite families, two candidates each: primary full-period h10 XS; full 1/5/10/20 horizons, three periods, all XS and per-symbol TS; identity reference only",
    "paired_ic": "Single primary: full-period h10 IC COMBO10 minus COMBO5; HAC20; remaining periods/horizons descriptive",
    "accounts": "IDENTITY_MATCHED, IDENTITY_FLOW5, IDENTITY_FLOW10; tails 10%, minimum 8, holding 10 sessions; unchanged execution and costs",
    "account_primary": "One paired full-period net daily return: IDENTITY_FLOW10 minus IDENTITY_FLOW5, one tick, two-sided HAC20, alpha=.05; two ticks saved only",
    "selection": "Exploratory after prior research and OOS were seen; no window selection from OOS",
}
BASELINE = "STICKY_NET_SHARE"
BASELINE_META = {BASELINE: {"family": "baseline", "sign": 1, "description": "Original sticky net share."}}
PROTOCOL = {
    "declared": "2026-11-26, before any extension or 2011-2021 crowding result was computed",
    "window": f"{START}..{END}, 2010 warm-up, research partition only",
    "universe": "shared member universe (src.data.member_universe), top-20 rank tables",
    "labels": "actual-contract open-to-open forward returns, horizons 1/5/10/20",
    "extension_primary": "6 full-period h10 cross-sectional IC tests, BH within the 6",
    "extension_full": "6 factors x 4 horizons x 3 periods XS + per-symbol TS, BH within family",
    "extension_residual_primary": "6 full-period h10 residual IC tests controlling STICKY_NET_SHARE, BH within the 6",
    "crowding_primary": "CROWD1/2/3 reversal sign, full-period h1 and h5 cross-sectional IC (6 tests), BH within the 6",
    "crowding_full": "h1/h5 x 3 periods XS + per-symbol TS, BH within family",
    "min_names": MIN_NAMES, "min_days": MIN_DAYS, "exploratory": True,
    "factors": {name: meta["description"] for name, meta in {**EXTENSION_META, **CROWDING_META}.items()},
}
FOLLOWUP_PROTOCOL = {
    "declared": "2026-11-26, after the extension IC and account results, before any follow-up result was computed",
    "window": PROTOCOL["window"],
    "followup_primary": "STICKY_PERSIST_IDENTITY and STICKY_FLOW5 full-period h10 cross-sectional IC, BH within the 2",
    "followup_full": "2 factors x 4 horizons x 3 periods XS + per-symbol TS, BH within family",
    "followup_residual_primary": "2 full-period h10 residual IC tests controlling STICKY_NET_SHARE, BH within the 2",
    "seat_mask_choice": "intersection of STICKY_PERSIST and STICKY_IDENTITY masks, chosen after a coverage-only check "
                        "(98.6% of symbol-days with a qualifying seat, no returns used)",
    "exploratory": True,
    "factors": {name: meta["description"] for name, meta in FOLLOWUP_META.items()},
}
COMBO = "STICKY_PI_FLOW"
COMBO_PARTS = ("STICKY_PERSIST_IDENTITY", "STICKY_FLOW5")
COMBO_META = {
    COMBO: {"family": "combo", "sign": 1,
            "description": "Equal-weight mean of cross-sectional percentile ranks of STICKY_PERSIST_IDENTITY and STICKY_FLOW5."},
    **{name: {**FOLLOWUP_META[name], "family": "reference"} for name in COMBO_PARTS},
}
COMBO_PROTOCOL = {
    "declared": "2026-11-26, after the follow-up IC and account results showed sub-period complementarity, "
                "before any combination result was computed",
    "window": PROTOCOL["window"],
    "combo_rule": COMBO_META[COMBO]["description"] + " Fixed 1:1 weights, no optimisation.",
    "combo_primary": "STICKY_PI_FLOW full-period h10 cross-sectional IC (single test)",
    "combo_secondary": "HAC paired daily h10 IC differences STICKY_PI_FLOW minus each component, 3 periods; "
                       "daily IC correlation and mean cross-sectional factor correlation of the components",
    "exploratory": True,
}
COMBO_SERIES = (
    (BASELINE, "原粘性净多（基线）", "#7a8a96"),
    ("STICKY_PERSIST_IDENTITY", "持久长期粘性身份", "#B76569"),
    ("STICKY_FLOW5", "粘性席位持仓流 · 5 日平滑", "#5B8FA8"),
    (COMBO, "身份 + 持仓流等权组合", "#147D89"),
)
EXTENSION_SERIES = (
    (BASELINE, "原粘性净多（基线）", "#7a8a96"),
    ("STICKY_TOP20", "榜内换手门槛粘性", "#147D89"),
    ("STICKY_Z250", "粘性净多 · 品种内 z 分数", "#C27D4E"),
    ("STICKY_CHG20", "粘性净多 · 20 日变化", "#B76569"),
    ("STICKY_FLOW", "粘性席位持仓流", "#5B8FA8"),
    ("STICKY_IDENTITY", "长期粘性身份席位", "#6E9C7B"),
    ("SEAT_SKILL_NET", "历史记录为正的席位", "#8D83B5"),
)
CROWD_COLORS = {"CROWD1": "#147D89", "CROWD2": "#C27D4E", "CROWD3": "#8D83B5"}


def build_inputs(jobs: int, extra_names: tuple[str, ...] = ()):
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel.date, "扩展研究面板")
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(["date", "symbol"])
    eligible, dominant = D.member_universe(panel, pd.Timestamp(START), "research")
    bars = D.load_contracts("research", for_execution=True)
    returns = return_matrix(dominant, bars.set_index(["contract", "date"])["open"])

    def one(symbol):
        tables = member_tables(symbol, eligible.index, top20=True, partition="research")
        market = market_inputs(panel, symbol).reindex(eligible.index)
        return extension_factors(tables, market, returns[symbol])

    symbols = list(eligible.columns)
    parts = Parallel(n_jobs=jobs, prefer="threads")(delayed(one)(s) for s in symbols)
    names = [BASELINE, *EXTENSION_META, *CROWDING_META, *FOLLOWUP_META, *extra_names]
    scores = {name: pd.DataFrame({s: part[name] for s, part in zip(symbols, parts)}, index=eligible.index)
              .where(eligible).loc[START:END] for name in names}
    labels = {h: forward_returns(returns, h).reindex_like(scores[BASELINE]) for h in HORIZONS}
    return scores, labels


def residual_incremental(scores: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame],
                         names=tuple(EXTENSION_META)) -> pd.DataFrame:
    rows = []
    control = scores[BASELINE]
    for name in names:
        residual = rank_residual(scores[name], control, MIN_NAMES)
        for h, label in labels.items():
            ic, n = rank_corr(residual, label.reindex_like(residual), MIN_NAMES)
            for period, (left, right) in PERIODS.items():
                rows.append(dict(factor=name, control=BASELINE, horizon=h, period=period,
                                 avg_names=n.loc[left:right].mean(),
                                 **mean_inference(ic.loc[left:right], max(20, 2 * h), MIN_DAYS)))
    result = pd.DataFrame(rows)
    primary = result.horizon.eq(10) & result.period.eq("2011-2021")
    result["q_primary"] = fixed_bh(result.loc[primary, "p"]).reindex(result.index)
    result["q_all"] = fixed_bh(result.p)
    return result


def combo_diagnostics(scores: dict[str, pd.DataFrame], daily: pd.DataFrame) -> pd.DataFrame:
    """两成分的互补程度与组合相对各成分的配对 IC 差。"""
    rows = []
    left, right = COMBO_PARTS
    factor_corr, _ = rank_corr(scores[left], scores[right], MIN_NAMES)
    for period, (start, end) in PERIODS.items():
        rows.append(dict(statistic="cross_sectional_factor_corr", horizon=np.nan, period=period,
                         value=factor_corr.loc[start:end].mean()))
    for h in HORIZONS:
        matrix = daily.loc[daily.horizon.eq(h) & daily.period.eq("2011-2021")].pivot(
            index="date", columns="factor", values="ic")
        for period, (start, end) in PERIODS.items():
            part = matrix.loc[start:end]
            rows.append(dict(statistic="daily_ic_corr", horizon=h, period=period,
                             value=part[left].corr(part[right])))
            for other in COMBO_PARTS:
                stats = mean_inference(part[COMBO] - part[other], max(20, 2 * h), MIN_DAYS)
                rows.append(dict(statistic=f"ic_gain_vs_{other}", horizon=h, period=period,
                                 value=stats.pop("mean_ic"), **stats))
    return pd.DataFrame(rows)


def timeseries_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """逐品种时序 IC 的横向汇总：均值、可估品种数与 q < 0.05 的正负个数。"""
    usable = ts.loc[ts.ic.notna()]
    grouped = usable.groupby(["factor", "horizon", "period"])
    return pd.DataFrame({
        "mean_ts_ic": grouped.ic.mean(),
        "median_ts_ic": grouped.ic.median(),
        "symbols": grouped.size(),
        "positive_share": grouped.ic.apply(lambda x: x.gt(0).mean()),
        "significant_positive": grouped.apply(lambda x: int((x.q_all.lt(.05) & x.ic.gt(0)).sum()), include_groups=False),
        "significant_negative": grouped.apply(lambda x: int((x.q_all.lt(.05) & x.ic.lt(0)).sum()), include_groups=False),
    }).reset_index()


def plot_crowding(daily: pd.DataFrame, summary: pd.DataFrame, ts_summary: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    full = daily.loc[daily.period.eq("2011-2021")]
    with plt.rc_context({**_FONTS, "text.color": _INK, "axes.labelcolor": _MUTED,
                         "xtick.color": _MUTED, "ytick.color": _MUTED}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), facecolor=_SURFACE,
                                       gridspec_kw={"width_ratios": [1.35, 1.0]})
        for axis in (ax1, ax2):
            axis.set_facecolor(_SURFACE)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#d7e0e6")
            axis.tick_params(length=0, labelsize=9)
        ax1.axvspan(full.date.min(), pd.Timestamp("2016-01-04"), color="#f1efea", zorder=0, lw=0)
        ax1.axhline(0, color="#b9b7b0", lw=.8)
        for name, color in CROWD_COLORS.items():
            for h, style in ((1, "-"), (5, "--")):
                ic = full.loc[full.factor.eq(name) & full.horizon.eq(h)].set_index("date")["ic"].dropna()
                ax1.plot(ic.index, ic.cumsum(), color=color, ls=style, lw=1.8, label=f"{name} · {h} 日")
        ax1.set_title("累计截面 IC（已按反转方向取号）· 灰底为 2011–2015", loc="left", fontsize=11.5, fontweight="bold", pad=10)
        ax1.set_ylabel("累计 IC（逐日均值之和）", fontsize=9.5)
        ax1.legend(loc="lower left", frameon=False, fontsize=8.6, ncol=2)
        ax1.grid(axis="y", color=_GRID, lw=.7)
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        xs = summary.loc[summary.period.eq("2011-2021")].set_index(["factor", "horizon"])
        ts = ts_summary.loc[ts_summary.period.eq("2011-2021")].set_index(["factor", "horizon"])
        bars = [("截面 1 日", xs, 1, "mean_ic", "#147D89"), ("截面 5 日", xs, 5, "mean_ic", "#5B8FA8"),
                ("时序 1 日（品种均值）", ts, 1, "mean_ts_ic", "#C27D4E"),
                ("时序 5 日（品种均值）", ts, 5, "mean_ts_ic", "#E0A878")]
        height = .8 / len(bars)
        for i, (label, table, h, column, color) in enumerate(bars):
            for j, name in enumerate(CROWD_COLORS):
                value = float(table.loc[(name, h), column])
                y = j + (i - (len(bars) - 1) / 2) * height
                ax2.barh(y, value, height=height * .9, color=color, label=label if j == 0 else None)
                tag = f"{value:+.4f}"
                if table is xs and xs.loc[(name, h), "p"] < .05:
                    tag += "*"
                ax2.annotate(tag, (value, y), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                             ha="left" if value >= 0 else "right", va="center", fontsize=8)
        ax2.axvline(0, color="#9aa8b2", lw=.8)
        ax2.set_yticks(range(len(CROWD_COLORS)), list(CROWD_COLORS))
        ax2.invert_yaxis()
        span = max(.02, float(np.nanmax(np.abs([*xs["mean_ic"], *ts["mean_ts_ic"]]))) * 1.6)
        ax2.set_xlim(-span, span)
        ax2.set_title("2011–2021 平均 IC（* 为截面 p < 0.05）", loc="left", fontsize=11.5, fontweight="bold", pad=10)
        ax2.set_xlabel("平均 Spearman IC", fontsize=9.5)
        ax2.legend(loc="lower right", frameon=False, fontsize=8.4)
        ax2.grid(axis="x", color=_GRID, lw=.7)
        fig.text(.018, .95, "拥挤度三因子 · 截面与时序 IC · 2011–2021", fontsize=14.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2.8)
        fig.savefig(output, dpi=180, facecolor=_SURFACE)
        plt.close(fig)


def run(out: Path, jobs: int = 8) -> None:
    C.assert_research_only(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2))
    scores, labels = build_inputs(jobs)

    extension = {name: scores[name] for name in (BASELINE, *EXTENSION_META)}
    summary, daily, ts = evaluate_factors(extension, labels, {**BASELINE_META, **EXTENSION_META})
    summary.to_csv(out / "extension_ic.csv", index=False)
    ts.to_csv(out / "extension_timeseries_ic.csv", index=False)
    timeseries_summary(ts).to_csv(out / "extension_timeseries_summary.csv", index=False)
    residual_incremental(scores, labels).to_csv(out / "extension_residual_ic.csv", index=False)
    plot_redefinition_ic(daily, summary, out / "extension_ic.png", series=EXTENSION_SERIES,
                         title="粘性净多扩展因子 · 2011–2021")

    crowd = {name: scores[name] for name in CROWDING_META}
    c_summary, c_daily, c_ts = evaluate_factors(crowd, {h: labels[h] for h in (1, 5)}, CROWDING_META)
    c_ts_summary = timeseries_summary(c_ts)
    c_summary.to_csv(out / "crowding_ic.csv", index=False)
    c_ts.to_csv(out / "crowding_timeseries_ic.csv", index=False)
    c_ts_summary.to_csv(out / "crowding_timeseries_summary.csv", index=False)
    plot_crowding(c_daily, c_summary, c_ts_summary, out / "crowding_ic.png")

    (out / "followup_protocol.json").write_text(json.dumps(FOLLOWUP_PROTOCOL, ensure_ascii=False, indent=2))
    followup = {name: scores[name] for name in (BASELINE, *FOLLOWUP_META)}
    f_summary, _, f_ts = evaluate_factors(followup, labels, {**BASELINE_META, **FOLLOWUP_META})
    f_summary.to_csv(out / "followup_ic.csv", index=False)
    f_ts.to_csv(out / "followup_timeseries_ic.csv", index=False)
    timeseries_summary(f_ts).to_csv(out / "followup_timeseries_summary.csv", index=False)
    residual_incremental(scores, labels, tuple(FOLLOWUP_META)).to_csv(out / "followup_residual_ic.csv", index=False)

    (out / "combo_protocol.json").write_text(json.dumps(COMBO_PROTOCOL, ensure_ascii=False, indent=2))
    combo = {BASELINE: scores[BASELINE], **{name: scores[name] for name in COMBO_PARTS},
             COMBO: rank_composite(*(scores[name] for name in COMBO_PARTS))}
    k_summary, k_daily, k_ts = evaluate_factors(combo, labels, {**BASELINE_META, **COMBO_META})
    k_summary.to_csv(out / "combo_ic.csv", index=False)
    k_ts.to_csv(out / "combo_timeseries_ic.csv", index=False)
    timeseries_summary(k_ts).to_csv(out / "combo_timeseries_summary.csv", index=False)
    diagnostics = combo_diagnostics(combo, k_daily)
    diagnostics.to_csv(out / "combo_diagnostics.csv", index=False)
    plot_redefinition_ic(k_daily, k_summary, out / "combo_ic.png", series=COMBO_SERIES,
                         title="持久长期粘性身份与粘性席位持仓流的等权组合 · 2011–2021", bar_legend="upper right")

    columns = ["factor", "horizon", "mean_ic", "p", "q_primary", "q_all", "days"]
    print(diagnostics.round(4).to_string(index=False))
    for table in (summary, f_summary, k_summary):
        print(table.loc[table.period.eq("2011-2021") & table.horizon.eq(10), columns].round(5).to_string(index=False))
    print(c_summary.loc[c_summary.period.eq("2011-2021"), columns].round(5).to_string(index=False))


def run_smoothing_study(out: Path, jobs: int) -> None:
    """仅研究预测内容；共同样本和检验族固定，不构造仓位。"""
    C.assert_research_only(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(SMOOTHING_PROTOCOL, ensure_ascii=False, indent=2))
    scores, labels = build_inputs(jobs, extra_names=("STICKY_FLOW10",))
    parts = ("STICKY_PERSIST_IDENTITY", "STICKY_FLOW5", "STICKY_FLOW10")
    common = np.logical_and.reduce([np.isfinite(scores[name]) for name in parts])
    matched = {name: scores[name].where(common) for name in parts}
    matched["STICKY_PI_FLOW5"] = rank_composite(matched[parts[0]], matched[parts[1]])
    matched["STICKY_PI_FLOW10"] = rank_composite(matched[parts[0]], matched[parts[2]])
    metadata = {name: dict(sign=1, family="reference" if name == parts[0] else
                           "combo_smoothing" if name.startswith("STICKY_PI_") else "flow_smoothing")
                for name in matched}
    summary, daily, ts = evaluate_factors(matched, labels, metadata)
    summary.to_csv(out / "factor_ic.csv", index=False)
    ts.to_csv(out / "timeseries_ic.csv", index=False)
    rows = []
    for h in HORIZONS:
        for period in PERIODS:
            part = daily.loc[daily.horizon.eq(h) & daily.period.eq(period)].pivot(
                index="date", columns="factor", values="ic")
            stats = mean_inference(part.STICKY_PI_FLOW10 - part.STICKY_PI_FLOW5, max(20, 2 * h), MIN_DAYS)
            rows.append(dict(horizon=h, period=period, primary=(h == 10 and period == "2011-2021"), **stats))
    pd.DataFrame(rows).to_csv(out / "paired_ic_gain.csv", index=False)
    series = ((parts[0], "身份（同样本）", "#7a8a96"),
              ("STICKY_PI_FLOW5", "身份 + 5 日流", "#B76569"),
              ("STICKY_PI_FLOW10", "身份 + 10 日流", "#147D89"))
    plot_redefinition_ic(daily, summary, out / "comparison.png", series=series,
                         title="身份 + 持仓流平滑窗口 · 研究期 5 日 / 10 日同样本探索")
    print(summary.loc[summary.period.eq("2011-2021") & summary.horizon.eq(10),
                      ["factor", "mean_ic", "p", "q_primary", "days"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--flow-smoothing", action="store_true", help="独立的研究期 5/10 日同样本比较")
    args = parser.parse_args()
    if args.flow_smoothing:
        run_smoothing_study(C.study_dir("member_flow_smoothing"), args.jobs)
    else:
        run(C.study_dir(OUT_NAME), args.jobs)


if __name__ == "__main__":
    main()

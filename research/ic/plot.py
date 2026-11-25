"""绘制主要因子 IC 与多重检验结果。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from research.ic.study import _bh


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("符号检验样本数无效")
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2 ** trials)


def _factor_results(xs_dir: Path, member_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paper_path = xs_dir / "paper_ic.csv"
    five_day_path = xs_dir / "five_day_ic.csv"
    member_path = member_dir / "ic_by_symbol.csv"
    for path in (paper_path, five_day_path, member_path):
        C.assert_research_only(path)
        if not path.is_file():
            raise FileNotFoundError(f"缺少因子 IC 结果文件: {path}")

    paper = pd.read_csv(paper_path)
    five_day = pd.read_csv(five_day_path)
    member = pd.read_csv(member_path)
    next_day = paper.loc[
        paper.scope.eq("all") & paper.norm.eq("raw") & paper.horizon.eq("d1")
    ]
    five_day = five_day.loc[
        five_day.scope.eq("all") & five_day.norm.eq("raw") & five_day.window.eq("d5")
    ]
    xs_rows = []
    for horizon, table, factors in (
        ("次日截面", next_day, ["K_DS", "SLOPE_SHORT_20", "K_DB", "K_NET_SHARE",
                               "FLOW_CONTROL", "CROWD1_TAIL", "POWER_20_FILLED"]),
        ("五日截面", five_day, ["FLOW_CONTROL", "K_DS", "POWER_20_FILLED",
                                "K_NET_SHARE", "SPIDER_FREE"]),
    ):
        for factor in factors:
            row = table.loc[table.factor.eq(factor)]
            if len(row) != 1:
                raise ValueError(f"{horizon} IC 结果中 {factor} 应恰有一行，实际为 {len(row)} 行")
            value = row.iloc[0]
            xs_rows.append({
                "horizon": horizon,
                "factor": "-K_DS" if factor == "K_DS" else factor,
                "mean_ic": float(value.mean_ic),
                "q_primary": float(value.q_primary) if pd.notna(value.q_primary) else np.nan,
                "q_all": float(value.q_all),
            })

    eligible = member.loc[member.n_pairs.ge(120)].copy()
    rows = []
    for factor, label in (("K_DS", "-K_DS"), ("FLOW_CONTROL", "FLOW_CONTROL")):
        group = eligible.loc[eligible.factor.eq(factor), "signed_ic"].dropna()
        n = len(group)
        successes = int(group.gt(0).sum())
        rows.append({
            "factor": label,
            "mean_ic": float(group.mean()),
            "positive": successes,
            "n": n,
            "p": _binomial_upper_tail(successes, n),
        })
    time_series = pd.DataFrame(rows)
    time_series["q_all"] = _bh(
        eligible.groupby("factor")["signed_ic"].apply(
            lambda values: _binomial_upper_tail(int(values.gt(0).sum()), int(values.notna().sum()))
            if values.notna().any() else np.nan
        )
    ).reindex(time_series.factor.map({"-K_DS": "K_DS", "FLOW_CONTROL": "FLOW_CONTROL"})).to_numpy()
    return pd.DataFrame(xs_rows), time_series


def plot_factor_significance(xs_dir: Path, member_dir: Path, output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    cross_section, time_series = _factor_results(xs_dir, member_dir)
    palette = {
        "all": "#147D89",
        "primary": "#D18B38",
        "none": "#A7B3BC",
    }

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "text.color": "#293e4c",
        "axes.labelcolor": "#5c6d79",
        "xtick.color": "#657682",
        "ytick.color": "#657682",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    }):
        figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0, 1.05]})
        labels = {
            "-K_DS": r"$-K_{DS}$",
            "SLOPE_SHORT_20": r"$\mathrm{SLOPE\_SHORT}_{20}$",
            "K_DB": r"$K_{DB}$",
            "K_NET_SHARE": r"$K_{\mathrm{NET}}$",
            "FLOW_CONTROL": r"$\mathrm{FLOW}$",
            "CROWD1_TAIL": r"$\mathrm{CROWD1}$",
            "POWER_20_FILLED": r"$\mathrm{POWER}_{20}$",
            "SPIDER_FREE": r"$\mathrm{SPIDER}$",
        }
        for axis, horizon in zip(axes[:2], ("次日截面", "五日截面")):
            rows = cross_section.loc[cross_section.horizon.eq(horizon)].sort_values("mean_ic")
            colors = [
                palette["all"] if row.q_all < 0.05 else
                palette["primary"] if pd.notna(row.q_primary) and row.q_primary < 0.05 else
                palette["none"]
                for row in rows.itertuples()
            ]
            axis.barh(rows.factor.map(labels), rows.mean_ic, color=colors, height=.52)
            axis.axvline(0, color="#8c9ba5", linewidth=.8)
            axis.set_title(horizon, loc="left", fontsize=11, fontweight="bold", pad=10, color="#293e4c")
            axis.set_xlabel("方向调整后的平均 IC", labelpad=6)
            axis.grid(axis="x", color="#e6ecf0", linewidth=.7)
            axis.set_axisbelow(True)
            low_ic, high_ic = float(rows.mean_ic.min()), float(rows.mean_ic.max())
            span = max(high_ic - low_ic, 0.01)
            axis.set_xlim(
                min(0.0, low_ic) - (max(span * .36, .009) if low_ic < 0 else .003),
                max(0.0, high_ic) + max(span * .36, .009),
            )
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.spines["bottom"].set_color("#d7e0e6")
            axis.tick_params(axis="y", length=0, pad=6)
            for index, row in enumerate(rows.itertuples()):
                axis.annotate(
                    f"q={row.q_all:.3f}",
                    (row.mean_ic, index),
                    xytext=(5 if row.mean_ic >= 0 else -5, 0),
                    textcoords="offset points",
                    ha="left" if row.mean_ic >= 0 else "right",
                    va="center",
                    fontsize=8.2,
                    color="#52636f",
                )

        axis = axes[2]
        rows = time_series.sort_values("mean_ic")
        colors = [palette["all"] if q < 0.05 else palette["none"] for q in rows.q_all]
        axis.barh(rows.factor.map(labels), rows.mean_ic, color=colors, height=.42)
        axis.axvline(0, color="#8c9ba5", linewidth=.8)
        axis.set_title("逐品种时序", loc="left", fontsize=11, fontweight="bold", pad=10, color="#293e4c")
        axis.set_xlabel("平均逐品种 IC", labelpad=6)
        axis.grid(axis="x", color="#e6ecf0", linewidth=.7)
        axis.set_axisbelow(True)
        high_ts = max(0.0, float(rows.mean_ic.max()))
        axis.set_xlim(min(0.0, float(rows.mean_ic.min())) - 0.002, max(high_ts * 1.95, 0.045))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#d7e0e6")
        axis.tick_params(axis="y", length=0, pad=6)
        for index, row in enumerate(rows.itertuples()):
            axis.annotate(
                f"{row.positive}/{row.n} 为正，q={row.q_all:.3f}",
                (row.mean_ic, index),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.2,
                color="#52636f",
            )

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette["all"], label="全检验校正 q < 0.05"),
            plt.Rectangle((0, 0), 1, 1, color=palette["primary"], label="主检验校正 q < 0.05"),
            plt.Rectangle((0, 0), 1, 1, color=palette["none"], label="未通过相应校正"),
        ]
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                      bbox_to_anchor=(.52, .02), fontsize=9.2)
        figure.suptitle("会员持仓因子的 IC 与显著性", x=.03, y=.958, ha="left", va="top",
                        fontsize=14.5, fontweight="bold", color="#293e4c")
        figure.text(.03, .892, r"2016–2021；截面为 HAC $t$，时序为方向符号检验后的 BH $q$",
                    va="top", fontsize=9.2, color="#697c89")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="y", labelsize=9.2)
        figure.subplots_adjust(left=.135, right=.975, top=.78, bottom=.18, wspace=.54)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
        plt.close(figure)
    return output


def run(xs_dir: Path, member_dir: Path, output: Path) -> Path:
    return plot_factor_significance(xs_dir, member_dir, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制因子 IC 显著性图")
    parser.parse_args()
    output = C.study_dir("xs_position") / "factor_significance.png"
    print(f"因子显著性图写入 {run(C.study_dir('xs_position'), C.study_dir('member_ic'), output)}")


if __name__ == "__main__":
    main()

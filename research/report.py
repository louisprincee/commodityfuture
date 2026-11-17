"""由预测生成论文 Exhibit 3–8 对应的表与图。只读研究期数据集与已保存的预测。"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from research import features as F
from research import portfolio as P

BASELINES = F.PROTOCOL["portfolio"]["baselines"]
LABELS = [f"{h}d" for h in F.HORIZONS] + ["Ensemble"]
# 参考调色板（dataviz references/palette.md）
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def wide(ds: pd.DataFrame, col: str, mask: pd.Series | None = None) -> pd.DataFrame:
    sub = ds if mask is None else ds[mask]
    return sub.pivot(index="date", columns="symbol", values=col).sort_index()


def scores(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """各期限预测的截面百分位，以及八期限平均后再排序的集成得分。"""
    out = {}
    for h in F.HORIZONS:
        p = pred[pred.horizon == h].pivot(index="date", columns="symbol", values="pred")
        out[f"{h}d"] = p.rank(axis=1, pct=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 某品种当日八个期限均无预测
        stack = np.nanmean(np.stack([out[f"{h}d"].to_numpy() for h in F.HORIZONS]), axis=0)
    ens = pd.DataFrame(stack, index=out["5d"].index, columns=out["5d"].columns)
    out["Ensemble"] = ens.rank(axis=1, pct=True)
    return out


def rank_ic(score: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    s = score.rank(axis=1)
    f = fwd.reindex_like(score).where(score.notna()).rank(axis=1)
    return s.corrwith(f, axis=1, method="pearson").dropna()


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(out: Path, perf: pd.DataFrame, rets: pd.DataFrame, corr: pd.DataFrame,
            fam_imp: pd.DataFrame, expo: pd.DataFrame) -> None:
    span = f"{rets.index.min().year}–{rets.index.max().year}"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2})

    # 1. 各期限净夏普（镜像 J 形检验）
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _style(ax)
    sharpe = perf.loc[LABELS, "sharpe"]
    colors = [BLUE] * len(F.HORIZONS) + [ORANGE]
    ax.bar(range(len(LABELS)), sharpe.values, color=colors, width=0.6)
    for i, v in enumerate(sharpe.values):
        ax.text(i, v + (0.03 if v >= 0 else -0.03), f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_ylabel("Sharpe (net, 5 bp)")
    ax.set_title(f"Net Sharpe ratio by target horizon, {span} walk-forward", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "sharpe_by_horizon.png", dpi=150)
    plt.close(fig)

    # 2. 累计净收益：八个期限灰线，集成蓝线
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _style(ax)
    cum = rets.cumsum()
    for lab in LABELS[:-1]:
        ax.plot(cum.index, cum[lab], color=MUTED, linewidth=1)
    # 末端标签按数值排序后错开，避免重叠
    ends = cum[LABELS[:-1]].iloc[-1].sort_values()
    gap = 0.035 * (cum.max().max() - cum.min().min())
    placed = []
    for lab, v in ends.items():
        y = v if not placed or v - placed[-1] >= gap else placed[-1] + gap
        placed.append(y)
        ax.text(cum.index[-1], y, f" {lab}", fontsize=7, color=INK2, va="center")
    ax.plot(cum.index, cum["Ensemble"], color=BLUE, linewidth=2)
    ax.text(cum.index[0], cum["Ensemble"].max(), "Ensemble", fontsize=8, color=BLUE, va="bottom")
    ax.set_ylabel("Cumulative net return (sum)")
    ax.set_title("Long–short net returns; grey lines are single horizons", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    # 3. 组合相关性热图（Exhibit 6）
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for name, table, title, fmt in (
        ("strategy_correlation.png", corr, "Correlation of daily net long–short returns", "{:.2f}"),
        ("family_importance.png", fam_imp, "Normalized gain importance by feature family (mean over test years)", "{:.0%}"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        im = ax.imshow(table.values, cmap=cmap, aspect="auto",
                       vmin=0 if name.startswith("family") else min(0, table.values.min()), vmax=table.values.max())
        ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8, color=INK2)
        ax.set_yticks(range(table.shape[0]), table.index, fontsize=8, color=INK2)
        hi = table.values.max()
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.6 * hi else INK)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    # 4. 集成组合板块净敞口（Exhibit 8），小多图
    net = expo["net"].resample("ME").mean()
    secs = list(net.columns)
    ncol = 4
    nrow = int(np.ceil(len(secs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.0 * nrow), sharex=True, sharey=True)
    for ax, sec in zip(np.ravel(axes), secs):
        _style(ax)
        ax.fill_between(net.index, net[sec], 0, color=BLUE, alpha=0.25, linewidth=0)
        ax.plot(net.index, net[sec], color=BLUE, linewidth=1.2)
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_title(sec, fontsize=8, loc="left", color=INK)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    for ax in np.ravel(axes)[len(secs):]:
        ax.set_visible(False)
    fig.suptitle("Ensemble net exposure by sector (monthly mean of daily weights)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "sector_net_exposure.png", dpi=150)
    plt.close(fig)


def run(ds: pd.DataFrame, out: Path) -> None:
    C.assert_no_holdout_dates(ds["date"], "报告数据集")
    pred = pd.read_parquet(out / "predictions.parquet")
    imp = pd.read_csv(out / "feature_importance.csv", index_col=[0, 1])
    rets_wide = wide(ds, "ret1")
    rolls = wide(ds, "roll").eq(True)
    sc = scores(pred)
    start, end = pred["date"].min(), pred["date"].max()

    rows, daily, yearly, expo = [], {}, {}, None
    for lab, s in sc.items():
        w = P.score_to_weights(s)
        for delay, cost in ((0, P.COST_BPS), (0, 0.0), (1, P.COST_BPS)):
            bt = P.backtest(w, rets_wide, rolls, delay=delay, cost_bps=cost)
            bt = bt.loc[start:end]
            key = "net" if cost else "gross"
            m = P.metrics(bt[key], bt["turnover"])
            rows.append({"model": lab, "timing": f"delay{delay}", "cost_bps": cost, **m,
                         "avg_names": int(s.notna().sum(axis=1).mean())})
            if delay == 0 and cost:
                daily[lab] = bt["net"]
                yearly[lab] = bt["net"].groupby(bt.index.year).sum()
        if lab == "Ensemble":
            expo = P.exposures(w.loc[start:end], ds.drop_duplicates("symbol").set_index("symbol")["sector"])

    # 单因子基线：同一构建方式，信号方向事前给定
    elig = ds["eligible"] & ds["date"].between(start, end)
    for feat, sign in BASELINES.items():
        s = wide(ds, feat, elig) * sign
        w = P.score_to_weights(s)
        bt = P.backtest(w, rets_wide, rolls, delay=0).loc[start:end]
        rows.append({"model": f"baseline {'-' if sign < 0 else ''}{feat}", "timing": "delay0",
                     "cost_bps": P.COST_BPS, **P.metrics(bt["net"], bt["turnover"]),
                     "avg_names": int(s.notna().sum(axis=1).mean())})
        daily[f"baseline {feat}"] = bt["net"]
    ew = wide(ds, "ret1", elig).mean(axis=1).loc[start:end]
    rows.append({"model": "equal-weight long-only", "timing": "delay0", "cost_bps": 0.0, **P.metrics(ew)})

    perf = pd.DataFrame(rows)
    perf.to_csv(out / "performance_all.csv", index=False)
    main = perf[(perf.timing == "delay0") & (perf.cost_bps == P.COST_BPS)].set_index("model")
    main.to_csv(out / "performance.csv")
    rets = pd.DataFrame(daily)
    rets.to_csv(out / "daily_net_returns.csv")
    corr = rets[LABELS].corr()
    corr.to_csv(out / "strategy_correlation.csv")
    pd.DataFrame(yearly).to_csv(out / "annual_net_returns.csv")

    # 预测的截面秩 IC（与 Exhibit 无对应，用于判断信号是否来自预测力）
    fwd = {h: wide(ds, f"fwd{h}") for h in F.HORIZONS}
    ic_rows = []
    for lab, s in sc.items():
        hs = F.HORIZONS if lab == "Ensemble" else [int(lab[:-1])]
        for h in hs:
            ic = rank_ic(s, fwd[h])
            ic_rows.append({"model": lab, "target_horizon": h, "mean_ic": ic.mean(),
                            "ic_ir": ic.mean() / ic.std(), "share_positive": (ic > 0).mean(), "days": len(ic)})
    pd.DataFrame(ic_rows).to_csv(out / "rank_ic.csv", index=False)

    # 特征重要性（Exhibit 4）与族汇总
    imp.to_csv(out / "feature_importance.csv")
    top = []
    for h in F.HORIZONS:
        mean = imp.loc[h].mean().sort_values(ascending=False)
        for rank, (feat, v) in enumerate(mean.head(10).items(), 1):
            top.append({"horizon": f"{h}d", "rank": rank, "feature": feat, "mean_importance": v})
    pd.DataFrame(top).to_csv(out / "top10_features.csv", index=False)
    fam = imp.T.groupby(lambda f: next(x for x in F.FAMILIES if f.startswith(x) and f[len(x):].isdigit())).sum().T
    fam_imp = fam.groupby(level="horizon").mean()
    fam_imp.index = [f"{h}d" for h in fam_imp.index]
    fam_imp = fam_imp[F.FAMILIES]
    fam_imp.to_csv(out / "family_importance.csv")
    unused = (imp.groupby(level="horizon").mean() == 0).sum(axis=1)
    unused.to_csv(out / "unused_features.csv", header=["features_with_zero_importance"])

    # 特征截面百分位相关（Exhibit 3）
    feat_corr = ds.loc[ds["eligible"], F.FEATURES].corr()
    feat_corr.to_csv(out / "feature_correlation.csv")
    expo.to_csv(out / "ensemble_sector_exposure.csv")

    figures(out, main, rets.loc[:, LABELS], corr, fam_imp, expo)
    summary = {
        "test_window": [str(start.date()), str(end.date())],
        "performance_net_delay0": main.loc[LABELS, ["ann_return", "ann_vol", "sharpe", "max_drawdown",
                                                   "daily_turnover"]].round(4).to_dict(orient="index"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(main[["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "success_rate",
                "daily_turnover", "t_stat"]].round(3).to_string())

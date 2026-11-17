"""持有期与预测期限对齐的第二设定（config/research/holding_aligned.json）。只读已保存预测与研究期数据集。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config as C
from research import features as F
from research import portfolio as P
from research import report as R

PROTOCOL = json.loads((C.CONFIG_DIR / "research" / "holding_aligned.json").read_text(encoding="utf-8"))
HORIZONS = PROTOCOL["construction"]["horizons"]
VARIANTS = ("cohort", "renormalized")


def held_books(sc: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    """各期限按 H 日重叠持有的权重，及八个期限持仓的等权集成。"""
    books = {}
    for h in HORIZONS:
        books[f"{h}d"] = P.hold_cohorts(P.score_to_weights(sc[f"{h}d"]), h)
    cols = sorted(set().union(*(b.columns for b in books.values())))
    idx = books["5d"].index
    books["Ensemble"] = sum(b.reindex(index=idx, columns=cols).fillna(0.0) for b in books.values()) / len(HORIZONS)
    return {
        "cohort": books,
        "renormalized": {k: P.normalize_legs(v) for k, v in books.items()},
    }


def run(ds: pd.DataFrame, pred_dir: Path, out: Path) -> None:
    C.assert_no_holdout_dates(ds["date"], "持有期设定数据集")
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(pred_dir / "predictions.parquet")
    rets = R.wide(ds, "ret1")
    rolls = R.wide(ds, "roll").eq(True)
    start, end = pred["date"].min(), pred["date"].max()
    books = held_books(R.scores(pred))

    rows, daily, yearly = [], {}, {}
    for variant in VARIANTS:
        for lab, w in books[variant].items():
            for delay, cost in ((0, P.COST_BPS), (0, 0.0), (1, P.COST_BPS)):
                bt = P.backtest(w, rets, rolls, delay=delay, cost_bps=cost).loc[start:end]
                key = "net" if cost else "gross"
                rows.append({"variant": variant, "model": lab, "timing": f"delay{delay}", "cost_bps": cost,
                             **P.metrics(bt[key], bt["turnover"]), "avg_gmv": bt["gmv"].mean()})
                if variant == "cohort" and delay == 0 and cost:
                    daily[lab] = bt["net"]
                    yearly[lab] = bt["net"].groupby(bt.index.year).sum()
    perf = pd.DataFrame(rows)
    perf.to_csv(out / "performance_all.csv", index=False)

    # 与日度调仓结果并列（replication.json 的主口径）
    base = pd.read_csv(pred_dir / "performance.csv", index_col="model")
    main = perf[(perf.timing == "delay0") & (perf.cost_bps == P.COST_BPS)]
    cmp = pd.DataFrame({
        "daily_sharpe": base.loc[R.LABELS, "sharpe"],
        "daily_turnover": base.loc[R.LABELS, "daily_turnover"],
        **{f"{v}_sharpe": main[main.variant == v].set_index("model").loc[R.LABELS, "sharpe"] for v in VARIANTS},
        **{f"{v}_turnover": main[main.variant == v].set_index("model").loc[R.LABELS, "daily_turnover"]
           for v in VARIANTS},
    })
    cmp.to_csv(out / "comparison_with_daily.csv")
    rets_daily = pd.DataFrame(daily)
    rets_daily.to_csv(out / "daily_net_returns.csv")
    rets_daily.corr().to_csv(out / "strategy_correlation.csv")
    pd.DataFrame(yearly).to_csv(out / "annual_net_returns.csv")
    figure(out, cmp, f"{start.year}–{end.year}")
    pd.set_option("display.width", 200)
    print(main.set_index(["variant", "model"])[["ann_return", "ann_vol", "sharpe", "max_drawdown",
                                                 "daily_turnover", "avg_gmv", "t_stat"]].round(3).to_string())
    print(cmp.round(3).to_string())


def figure(out: Path, cmp: pd.DataFrame, span: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    (out / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    R._style(ax)
    x = np.arange(len(R.LABELS))
    ax.bar(x - 0.18, cmp["daily_sharpe"], width=0.34, color=R.MUTED, label="daily rebalance")
    ax.bar(x + 0.18, cmp["cohort_sharpe"], width=0.34, color=R.BLUE, label="hold H days (cohort)")
    ax.axhline(0, color=R.INK2, linewidth=0.8)
    ax.set_xticks(x, R.LABELS)
    ax.set_ylabel("Sharpe (net, 5 bp)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"Net Sharpe: daily rebalance vs holding for the forecast horizon, {span}",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "sharpe_daily_vs_held.png", dpi=150)
    plt.close(fig)

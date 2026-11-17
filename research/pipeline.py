"""复现流程：panel（日面板与特征）、train（滚动训练与预测）、report（组合、表格与图）。"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import config as C
import data as D
from research import features as F

PANEL_DIR = C.study_dir("panel")
OUT_DIR = C.study_dir("replication")


def run_panel() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = D.build_daily_panel("research")
    panel.to_parquet(PANEL_DIR / "daily_panel.parquet", index=False)
    ds, raw = F.build_dataset(panel)
    C.assert_no_holdout_dates(ds["date"], "数据集")
    ds.to_parquet(PANEL_DIR / "dataset.parquet", index=False)
    elig = ds[ds["eligible"]]
    by_year = elig.groupby(elig["date"].dt.year)["symbol"].agg(["nunique"])
    by_year["avg_daily"] = elig.groupby([elig["date"].dt.year, "date"]).size().groupby(level=0).mean().round(1)
    by_year.to_csv(PANEL_DIR / "universe_by_year.csv")
    members = elig.groupby(elig["date"].dt.year)["symbol"].unique().apply(sorted)
    (PANEL_DIR / "universe_members.json").write_text(
        json.dumps({int(k): list(v) for k, v in members.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    coverage = elig[F.FEATURES].notna().groupby(elig["date"].dt.year).mean().T.round(3)
    coverage.to_csv(PANEL_DIR / "feature_coverage_by_year.csv")
    print(by_year.to_string())
    print(f"日面板 {panel.shape}，数据集 {ds.shape}，用时 {time.time() - t0:.0f}s → {PANEL_DIR}")


def load_dataset() -> pd.DataFrame:
    path = PANEL_DIR / "dataset.parquet"
    C.assert_research_only(path)
    ds = pd.read_parquet(path)
    C.assert_no_holdout_dates(ds["date"], "数据集")
    return ds


def main(cmd: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m research {cmd}")
    parser.add_argument("--horizons", nargs="+", type=int, default=F.HORIZONS)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if cmd in ("panel", "all"):
        run_panel()
    if cmd in ("train", "all"):
        from research import model
        model.run(load_dataset(), args.horizons, OUT_DIR, jobs=args.jobs)
    if cmd in ("report", "all"):
        from research import report
        report.run(load_dataset(), OUT_DIR)
    if cmd in ("holding", "all"):
        from research import holding
        holding.run(load_dataset(), OUT_DIR, C.study_dir("holding_aligned"))
    return 0

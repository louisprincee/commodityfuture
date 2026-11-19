"""研究入口：python -m research <命令>。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
import data as D
from research import portfolio as P

PANEL_DIR = C.study_dir("panel")
COMMANDS = ("panel", "member-ic", "paper-ic", "hold-five", "risk-budget")


def run_panel() -> pd.DataFrame:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel = D.build_daily_panel("research")
    C.assert_no_holdout_dates(panel["date"], "日面板")
    panel["eligible"] = P.eligibility(panel)
    panel.to_parquet(PANEL_DIR / "daily_panel.parquet", index=False)
    eligible = panel[panel["eligible"]]
    by_year = eligible.groupby(eligible["date"].dt.year)["symbol"].agg(["nunique"])
    by_year["avg_daily"] = eligible.groupby(
        [eligible["date"].dt.year, "date"]).size().groupby(level=0).mean().round(1)
    by_year.to_csv(PANEL_DIR / "universe_by_year.csv")
    members = eligible.groupby(eligible["date"].dt.year)["symbol"].unique().apply(sorted)
    (PANEL_DIR / "universe_members.json").write_text(
        json.dumps({int(year): list(symbols) for year, symbols in members.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(by_year.to_string())
    print(f"日面板 {panel.shape}，用时 {time.time() - started:.0f}s → {PANEL_DIR}")
    return panel


def load_panel() -> pd.DataFrame:
    path = PANEL_DIR / "daily_panel.parquet"
    if not path.exists():
        return run_panel()
    C.assert_research_only(path)
    panel = pd.read_parquet(path)
    C.assert_no_holdout_dates(panel["date"], "日面板")
    return panel


def main(cmd: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m research {cmd}")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if cmd == "panel":
        run_panel()
        return 0
    panel = load_panel()
    if cmd == "member-ic":
        from research import ic_study
        ic_study.run_member_ic(panel, C.study_dir("member_ic"), jobs=args.jobs)
        return 0
    out = C.study_dir("xs_position")
    if cmd == "paper-ic":
        from research import ic_study
        ic_study.run_paper_ic(panel, C.RESEARCH_FEE_PATH, out, jobs=args.jobs)
    elif cmd == "hold-five":
        from research import ic_study
        ic_study.run_hold_five(panel, C.RESEARCH_FEE_PATH, out, jobs=args.jobs)
    elif cmd == "risk-budget":
        from research import risk_budget
        risk_budget.run(panel, C.RESEARCH_FEE_PATH, out, jobs=args.jobs)
    return 0


def cli(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(f"用法: python -m research {{{'|'.join(COMMANDS)}}} [参数]")
        return 2
    return main(argv[0], argv[1:])


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))

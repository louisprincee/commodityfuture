"""研究入口：python -m research <命令>。"""
from __future__ import annotations

import argparse
import importlib
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
COMMANDS = ("check", "panel", "member-ic")
EXPERIMENTS = {
    "xs": "xs_position", "paper-ic": "paper_ic", "execution": "execution_study",
    "hold-five": "hold_study", "timing": "timing_study",
    "event-timing": "event_timing",
    "position-diagnostics": "position_diagnostics",
}


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
    elif cmd == "member-ic":
        from research import member_ic
        member_ic.run(load_panel(), C.study_dir("member_ic"), jobs=args.jobs)
    elif cmd in EXPERIMENTS:
        study = importlib.import_module(f"research.experiments.{EXPERIMENTS[cmd]}")
        study.run(load_panel(), C.RESEARCH_FEE_PATH, C.study_dir("xs_position"), jobs=args.jobs)
    return 0


def cli(argv: list[str]) -> int:
    if not argv or (argv[0] not in COMMANDS and argv[0] not in EXPERIMENTS):
        print(f"用法: python -m research {{{'|'.join(COMMANDS)}}} [参数]")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        from research import data_check
        return data_check.run()
    return main(cmd, rest)


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))

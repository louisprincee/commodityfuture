"""因子研究入口：python -m research <命令>。只计算因子与 IC，不构建持仓。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
import data as D

COMMANDS = ("panel", "member-ic", "paper-ic", "five-day-ic")


def main(cmd: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m research {cmd}")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if cmd == "panel":
        D.build_panel_cache()
        return 0
    from research import ic_study
    panel = D.load_panel()
    if cmd == "member-ic":
        ic_study.run_member_ic(panel, C.study_dir("member_ic"), jobs=args.jobs)
    elif cmd == "paper-ic":
        ic_study.run_paper_ic(panel, C.study_dir("xs_position"), jobs=args.jobs)
    elif cmd == "five-day-ic":
        ic_study.run_five_day_ic(panel, C.study_dir("xs_position"), jobs=args.jobs)
    return 0


def cli(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(f"用法: python -m research {{{'|'.join(COMMANDS)}}} [参数]")
        return 2
    return main(argv[0], argv[1:])


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))

"""因子研究入口：python -m research <命令>。只计算因子与 IC，不构建持仓。"""
from __future__ import annotations

import argparse
import sys

import config as C
import data as D

COMMANDS = ("panel", "member-ic", "paper-ic", "five-day-ic", "candidate-ic", "xs-robustness",
            "member-smoothing", "member-complementarity", "member-extensions", "plot-ic")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m research", description="商品期货因子与预测检验")
    parser.add_argument("study", choices=COMMANDS)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    cmd = args.study
    if cmd == "panel":
        D.build_panel_cache()
        return 0
    if cmd == "plot-ic":
        from research import plot_ic
        output = C.study_dir("xs_position") / "factor_significance.png"
        print(f"因子显著性图写入 {plot_ic.run(C.study_dir('xs_position'), C.study_dir('member_ic'), output)}")
        return 0
    if cmd == "candidate-ic":
        from research import economic_ic
        economic_ic.run(D.load_panel(), C.study_dir("economic_ic"))
        return 0
    if cmd == "xs-robustness":
        from research import xs_robustness
        xs_robustness.run(C.study_dir("xs_robustness"))
        return 0
    if cmd == "member-smoothing":
        from research import member_smoothing
        out = C.study_dir("member_smoothing")
        result = member_smoothing.run(out, args.jobs)
        print(f"{len(result)} 项 IC 写入 {out}")
        return 0
    if cmd == "member-complementarity":
        from research import member_complementarity
        out = C.study_dir("member_complementarity")
        result = member_complementarity.run(out, args.jobs)
        print(f"{len(result)} 项持仓条件价差写入 {out}")
        return 0
    if cmd == "member-extensions":
        from research import member_extensions
        out = C.study_dir("member_extensions")
        result = member_extensions.run(out, args.jobs)
        print(f"{len(result)} 项持仓补充因子检验写入 {out}")
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

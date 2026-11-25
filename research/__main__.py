"""因子研究入口：python -m research <命令>。只计算因子与 IC，不构建持仓。"""
from __future__ import annotations

import argparse
import sys

import config as C
import data as D

COMMANDS = ("panel", "member-ic", "paper-ic", "five-day-ic", "candidate-ic", "xs-robustness",
            "member-smoothing", "member-complementarity", "chain-ic", "plot-ic")


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
        from research.ic import plot as plot_ic
        output = C.study_dir("xs_position") / "factor_significance.png"
        print(f"因子显著性图写入 {plot_ic.run(C.study_dir('xs_position'), C.study_dir('member_ic'), output)}")
        return 0
    if cmd == "candidate-ic":
        from research.ic import economic as economic_ic
        economic_ic.run(D.load_panel(), C.study_dir("economic_ic"))
        return 0
    if cmd == "xs-robustness":
        from research.ic import robustness as xs_robustness
        xs_robustness.run(C.study_dir("xs_robustness"))
        return 0
    if cmd == "member-smoothing":
        from research.member import smoothing as member_smoothing
        out = C.study_dir("member_smoothing")
        result = member_smoothing.run(out, args.jobs)
        print(f"{len(result)} 项 IC 写入 {out}")
        return 0
    if cmd == "member-complementarity":
        from research.member import complementarity as member_complementarity
        out = C.study_dir("member_complementarity")
        result = member_complementarity.run(out, args.jobs)
        print(f"{len(result)} 项持仓条件价差写入 {out}")
        return 0
    if cmd == "chain-ic":
        from research.member import chain as member_behavior
        out = C.study_dir("member_behavior")
        result = member_behavior.run_chain_ic(out, args.jobs)
        print(result.round(4).to_string(index=False))
        return 0
    from research.ic import study as ic_study
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

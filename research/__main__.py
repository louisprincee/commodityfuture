"""研究入口：python -m research <命令>。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COMMANDS = ("check", "panel", "train", "report", "holding", "all")


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(f"用法: python -m research {{{'|'.join(COMMANDS)}}} [参数]")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        from research import data_check
        return data_check.run()
    from research import pipeline
    return pipeline.main(cmd, rest)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

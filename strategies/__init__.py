"""当前策略定义。回测账户在 backtest，因子计算在 research。"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass(frozen=True)
class Targets:
    weights: pd.DataFrame
    dominant: pd.DataFrame
    selection: pd.DataFrame | None = None

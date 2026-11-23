"""研究对照：只用粘性席位净多排名，两端各 20% 等权多空，10 日重叠持有。"""
import pandas as pd

from research.member_complementarity import load_inputs
from strategies import Targets
from strategies.sticky_core import hold_cohorts, tail_weights

STRATEGY_ID = "sticky_only"
MAIN_BOOK = "STICKY_ONLY_10D_2X"
EQUITY_CURVE_FILE = "equity_curves.csv"
HORIZON = 10
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int, *, horizon: int = HORIZON) -> Targets:
    """在流动性池且粘性净多有效的品种内按粘性净多选样。"""
    sticky, _, _, dominant = load_inputs(jobs, partition=partition, common_only=False, panel=panel)
    fresh = tail_weights(sticky.reindex(columns=sticky.columns.sort_values()))
    return Targets(hold_cohorts(fresh, horizon=horizon).loc[start:], dominant)

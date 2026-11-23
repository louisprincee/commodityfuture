"""粘性净多两端用跨品种一致性残差对称筛选。"""
import pandas as pd

from research.member_behavior import load_chain_inputs, residual_scores
from research.member_complementarity import asymmetric_selections, sticky_quintiles
from strategies import Targets
from strategies.sticky_core import hold_cohorts, tail_weights

STRATEGY_ID = "sticky_chain"
MAIN_BOOK = "STICKY_CHAIN_RESIDUAL"
EQUITY_CURVE_FILE = "equity_curves.csv"
HORIZON = 10
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20


def confirmed_weights(sticky, long_score, short_score) -> pd.DataFrame:
    """先按粘性净多分组，再在两端分别按各自的确认分数选样。"""
    long, short = asymmetric_selections(sticky_quintiles(sticky), long_score, short_score)
    return tail_weights(long.astype(float).where(long | short), tail=0.5, min_names=2)


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int, *, horizon: int = HORIZON) -> Targets:
    sticky, chain, dominant = load_chain_inputs(panel, jobs, partition=partition)
    residual = residual_scores(chain, sticky)
    fresh = confirmed_weights(sticky.where(residual.notna()), residual, residual)
    return Targets(hold_cohorts(fresh, horizon=horizon).loc[start:], dominant)

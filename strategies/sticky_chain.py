"""粘性净多两端用跨品种一致性残差对称筛选。"""
import pandas as pd

from research.member_behavior import load_chain_inputs, residual_scores
from strategies import Targets
from strategies.common import confirmed_weights, hold_cohorts

STRATEGY_ID = "sticky_chain"
MAIN_BOOK = "STICKY_CHAIN_RESIDUAL"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    sticky, chain, dominant = load_chain_inputs(panel, jobs, partition=partition)
    residual = residual_scores(chain, sticky)
    fresh = confirmed_weights(sticky.where(residual.notna()), residual)
    return Targets(hold_cohorts(fresh).loc[start:], dominant)

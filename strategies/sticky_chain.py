"""粘性净多两端各取 20% 候选，再用跨品种一致性残差筛到约 10%。"""
import pandas as pd

from research.member.chain import load_chain_inputs, residual_scores
from strategies import Targets
from strategies.common import confirmed_weights
from strategies.common import hold_cohorts

TAIL_SHARE = 0.10
STRATEGY_ID = "sticky_chain"
MAIN_BOOK = "STICKY_CHAIN_RESIDUAL"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    sticky, chain, dominant = load_chain_inputs(panel, jobs, partition=partition, start=start)
    residual = residual_scores(chain, sticky)
    fresh = confirmed_weights(sticky.where(residual.notna()), residual, tail_share=TAIL_SHARE)
    return Targets(hold_cohorts(fresh).loc[start:], dominant)

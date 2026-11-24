"""会员持仓主策略：只用粘性席位净多排名，两端各 10% 等权多空，10 日重叠持有。"""
import pandas as pd

from research.member_complementarity import load_seat_inputs
from strategies import Targets
from strategies.common import hold_cohorts, tail_weights

TAIL_SHARE = 0.10
STRATEGY_ID = "sticky_only"
MAIN_BOOK = "STICKY_ONLY"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    """在流动性池且粘性净多有效的品种内按粘性净多选样。"""
    sticky, _, dominant = load_seat_inputs(panel, jobs, partition=partition)
    fresh = tail_weights(sticky, tail_share=TAIL_SHARE)
    return Targets(hold_cohorts(fresh).loc[start:], dominant)

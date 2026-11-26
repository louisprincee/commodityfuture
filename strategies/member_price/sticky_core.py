"""STICKY_CORE：基差动量、低偏度与粘性席位净多头占比的三因子截面组合，两端等权多空与 10 日重叠持有。"""
import pandas as pd

from strategies import Targets
from strategies.member_price.common import RANKED_FACTORS, ranked_targets

STRATEGY_ID = "sticky_core"
MAIN_BOOK = "STICKY_CORE"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    return ranked_targets(panel, start, partition, jobs, RANKED_FACTORS)

"""STICKY_CORE_H20：STICKY_CORE 的持有期变种，只把 10 日重叠持有改为 20 日。"""
import pandas as pd

from strategies import Targets
from strategies.member_price.common import RANKED_FACTORS, ranked_targets

STRATEGY_ID = "sticky_core_h20"
MAIN_BOOK = "STICKY_CORE_H20"
EQUITY_CURVE_FILE = "equity_curves.csv"
HOLD = 20


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    return ranked_targets(panel, start, partition, jobs, RANKED_FACTORS, horizon=HOLD)

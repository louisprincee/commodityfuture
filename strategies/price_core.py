"""研究对照：CORE 去掉粘性净多，只用基差动量和负向偏度打分。

候选池、选样、10 日重叠持有、2X 乘数和 20% 单品种上限都与 STICKY_CORE 相同，
候选池仍要求粘性净多有效，所以两者的差异只来自打分是否包含粘性净多。只运行研究期。
"""
import pandas as pd

from strategies import Targets
from strategies import sticky_core as core

STRATEGY_ID = "price_core"
MAIN_BOOK = "PRICE_CORE"
EQUITY_CURVE_FILE = "equity_curves.csv"
SCORE_FACTORS = ("BASIS_MOM_242", "NEG_SKEW_242")
HORIZON = core.HORIZON
TARGET_MULTIPLIER = core.TARGET_MULTIPLIER
NAME_CAP = core.NAME_CAP


def build_targets(panel: pd.DataFrame, bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int, *, horizon: int = HORIZON) -> Targets:
    targets = core.build_targets(panel, bars, start, partition, jobs, horizon=horizon,
                                 score_factors=SCORE_FACTORS)
    return Targets(targets.weights, targets.dominant, selection=targets.selection)

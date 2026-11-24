"""研究对照：CORE 去掉粘性净多，只用基差动量和负向偏度打分。

候选池、选样、10 日重叠持有、2X 乘数和 20% 单品种上限都与 STICKY_CORE 相同，
候选池仍要求粘性净多有效，所以两者的差异只来自打分是否包含粘性净多。只运行研究期。
"""
import pandas as pd

from strategies import Targets
from strategies.common import PRICE_FACTORS, ranked_targets

STRATEGY_ID = "price_core"
MAIN_BOOK = "PRICE_CORE"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    if partition != "research":
        raise ValueError("价格因子对照只运行研究期")
    return ranked_targets(panel, start, partition, jobs, PRICE_FACTORS)

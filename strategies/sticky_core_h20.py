"""STICKY_CORE_H20：STICKY_CORE 的持有期变种，只把 10 日重叠持有改为 20 日。

因子、合成、两端各 10% 选样、目标权重乘数 2 和单品种上限 20% 均与 STICKY_CORE 相同。
持有期是在看过研究期 5/10/20 日合成 IC 之后改的，属于事后新规格，单独命名、单独报告。
2026-11-25 按用户要求在规则冻结后读取 2022–2025 样本外一次，读取后不再修改任何参数。
"""
import pandas as pd
from strategies import Targets
from strategies.common import RANKED_FACTORS, ranked_targets

STRATEGY_ID = "sticky_core_h20"
MAIN_BOOK = "STICKY_CORE_H20"
EQUITY_CURVE_FILE = "equity_curves.csv"
HOLD = 20


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    return ranked_targets(panel, start, partition, jobs, RANKED_FACTORS, horizon=HOLD)

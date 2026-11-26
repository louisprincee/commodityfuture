"""会员持仓主策略：只用粘性席位净多排名，两端各 10% 等权多空，10 日重叠持有。"""
import pandas as pd

from research.factors.member import load_sticky_inputs
from strategies import Targets
from strategies.common import hold_cohorts, tail_weights

TAIL_SHARE = 0.10
STRATEGY_ID = "sticky_only"
MAIN_BOOK = "STICKY_ONLY"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    """在流动性池且粘性净多有效的品种内按粘性净多选样。"""
    sticky, dominant = load_sticky_inputs(panel, start, partition, jobs)
    fresh = tail_weights(sticky, tail_share=TAIL_SHARE)
    selection = pd.DataFrame({
        "candidates": sticky.notna().sum(axis=1),
        "long_slots": fresh.gt(0).sum(axis=1),
        "short_slots": fresh.lt(0).sum(axis=1),
        "fresh_gross": fresh.abs().sum(axis=1),
        "target_gross": hold_cohorts(fresh).abs().sum(axis=1),
    }).rename_axis("date").reset_index()
    return Targets(hold_cohorts(fresh).loc[start:], dominant, selection)

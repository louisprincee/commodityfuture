"""研究对照：持仓分端规则单独成书，不用价格因子，10 日重叠持有。"""
import pandas as pd

from research.member_complementarity import MIN_NAMES, load_inputs
from strategies import Targets
from strategies.sticky_core import hold_cohorts
from strategies.sticky_price_blend import flow_share_daily_weights

STRATEGY_ID = "sticky_split"
MAIN_BOOK = "STICKY_SPLIT_10D_2X"
EQUITY_CURVE_FILE = "equity_curves.csv"
HORIZON = 10
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int, *, horizon: int = HORIZON) -> Targets:
    """多头取粘性净多前 20% 中五日持仓流高的一半，空头取后 20% 中席位份额低的一半。"""
    sticky, flow, size, dominant = load_inputs(jobs, partition=partition, common_only=False, panel=panel)
    columns = sticky.columns.sort_values()
    sticky, flow, size = (frame.reindex(columns=columns) for frame in (sticky, flow, size))
    common = sticky.notna() & flow.notna() & size.notna()
    common = common & common.sum(axis=1).ge(MIN_NAMES).to_numpy()[:, None]
    fresh = flow_share_daily_weights(sticky.where(common), flow.where(common), size.where(common))
    return Targets(hold_cohorts(fresh, horizon=horizon).loc[start:], dominant)

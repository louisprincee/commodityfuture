"""冻结的研究备选。持仓与价格各自选样，再以 50/50 合并目标权重。"""
import pandas as pd

from research.member_complementarity import MIN_NAMES, asymmetric_selections, load_inputs, sticky_quintiles
from research.xs_robustness import load_price_factors
from strategies import Targets
from strategies.sticky_core import hold_cohorts, tail_weights

STRATEGY_ID = "sticky_price_blend"
MAIN_BOOK = "STICKY_PRICE_BLEND_10D_2X"
EQUITY_CURVE_FILE = "equity_curves.csv"
HORIZON = 10
TARGET_MULTIPLIER = 2.0
NAME_CAP = 0.20


def flow_share_daily_weights(sticky, flow, size) -> pd.DataFrame:
    long, short = asymmetric_selections(sticky_quintiles(sticky), flow, size)
    return tail_weights(long.astype(float).where(long | short), tail=0.5, min_names=2)


def daily_target_weights(sticky, flow, size, basis, neg_skew) -> pd.DataFrame:
    """以两个子组合共同可用的品种为池，对冲后的名义权重保留净额。"""
    common = sticky.notna() & flow.notna() & size.notna() & basis.notna() & neg_skew.notna()
    common = common & common.sum(axis=1).ge(MIN_NAMES).to_numpy()[:, None]
    holding_weights = flow_share_daily_weights(sticky.where(common), flow.where(common), size.where(common))
    price_score = (basis.where(common).rank(axis=1, pct=True) + neg_skew.where(common).rank(axis=1, pct=True)) / 2
    return (holding_weights + tail_weights(price_score)) / 2


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int, *, horizon: int = HORIZON) -> Targets:
    """构造以共同候选池净额合并的日频权重。"""
    sticky, flow, size, dominant = load_inputs(jobs, partition=partition, common_only=False, panel=panel)
    columns = sticky.columns.sort_values()
    sticky, flow, size = (frame.reindex(columns=columns) for frame in (sticky, flow, size))
    basis, neg_skew = load_price_factors(sticky.index, columns, partition=partition, panel=panel)
    fresh = daily_target_weights(sticky, flow, size, basis, neg_skew)
    return Targets(hold_cohorts(fresh, horizon=horizon).loc[start:], dominant)

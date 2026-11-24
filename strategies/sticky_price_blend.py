"""一致性残差持仓组合和价格组合各占一半，同品种相反仓位相抵。"""
import pandas as pd

from research.member_behavior import load_chain_inputs, residual_scores
from research.member_complementarity import MIN_NAMES
from research.xs_robustness import load_price_factors
from strategies import Targets
from strategies.common import confirmed_weights, hold_cohorts, tail_weights

TAIL_SHARE = 0.10
STRATEGY_ID = "sticky_price_blend"
MAIN_BOOK = "STICKY_PRICE_BLEND_RESIDUAL"
EQUITY_CURVE_FILE = "equity_curves.csv"


def blend_weights(sticky, residual, basis, neg_skew) -> pd.DataFrame:
    """在共同池选样，保留两套目标权重相抵后的净额。"""
    common = residual.notna()
    holding_weights = confirmed_weights(sticky.where(common), residual, tail_share=TAIL_SHARE)
    price_score = (basis.where(common).rank(axis=1, pct=True) + neg_skew.where(common).rank(axis=1, pct=True)) / 2
    return (holding_weights + tail_weights(price_score, tail_share=TAIL_SHARE)) / 2


def daily_target_weights(sticky, basis, neg_skew, chain) -> pd.DataFrame:
    prices_valid = basis.notna() & neg_skew.notna()
    residual = residual_scores(chain.where(prices_valid), sticky.where(prices_valid))
    residual = residual.where(residual.notna().sum(axis=1).ge(MIN_NAMES), axis=0)
    return blend_weights(sticky, residual, basis, neg_skew)


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    sticky, chain, dominant = load_chain_inputs(panel, jobs, partition=partition)
    basis, neg_skew = load_price_factors(sticky.index, sticky.columns, partition=partition, panel=panel)
    fresh = daily_target_weights(sticky, basis, neg_skew, chain)
    return Targets(hold_cohorts(fresh).loc[start:], dominant)

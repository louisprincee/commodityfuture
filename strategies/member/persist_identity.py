"""持久粘性身份：按 STICKY_PERSIST_IDENTITY 排名，仅要求自身分数有效，不以持仓流或基线因子筛样本。

2026-11-26 用户指定的独立样本规格；不同于旧组合实验的 STICKY_PI_REF。
"""
import numpy as np
import pandas as pd

from research.factors.member import load_extension_inputs, rank_composite
from strategies import Targets
from strategies.common import HORIZON, TAIL_SHARE, hold_cohorts, tail_weights

STRATEGY_ID = "persist_identity"
MAIN_BOOK = "PERSIST_IDENTITY"
LABEL = "持久粘性身份"
EQUITY_CURVE_FILE = "equity_curves.csv"
FACTOR = "STICKY_PERSIST_IDENTITY"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    signals, dominant = load_extension_inputs(panel, start, jobs, partition=partition, names=(FACTOR,))
    return _targets(signals[FACTOR], dominant, start)


def _targets(score: pd.DataFrame, dominant: pd.DataFrame, start: pd.Timestamp) -> Targets:
    common = np.isfinite(score)
    fresh = tail_weights(score.where(common), TAIL_SHARE)
    selection = pd.DataFrame({
        "candidates": common.sum(axis=1),
        "long_slots": fresh.gt(0).sum(axis=1),
        "short_slots": fresh.lt(0).sum(axis=1),
        "sample_hash": pd.util.hash_pandas_object(common, index=True),
        "fresh_gross": fresh.abs().sum(axis=1),
    }).rename_axis("date").reset_index()
    return Targets(hold_cohorts(fresh, HORIZON).loc[start:], dominant, selection)


def build_smoothing_study(panel: pd.DataFrame, start: pd.Timestamp, jobs: int) -> dict[str, Targets]:
    """用户授权的研究期探索；三个账本同样本，不注册为现役策略。"""
    names = (FACTOR, "STICKY_FLOW5", "STICKY_FLOW10")
    signals, dominant = load_extension_inputs(panel, start, jobs, names=names)
    common = np.logical_and.reduce([np.isfinite(signals[name]) for name in names])
    identity, flow5, flow10 = (signals[name].where(common) for name in names)
    scores = {"IDENTITY_MATCHED": identity,
              "IDENTITY_FLOW5": rank_composite(identity, flow5),
              "IDENTITY_FLOW10": rank_composite(identity, flow10)}
    return {name: _targets(score, dominant, start) for name, score in scores.items()}

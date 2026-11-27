"""持久粘性身份 + 持仓流否决：按 STICKY_PERSIST_IDENTITY 取两端 10%，5 日持仓流只否决开仓。

2026-11-27 用户指定为现役策略。规则与 GATE_PROTOCOL 中的 IDENTITY_VETO 账本相同，
在研究期四个门控账本的结果之后选定，因此研究期结果是选择后的样本内证据。
候选与 PERSIST_IDENTITY 相同：只要求身份分数有效；持仓流缺失时不否决，空出的名额不补。
"""
import numpy as np
import pandas as pd

from research.factors.member import load_extension_inputs
from strategies import Targets
from strategies.common import HORIZON, gate_weights, hold_cohorts
from strategies.member.persist_identity import FACTOR, GATE_FLOW, GATE_VETO

STRATEGY_ID = "persist_identity_flow_veto"
MAIN_BOOK = "PERSIST_IDENTITY_FLOW_VETO"
LABEL = "持久身份 + 持仓流否决"
EQUITY_CURVE_FILE = "equity_curves.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    signals, dominant = load_extension_inputs(panel, start, jobs, partition=partition, names=(FACTOR, GATE_FLOW))
    return _targets(signals[FACTOR], signals[GATE_FLOW], dominant, start)


def _targets(score: pd.DataFrame, flow: pd.DataFrame, dominant: pd.DataFrame, start: pd.Timestamp) -> Targets:
    fresh = gate_weights(score, flow, veto=True, buffer=False, veto_share=GATE_VETO)
    selection = pd.DataFrame({
        "candidates": np.isfinite(score).sum(axis=1),
        "long_slots": fresh.gt(0).sum(axis=1),
        "short_slots": fresh.lt(0).sum(axis=1),
        "fresh_gross": fresh.abs().sum(axis=1),
    }).rename_axis("date").reset_index()
    return Targets(hold_cohorts(fresh, HORIZON).loc[start:], dominant, selection)

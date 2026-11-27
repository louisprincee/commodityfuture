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
CONTROL_BOOK = "PERSIST_IDENTITY"
OOS_PROTOCOL = {
    "declared": "2026-11-27, at the user's explicit request, before any 2022-2025 result of this strategy was computed",
    "rule": "Frozen as committed in f2d9627: STICKY_PERSIST_IDENTITY 10% tails, STICKY_FLOW5 veto at percentile "
            f"{GATE_VETO:.2f} / {1 - GATE_VETO:.2f}, missing flow never vetoes, vetoed slots not refilled, "
            f"{HORIZON}-session overlapping cohorts, target multiplier 2, unchanged costs and non-delivery execution.",
    "window": "Independent OOS account 2022-01-01..2025-12-31 with research-period warm-up, as for the other active "
              "strategies; the 2011-2025 continuous account is descriptive only.",
    "primary": f"One test: OOS daily net return {MAIN_BOOK} minus {CONTROL_BOOK}, one tick, two-sided HAC 20 lags, "
               "alpha .05. Both books come from the same OOS run.",
    "secondary": "Descriptive only: each book's net annual return, Sharpe, max drawdown, turnover; volatility-matched "
                 "gain; turnover ratio; two ticks saved, not reported in tables.",
    "caveats": "The veto rule was chosen after the research-period gate study. The PERSIST_IDENTITY and 1:1 "
               "identity+flow OOS results were read before this declaration.",
    "no_revision": "After reading, no change to signs, windows, veto percentile, slot rules, holding days or leverage.",
}


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

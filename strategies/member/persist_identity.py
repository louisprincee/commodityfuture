"""持久粘性身份：按 STICKY_PERSIST_IDENTITY 排名，仅要求自身分数有效，不以持仓流或基线因子筛样本。

2026-11-26 用户指定的独立样本规格；不同于旧组合实验的 STICKY_PI_REF。
"""
import numpy as np
import pandas as pd

from research.factors.member import load_extension_inputs, rank_composite
from strategies import Targets
from strategies.common import HORIZON, TAIL_SHARE, gate_weights, hold_cohorts, tail_weights

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


GATE_FLOW = "STICKY_FLOW5"
GATE_BAND = 0.20
GATE_VETO = 0.20
GATE_PROTOCOL = {
    "declared": "2026-11-27, at the user's request, before any gate result was computed",
    "status": "Exploratory research-only account experiment; not an active strategy; no OOS access. "
              "Declared after the identity and 1:1 identity+flow OOS results had been read.",
    "window": "2011-01-04..2021-12-31; 2010 warm-up; research partition only",
    "sample": "Candidates are names with finite STICKY_PERSIST_IDENTITY, exactly as PERSIST_IDENTITY. "
              "STICKY_FLOW5 never shrinks the sample: a missing flow value never vetoes.",
    "flow_percentile": "Cross-sectional percentile rank (pct=True, ascending) of STICKY_FLOW5 among candidates "
                       "with finite flow; no veto on a day with fewer than 8 finite flow values.",
    "veto": "A long candidate with flow percentile <= 0.20, or a short candidate with flow percentile >= 0.80, "
            "is not opened. Vetoed slots are not refilled from outside the 10% tail.",
    "buffer": "Each day's cohort replaces the cohort opened 10 sessions earlier. An incumbent name of the expiring "
              "cohort is kept in the new cohort while its identity rank stays within the top (bottom) "
              "ceil(0.20 n) and, in books with the veto, it is not vetoed. Remaining slots up to ceil(0.10 n) "
              "per leg are filled with the best-ranked non-vetoed names inside the 10% tail.",
    "weights": "Equal weight within each leg, 0.5 gross per leg; a cohort with an empty leg is flat "
               "(existing equal_leg_weights rule). 10 overlapping cohorts, next-open execution, "
               "target multiplier 2, unchanged costs and non-delivery execution.",
    "books": "IDENTITY_CONTROL (no veto, no buffer; equals PERSIST_IDENTITY), IDENTITY_VETO, IDENTITY_BUFFER, "
             "IDENTITY_GATE (veto + buffer)",
    "primary": "One test: full-period net daily return IDENTITY_GATE minus IDENTITY_CONTROL, one tick, "
               "two-sided HAC 20 lags, alpha .05.",
    "design_condition": "IDENTITY_GATE mean daily turnover must not exceed IDENTITY_CONTROL; reported beside the "
                        "primary test. A gate that raises turnover fails its design goal whatever its return.",
    "secondary": "Descriptive only: VETO and BUFFER minus CONTROL; 2011-2015 and 2016-2021; gross versus cost "
                 "decomposition; two ticks saved, not reported in tables.",
    "fixed_parameters": {"flow": GATE_FLOW, "veto_percentile": GATE_VETO, "buffer_band": GATE_BAND,
                         "tail_share": TAIL_SHARE, "holding_days": HORIZON},
}


def build_gate_study(panel: pd.DataFrame, start: pd.Timestamp, jobs: int) -> dict[str, Targets]:
    """用户授权的研究期探索（GATE_PROTOCOL）；不注册为现役策略，不读 OOS。"""
    signals, dominant = load_extension_inputs(panel, start, jobs, names=(FACTOR, GATE_FLOW))
    score, flow = signals[FACTOR], signals[GATE_FLOW]
    books = {}
    for book, veto, buffer in (("IDENTITY_CONTROL", False, False), ("IDENTITY_VETO", True, False),
                               ("IDENTITY_BUFFER", False, True), ("IDENTITY_GATE", True, True)):
        fresh = gate_weights(score, flow, veto=veto, buffer=buffer, veto_share=GATE_VETO, band_share=GATE_BAND)
        selection = pd.DataFrame({
            "candidates": np.isfinite(score).sum(axis=1),
            "long_slots": fresh.gt(0).sum(axis=1),
            "short_slots": fresh.lt(0).sum(axis=1),
            "fresh_gross": fresh.abs().sum(axis=1),
        }).rename_axis("date").reset_index()
        books[book] = Targets(hold_cohorts(fresh, HORIZON).loc[start:], dominant, selection)
    return books


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

"""固定的纯会员增量试验：同样本 Sticky、Flow 与两阶段确认，保留完整样本参考。"""
import numpy as np
import pandas as pd

from research.factors.member import load_sticky_flow_inputs
from strategies import Targets
from strategies.common import HORIZON, MIN_NAMES, TAIL_SHARE, confirmed_weights, hold_cohorts, tail_weights

STICKY_BOOK = "STICKY_MATCHED"
FLOW_BOOK = "FLOW_MATCHED"
COMBINED_BOOK = "STICKY_FLOW_MATCHED"
REFERENCE_BOOK = "STICKY_FULL_REFERENCE"


def build_books(panel, start, jobs):
    sticky, flow, residual, dominant = load_sticky_flow_inputs(panel, start, jobs)
    common = np.isfinite(sticky) & np.isfinite(flow) & np.isfinite(residual)
    matched_sticky, matched_flow = sticky.where(common), flow.where(common)
    fresh = {
        STICKY_BOOK: tail_weights(matched_sticky),
        FLOW_BOOK: tail_weights(matched_flow),
        COMBINED_BOOK: confirmed_weights(matched_sticky, residual.where(common),
                                         tail_share=TAIL_SHARE, min_names=MIN_NAMES),
        REFERENCE_BOOK: tail_weights(sticky),
    }
    books = {}
    for name, values in fresh.items():
        mask = sticky.notna() if name == REFERENCE_BOOK else common
        candidates = mask.sum(axis=1)
        selection = pd.DataFrame({"candidates": candidates, "long_slots": values.gt(0).sum(axis=1),
                                  "short_slots": values.lt(0).sum(axis=1),
                                  "sample_hash": pd.util.hash_pandas_object(mask, index=True),
                                  "fresh_gross": values.abs().sum(axis=1)}).rename_axis("date").reset_index()
        books[name] = Targets(hold_cohorts(values, HORIZON).loc[start:], dominant, selection)
    return books

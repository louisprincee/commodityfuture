"""研究期实验：脱离成交排名的持久粘性席位重定义（STICKY_PERSIST / PURE）与排名缓冲带对照。"""
import numpy as np
import pandas as pd

from research.factors.member import load_persist_inputs
from strategies import Targets
from strategies.common import HORIZON, TAIL_SHARE, buffered_tail_weights, hold_cohorts, tail_weights

BASELINE_BOOK = "STICKY_ONLY_REF"
BASELINE_BUF_BOOK = "STICKY_ONLY_BUF"
PERSIST_BOOK = "STICKY_PERSIST"
PERSIST_PURE_BOOK = "STICKY_PERSIST_PURE"
PERSIST_BUF_BOOK = "STICKY_PERSIST_BUF"


def build_books(panel: pd.DataFrame, start: pd.Timestamp, jobs: int) -> dict[str, Targets]:
    """在 2011–2021 研究期同一流动性池与执行成本下对比粘性重定义与排名缓冲带。"""
    signals, dominant = load_persist_inputs(panel, start, jobs)
    base = signals["STICKY_NET_SHARE"]
    persist = signals["STICKY_PERSIST"]
    pure = signals["STICKY_PERSIST_PURE"]

    fresh = {
        BASELINE_BOOK: tail_weights(base, TAIL_SHARE),
        BASELINE_BUF_BOOK: buffered_tail_weights(base, enter_share=TAIL_SHARE, exit_share=0.20),
        PERSIST_BOOK: tail_weights(persist, TAIL_SHARE),
        PERSIST_PURE_BOOK: tail_weights(pure, TAIL_SHARE),
        PERSIST_BUF_BOOK: buffered_tail_weights(persist, enter_share=TAIL_SHARE, exit_share=0.20),
    }
    raw_scores = {
        BASELINE_BOOK: base,
        BASELINE_BUF_BOOK: base,
        PERSIST_BOOK: persist,
        PERSIST_PURE_BOOK: pure,
        PERSIST_BUF_BOOK: persist,
    }
    books = {}
    for name, values in fresh.items():
        mask = np.isfinite(raw_scores[name])
        selection = pd.DataFrame({
            "candidates": mask.sum(axis=1),
            "long_slots": values.gt(0).sum(axis=1),
            "short_slots": values.lt(0).sum(axis=1),
            "sample_hash": pd.util.hash_pandas_object(mask, index=True),
            "fresh_gross": values.abs().sum(axis=1),
        }).rename_axis("date").reset_index()
        books[name] = Targets(hold_cohorts(values, HORIZON).loc[start:], dominant, selection)
    return books

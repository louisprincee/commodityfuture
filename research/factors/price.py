"""基差动量和负向偏度。策略与截面检验共用同一实现。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import data as D


def basis_momentum(ret1: pd.DataFrame, ret2: pd.DataFrame, window: int = 242) -> pd.DataFrame:
    """主力与次主力各自同合约日收益的 window 日复合收益之差；任一侧不满窗为缺失。"""
    def compound(r: pd.DataFrame) -> pd.DataFrame:
        return np.expm1(np.log1p(r).rolling(window, min_periods=window).sum())
    return compound(ret1) - compound(ret2)


def rolling_skew(ret1: pd.DataFrame, window: int = 242, min_periods: int = 200) -> pd.DataFrame:
    return ret1.rolling(window, min_periods=min_periods).skew()


def load_price_factors(index: pd.Index, columns: pd.Index, *, partition: str = "research",
                       panel: pd.DataFrame | None = None):
    """在完整预热日历上计算固定的基差动量和负偏度，再对齐到评价品种日。"""
    if partition == "research":
        panel = D.load_panel() if panel is None else panel
        C.assert_no_holdout_dates(panel["date"], "价格因子研究期日面板")
    elif partition == "oos":
        C.assert_oos_dates(index, "价格因子样本外日历")
        panel = D.load_evaluation_panel() if panel is None else panel
    elif partition == "combined":
        panel = D.load_evaluation_panel() if panel is None else panel
    else:
        raise ValueError(f"未知分区 {partition}")
    panel = panel.loc[panel["symbol"].isin(C.COMMODITY_SYMBOLS)]
    ret1 = panel.pivot(index="date", columns="symbol", values="ret1")
    ret2 = panel.pivot(index="date", columns="symbol", values="ret2")
    basis = basis_momentum(ret1, ret2).reindex(index=index, columns=columns)
    neg_skew = -rolling_skew(ret1).reindex(index=index, columns=columns)
    return basis, neg_skew

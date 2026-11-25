"""因子研究设定。会员持仓因子的方向、窗口与检验口径在看到收益之前固定。"""
from __future__ import annotations

# 流动性池定义在 src/config.py 的 UNIVERSE，研究、策略与回测共用。

# 兴业证券（2026-08-05）：商品与国债为拥挤反转，分位阈值不含当日。
CROWDING = {
    "q_high": 0.9,
    "q_low": 0.1,
    "confirm_days": 3,
    "direction": "reversal",
    # 原文按品种展示多档窗口并选取较优者。此处只保留一个事前窗口，两套调仓规则都报告。
    "reference_window": 250,
}

# 一致性样本池：两榜同时连续在榜的会员不足，未上榜一侧按该榜末位持仓的一半填补。
CONSISTENCY = {
    "camp_size": 5,
    "min_pool": 10,
}

XS_POSITION = {
    "test_start": "2016-01-04",
    "min_names": 8,
    # 品种内标准化只用历史数据，不含当日。
    "zscore_window": 250,
    "zscore_min_periods": 60,
}


# 2026-11-25 扩展会员研究已确认；完整定义及检验族由统一研究入口冻结。
MEMBER_REVIEW = {
    "specification": "member_review",
    "start": "2011-01-04",
    "end": "2021-12-31",
    "primary_horizon": 10,
    "horizons": (1, 5, 10, 20),
    "min_names": 8,
    "min_days": 120,
}

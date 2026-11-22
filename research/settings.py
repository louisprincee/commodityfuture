"""因子研究设定。会员持仓因子的方向、窗口与检验口径在看到收益之前固定。"""
from __future__ import annotations

# 流动性池定义在 src/config.py 的 UNIVERSE，研究、策略与回测共用。

# 兴业证券（2026-08-05）：商品与国债为拥挤反转，分位阈值不含当日。
CROWDING = {
    "q_high": 0.9,
    "q_low": 0.1,
    "confirm_days": 3,
    "crowd3_window": 3,
    "crowd3_min_appear": 2,
    "direction": "reversal",
    # 原文按品种展示多档窗口并选取较优者。此处只保留一个事前窗口，两套调仓规则都报告。
    "reference_window": 250,
}

# 一致性样本池：两榜同时连续在榜的会员不足，未上榜一侧按该榜末位持仓的一半填补。
CONSISTENCY = {
    "camp_size": 5,
    "min_pool": 10,
    "pool_rule": "any side ranked every day in window; missing side = half of that side's 20th-rank volume",
}

# 截面持仓因子。符号来自两篇报告的经济方向，不由研究期收益决定。
# 拥挤度为反转；一致性合力、净多头占比变化、会员持仓流为顺势。
XS_POSITION = {
    "test_start": "2016-01-04",
    "slippage_ticks": 1.0,
    "min_names": 8,
    "min_names_paper": 4,
    "ts_window": 250,
    "consistency_windows": [20, 60],
    "net_windows": [20, 60],
    "mom_window": 20,
    "signs": {
        "CROWD1": -1,
        "CROWD2": -1,
        "CROWD3": -1,
        "POWER_20": 1,
        "POWER_60": 1,
        "NET_CHG_20": 1,
        "NET_CHG_60": 1,
        "FLOW": 1,
        "FLOW_ACTIVE": 1,
        "MOM20": 1,
    },
    # 每类一个代表：换手拥挤、短窗合力、开源重点的 60 日净多头变化、只计成交榜会员的持仓流。
    "core": ["CROWD1", "POWER_20", "NET_CHG_60", "FLOW_ACTIVE"],
    "composite_min": {"core": 3, "all": 5},
    # 兴业商品组合的 8 个品种，只作原文规则的对照，不参与选因子。
    "paper_symbols": ["AG", "CU", "RB", "M", "MA", "TA", "V", "FU"],
    # 第二设定：在看到第一设定日频换手普遍高于 0.6 之后加入。持有期固定，不搜索。
    "later_hold_days": 5,
    # 第四设定：在 5 日持有和第三设定之后，把重叠持有扩到 10 日和 60 日。
    # 60 日对应开源净多头占比变化的回看长度，这里用作持有期，不改 NET_CHG 的窗口。
    "hold_horizons": [10, 60],
    # 第三设定：品种内标准化，以及两条降换手规则。参数在看到该设定收益之前固定。
    # 标准化用过去 250 日、至少 60 日的均值和标准差，不含当日。
    "zscore_window": 250,
    "zscore_min_periods": 60,
    "tail": 0.20,       # 只持有截面最高、最低各 20%，中间空仓，腿内等权
    "rank_band": 0.10,  # 分位需越过中位数 ±10 个百分点才改变多空方向
}

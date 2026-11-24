"""STICKY_CORE：基差动量、低偏度与粘性席位净多头占比的三因子截面组合，两端等权多空与 10 日重叠持有。

构造顺序
  因子   使用 t 日及之前的日线和会员排名：
           1. BASIS_MOM_242   (+): 主力与次主力 242 日复合收益之差（期限结构斜率动量）
           2. NEG_SKEW_242    (+): 主力 242 日（至少 200 日）日收益偏度取反（厌恶彩票/尾部风险）
           3. STICKY_NET_SHARE(+): 剔除高换手活跃席位（CROWD1 门槛）后的前 20 名净多头占比
  合成   在当日流动性池且三因子均有效的品种内分别计算百分位秩，等权平均。
  选样   有效候选至少 8 个品种时，取合成得分前 20% 做多、后 20% 做空，不设拥挤度硬门槛否决。
  持有   每日新鲜多空各分配 0.50 名义权重，经 10 个交易日重叠持有形成当期组合，次日开盘成交。
  下单   目标权重乘数 2，单品种目标上限 20%。
"""
import pandas as pd
from strategies import Targets
from strategies.common import RANKED_FACTORS, ranked_targets

STRATEGY_ID = "sticky_core"
MAIN_BOOK = "STICKY_CORE"
EQUITY_CURVE_FILE = "equity_curves_d0_t1.csv"


def build_targets(panel: pd.DataFrame, _bars: pd.DataFrame, start: pd.Timestamp,
                  partition: str, jobs: int) -> Targets:
    return ranked_targets(panel, start, partition, jobs, RANKED_FACTORS)

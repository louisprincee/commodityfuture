# 中国商品期货会员持仓研究

本仓库研究中国商品期货会员持仓与经济先验因子。因子定义参考兴业证券《基于期货会员持仓的拥挤度与一致性 CTA 择时策略研究》和开源证券《蜘蛛网策略复盘与期货会员行为分析》。`research/` 负责因子与检验，`strategies/` 定义策略，`backtest/engine.py` 统一负责回测执行。会员持仓主策略是 Sticky-only，一致性残差对称版随组回测。

4 套策略的回测表现见[研究报告](docs/research.md#52-4-套策略回测结果)。

## 目录

| 路径 | 内容 |
|---|---|
| `download/` | 米筐商品期货下载：分钟行情、逐合约日行情、会员持仓排名、手续费与执行参数 |
| `src/config.py` | 路径、样本分区、品种与板块、流动性池门槛、样本外读取守卫 |
| `src/data.py` | 合约日行情、主力映射、会员排名、日面板与时点流动性池 |
| `src/costs.py` | 合约手续费、保证金、成交容量与价格限制 |
| `research/factors/` | 会员持仓指标，以及基差动量、负向偏度 |
| `research/ic/` | 时序与截面 IC、经济先验检验、稳健性检验和 IC 图 |
| `research/member/` | 席位平滑、条件价差，以及跨品种一致性与残差 |
| `research/settings.py` | 因子方向、窗口与检验口径 |
| `strategies/common.py` | 共享的截面选样、权重和持有规则 |
| `strategies/sticky_only.py` | 只用粘性净多，两端各 10%，10 日重叠持有 |
| `strategies/sticky_chain.py` | 粘性净多两端各 20% 候选，再用一致性残差筛到约 10% |
| `strategies/sticky_core.py` | 等权合成基差动量、负向偏度和粘性净多，两端各 10% |
| `strategies/sticky_core_h20.py` | 与 STICKY_CORE 相同，重叠持有改为 20 日 |
| `backtest/engine.py` | 整数手执行、账户核算、绩效指标和策略对照图 |
| `docs/research.md` | 因子研究及现行策略表现 |
| `tests/` | 合成数据测试 |

`data/research/` 存放 2010-01-04 至 2021-12-31 的研究数据，`data/oos/` 存放 2022–2025 的封存数据。两个分区下分别用 `minute/` 存按品种的分钟 Parquet，`daily/` 存逐合约行情、主力映射及会员榜，`execution/` 存手续费及执行参数。原始分钟总表和旧版因子缓存已删除；研究代码仍拒绝读取 `data/_archive/` 下的任何路径。

`download.getMinutes` 和 `download.getExternal` 的 `--datapath` 指向数据总目录，跨分区日期区间分别落盘。分钟下载按交易日分区，只下载现有品种分片首尾之外的请求区间，再合并保存。需要单独保留某次下载时，可用 `getMinutes --standalone` 写入对应分区的 `minute/segments/`。

## 运行

```bash
python -m src.data
python -m research.ic.study member-ic --jobs 8
python -m research.ic.study paper-ic --jobs 8
python -m research.ic.study five-day-ic --jobs 8
python -m research.ic.economic
python -m research.ic.robustness
python -m research.member.chain --jobs 8
python -m research.member.smoothing --jobs 8
python -m research.member.complementarity --jobs 8
python -m research.ic.plot
python -m backtest.engine --research --jobs 8
python -m backtest.engine --family member --oos --jobs 8
python -m backtest.engine --start 2016 --end 2025
python -m pytest tests -p no:cacheprovider
```

策略分为 `member` 会员持仓组（`sticky_only`、`sticky_chain`）与 `member_price` 价格拓展组（`sticky_core`、`sticky_core_h20`）。两组沿用相同的输出结构。

```
runs/strategies/<类别>/<策略>/{research,oos,combined}/
runs/strategies/<类别>/comparison/{research,oos,2016-2025}/
runs/strategies/benchmark/{research,oos}/
```

`--family` 选类别，省略时运行全部策略。给出策略名时只运行该策略。各策略的 `performance.png` 和 `yearly/YYYY.png` 共用由回测引擎管理的等权多头基准，基准从有会员榜的品种中按日应用流动性门槛。完整区间是同一账户从 2016 年连续做到 2025 年。研究期和样本外仍各自开户，总览图上的两个夏普取这两次结果。每类的 `comparison/2016-2025/plots/overview.png` 显示该类策略和基准，浅色标记样本外。

完整期运行全部策略时，另生成 `runs/strategies/comparison/2016-2025/plots/overview.png` 和 `runs/strategies/summary.csv`。报告中的结果链接指向本地生成的文件，仅克隆仓库时需要先生成这些结果。

除 `sticky_core_h20` 为 20 日外，策略都是 10 个交易日重叠持有。目标权重乘数为 2，策略名不再标注这项。

研究期回测与因子评价为 2016–2021。2010–2015 只用于指标预热。2022–2025 是封存样本外分区，不用于因子或参数选择；只有在明确请求并预先声明评估方案后，才可评价已冻结的规则。流动性资格使用过去 242 个交易日成交额中位数，不再叠加上市天数或 60 日有效观测门槛。本地 `data/` 与 `runs/` 未纳入版本控制。

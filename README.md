# 中国商品期货会员持仓研究

本仓库研究中国商品期货会员持仓与经济先验因子。因子定义参考兴业证券《基于期货会员持仓的拥挤度与一致性 CTA 择时策略研究》（2026-08-05）和开源证券《蜘蛛网策略复盘与期货会员行为分析》（2025-03-07）。`research/` 只做因子构建与 IC 检验；`strategies/` 仅保留当前 STICKY_CORE 策略；`backtest/` 是成本、换手与整数手账户的回测引擎。

完整记录见 [docs/research.md](docs/research.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `download/` | 米筐商品期货下载：分钟行情、逐合约日行情、会员持仓排名、手续费与执行参数 |
| `src/config.py` | 路径、样本分区、品种与板块、流动性池门槛、样本外读取守卫 |
| `src/data.py` | 合约日行情、主力映射、会员排名、日面板与时点流动性池 |
| `research/factors.py` | 兴业拥挤度、一致性与开源蜘蛛网、净多头占比等会员持仓指标 |
| `research/ic_study.py` | 会员持仓因子的逐品种时序 IC、全市场与板块截面 IC、5 日预测 IC |
| `research/economic_ic.py` | 期限结构、12-1 月动量、仓单变化等经济先验因子检验 |
| `research/settings.py` | 因子方向、窗口与检验口径 |
| `research/xs_robustness.py` | 截面因子稳健性检验：5/10/20 日 IC、两腿收益、趋势与波动状态、逐年一致性 |
| `strategies/sticky_core.py` | 当前唯一主策略：STICKY_CORE_2X，基差动量、负向偏度与粘性席位净多头占比截面等权合成，两端等权多空、10 日重叠持有；权重乘数 2.0X、单品种名义上限 20%；另报动态等权基准 |
| `backtest/engine.py` | 回测引擎：整数手账户、权重成本与换手、绩效指标，以及 `python -m backtest.engine` 入口 |
| `docs/research.md` | 研究报告 |
| `tests/` | 合成数据测试 |

`data/research/` 存放 2010-01-04 至 2021-12-31 的研究数据，`data/oos/` 存放 2022–2025 的封存数据。两个分区下分别用 `minute/` 存按品种的分钟 Parquet，`daily/` 存逐合约行情、主力映射及会员榜，`execution/` 存手续费及执行参数。原始分钟总表和旧版因子缓存已删除；研究代码仍拒绝读取 `data/_archive/` 下的任何路径。

分钟与日频下载命令的 `--datapath` 指向数据总目录，跨分区日期区间分别落盘。分钟下载按交易日分区，只下载现有品种分片首尾之外的请求区间，再合并保存。需要单独保留一次下载时，可用 `getMinutes --standalone` 写入对应分区的 `minute/segments/`。

## 运行

```bash
python -m research panel
python -m research member-ic --jobs 8
python -m research paper-ic --jobs 8
python -m research five-day-ic --jobs 8
python -m research candidate-ic
python -m research xs-robustness
python -m backtest.engine sticky_core --partition research --jobs 8
python -m backtest.engine sticky_core --partition oos --jobs 8
python -m backtest.engine sticky_core --partition combined
python -m pytest tests -p no:cacheprovider
```

因子定义、公式、检验和策略经济故事统一记录在 [docs/research.md](docs/research.md)。因子 IC 输出在 `runs/research/member_ic/`（会员因子时序）、`runs/research/xs_position/`（会员因子截面与 5 日）、`runs/research/economic_ic/`（经济先验因子）与 `runs/research/xs_robustness/`（截面稳健性）。当前策略回测统一输出在 `runs/strategies/sticky_core/{research,oos,combined}/`。

研究期回测与因子评价为 2016–2021。2010–2015 只用于指标预热。2022–2025 是封存样本外分区，不用于因子或参数选择；只有在明确请求并预先声明评估方案后，才可评价已冻结的规则。本地 `data/` 与 `runs/` 未纳入版本控制。

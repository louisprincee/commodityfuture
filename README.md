# 中国商品期货会员持仓研究

本仓库研究中国商品期货会员持仓。因子定义来自兴业证券《基于期货会员持仓的拥挤度与一致性 CTA 择时策略研究》（2026-08-05）和开源证券《蜘蛛网策略复盘与期货会员行为分析》（2025-03-07）。`research/` 只做因子构建与 IC 检验；`strategies/` 存放待选策略的持仓规则；`backtest/` 是成本、换手与整数手账户的回测引擎。

完整记录见 [docs/research.md](docs/research.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `download/` | 米筐商品期货下载：分钟行情、逐合约日行情、会员持仓排名、手续费与执行参数 |
| `src/config.py` | 路径、样本分区、品种与板块、流动性池门槛、样本外读取守卫 |
| `src/data.py` | 合约日行情、主力映射、会员排名、日面板与时点流动性池 |
| `research/factors.py` | 兴业拥挤度、一致性与开源蜘蛛网、净多头占比等会员持仓指标 |
| `research/ic_study.py` | 逐品种时序 IC、全市场与板块截面 IC、5 日预测 IC |
| `research/settings.py` | 因子方向、窗口与检验口径 |
| `strategies/rules.py` | 信号到持仓的通用规则：截面权重、重叠持有、两端缓冲、分位事件、择时方向、预算与对冲 |
| `strategies/buffer.py`、`hold_five.py`、`risk_budget.py` | 三组待选策略 |
| `backtest/engine.py` | 回测引擎：整数手账户、权重成本与换手、绩效指标，以及 `python -m backtest.engine` 入口 |
| `docs/research.md` | 研究报告 |
| `tests/` | 合成数据测试 |

`data/research/` 存放 2010-01-04 至 2021-12-31 的研究数据，`data/oos/` 存放 2022–2025 的封存数据。两个分区下分别用 `minute/` 存按品种的分钟 Parquet，`daily/` 存逐合约行情、主力映射及会员榜，`execution/` 存手续费及执行参数。`data/_archive/` 保留跨分区的原始分钟总表和历史缓存。

分钟与日频下载命令的 `--datapath` 指向数据总目录，跨分区日期区间分别落盘。分钟下载按交易日分区，只下载现有品种分片首尾之外的请求区间，再合并保存。需要单独保留一次下载时，可用 `getMinutes --standalone` 写入对应分区的 `minute/segments/`。

## 运行

```bash
python -m research panel
python -m research member-ic --jobs 8
python -m research paper-ic --jobs 8
python -m research five-day-ic --jobs 8
python -m backtest.engine buffer --jobs 8
python -m backtest.engine hold-five --jobs 8
python -m backtest.engine risk-budget --jobs 8
python -m pytest tests -p no:cacheprovider
```

因子 IC 输出在 `runs/research/member_ic/`（时序）与 `runs/research/xs_position/`（截面与 5 日），策略回测输出在 `runs/strategies/<策略>/`。

研究期回测为 2016–2021。2010–2015 只用于指标预热。2022–2025 样本外分区不用于评估。本地 `data/` 与 `runs/` 未纳入版本控制。

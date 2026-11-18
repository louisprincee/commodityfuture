# 中国商品期货会员持仓研究

本仓库研究中国商品期货会员持仓。因子定义来自兴业证券《基于期货会员持仓的拥挤度与一致性 CTA 择时策略研究》（2026-08-05）和开源证券《蜘蛛网策略复盘与期货会员行为分析》（2025-03-07）。当前入口逐品种计算研报指标，并衡量其与下一段开盘到开盘收益的秩 IC。历史截面和账户实验保存在 `research/experiments/`。

完整记录见 [docs/research.md](docs/research.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `download/` | 米筐商品期货下载：分钟行情、逐合约日行情、会员持仓排名、手续费 |
| `src/config.py` | 路径、样本分区、品种与板块、样本外读取守卫 |
| `src/data.py` | 合约日行情、主力映射与会员排名 |
| `research/settings.py` | 品种池、因子符号与窗口 |
| `research/member_ic.py` | 逐品种、逐日研报指标与开盘到开盘秩 IC |
| `research/paper_factors.py`、`research/member_study.py` | 研报指标及会员榜计算 |
| `research/experiments/` | 已完成的截面、持有期和账户实验 |
| `research/portfolio.py` | 日面板筛选及历史组合使用的共用计算 |
| `research/data_check.py` | 本地数据完整性检查 |
| `docs/research.md` | 研究报告 |
| `tests/` | 合成数据测试 |

`data/research/` 存放 2010-01-04 至 2021-12-31 的研究数据，`data/oos/` 存放 2022–2025 的封存数据。两个分区下分别用 `minute/` 存按品种的分钟 Parquet，`daily/` 存逐合约行情、主力映射及可用的会员榜，`execution/` 存手续费及可用的执行参数。`data/_archive/` 保留跨分区的原始分钟总表和历史缓存。

分钟与日频下载命令的 `--datapath` 指向数据总目录，跨分区日期区间分别落盘。分钟下载按交易日分区，只下载现有品种分片首尾之外的请求区间，再合并保存。需要单独保留一次下载时，可用 `getMinutes --standalone` 写入对应分区的 `minute/segments/`。

## 运行

```bash
python -m research check
python -m research panel
python -m research member-ic --jobs 8
python -m research event-timing --jobs 8
python -m pytest tests -p no:cacheprovider
```

逐品种 IC 输出在 `runs/research/member_ic/`，异常持仓事件账户输出在 `runs/research/xs_position/event_timing/`。历史研究的复算命令见 [研究报告](docs/research.md)。

研究期回测为 2016–2021。2010–2015 只用于指标预热。2022–2025 样本外分区不用于评估。本地 `data/` 未纳入版本控制。

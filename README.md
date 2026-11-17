# 中国商品期货机器学习复现

本仓库在中国商品期货上复现 Guida (2025)《Machine Learning in Commodity Futures: Bridging Data, Theory, and Return Predictability》（CFA Institute Research Foundation）：用九类理论与技术特征 × 14 个回看窗口共 126 个特征，以 XGBoost 回归树逐年扩展窗口训练八个预测期限（5–200 日），按截面得分构建多空组合，并对八个期限做集成。

完整研究报告见 [docs/research.md](docs/research.md)，包括数据检查结论、复现口径、与论文结论的逐项对照及局限。

## 目录

| 路径 | 内容 |
|---|---|
| `download/` | 米筐商品期货下载：分钟行情、逐合约日行情与主力映射等外部数据、手续费 |
| `src/config.py` | 路径、样本分区、品种与板块、样本外读取守卫 |
| `src/data.py` | 合约日行情与主力 / 次主力日面板 |
| `config/research/replication.json` | 事前声明的复现协议 |
| `research/data_check.py` | 本地数据完整性与一致性检查 |
| `research/features.py` | 126 个特征与前瞻收益标签 |
| `research/model.py` | 逐年滚动训练、网格选参、标签期限清洗 |
| `research/portfolio.py` | 多空权重、换手与成本、绩效指标 |
| `research/report.py` | 论文 Exhibit 3–8 对应的表与图 |
| `research/holding.py` | 第二设定：持有期与预测期限对齐 |
| `runs/research/data_check/` | 数据检查结果 |
| `runs/research/replication/` | 预测、重要性、绩效、相关性、敞口与图 |
| `runs/research/holding_aligned/` | 第二设定的绩效、相关性与图 |
| `tests/` | 合成数据测试 |

## 运行

```bash
python -m research check
python -m research panel
python -m research train --jobs 10
python -m research report
python -m research holding
python -m pytest tests -p no:cacheprovider
```

研究结果依赖未纳入版本控制的本地 `data/`；研究期回测为 2016–2021，2010–2015 只用于特征预热与训练；2022–2025 样本外分区未用于评估。

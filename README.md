# 中国商品期货会员持仓研究

保留四套现有策略：**Sticky-only、Sticky-chain、Sticky-core、Sticky-core-h20**。当前主研究期覆盖 **2011–2021（2010 年预热）**，历史对照与完整发现见[研究报告](docs/research.md)。

分层保持清晰：`research/` 只计算因子和统计；`strategies/` 形成目标权重；`backtest/` 负责账户、整数手、非交割执行、历史费用和绩效。三层共用 `src/` 的日面板、时点流动性池和分区守卫。

## 常用入口

```bash
PYTHONDONTWRITEBYTECODE=1 python -m research.member.review --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --start 2011 --end 2021 --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --experiment sticky-flow --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```

说明：
- `research.member.review`：在 **2011–2021（2010 预热）** 检验会员分组（连续上榜 `RUN_LONG`、方向稳定 `DIR_STABLE`）、持仓流相对粘性净多的增量信息，以及 0 / 0.5 / 0.8 / 1.0 末位比例截尾敏感性，输出至 `runs/research/member_review/`。
- `backtest.engine --start 2011 --end 2021`：用统一的非交割执行与历史手续费重跑四套现有策略及等权基准，结果保存在 `runs/strategies/{member,member_price}/<策略>/research_2011_2021/` 与 `runs/strategies/comparison/research_2011_2021/`。
- `backtest.engine --experiment sticky-flow`：运行 `Sticky`、`Flow` 与 `Sticky + Flow` 同样本配对增量回测，结果保存在 `runs/strategies/member/sticky_flow/` 与 `runs/strategies/comparison/sticky_flow/`。
- `backtest.engine --research` / `--oos` / `--start 2016 --end 2025`：保留原有 2016–2021 与 2016–2025 入口，便于复现早期对比结果。

刷新历史执行参数或合约规格时限定研究分区日期：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m download.getExecutionParameters --partition research --start-date 20100104 --end-date 20211231
PYTHONDONTWRITEBYTECODE=1 python -m download.getContractSpecifications --start-date 20100104 --end-date 20211231
```

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `src/config.py`、`src/data.py` | 品种范围、分区守卫、日面板与 242 日成交额中位数流动性资格 |
| `src/costs.py` | 历史手续费（同合约前值优先、其次同品种无冲突前值）、保证金、容量与跌停约束 |
| `research/factors/member.py` | 研报指标（拥挤度/一致性/蜘蛛网）、粘性净多及会员画像/流因子 |
| `research/factors/price.py` | 基差动量 `BASIS_MOM_242` 与负向偏度 `NEG_SKEW_242` |
| `research/ic/` | `study.py`（IC 与收益标签）、`statistics.py`（HAC/BH/残差）、`robustness.py`、`economic.py`、`plot.py` |
| `research/member/` | `review.py`（2011–2021 扩展会员研究）、`chain.py`、`complementarity.py`、`smoothing.py` |
| `strategies/` | `sticky_only`、`sticky_chain`、`sticky_core`、`sticky_core_h20`、`sticky_flow` 与共用选样 `common.py` |
| `backtest/engine.py` | 整数手非交割回测引擎、绩效表与策略对比总入口 |
| `research/history/turnover_spread.py` | 已退役差值 `STICKY_MINUS_SPEC` 的复现归档 |
| `docs/research.md`、`docs/history.md` | 唯一活跃研究报告与口径变更/退役记录 |
| `tests/` | 四个合成数据测试集（`test_data`、`test_research`、`test_strategies`、`test_backtest`） |

## 数据与规范

- `data/research/` 存放 2010-01-04–2021-12-31 数据；`data/oos/` 存放封存的 2022–2025；`data/_archive/` 禁止研究读取。
- 凭证只由鉴权函数读取，不输出、不提交；不改写原始行情与排名记录；不添加未经批准的新依赖。
- 本地 `data/`、`runs/` 已通过 `.gitignore` 忽略，克隆后可通过上述入口重新生成全部本地结果。

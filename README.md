# 中国商品期货会员持仓研究

保留四套现有策略：**Sticky-only、Sticky-chain、Sticky-core、Sticky-core-h20**。当前研究覆盖 **2011–2021，2010 年预热**。完整结果、执行资料缺口及公开来源见[研究报告](docs/research.md)。

`research/` 只计算因子和检验；`strategies/` 形成目标权重；`backtest/` 负责账户、整数手、执行和成本。三层共用 `src/` 的日面板、时点流动性资格和分区守卫。

## 当前入口

```bash
PYTHONDONTWRITEBYTECODE=1 python -m research.member.review --jobs 4
PYTHONDONTWRITEBYTECODE=1 python run.py --jobs 4
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --basetemp=/private/tmp/commodityfuture-synthetic-tests
```

当前会员因子实现保留 16 个规格：基线 2 个、连续上榜与方向稳定候选 2 个、持仓流 4 个，以及末位比例 0/0.5/0.8/1.0 的两组敏感性 8 个。当前入口专门检验八种插补的共同样本差异和持仓流对粘性净多的增量信息，结果在 `runs/research/member_imputation_v3/`。其余会员分组、榜内换手、拥挤度一日/五日时序与截面检验的完整证据保留在 `runs/research/member_review_v2/`，不因筛选后只留两个候选而重算其原有校正值。

当前四套账户分别保存在 `runs/strategies/{member,member_price}/<策略>/research_2011_2021_execution_v1/`；汇总及总图在 `runs/strategies/comparison/research_2011_2021_execution_v1/`。初始权益 1000 万元，目标权重乘数 2、整数手；缺失手续费先沿用同合约前值，没有时使用同品种此前最近且无冲突的费率，成交保存来源合约、日期及沿用标记。每边一跳是报告情景，两跳仅留结果文件。没有任何前值的缺失记录仍不补造；先前 `research_2011_2021_fee_carry/` 和 `research_2011_2021_audited/` 结果单独保留。

四套策略和基准统一采用非交割账户：实际合约期限由米筐元数据提供；新目标仓位上限为 5 日历史平仓容量，已有超限仓位逐日减仓；最晚持有日前 5 个交易时段停止加仓并退出，研究结束日同样预留退出窗口。平仓遵守容量和涨跌停，无法如期退出则抛出执行失败。当前四套研究期回测及基准均无过期仓位、陈旧估值或期末残仓；早期缺失参数取消相关开仓，仍以 `partial_execution_data` 标记资料局限。

刷新历史执行参数时，日期受研究分区限制：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m download.getExecutionParameters --partition research --start-date 20100104 --end-date 20211231
PYTHONDONTWRITEBYTECODE=1 python -m download.getContractSpecifications --start-date 20100104 --end-date 20211231
```

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/config.py`、`src/data.py` | 商品范围、分区、日面板与完整 242 日成交额中位数资格 |
| `src/costs.py` | 手续费、保证金、容量、价格限制和合约规格 |
| `research/factors/member.py` | 原始会员持仓、拥挤度、一致性和蜘蛛网 |
| `research/factors/member_profiles.py` | 当前保留的 16 个会员因子规格 |
| `research/ic/statistics.py` | HAC、固定族 BH、时序标准化秩回归近似检验 |
| `research/member/review.py` | 共同样本插补比较、流的增量信息与 IC 图 |
| `research/settings.py` | 当前与历史入口的固定参数 |
| `strategies/sticky_*.py` | 四套现有策略，均保持可运行 |
| `backtest/engine.py` | 账户执行及历史默认入口 |
| `run.py` | 四套策略的 2011–2021 批量编排与执行覆盖 |
| `research/history/` | 已退役差值研究的复现源码 |
| `docs/research.md` | 唯一活跃报告，附旧结果历史快照 |
| `docs/history.md` | 退役设计和口径变更记录 |
| `tests/` | 仅保留原有四个测试文件，新会员验证合并在 `test_research.py`，全部使用合成数据 |

## 数据与历史结果

`data/research/` 存放 2010-01-04–2021-12-31 数据；`data/oos/` 存放封存的 2022–2025。日行情、会员排名、分钟与执行参数分开缓存；本轮未读取封存分区。2010 年用于完整窗口预热，不缩短窗口补造年初样本。`data/_archive/` 仍禁止研究读取。

`download.getMinutes` 和 `download.getExternal` 的 `--datapath` 指向数据总目录，跨分区请求分别落盘；分钟数据按交易日增量合并。原始分钟总表和旧版因子缓存已删除。凭证只由鉴权函数读取，不输出、不移动、不纳入版本控制；不添加未经批准的新依赖。

旧命令 `research.ic.study`、`research.ic.robustness`、`research.member.smoothing/complementarity/chain` 及 `backtest.engine --research` 保留 2016–2021 默认起点。现行预测标签已与账户持有合约对齐；旧发表数字和 CSV 没有覆盖，不能把新旧标签的检验值混为同一规格。

手续费优先读取米筐实际合约历史交易参数；接口有记录不代表每个字段都有值。另有来源明确、限定日期的手续费补缺配置。用户已批准缺失时采用此前最近的费率，同合约优先，其次同品种无冲突的历史记录，不使用未来费率倒填或跨更名代码套用；此项是回测成本假设，原始 vendor 文件不改写。合约规格独立于手续费覆盖下载，旧代码合约行情保留到实际结束以支持平仓。

差值 `STICKY_MINUS_SPEC` 已退出活跃研究，不显著及不利结果保留。四套策略均在当前目录；恢复策略后留下的历史辅助重复文件已按授权清理。[历史记录](docs/history.md)说明具体变化，[清理记录](runs/research/cleanup_candidates.csv)列出本次删除的 17 个文件。后续新增删除项仍先征得授权。本地 `data/`、`runs/` 未纳入版本控制，克隆仓库后须生成本地结果。

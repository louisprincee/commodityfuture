# 中国商品期货会员持仓研究

当前保留三套策略：**粘性净多**、**持久粘性身份**、**粘性净多 + 基差动量 + 负偏度**。主研究期为 **2011–2021（2010 年预热）**，完整发现见 [研究报告](docs/research.md)。

分层保持清晰：`research/` 只计算因子和统计；`strategies/` 形成目标权重；`backtest/` 负责账户、整数手、非交割执行、历史费用和绩效。三层共用 `src/` 的日面板、时点流动性池和分区守卫。

## 常用入口

```bash
PYTHONDONTWRITEBYTECODE=1 python -m research.member.review --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --start 2011 --end 2021 --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine persist_identity --research --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```

说明：

- `research.member.review`：在 2011–2021（2010 预热）检验会员分组、持仓流增量与末位比例插补敏感性，输出至 `runs/research/member_review/`。
- `research.member.extensions`：展示保留的粘性身份、持仓流及 5 日平滑因子。当前 CSV 与图仅包含保留因子，完整原检验族保存于对应研究目录的 `history/`，校正值不因展示筛选而重算。
- `backtest.engine --research` / `--oos` / `--start 2011 --end 2025`：用统一的非交割执行与历史手续费重跑现役策略及等权基准。每个策略单独存放于 `runs/strategies/<策略>/{research,oos,2011-2025}/`，基准在 `runs/strategies/benchmark/`，两套纯会员策略都在运行范围内时另写 `runs/strategies/comparison/`。
- `backtest.engine persist_identity --research`：运行持久粘性身份策略，只要求自身信号有效。结果保存在 `runs/strategies/persist_identity/`。
- `backtest.engine sticky_basis_skew --research`：运行粘性净多、基差动量与负偏度的等权排名组合。持有天数只在 `strategies/member_price/sticky_basis_skew.py` 顶部的 `HOLDING_DAYS` 修改，默认 10 日；非默认持有期结果写入 `holding_<天数>/`。该策略不进入对比图或报告绩效表。
- `backtest.engine --start 2016 --end 2021`：旧研究期入口，结果写入各策略的 `legacy_research/`。

刷新历史执行参数或合约规格时限定研究分区日期：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m download.getExecutionParameters --partition research --start-date 20100104 --end-date 20211231
PYTHONDONTWRITEBYTECODE=1 python -m download.getContractSpecifications --start-date 20100104 --end-date 20211231
```

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `src/config.py`、`src/data.py` | 品种范围、分区守卫、日面板与流动性资格 |
| `src/costs.py` | 历史手续费、保证金、容量与跌停约束 |
| `research/factors/member.py` | 研报指标、粘性净多及会员画像与流因子 |
| `research/factors/price.py` | 基差动量与负向偏度 |
| `research/ic/` | IC、收益标签、HAC 与 BH |
| `research/member/` | 2011–2021 会员研究入口 |
| `strategies/member/` | 纯会员策略 `sticky_net`、`persist_identity` |
| `strategies/member_price/` | `sticky_basis_skew` |
| `backtest/engine.py` | 整数手非交割回测与策略对比总入口 |
| `docs/research.md`、`docs/history.md` | 活跃研究报告与口径变更记录 |
| `tests/` | 合成数据单元测试 |

## 数据与规范

- `data/research/` 存放 2010-01-04–2021-12-31；`data/oos/` 存放 2022–2025，仅限显式授权且事先声明协议的评价，已读过的结果不能用于反调规格。
- 凭证只由鉴权函数读取，不输出、不提交；不改写原始行情与排名记录；不添加未经批准的新依赖。
- 本地 `data/`、`runs/` 已通过 `.gitignore` 忽略，克隆后可通过上述入口重新生成。

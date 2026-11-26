# 历史变更与已退役研究

本文件记录已退役设计及重要口径演进；活跃报告仍以 `docs/research.md` 为准。

## 2026-11-26

- **更名合约留到各自最后交易日**：甲醇 `ME`、动力煤 `TC`、菜油 `RO`、强麦 `WS`、早籼稻 `ER` 不再在后继代码挂牌日从研究面板删除。挂牌之后旧月份仍进入该品种的成交额、持仓、主力候选和会员榜。成交额按金额相加；当日合约乘数不一致时，持仓和成交量按吨相加，会员榜先乘各自乘数再相加并重新取前 20 名。重叠期主力用前一交易日吨持仓，超过在任主力 1.1 倍才于下一交易日切换。乘数只取米筐合约规格，不手填。
- **清理已退役与早期快照脚本（用户批准）**：删除已退役的 `research/history/turnover_spread.py`（并移除空目录 `research/history/`），以及早期 2016–2021 快照复现脚本与静态表（`research/ic/economic.py`、`research/ic/robustness.py`、`research/ic/plot.py`、`research/member/smoothing.py`、`research/member/public_relationships.csv`）；将 `research/member/{chain,complementarity}.py` 的常量与 `load_daily_flow` 依赖内联至活跃模块；保留 `config/historical_fee_rules.csv` 独立存在以避免改写 `data/` 下原始米筐下载费率表。

## 2026-11-25

- **退役差值因子 `STICKY_MINUS_SPEC`**：原值 1:1 相减在 2010–2021 全样本未带来统计显著改善（10 日配对 IC 相对粘性净多为 `-0.0117`，`p=0.0743`）。相关复现代码归档至 `research/history/turnover_spread.py`，不显著与不利统计保留在 `runs/research/turnover_spread/`。
- **保留全部四套现有策略**：`sticky_only`、`sticky_chain`、`sticky_core`、`sticky_core_h20` 均维持可用，不因单次诊断结果删除。
- **研究期扩展与执行对齐（2011–2021，2010 预热）**：
  - 预测标签对齐账户执行时点：t 日收盘信号对应 t 日主力映射并在 t+1 开盘入场；
  - 修正未定义市场换手率与部分缺失增减量时的计算处理，保留更名前旧代码合约（`TC/ME/RO/WS/ER`）行情直至实际存续结束；
  - 经用户批准，缺失历史手续费采用同合约前值优先、其次同品种此前最近且无冲突费率的承接估计（不倒填未来费率、不跨更名品种套用，也不改写原始 vendor 记录）；
  - 统一采取非交割账户约束：依据实际合约到期和交割条款划定退出截止日，设 5 日平仓容量限制与最后 5 个交易日停止开新并退出窗口，无法退出时显式抛出 `ExecutionFailure`。
- **代码与产出路径收敛**：将早期独立脚本 `run.py` 及重复的 `labels.py` / `member_profiles.py` / `labels_v1.py` 分别合并至 `backtest/engine.py`、`research/ic/study.py`、`research/factors/member.py` 与 `research/history/turnover_spread.py`；清理 `v1/v2/v3` 及 `_audited/_fee_carry/_execution_v1` 临时实验目录，2011–2021 回测统一落盘于 `research_2011_2021/`，会员扩展检验统一落盘于 `runs/research/member_review/`。

# 历史变更与已退役研究

本文件记录已退役设计及重要口径演进；活跃报告仍以 `docs/research.md` 为准。

## 2026-11-25

- **退役差值因子 `STICKY_MINUS_SPEC`**：原值 1:1 相减在 2010–2021 全样本未带来统计显著改善（10 日配对 IC 相对粘性净多为 `-0.0117`，`p=0.0743`）。相关复现代码归档至 `research/history/turnover_spread.py`，不显著与不利统计保留在 `runs/research/turnover_spread/`。
- **保留全部四套现有策略**：`sticky_only`、`sticky_chain`、`sticky_core`、`sticky_core_h20` 均维持可用，不因单次诊断结果删除。
- **研究期扩展与执行对齐（2011–2021，2010 预热）**：
  - 预测标签对齐账户执行时点：t 日收盘信号对应 t 日主力映射并在 t+1 开盘入场；
  - 修正未定义市场换手率与部分缺失增减量时的计算处理，保留更名前旧代码合约（`TC/ME/RO/WS/ER`）行情直至实际存续结束；
  - 经用户批准，缺失历史手续费采用同合约前值优先、其次同品种此前最近且无冲突费率的承接估计（不倒填未来费率、不跨更名品种套用，也不改写原始 vendor 记录）；
  - 统一采取非交割账户约束：依据实际合约到期和交割条款划定退出截止日，设 5 日平仓容量限制与最后 5 个交易日停止开新并退出窗口，无法退出时显式抛出 `ExecutionFailure`。
- **代码与产出路径收敛**：将早期独立脚本 `run.py` 及重复的 `labels.py` / `member_profiles.py` / `labels_v1.py` 分别合并至 `backtest/engine.py`、`research/ic/study.py`、`research/factors/member.py` 与 `research/history/turnover_spread.py`；清理 `v1/v2/v3` 及 `_audited/_fee_carry/_execution_v1` 临时实验目录，2011–2021 回测统一落盘于 `research_2011_2021/`，会员扩展检验统一落盘于 `runs/research/member_review/`。

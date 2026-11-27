# 历史变更与已退役研究

本文件记录已退役设计及重要口径演进；活跃报告仍以 `docs/research.md` 为准。

## 2026-11-26

- **活跃报告与图表按保留名单收敛（用户要求）**：删除活跃报告中低调仓强度、单独低净敞口过滤、榜内换手门槛、z 分数、20 日变化、历史表现筛选、拥挤度/一致性/蜘蛛网、全榜净多及已止步组合的定义、行、图与解释。保留原粘性净多、高换手与插补对照、持仓流残差、持久同向及净敞口敏感性对照、长期身份、持久长期身份、原始及 5 日平滑持仓流。重画三张相关 IC 图；完整原检验族先保存到各研究目录 `history/`，当前 CSV 再导出保留行，q 值原样承接，不重划检验族。旧拥挤度与组合图也移入历史目录，不再由默认入口绘制或在活跃报告展示。研究期 5/10 日组合实验和先前检验记录保留于历史记录，现役策略及账户结果不变。

- **`runs/` 顶层元数据归入 `meta/`**：`evaluation_protocol.json`、`execution_audit.csv`、`manifest.json` 移至 `runs/meta/`；活跃报告链接同步更新。
- **三套策略按因子规则重命名**：`sticky_only` → `sticky_net`（账本 `STICKY_NET`）、`sticky_identity` → `persist_identity`（账本 `PERSIST_IDENTITY`）、`sticky_basis_momentum_neg_skew` → `sticky_basis_skew`（账本 `STICKY_BASIS_SKEW`）。因子名 `STICKY_NET_SHARE`、`STICKY_PERSIST_IDENTITY` 等不变；结果目录同步更名，对比图改用中文标签。样本外对比图改为与研究期同款总览版式，坐标按整年刻度。
- **2016 年前后符号切换诊断图**：`member_review` 新增粘性组构成的逐年统计与图（未上成交榜席位占比、粘性持仓占比，并对照两组 10 日 IC），只使用持仓与成交榜，不涉及收益。
- **图表统计对账**：对比图原先以 `pct_change` 丢弃账户首日，导致图中的年化/Sharpe 与账本汇总有细微差别；改为从初始资金计算首日收益，分段 Sharpe 保留对应的前一期权益基数。重画三组 member 对比图，不改变任何交易、净值或检验结果。

- **价格策略持有天数改为源码常量（用户指定）**：`sticky_basis_momentum_neg_skew` 删除 API `holding_days` 与 CLI `--holding-days`，改为只读策略文件顶部 `HOLDING_DAYS`（默认 10）；回测引擎的持有期、`holding_<days>/` 目录与图标题均从该常量取得，规则与已有结果不变。
- **身份独立样本与价格策略合并（本轮用户指定）**：身份账户取消 FLOW5 和原粘性净多有效性的额外筛选，仅要求自身 `STICKY_PERSIST_IDENTITY` 有限，另保留共用时点流动性资格。账户由旧 `STICKY_PI_REF` 改名为 `STICKY_PERSIST_IDENTITY`，作为新样本规格先写入 `runs/evaluation_protocol.json` 再重跑；已读 OOS 仅作描述性评价。原 `sticky_core` 与 `sticky_core_h20` 合为 `sticky_basis_momentum_neg_skew`（`BASIS_MOM_242 + NEG_SKEW_242 + STICKY_NET_SHARE`），API `holding_days` / CLI `--holding-days` 支持自选，默认 10 日；非默认结果放 `holding_<days>/` 防止覆盖。按用户要求删除旧价格策略重复源码与结果；现役共三套，只有两套 member 策略参与对比。
- **全量刷新与执行审计**：三个现役策略与基准分别重跑 research、OOS、2011–2025 连续账户；同时重建研究面板，重跑会员复核、扩展、互补与保留的原研报 IC 入口。15 个账本/区间对含研究期平滑实验，共 30 个滑点情景，期末残仓与陈旧估值均为 0。1 跳整数手、资金与费用对账、历史费用来源日期、合约期限及最后 5 日停止新开审计通过；连续账户共 4 笔平仓的成交日缺少期限记录，依据此前已知的同一实际合约条款核验，未倒填未来数据或新增开仓。
- **用户授权的 10 日持仓流平滑探索**：研究期内固定身份、FLOW5、FLOW10 共同有效样本，比较身份、身份加 5 日流、身份加 10 日流；不注册为现役策略，不读 OOS。10 日组合净年化 7.92%、Sharpe 0.99，相比 5 日组合 10.74%、1.38；唯一账户主检验的年化净收益差 −2.82 个百分点，95% HAC 区间 [−5.22, −0.41]，p = 0.0215。组合 IC 差 −0.0047，p = 0.190；换手仅小幅下降，研究期探索不支持替换，完整结果保留。
- **清理与修复旧研究产物**：用户单独明确授权删除 `runs/research/xs_robustness/` 中 7 个 CSV 和 `runs/research/xs_position/factor_significance.png`，其生成代码已退役。另发现 `retained_profile_evidence.csv` 只在一个已不存在的旧文件存在时才更新，导致其部分统计与当前输入不符；改为每次从当前信号重算两组已筛选画像的描述性统计，`q_primary/q_all` 留空，不将原 12 项检验族缩成 2 项重做 BH。该表不再宣称保留的历史校正值对应当前输入。所有当前产物的源码、输入元数据与结果摘要写入 `runs/manifest.json`。

- **按用户指定名单收敛为四套策略**：保留 `sticky_only`、`sticky_identity`、`sticky_core`、`sticky_core_h20`。身份策略从原组合实验中提取 `STICKY_PI_REF`，原样保留身份、原粘性净多和 FLOW5 同时有效的样本筛选、10% 两端、10 日持有与全部执行参数；FLOW5 数值不参与排序。其研究期和 OOS 原始 CSV 移至 `runs/strategies/sticky_identity/{research,oos}/`，OOS 冻结协议与身份成交审计随迁。根据用户明确删除授权，删除 `sticky_flow`、`sticky_persist`、`sticky_persist_buf`、`sticky_extension`、`sticky_followup`、`sticky_combo` 的策略实现及策略回测结果，移除 `--experiment` 入口、无调用的账户实验函数与对应测试。`runs/research/` 的因子结果、协议和图表不变。比较目录重建为仅含 Sticky-only、身份和基准的研究期/OOS描述性比较，旧持续粘性配对检验与图表删除；未新增 OOS 假设。
- **删除依据与历史收益口径澄清**：上述删除来自用户的保留名单，并不意味着其他候选在研究期从未跑赢 Sticky-only。统一单边 1 跳下，Sticky-only 净年化约 4.80%，持续粘性约 6.96%、持续粘性缓冲约 6.86%、旧 Sticky+Flow 约 5.98%、长期粘性身份（不含近期持续筛选）约 5.93%、身份与持仓流组合约 10.74%。其研究期相对基线增量未通过预先声明的显著性门槛；账面收益更高不等于显著改善。退役账户 CSV 已按授权删除，本条仅保留历史判断记录。
- **身份加持仓流的已授权 OOS 结论**：读取前冻结 1:1 截面排名组合，固定 10 日持有与相同共同样本。2022–2025 共 969 日，组合净年化 2.79%、Sharpe 0.38、回撤 −14.41%；仅身份净年化 10.14%、Sharpe 1.11、回撤 −9.40%。组合减身份的年化净收益差 −7.35 个百分点（95% HAC 区间 [−15.21, +0.51]，p = 0.0667），其中毛差 −6.33 个百分点，新增成本 1.02 个百分点。研究期该组合相对身份的净增量 +0.56 个百分点，p = 0.822。组合未证明有正向净收益增量，不把 OOS 退化全归因于手续费；此次删除前已完整报告。只保留身份结果，其他账户明细按本次授权删除，不归档到另一个策略结果目录。
- **身份定义的现有实现限制**：复核发现 `_persist_mask` 的方向均值设 `rolling(20, min_periods=20)`，缺榜日为缺失，因此实际要求最近 20 日均有有效上榜方向，比文字中的“20 日至少上榜 15 日”更严格。既有结果评价的是该严格实现；此次仅更正说明并在原有测试中覆盖缺榜边界，不在读取 OOS 后改变筛选。10 日平滑持仓流尚未新建策略或回测，后续属于新探索规格，不能据已读 OOS 选择窗口。

- **粘性净多扩展研究（研究期，探索性）**：新增 `research/member/extensions.py` 与 `research.factors.member.extension_factors`，规格与检验族先写入 `runs/research/member_extensions/protocol.json`，只读取 2011–2021。五个方向为榜内换手门槛、品种内 z 分数与 20 日变化、粘性席位持仓流、长期粘性身份与席位历史记录，以及拥挤度三因子 1/5 日截面与时序 IC 的重做。`STICKY_FLOW` 在主族、全规格族与残差族中均显著，`STICKY_IDENTITY` 只通过主族；其余候选与 `CROWD1/2/3` 均不显著，止步。`plot_redefinition_ic` 增加 `series`/`title` 参数以复用。
- **粘性席位持仓流与长期粘性身份的研究期账户对照**：新增 `strategies/member/sticky_extension.py` 与 `--experiment sticky-extension`，规则在账户结果之前写入 `docs/research.md` 3.5 节。三个试验账本（只用持仓流、粘性净多 + 持仓流确认、长期粘性身份）相对共同样本对照的 2011–2021 净收益增量均不显著（主族 q = 0.804）。只用持仓流的账本 Sharpe 1.02，但敞口低、换手高，净年化低于对照；长期粘性身份的账本净增量 +1.14%，集中在 2016–2021。均未升为正式策略。
- **后续规格：持仓流降频与持久长期粘性身份（研究期）**：看过上述结果后新命名两条规格，规则先写入 `docs/research.md` 3.5 节与 `runs/research/member_extensions/followup_protocol.json`。因子 `STICKY_PERSIST_IDENTITY`（`STICKY_PERSIST` 与 `STICKY_IDENTITY` 席位的交集，只看过覆盖率后选定）与 `STICKY_FLOW5` 的 IC 均显著。新增 `strategies/member/sticky_followup.py` 与 `--experiment sticky-followup`：每 5 日调仓和 5 日平滑都没有降低持仓流的换手（约 31%），净增量接近零；`STICKY_PERSIST_IDENTITY` 账本 Sharpe 1.27，净增量 +5.18%，但主检验未通过（p = 0.057，q = 0.171），且增量集中在 2020–2021。均未升为正式策略。`STICKY_PERSIST` 的判定抽成共用函数 `_persist_mask`，原重定义因子数值不变。
- **组合规格：持久长期粘性身份 + 粘性席位持仓流（研究期）**：看到两者前后半段互补后新命名 `STICKY_PI_FLOW`（两者截面百分位排名等权平均），规则先写入 `docs/research.md` 3.5 节与 `combo_protocol.json`。新增 `rank_composite`、`strategies/member/sticky_combo.py` 与 `--experiment sticky-combo`。组合 10 日 IC +0.0530，前后半段均约 +0.05，两成分几乎不相关；账户 Sharpe 1.38、回撤 −10.95%，但相对只用身份的净增量 +0.56% 不显著（p = 0.822），换手升高抵消了大部分毛增量，2 跳时 Sharpe 略低于只用身份。未升为正式策略。
- **`STICKY_PERSIST_BUF` 升为现役纯会员策略并声明样本外协议**：按用户要求把研究期实验账本 `STICKY_PERSIST_BUF` 原样升为 `strategies/member/sticky_persist_buf.py`。规则冻结与评估协议（唯一主检验：样本外逐日净收益差相对 `STICKY_ONLY`，HAC 20 阶双侧）先写入 `docs/research.md` 4.1.1 节，之后才读取样本外。样本外主检验为不利且显著：年化净收益差 −7.14%（95% HAC 区间 [−14.13%, −0.15%]，p = 0.045）；按协议不调整规则，策略保留可运行。
- **退役策略 `STICKY_CHAIN_RESIDUAL`**：按用户本次明确要求，从可运行策略注册表与默认回测中移除 `sticky_chain`；随后按用户要求一并删除产业链研究代码 `research/member/chain.py`、其测试、`runs/research/member_behavior/` 与全部账本，活跃文档不再提及。
- **回测产物按策略单独存放**：`runs/strategies/` 去掉 `member/`、`member_price/` 类别层，改为 `runs/strategies/<策略>/{research,oos,2011-2025}/`；基准在 `benchmark/`，纯会员策略对比统一写入 `comparison/{research,oos,2011-2025}/`（含价格因子的 `sticky_core`、`sticky_core_h20` 只保留源码与各自单策略图，不进对比与活跃文档）（同一组文件：`summary.csv`、`equity_curves.csv`、`yearly_returns.csv`、`overview.png`），不再输出分类别对比与顶层 `summary.csv`。研究期实验写入 `runs/strategies/{sticky_flow,sticky_persist}/research/`，各账本与该实验的基准为子目录。新 `comparison/` 由既有净值曲线重新汇总，未重跑回测；旧布局的对比产物经用户批准删除。删除 `download/`、`strategies/member/`、`strategies/member_price/` 中无内容的 `__init__.py`（按命名空间包导入）。
- **策略按纯会员与含价格因子分目录隔离**：将纯会员分支（`sticky_only`、`sticky_chain`、`sticky_flow`、`sticky_persist`）移至 `strategies/member/`，将保留的含价格因子分支（`sticky_core`、`sticky_core_h20`）及其复合 helper 隔离至 `strategies/member_price/`，`strategies/common.py` 不再依赖价格或互补旧模块；活跃研究报告 `docs/research.md` 仅体现纯会员研究与回测结果。
- **统一研究期/样本外与实盘容量缓冲（方案 B）**：回测默认研究期统一为 `2011–2021`（`2010` 年预热），样本外为 `2022–2025`，`--start 2016 --end 2021` 作为旧研究入口保留代码复现能力并清理旧目录产物；执行账本引入 `TARGET_CAPACITY_DAYS = 3` 与 `LIQUIDATION_DAYS = 5` 解耦设计（持仓上限收紧为 3 日平仓容量，保留最后 5 日强平窗口），消除大资金跨十年连续复利在 `ZC2205` 缩量交割前的尾部残仓问题。
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

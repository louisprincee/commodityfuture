# 商品期货会员持仓因子研究

本研究围绕中国商品期货交易所每日披露的前 20 名会员持仓与成交榜单展开，核心目标是检验低换手“粘性席位”净多头寸的截面预测能力，并构建贴近机构实盘非交割约束的跨品种纯会员策略。

**核心结论：**

- **粘性席位净多具备正向预测力，但原定义存在周期分化**：在 `2011–2021` 统一研究期内，按当日换手门槛切分出的原粘性净多整体有效，但在前半段（`2011–2015`）走弱、后半段（`2016–2021`）显著走强。
- **脱离成交排名的“持久粘性”重定义显著改善纯会员分支**：榜单未披露侧的末位比例插补敏感性检验表明，默认可见口径记零已足够稳健；真正有效的是摆脱当日成交量截断，改用“20 日连续上榜与净方向一致性”定义粘性席位（`STICKY_PERSIST`）。新定义修复了前半段预测力，并显著抬升纯会员研究期收益与夏普；叠加排名进出缓冲带后，最大回撤进一步减半。
- **持仓流残差在实盘换手约束下未形成显著净增量**：控制粘性存量后，披露净持仓流虽有正向 IC 增量，但信号切换较快，配对回测中被新增换手摩擦吞噬，暂不单列策略。
- **纯会员策略在研究期与样本外均稳健跑赢基准**：`STICKY_ONLY` 与 `STICKY_CHAIN_RESIDUAL` 在 `2011–2021` 研究期、`2022–2025` 样本外以及 `2011–2025` 连续复利账本中均以零残仓通过实盘容错约束，其中 `STICKY_ONLY` 样本外表现保持韧性。

---

## 一、研究设定与统一实盘模拟规范

研究范围严格限定为中国商品期货（上期所、大商所、郑商所、上期能源及广期所实物商品品种），金融期货仅作为黑名单在下载与校验环节显式剔除。**回测统一采用“研究期 + 样本外”两段标准分区**（并附连续复利全区间账本检验执行稳定性），因子研究阶段才进一步拆看前半段与后半段。

| 分区 | 交易日区间 | 用途与定位 |
| --- | --- | --- |
| 研究期（`research`） | `2011-01-04` – `2021-12-31` | 以完整 `2010` 年行情与榜单预热，用于因子研究与策略评估 |
| 样本外（`oos`） | `2022-01-01` – `2025-12-31` | 全部规则冻结后的独立样本外检验，不根据其结果反调参数 |
| 连续期（`combined`） | `2011-01-04` – `2025-12-31` | 同一账户跨过样本边界连续复利，检验放大资金下的持仓与退场容量边界 |

### 1.1 品种池与防前视时序

每日收盘后重算流动性池：过去一年成交额中位数达到 **5 亿元**、当日主力合约有效成交且收益合法的品种进入截面。对历史更名品种（`ME/MA`、`TC/ZC`、`RO/OI`、`WS/WH`、`ER/RI`），保留旧代码合约直至最后交易日，统一乘数口径并映射至研究标的，不改写原始成交价。

所有因子仅利用当日收盘已知的行情与榜单计算，权重于次日开盘成交；前瞻收益标签严格按同一实际合约的开盘价复利计算，跨换月不引入价差跳变，也不从封存样本外年份补齐研究期末标签。截面 IC 采用每日截面 Spearman 相关性检验，时间序列显著性采用 Newey–West HAC 标准误，并在事先划定的检验族内做 Benjamini–Hochberg（BH）校正。

### 1.2 实盘口径回测与非交割容量安全规范

纯会员策略与等权多头基准均在统一账本上按整数手运行，采用贴近机构非交割实盘的硬约束（包括“方案 B”容量安全冗余设置，将建仓容忍上限与交割前强平窗口解耦）：

| 执行维度 | 统一规则说明 |
| --- | --- |
| 资金与头寸 | 初始 1,000 万空仓起步；目标权重乘数 `2.0`（基准 `1.0`），单品种上限 `20%`（基准 `10%`），统一保证金率 `20%`，总保证金上限 `50%` |
| 成交、滑点与费用 | 次日开盘成交、结算盯市；换月先平旧再开新并双边计费；主报告按单边 `1` 跳滑点（`2` 跳结果另存 CSV） |
| 历史手续费承接 | 优先取逐日实际费率；早期缺失按“同合约前值优先、同品种此前无冲突最近费率次之”承接，严禁未来倒填，彻底缺失期禁开新仓 |
| 成交容量与强平退出 | 单日开平容量为合约截至昨日过去 20 日中位成交量的 `1%`；建仓持仓上限收紧为 `3` 日平仓容量（预留流动性冗余），到期前最后 `5` 个交易日停开并强制清仓 |

---

## 二、会员持仓因子体系与经济学逻辑

交易所每日公布多头、空头与成交前 20 名席位，我们将其拆为**存量方向**、**换手分类**与**边际增减流**三层信息。

### 2.1 核心指标定义

记品种 $i$、交易日 $t$、会员 $j$ 在多空榜持仓为 $L_{j,t}$、$S_{j,t}$，披露增减量为 $\Delta L_{j,t}$、$\Delta S_{j,t}$，成交榜成交量为 $V_{j,t}$，市场单边总持仓与总成交量为 $OI_{i,t}$、$VOL_{i,t}$。

- **全榜净多占比 `K_NET_SHARE`**：衡量同时披露双榜合约上全部上榜席位的净偏多程度：
  $$\operatorname{K\_NET\_SHARE}_{i,t} = \frac{\sum_{c\in\mathcal{C}_{i,t}}\sum_j\left(L_{j,c,t}-S_{j,c,t}\right)}{\sum_{c\in\mathcal{C}_{i,t}}OI_{c,t}}$$
- **会员持仓流 `FLOW_CONTROL`**：刻画榜单披露净头寸的单日边际变化（缺失增减量时保持中性零）：
  $$\operatorname{FLOW\_CONTROL}_{i,t} = \frac{\sum_j\Delta L_{j,t}-\sum_j\Delta S_{j,t}}{OI_{i,t}}$$
- **粘性席位净多 `STICKY_NET_SHARE`**：取多空持仓榜前 20 名，定义可见持仓 $G_{j,t}=L_{j,t}+S_{j,t}$（默认一侧未上榜按可见口径记零）。以市场换手率 $VOL_{i,t}/OI_{i,t}$ 划出高换手席位 $\mathcal{H}_{i,t}$，其余持仓席位（含未入成交榜者）归入粘性组 $\mathcal{P}_{i,t}$，计算粘性组净多占比：
  $$\operatorname{STICKY\_NET\_SHARE}_{i,t} = \frac{\sum_{j\in\mathcal{P}_{i,t}}\left(L_{j,t}-S_{j,t}\right)}{\sum_{j\in\mathcal{P}_{i,t}}\left(L_{j,t}+S_{j,t}\right)}$$
  对应地，高换手组内计算结果记为 `SPEC_NET_SHARE`。与研报 `CROWD1` 要求多、空、成交三榜同时有效不同，粘性净多保留了只见持仓的席位，且两组各以组内持仓为分母，并不互为相反数。

### 2.2 经济学解释与产业链扩展线索

- **高频换手与低频留仓的行为差异**：高频活跃席位日内进出频繁，收盘净头寸往往是日内交易或被动做市提供流动性的副产品，方向预测偏弱甚至略微反转；低换手粘性席位跨日留仓比例高，背后多为产业套保与基本面中长线资金，信息逐步反映到价格的过程使其在中短期持仓窗期内维持正向预测力。
- **跨板块产业链一致性残差（`CHAIN_RESIDUAL`）**：将席位在同板块其他品种的标准化净头寸加总，仅保留本品种与板块其余品种同向的粘性席位构成一致性净多，并在每日截面上剥离普通粘性净多排名后的残差。该因子用于挑选多空方向一致度超出单品种本身水平的品种（[chain_ic.csv](../runs/research/member_behavior/chain_ic.csv)）。
- **空头端粘性份额与产业套保溢价（`STICKY_SIZE`）**：在粘性净多最低的空头候选组内，粘性持仓集中度越高，未来 5 日相对反而越抗跌（[conditional_tests.csv](../runs/research/member_complementarity/conditional_tests.csv)），印证了卖方套保支付风险溢价导致空头端不宜盲目重仓的特征。

---

## 三、研究期深度排查与纯会员新进展（2011–2021）

### 3.1 前后半段符号分化与累计 IC 诊断

拉长周期至 2011 年后，按原成交门槛切分的粘性与高换手两组呈现出清晰的前后段镜像切换：粘性组在前半段整体走平或偏弱、2016 年之后持续走强；高换手组恰好相反。

| 共同有效样本（10 日 IC） | 2011–2021 全期 | 2011–2015 前半段 | 2016–2021 后半段 | 日均覆盖品种 |
| --- | ---: | ---: | ---: | ---: |
| `IMP00_STICKY`（原定义粘性） | +0.0199 | −0.0037 | +0.0396 | 30.1 |
| `IMP00_SPEC`（原定义高换手） | −0.0030 | +0.0170 | −0.0198 | 30.0 |

![粘性与高换手席位净多的累计截面 IC](../runs/research/member_review/cumulative_ic.png)

跨期对比发现，即便固定同一批核心品种或逐一剔除板块，该转换依然存在；同时前半段中因未入成交榜而被迫归入粘性组的席位比例显著更高，说明依赖单日成交量榜单来识别低换手席位在早期市场会产生噪声混入。

### 3.2 榜单未披露一侧的插补敏感性检验

针对“单边上榜席位未披露一侧默认记零”的疑虑，我们在共同样本下将缺失侧按当日另一侧末位持仓的 `0 / 0.5 / 0.8 / 1.0` 倍做比例填充：

![四档末位比例插补 IC 与 95% 置信区间](../runs/research/member_review/imputation_comparison.png)

- **结论**：粘性组四档 IC 均窄幅收敛且两两配对差值不显著，说明现有可见口径记零足够稳健；高换手组对插补敏感但自身各档均不显著（完整检验见 [factor_ic.csv](../runs/research/member_review/factor_ic.csv) 与 [paired_differences.csv](../runs/research/member_review/paired_differences.csv)）。

### 3.3 持仓流残差与 `Sticky + Flow` 配对回测

每日剥离粘性存量影响后，净持仓流残差在前后半段均呈现显著正向 IC；但在同样本、同成本口径下对比回测发现，高频持仓流大幅抬高换手，额外摩擦成本基本抵消了毛收益增量，暂不独立新增为正式策略（详见 [flow_incremental.csv](../runs/research/member_review/flow_incremental.csv)、[summary.csv](../runs/strategies/comparison/sticky_flow/summary.csv) 与 [paired_return_gains.csv](../runs/strategies/comparison/sticky_flow/paired_return_gains.csv)）。

| 残差因子（控制粘性净多） | 10 日全期 IC | 95% HAC 区间 | 主族 q | 全规格 q | 2011–2015 | 2016–2021 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `FLOW_CONTROL`（净持仓流） | +0.0195 | `[+0.0103, +0.0287]` | `6.0e-5` | `2.1e-4` | +0.0276 | +0.0127 |
| `FLOW_CHANGE5`（5 日流变化） | +0.0113 | `[+0.0053, +0.0173]` | `2.2e-4` | `8.8e-4` | +0.0123 | +0.0105 |

| 2011–2021 配对回测（1 跳） | 净年化 | Sharpe | 最大回撤 | 日均换手 | 实际敞口 | 相对基线净增量 | 主族 q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `STICKY_MATCHED` | +4.80% | 0.57 | −24.23% | 17.14% | 1.37 | 基准 | — |
| `FLOW_MATCHED` | +1.62% | 0.43 | −9.05% | 30.85% | 0.78 | −3.18% | 0.424 |
| `STICKY_FLOW_MATCHED` | +5.97% | 0.71 | −21.29% | 27.11% | 1.59 | +1.18% | 0.424 |

![Sticky + Flow 增量试验净值与回撤](../runs/strategies/comparison/sticky_flow/overview.png)

### 3.4 粘性席位重定义与研究期实验结果

为彻底摆脱对当日成交量截断榜单的依赖，结合前期连续上榜与方向一致性的保留证据（[retained_profile_evidence.csv](../runs/research/member_review/retained_profile_evidence.csv)），本轮在研究期冻结测试了四组基于持仓行为本身的替代粘性定义：

- **`STICKY_PERSIST`（持久同向席位）**：过去 20 日至少上榜 15 天，且净头寸方向一致性不低于 0.80，完全不看成交榜。
- **`STICKY_LOW_ADJUST`（低调仓强度席位）**：过去 5 日平均持仓增减比例不高于同日同行中位数。
- **`STICKY_UNILATERAL`（高单边席位）**：原粘性组内剔除双边对称持仓席位。
- **`STICKY_PERSIST_PURE`（纯单边持久席位）**：在 `STICKY_PERSIST` 基础上进一步剔除双边对称持仓席位。

#### （1）截面 IC 检验

| 重定义粘性因子（2011–2021） | 1 日 IC | 5 日 IC | 10 日 IC | 20 日 IC | 10 日 p | 主族 q | 全 48 项 q | 2011–2015 | 2016–2021 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `STICKY_NET_SHARE`（基线） | +0.0200 | +0.0224 | +0.0199 | +0.0230 | 0.094 | — | — | −0.0037 | +0.0396 |
| `STICKY_PERSIST` | +0.0135 | +0.0225 | +0.0280 | +0.0343 | 0.025 | 0.053 | 0.047 | +0.0085 | +0.0444 |
| `STICKY_LOW_ADJUST` | +0.0166 | +0.0200 | +0.0207 | +0.0235 | 0.076 | 0.101 | 0.121 | +0.0033 | +0.0353 |
| `STICKY_UNILATERAL` | +0.0163 | +0.0195 | +0.0186 | +0.0206 | 0.115 | 0.115 | 0.173 | −0.0032 | +0.0369 |
| `STICKY_PERSIST_PURE` | +0.0114 | +0.0220 | +0.0264 | +0.0294 | 0.026 | 0.053 | 0.047 | +0.0126 | +0.0379 |

![粘性席位重定义 IC 与周期对比](../runs/research/member_review/seat_redefinition_ic.png)

结果显示，只有 `STICKY_PERSIST` 与 `STICKY_PERSIST_PURE` 通过全族多重校正，成功把早期负值修复为正值，并避免了对未披露成交量席位的主观归类（[seat_redefinition_ic.csv](../runs/research/member_review/seat_redefinition_ic.csv) · [seat_redefinition_gains.csv](../runs/research/member_review/seat_redefinition_gains.csv)）。

#### （2）账户级对照与排名缓冲带验证

在研究期内按同一非交割与成本框架对比原基线、持久粘性以及“进入前/后 10%、跌出前/后 20% 方平”的排名缓冲带（[summary.csv](../runs/strategies/comparison/sticky_persist/summary.csv) · [paired_return_gains.csv](../runs/strategies/comparison/sticky_persist/paired_return_gains.csv)）：

| 2011–2021 研究期实验账本 | 净年化 | Sharpe | 最大回撤 | 日均换手 | 期末权益（万元） | 相对基线净增量 (`11–15 / 16–21`) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `STICKY_ONLY_REF`（原基线） | +4.80% | 0.57 | −24.23% | 17.14% | 1,635.2 | 基准 |
| `STICKY_ONLY_BUF`（原定义 + 缓冲带） | +4.83% | 0.56 | −24.29% | 16.58% | 1,635.5 | `+0.03%` (`-0.35% / +0.34%`) |
| `STICKY_PERSIST` | +6.96% | 0.83 | −16.70% | 17.77% | 2,075.9 | `+2.16%` (`+3.19% / +1.30%`) |
| `STICKY_PERSIST_PURE` | +5.63% | 0.72 | −16.30% | 20.94% | 1,800.5 | `+0.83%` (`+3.24% / -1.18%`) |
| `STICKY_PERSIST_BUF`（持久 + 缓冲带） | +6.86% | 0.79 | −12.05% | 17.48% | 2,048.2 | `+2.06%` (`+3.47% / +0.89%`) |

![粘性重定义与排名缓冲带研究期对比](../runs/strategies/comparison/sticky_persist/overview.png)

实验清晰分离了两项机制的作用：**仅给原定义加缓冲带无法改善收益与回撤**，核心驱动力来自用“持续性 + 方向一致性”重定义粘性席位；而在新定义上叠加排名缓冲带，则在基本保留全部收益增量的前提下把最大回撤压低了一半。

### 3.5 已退役及止步方向

`STICKY_MINUS_SPEC`（粘性减高换手等权差值）在研究期内未表现出优于单用粘性组的增量，相关实现已退役归档至 [history.md](history.md)；研报 `CROWD1/2/3` 反转信号及静态公司股权关系表均未通过稳定性检验，不再扩展。

---

## 四、纯会员策略构建与全周期回测表现

### 4.1 纯会员策略规则

| 策略代号 | 持有期 | 截面选样逻辑 |
| --- | ---: | --- |
| `STICKY_ONLY` | 10 日 | 在流动性池内按 `STICKY_NET_SHARE` 排序，等权做多前 10%、做空后 10% |
| `STICKY_CHAIN_RESIDUAL` | 10 日 | 共同有效 ≥ 20 品种时，先按粘性取首尾 20%，再按产业链一致性残差各选一半 |

### 4.2 研究期、样本外与连续复利绩效总览（1 跳滑点）

| 策略 / 基准 | 研究期 `2011–2021` (净年化 / Sharpe / MDD / 换手) | 样本外 `2022–2025` (净年化 / Sharpe / MDD / 换手) | 连续期 `2011–2025` (净年化 / Sharpe / MDD) |
| --- | ---: | ---: | ---: |
| `STICKY_ONLY` | +4.80% / 0.57 / −24.23% / 17.1% | +10.07% / 1.06 / −12.09% / 25.8% | +6.33% / 0.72 / −24.23% |
| `STICKY_CHAIN_RESIDUAL` | +5.01% / 0.58 / −16.13% / 19.3% | +4.48% / 0.48 / −22.57% / 30.5% | +4.78% / 0.54 / −22.50% |
| `LONG_ONLY_BMK`（基准） | +2.50% / 0.34 / −33.83% / 3.3% | +0.26% / 0.04 / −15.05% / 4.6% | +1.87% / 0.25 / −33.83% |

[STICKY_ONLY 研究期 CSV](../runs/strategies/member/sticky_only/research/summary.csv) · [STICKY_ONLY 样本外 CSV](../runs/strategies/member/sticky_only/oos/summary.csv) · [STICKY_CHAIN 研究期 CSV](../runs/strategies/member/sticky_chain/research/summary.csv) · [STICKY_CHAIN 样本外 CSV](../runs/strategies/member/sticky_chain/oos/summary.csv)

![纯会员策略与等权基准 · 2011–2025 全区间表现](../runs/strategies/member/comparison/2011-2025/plots/overview.png)

![纯会员策略 · 研究期 2011–2021](../runs/strategies/member/comparison/research/plots/performance.png)

![纯会员策略 · 样本外 2022–2025](../runs/strategies/member/comparison/oos/plots/performance.png)

**核心结论：**
- **`STICKY_ONLY` 跨越周期保持韧性**：尽管在 `2011–2015` 商品熊市中表现温和，但在后半段与商品多头几乎走平的 `2022–2025` 样本外周期取得更高夏普与更低回撤。
- **`STICKY_CHAIN_RESIDUAL` 在样本外受换手摩擦压制**：研究期回撤显著低于 `STICKY_ONLY`，但样本外因换手率升至三成以上，净表现落后于单因子版本。

---

## 附录 A · 基础研报对照指标定义

记号同第 2.1 节，席位单日持仓变动记为 $D_{j,t}=\lvert L_{j,t}-L_{j,t-1}\rvert+\lvert S_{j,t}-S_{j,t-1}\rvert$：

- **拥挤度（`CROWD1/2/3`）**：多空持仓与成交均披露且大于零的席位中，挑选比值 $R^{(k)}_{j,t}\ge M^{(k)}_{i,t}$ 的席位组计算净多占比：$\text{CROWD1}$ 比较换手率；$\text{CROWD2}$ 比较单日持仓变动占成交量之比；$\text{CROWD3}$ 取近 3 日累计比值。
- **阵营合力（`POWER`）**：窗口内每天净头寸完整的会员按净多取首尾各 5 家，组内平均相关性与总净头寸斜率乘积之和。
- **蜘蛛网（`SPIDER`）**：按当日多空披露增减合计量异号判定方向。

原研报复现对照明细保留于 [five_day_ic.csv](../runs/research/xs_position/five_day_ic.csv)、[paper_ic.csv](../runs/research/xs_position/paper_ic.csv) 与 [factor_horizons.csv](../runs/research/xs_robustness/factor_horizons.csv)。

---

## 附录 B · 复现命令

```bash
# 1. 构建统一日频面板（含 2010 年预热）
PYTHONDONTWRITEBYTECODE=1 python -m src.data

# 2. 研究期（2011–2021）纯会员因子复核
PYTHONDONTWRITEBYTECODE=1 python -m research.member.review --jobs 8

# 3. 纯会员策略统一回测（研究期、样本外及 2011–2025 连续账户）
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --family member --research --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --family member --oos --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --family member --start 2011 --end 2025 --jobs 8

# 4. 研究期纯会员对照实验（Sticky + Flow 与 Sticky Persist + 缓冲带）
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --experiment sticky-flow --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --experiment sticky-persist --jobs 8

# 5. 全量合成数据单元测试
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```

# 商品期货会员持仓因子研究

本研究检验中国商品期货交易所前 20 名会员持仓与成交榜的预测能力，并评价会员信号在整数手、历史费用和非交割约束下的账户表现。

当前保留三套策略：**粘性净多**、**持久粘性身份**、**粘性净多 + 基差动量 + 负偏度**。本报告只比较前两套纯会员策略；含价格的三因子组合保留源码、账本与单策略图，不进入报告绩效表或对比图。用户授权的研究期持仓流平滑实验不注册为现役策略。退役与清理记录见 [history.md](history.md)。

保留名单来自用户的明确选择。账面收益高低、增量是否显著、样本外是否延续，是三件不同的事。

---

## 一、研究设定

研究范围为中国商品期货。金融期货仅作为黑名单在下载与校验环节剔除。回测采用研究期与样本外两段标准分区，并另跑一条跨边界的连续账户。

| 分区 | 区间 | 用途 |
| --- | --- | --- |
| 研究期 | 2011-01-04 – 2021-12-31 | 因子研究与策略评估；完整 2010 年预热 |
| 样本外 | 2022-01-01 – 2025-12-31 | 事前固定规格的描述性重跑；该分区此前已读，不是完全独立的确认样本 |
| 连续账户 | 2011-01-04 – 2025-12-31 | 同一账户跨过样本边界连续复利 |

### 1.1 品种池与时序

每日收盘后重算流动性池：过去 242 个交易日成交额中位数达到 5 亿元、当日主力合约有效成交且收益合法的品种进入截面。对历史更名品种保留旧代码合约直至最后交易日，统一乘数口径并映射至研究标的，不改写原始成交价。

因子只使用当日收盘已知的行情与榜单。权重于次日开盘成交。前瞻收益按同一实际合约开盘价复利，跨换月不引入价差跳变。截面 IC 为每日截面 Spearman 相关；时间序列显著性用 Newey–West HAC 标准误，并在事先划定的检验族内做 Benjamini–Hochberg 校正。

### 1.2 回测执行

纯会员策略与等权多头基准在统一账本上按整数手运行。

| 维度 | 规则 |
| --- | --- |
| 资金与头寸 | 初始 1,000 万空仓；目标权重乘数 2.0，基准为 1.0；单品种上限 20%，基准为 10%；统一保证金率 20%，总保证金上限 50% |
| 成交与费用 | 次日开盘成交、结算盯市；换月先平旧再开新并双边计费；主报告按单边 1 跳滑点，2 跳结果另存 |
| 历史手续费 | 优先取逐日实际费率；早期缺失按同合约前值优先、同品种此前无冲突最近费率次之承接；严禁未来倒填 |
| 容量与退出 | 单日开平容量为合约截至昨日过去 20 日中位成交量的 1%；建仓上限为 3 日平仓容量；到期前最后 5 个交易日停开并强制清仓 |

---

## 二、会员持仓因子定义

交易所每日公布多头、空头与成交前 20 名席位。记品种 \(i\)、交易日 \(t\)、会员 \(j\) 的多头持仓为 \(L_{j,t}\)、空头持仓为 \(S_{j,t}\)、披露增减量为 \(\Delta L_{j,t}\)、\(\Delta S_{j,t}\)，成交榜成交量为 \(V_{j,t}\)，市场单边总持仓与总成交量为 \(OI_{i,t}\)、\(VOL_{i,t}\)。

可见持仓与净头寸定义为

\[
G_{j,t}=L_{j,t}+S_{j,t},\qquad N_{j,t}=L_{j,t}-S_{j,t}.
\]

一侧未上榜时，该侧按可见口径记为零。

### 2.1 粘性与高换手分组

席位换手率与市场换手率分别为

\[
\tau_{j,t}=\frac{V_{j,t}}{G_{j,t}},\qquad \tau^{m}_{i,t}=\frac{VOL_{i,t}}{OI_{i,t}}.
\]

高换手组 \(\mathcal{H}_{i,t}\) 为同时上多空持仓榜与成交榜、且 \(\tau_{j,t}\ge\tau^{m}_{i,t}\) 的席位；其余有持仓的席位归入粘性组 \(\mathcal{P}_{i,t}\)，包括未上成交榜的席位。

粘性席位净多占比为

\[
\operatorname{STICKY\_NET\_SHARE}_{i,t}
=\frac{\sum_{j\in\mathcal{P}_{i,t}}N_{j,t}}{\sum_{j\in\mathcal{P}_{i,t}}G_{j,t}}.
\]

高换手组内同口径结果记为 \(\operatorname{SPEC\_NET\_SHARE}_{i,t}\)。两组各以组内持仓为分母，并不互为相反数。

经济学直觉是：高频活跃席位收盘净头寸多为日内交易副产品，方向预测偏弱；低换手粘性席位跨日留仓比例高，信息更可能逐步反映到价格。

### 2.2 全榜净多与持仓流

同时披露双榜的合约集合记为 \(\mathcal{C}_{i,t}\)。全榜净多占比为

\[
\operatorname{K\_NET\_SHARE}_{i,t}
=\frac{\sum_{c\in\mathcal{C}_{i,t}}\sum_j\bigl(L_{j,c,t}-S_{j,c,t}\bigr)}{\sum_{c\in\mathcal{C}_{i,t}}OI_{c,t}}.
\]

会员持仓流为

\[
\operatorname{FLOW\_CONTROL}_{i,t}
=\frac{\sum_j\Delta L_{j,t}-\sum_j\Delta S_{j,t}}{OI_{i,t}}.
\]

任一应披露增减量缺失时，该侧合计记为缺失，不补零。

### 2.3 粘性席位重定义

为减少对单日成交榜截尾的依赖，另定义四组基于持仓行为的粘性掩码。席位在 \(t\) 日有持仓记为 \(H_{j,t}=1\)，上榜记为 \(R_{j,t}=1\)。

**持久同向** \(\operatorname{STICKY\_PERSIST}\)：

\[
\sum_{u=0}^{19}R_{j,t-u}\ge 15,
\qquad
\left\lvert\frac{1}{20}\sum_{u=0}^{19}\operatorname{sign}(N_{j,t-u})\right\rvert\ge 0.8.
\]

当前实现中，方向均值要求 20 个有效上榜观测，因此实际比名义的“至少上榜 15 日”更严格。既有结果对应这一严格实现。

**低调仓强度** \(\operatorname{STICKY\_LOW\_ADJUST}\)：过去 5 日平均 \((|\Delta L|+|\Delta S|)/G\) 不高于同日同行中位数。

**去双边对锁** \(\operatorname{STICKY\_UNILATERAL}\)：原粘性组内剔除

\[
\frac{|N_{j,t}|}{G_{j,t}}<0.30
\]

的双边对称席位。\(\operatorname{STICKY\_PERSIST\_PURE}\) 在持久同向上再做同样剔除。

### 2.4 扩展因子

**榜内换手门槛** \(\operatorname{STICKY\_TOP20}\)：把门槛从全市场换手率改为同日成交榜席位 \(\tau_{j,t}\) 的中位数。

**品种内标准化与变化**：

\[
\operatorname{STICKY\_Z250}_{i,t}
=\frac{S_{i,t}-\mu_{i,t-1:t-250}}{\sigma_{i,t-1:t-250}},
\qquad
\operatorname{STICKY\_CHG20}_{i,t}=S_{i,t}-S_{i,t-20},
\]

其中 \(S_{i,t}\) 为原粘性净多，均值与标准差不含当日，至少 60 个历史观测。

**粘性席位持仓流**：

\[
\operatorname{STICKY\_FLOW}_{i,t}
=\frac{\sum_{j\in\mathcal{P}_{i,t}}\Delta L_{j,t}-\sum_{j\in\mathcal{P}_{i,t}}\Delta S_{j,t}}{OI_{i,t}}.
\]

任一粘性席位应有的增减量缺失时整项记为缺失。5 日与 10 日均值为

\[
\operatorname{STICKY\_FLOWk}_{i,t}
=\frac{1}{k}\sum_{u=0}^{k-1}\operatorname{STICKY\_FLOW}_{i,t-u},
\qquad k\in\{5,10\},
\]

需 \(k\) 个有效观测。

**长期粘性身份** \(\operatorname{STICKY\_IDENTITY}\)：过去 242 日不含当日，记上榜持仓天数为 \(D^{H}_{j,t}\)、其中被判为粘性的天数为 \(D^{P}_{j,t}\)。要求

\[
D^{H}_{j,t}\ge 60,
\qquad
\frac{D^{P}_{j,t}}{D^{H}_{j,t}}\ge 0.8,
\]

滚动窗口至少需 120 个历史日。今日有持仓的合格席位按净多占比聚合。

**持久长期粘性身份** \(\operatorname{STICKY\_PERSIST\_IDENTITY}\)：同时通过长期身份与近期持久同向的席位交集，再按

\[
\frac{\sum_{j\in\mathcal{M}_{i,t}}N_{j,t}}{\sum_{j\in\mathcal{M}_{i,t}}G_{j,t}}
\]

聚合到品种。取交集前只检查覆盖率，未使用收益。

**席位历史记录** \(\operatorname{SEAT\_SKILL\_NET}\)：席位过去 242 日

\[
\operatorname{sign}(N_{j,u-1})\times r_{u}
\]

的均值大于零，至少 60 个观测；\(r_u\) 为开盘到下一开盘、按前一日收盘主力映射的收益。

### 2.5 兴业证券拥挤度与一致性

记席位单日持仓变动

\[
D_{j,t}=\lvert L_{j,t}-L_{j,t-1}\rvert+\lvert S_{j,t}-S_{j,t-1}\rvert.
\]

只在多空持仓与成交均披露且大于零的席位中，挑选比值 \(R^{(k)}_{j,t}\ge M^{(k)}_{i,t}\) 的席位组，计算组内净多占比：

\[
\operatorname{CROWDk}_{i,t}
=\frac{\sum_{j\in\mathcal{A}^{(k)}_{i,t}}N_{j,t}}{\sum_{j\in\mathcal{A}^{(k)}_{i,t}}G_{j,t}}.
\]

三组门槛分别为：

- \(\operatorname{CROWD1}\)：席位换手率对市场换手率，\(R=V/G\)，\(M=VOL/OI\)；
- \(\operatorname{CROWD2}\)：单日持仓变动占成交量之比，\(R=D/V\)，\(M=|\Delta OI|/VOL\)；
- \(\operatorname{CROWD3}\)：近 3 日累计比值，分子分母分别对 \(D\) 与 \(V\)、\(|\Delta OI|\) 与 \(VOL\) 滚动求和。

商品采用反转取号。事件信号另要求连续 3 日收涨或收跌确认，分位阈值为过去 250 日、不含当日的 90% / 10% 分位。

一致性指标在窗口内净头寸完整的会员中，按净多取首尾各 5 家，计算组内平均相关性与总净头寸斜率，再合成阵营合力

\[
\operatorname{POWER}
=C_{\text{long}}\cdot\operatorname{Slope}_{\text{long}}
-C_{\text{short}}\cdot\bigl(-\operatorname{Slope}_{\text{short}}\bigr).
\]

只在一侧上榜时，未上榜一侧按当日该榜末位持仓的一半填补。

### 2.6 开源证券蜘蛛网

按当日多空披露增减合计量异号判定方向：

\[
\operatorname{SPIDER}_{i,t}
=
\begin{cases}
+1, & \sum\Delta L>0\ \text{且}\ \sum\Delta S<0,\\
-1, & \sum\Delta L<0\ \text{且}\ \sum\Delta S>0,\\
0, & \text{其他完整披露日}.
\end{cases}
\]

活跃口径另要求同日上成交榜。原研报复现明细保留于 [paper_ic.csv](../runs/research/xs_position/paper_ic.csv) 与 [five_day_ic.csv](../runs/research/xs_position/five_day_ic.csv)。

### 2.7 价格因子

基差动量为近月与次近月各自同合约日收益的 242 日复合收益之差：

\[
\operatorname{BASIS\_MOM\_242}_{i,t}
=\Bigl(\prod_{u=0}^{241}(1+r^{(1)}_{i,t-u})-1\Bigr)
-\Bigl(\prod_{u=0}^{241}(1+r^{(2)}_{i,t-u})-1\Bigr).
\]

负向偏度为近月日收益 242 日滚动偏度取负，至少 200 个观测：

\[
\operatorname{NEG\_SKEW\_242}_{i,t}=-\operatorname{Skew}\bigl(r^{(1)}_{i,t-241:t}\bigr).
\]

---

## 三、研究期因子结果

### 3.1 原定义粘性与高换手

拉长到 2011 年后，两组在前后半段呈镜像切换。

| 共同有效样本，10 日 IC | 2011–2021 | 2011–2015 | 2016–2021 |
| --- | ---: | ---: | ---: |
| 原粘性净多 | +0.0199 | −0.0037 | +0.0396 |
| 原高换手净多 | −0.0030 | +0.0170 | −0.0198 |

![粘性与高换手席位净多的累计截面 IC](../runs/research/member_review/cumulative_ic.png)

固定核心品种或逐一剔除板块后，该转换依然存在。前半段未入成交榜而归入粘性组的席位比例更高，说明单日成交量榜识别低换手席位在早期噪声更大。构成诊断见下图与 [sticky_composition_by_year.csv](../runs/research/member_review/sticky_composition_by_year.csv)。

![2016 年前后符号切换的诊断](../runs/research/member_review/sticky_composition.png)

### 3.2 未披露一侧插补

将缺失侧按当日另一侧末位持仓的 0 / 0.5 / 0.8 / 1.0 倍填充。粘性组四档 IC 窄幅收敛，两两配对差值不显著；高换手组对插补敏感，但各档均不显著。

![四档末位比例插补 IC 与 95% 置信区间](../runs/research/member_review/imputation_comparison.png)

完整检验见 [factor_ic.csv](../runs/research/member_review/factor_ic.csv) 与 [paired_differences.csv](../runs/research/member_review/paired_differences.csv)。

### 3.3 持仓流残差

控制原粘性净多后，净持仓流残差在前后半段均为正。账户实验已按用户要求删除，历史结论见 [history.md](history.md)。

| 残差因子，控制粘性净多 | 10 日全期 IC | 主族 q | 2011–2015 | 2016–2021 |
| --- | ---: | ---: | ---: | ---: |
| 净持仓流 | +0.0195 | \(6.0\times10^{-5}\) | +0.0276 | +0.0127 |
| 5 日流变化 | +0.0113 | \(2.2\times10^{-4}\) | +0.0123 | +0.0105 |

### 3.4 粘性席位重定义

| 因子，2011–2021 | 1 日 | 5 日 | 10 日 | 20 日 | 主族 q | 2011–2015 | 2016–2021 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原粘性净多 | +0.0200 | +0.0224 | +0.0199 | +0.0230 | — | −0.0037 | +0.0396 |
| 持久同向 | +0.0135 | +0.0225 | +0.0280 | +0.0343 | 0.053 | +0.0085 | +0.0444 |
| 低调仓强度 | +0.0166 | +0.0200 | +0.0207 | +0.0235 | 0.101 | +0.0033 | +0.0353 |
| 去双边对锁 | +0.0163 | +0.0195 | +0.0186 | +0.0206 | 0.115 | −0.0032 | +0.0369 |
| 持久 + 去对锁 | +0.0114 | +0.0220 | +0.0264 | +0.0294 | 0.053 | +0.0126 | +0.0379 |

![粘性席位重定义 IC 与周期对比](../runs/research/member_review/seat_redefinition_ic.png)

只有持久同向与持久 + 去对锁通过全族多重校正，并把早期负值修复为正值。完整结果见 [seat_redefinition_ic.csv](../runs/research/member_review/seat_redefinition_ic.csv)。

### 3.5 扩展研究

规格与检验族在任何结果之前写入 [protocol.json](../runs/research/member_extensions/protocol.json)，只使用研究期。

| 扩展因子，2011–2021 | 10 日 IC | 主族 q | 残差 10 日 IC | 2011–2015 | 2016–2021 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原粘性净多 | +0.0199 | — | — | −0.0037 | +0.0396 |
| 榜内换手门槛 | +0.0174 | 0.234 | −0.0024 | −0.0059 | +0.0370 |
| 品种内 z 分数 | +0.0069 | 0.577 | −0.0086 | +0.0037 | +0.0095 |
| 20 日变化 | +0.0050 | 0.577 | −0.0026 | +0.0145 | −0.0029 |
| 粘性席位持仓流 | +0.0341 | \(1.4\times10^{-12}\) | +0.0288 | +0.0468 | +0.0234 |
| 长期粘性身份 | +0.0301 | 0.038 | +0.0210 | +0.0078 | +0.0488 |
| 历史记录为正 | +0.0258 | 0.065 | +0.0206 | +0.0369 | +0.0165 |

![粘性净多扩展因子 IC](../runs/research/member_extensions/extension_ic.png)

粘性席位持仓流是本轮唯一在主族、全规格族和残差族中都显著的因子。长期粘性身份通过主族，增量集中在 2016 以后。榜内门槛、z 分数与 20 日变化止步。

后续规格在结果之前单独写定：

| 后续因子，2011–2021 | 10 日 IC | 主族 q | 残差 10 日 IC | 2011–2015 | 2016–2021 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 持久长期粘性身份 | +0.0376 | 0.0012 | +0.0293 | +0.0116 | +0.0595 |
| 粘性席位持仓流 5 日均值 | +0.0412 | \(6.1\times10^{-6}\) | +0.0331 | +0.0607 | +0.0248 |

两者前后半段强弱相反。等权截面排名组合 \(\operatorname{STICKY\_PI\_FLOW}\) 的 10 日 IC 为 +0.0530，两成分几乎不相关。该组合的策略账户已按用户授权删除；因子 IC 研究保留。

![持久长期粘性身份与粘性席位持仓流的等权组合 IC](../runs/research/member_extensions/combo_ic.png)

### 3.6 拥挤度三因子

按研报反转方向取号。主族为三因子 × 1 日、5 日的全期截面 IC。

| 因子，2011–2021 | 截面 1 日 | 截面 5 日 | 时序 1 日均值 | 时序 5 日均值 |
| --- | ---: | ---: | ---: | ---: |
| CROWD1 | +0.0041 | −0.0015 | +0.0073 | +0.0013 |
| CROWD2 | −0.0038 | −0.0085 | −0.0064 | −0.0168 |
| CROWD3 | −0.0060 | −0.0114 | −0.0074 | −0.0193 |

![拥挤度三因子截面与时序 IC](../runs/research/member_extensions/crowding_ic.png)

三个因子在 1 日和 5 日上均无显著截面或时序 IC。CROWD2/3 按反转取号后均值为负，即原始方向略偏顺势，与研报反转假设相反。

### 3.7 已退役及止步方向

粘性减高换手差值因子未表现出优于单用粘性组的增量。静态公司股权关系表未通过稳定性检验。榜内门槛、z 分数与 20 日变化止步于研究期 IC。以上方向均不再扩展。

---

## 四、当前策略与账户结果

### 4.1 两套纯会员策略

| 策略入口 | 账户名 | 排序规则 |
| --- | --- | --- |
| `sticky_net` | `STICKY_NET` | 原粘性席位净多占比 |
| `persist_identity` | `PERSIST_IDENTITY` | 持久长期粘性身份净多占比 |

共同选样与持有规则：当日有效候选不少于 8 个时，两端各取 10%；同分按固定品种顺序处理；每日建立十分之一批次，10 日重叠持有；次日开盘执行。

持久粘性身份先按行为筛席位，再把合格席位的当日持仓聚合到品种：

1. 过去 242 日不含当日，上榜日中至少 80% 属于原粘性组，上榜至少 60 日；
2. 今日有持仓，且通过近期持久同向筛选；
3. 交集内席位计算 \(\sum N/\sum G\)。

按用户要求，身份策略只要求自身分数有效，不再要求持仓流或原粘性净多同时有效。新账户名为 `PERSIST_IDENTITY`，区别于旧共同样本的 `STICKY_PI_REF`。规格在本轮 OOS 重跑前写入 [evaluation_protocol.json](../runs/meta/evaluation_protocol.json)。OOS 此前已读，结果仅作描述性评价。

### 4.2 账户结果

净年化为日均净收益 × 242。表内依次为净年化 / Sharpe / 最大回撤 / 日均换手。单边 1 跳。

| 策略 / 基准 | 研究期 2011–2021 | OOS 2022–2025 | 连续账户 2011–2025 |
| --- | --- | --- | --- |
| 粘性净多 | +4.80% / 0.57 / −24.23% / 17.14% | +10.07% / 1.06 / −12.09% / 25.78% | +6.33% / 0.72 / −24.23% / 19.55% |
| 持久粘性身份 | +10.17% / 1.27 / −14.13% / 17.43% | +9.97% / 1.09 / −9.22% / 26.65% | +10.53% / 1.25 / −14.13% / 20.09% |
| 等权多头基准 | +2.50% / 0.34 / −33.83% / 3.34% | +0.26% / 0.04 / −15.05% / 4.60% | +1.87% / 0.25 / −33.83% / 3.68% |

[研究期汇总](../runs/strategies/comparison/research/summary.csv) · [OOS 汇总](../runs/strategies/comparison/oos/summary.csv) · [连续账户汇总](../runs/strategies/comparison/2011-2025/summary.csv)

![当前纯会员策略 · 研究期](../runs/strategies/comparison/research/overview.png)

![当前纯会员策略 · 已读 OOS](../runs/strategies/comparison/oos/overview.png)

![当前纯会员策略 · 独立连续账户](../runs/strategies/comparison/2011-2025/overview.png)

前两个分区分别空仓开户；连续账户跨边界保持资金、仓位与持有队列。1/2 跳均保存于各自 `summary.csv`，报告只展示 1 跳。执行审计见 [execution_audit.csv](../runs/meta/execution_audit.csv)；源码与产物清单见 [manifest.json](../runs/meta/manifest.json)。

### 4.3 持仓流平滑探索

5 日持仓流组合的历史账户诊断未显示稳定正向净收益增量，退役结论见 [history.md](history.md)。本轮新增研究期 10 日平滑规格：在身份、FLOW5、FLOW10 同时有效的共同样本内比较，不读 OOS。

| 同样本研究期规格 | 10 日截面 IC | 净年化 | Sharpe | 最大回撤 | 日均换手 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 身份 | +0.0376 | +10.17% | 1.27 | −14.13% | 17.43% |
| 身份 + 5 日流 | +0.0530 | +10.74% | 1.38 | −10.95% | 28.99% |
| 身份 + 10 日流 | +0.0483 | +7.92% | 0.99 | −13.58% | 27.36% |

唯一账户主检验为 10 日组合减去 5 日组合的全期逐日净收益差。结果为不利方向，研究期探索不支持将 10 日版本替换 5 日。

![5/10 日持仓流平滑的共同样本预测比较](../runs/research/member_flow_smoothing/comparison.png)

![5/10 日持仓流平滑的研究期账户比较](../runs/strategies/identity_flow_smoothing/research/overview.png)

---

## 附录 · 复现命令

```bash
PYTHONDONTWRITEBYTECODE=1 python -m src.data
PYTHONDONTWRITEBYTECODE=1 python -m research.member.review --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m research.member.extensions --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --family member --start 2011 --end 2025 --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine sticky_basis_skew --research --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m research.member.extensions --flow-smoothing --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m backtest.engine --flow-smoothing-study --jobs 8
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```

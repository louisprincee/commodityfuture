# Chinese Commodity Futures Factor Research

## 1. Scope and Objective

The research covers Chinese commodity futures only. Financial futures are configuration exclusions rather than trading books.

The objective is to test whether a small set of pre-defined price and member-position factors has stable predictive content and can support one transparent cross-sectional strategy.

The active strategy is `STICKY_CORE_2X`.

## 2. Samples and Timing

| Partition | Date range | Use |
|---|---|---|
| Warm-up | 2010-01-04 to 2015-12-31 | Rolling-window warm-up only |
| Research | 2016-01-04 to 2021-12-31 | Parameter-free factor and strategy evaluation |
| Out of sample | 2022-01-04 to 2025-12-31 | Sealed partition evaluation after rule freeze |

The liquidity universe is formed at the close of day \(t\). It is not revised with future turnover:

- The median of daily turnover over the most recent 242 trading days must be at least RMB 500 million.
- A 242-day window is required. There is no separate listing-age or 60-observation threshold.
- The dominant contract must have positive volume on day \(t\).
- The dominant-contract return must be valid.
- EC is excluded.

A factor on day \(t\) uses only prices and member rankings known by the close of \(t\). Weights formed at the close of \(t\) first trade at the next session's open.

Forward labels use the next main contract, which avoids mixing one contract's open-to-open return across a roll.

## 3. Factor Definitions

### 3.1 Basis-momentum factor

For contract-specific close-to-close returns \(r^{(1)}\) and \(r^{(2)}\) on the dominant and second-dominant contracts:

\[
\operatorname{BASIS\_MOM}_{242,i,t}
=
\left(\prod_{\tau=t-241}^{t}(1+r^{(1)}_{i,\tau})-1\right)
-\left(\prod_{\tau=t-241}^{t}(1+r^{(2)}_{i,\tau})-1\right).
\]

The factor is missing unless both 242-day series are complete.

Economic interpretation: changes in the slope of the futures curve. A positive score means the dominant contract has outperformed the second-dominant contract over the trailing year.

### 3.2 Negative-skew factor

\[
\operatorname{NEG\_SKEW}_{242,i,t}
=-\operatorname{Skew}_{242}(r^{(1)}_{i,t})
\]

The skew window is 242 trading days with at least 200 valid observations.

Economic interpretation: risk compensation for lottery-like positive skew and tail-risk aversion. The sign is negative for raw skew and positive for negative-skew compensation.

### 3.3 Sticky-member net-share factor

For member \(j\), let \(L_{j,t}\), \(S_{j,t}\), and \(V_{j,t}\) be long positions, short positions, and volume.

The market turnover ratio is:

\[
\operatorname{MarketTurnover}_{i,t}
=
\frac{V^{market}_{i,t}}{OI^{market}_{i,t}}.
\]

A member is classified as high-turnover when:

\[
\frac{V_{j,t}}{L_{j,t}+S_{j,t}}
\ge
\operatorname{MarketTurnover}_{i,t}.
\]

The sticky set contains disclosed members outside that high-turnover set, including one-sided long or short positions that are valid and persistent:

\[
\operatorname{STICKY\_NET\_SHARE}_{i,t}
=
\frac{\sum_{j\in \operatorname{Sticky}_{i,t}}(L_{j,t}-S_{j,t})}
{\sum_{j\in \operatorname{Sticky}_{i,t}}(L_{j,t}+S_{j,t})}.
\]

Missing members and zero gross positions are excluded from both numerator and denominator.

Economic interpretation: persistent industrial hedgers and long-horizon institutions may show lower turnover than short-horizon active traders. The factor captures the visible net-position imbalance of the lower-turnover part of the top-20 member book.

### 3.4 Composite score and weights

Within the liquid candidate pool on day \(t\), rank the three factors cross-sectionally using percentile ranks:

\[
\operatorname{Score}_{i,t}
=
\frac{
\operatorname{PctRank}(\operatorname{BASIS\_MOM}_{i,t})
+
\operatorname{PctRank}(\operatorname{NEG\_SKEW}_{i,t})
+
\operatorname{PctRank}(\operatorname{STICKY\_NET\_SHARE}_{i,t})
}{3}.
\]

The equal-rank average avoids estimating in-sample factor weights.

If at least eight names are valid, select the highest and lowest \(\lceil20\%N_t\rceil\) names. Long names receive a combined \(+0.50\) target weight and short names receive a combined \(-0.50\) target weight.

Each fresh batch remains held for ten sessions:

\[
w_{i,t}^{held}
=
\frac{1}{10}\sum_{\ell=0}^{9}w_{i,t-\ell}^{fresh}.
\]

The close-formed weights trade at the next session's open.

## 4. Factor Evidence

### 4.1 Predictive test protocol

The member-factor study tested 80 original and sensitivity indicators. It used:

- Next-open label and five-session open-to-open label.
- The same commodity universe, factor directions, and minimum pair requirement.
- Time-series correlation for each symbol.
- Cross-sectional rank correlation for the full market.
- Benjamini-Hochberg correction within families defined before seeing results.

The main economic-factor family contains the roll yield, 12-1 momentum, and warehouse drawdown.

### 4.2 Economic-factor results

| Factor | Horizon | Mean IC | HAC t | Primary q | Full q |
|---|---|---:|---:|---:|---:|
| Roll yield | 1 day | +0.02200 | +3.57 | 0.00107 | 0.00214 |
| 12-1 momentum | 1 day | +0.02464 | +3.27 | 0.00163 | 0.00326 |
| Warehouse drawdown | 1 day | +0.00483 | +1.15 | 0.24855 | 0.29826 |

Roll yield and 12-1 momentum pass their predeclared economic-factor families. Warehouse drawdown does not.

### 4.3 Cross-sectional robustness

The five- and ten-session robustness tests use 5, 10, and 20-session labels. The following factors pass the 10-session robustness record.

| Factor | 10-day IC | HAC t | Long-leg excess | Short-leg excess | Positive years |
|---|---:|---:|---:|---:|---:|
| Roll yield | +0.0457 | +2.58 | +18.9 bp | +28.8 bp | 5/6 |
| 12-1 momentum | +0.0508 | +2.25 | +29.0 bp | +27.3 bp | 6/6 |
| Basis momentum | +0.0421 | +2.61 | +29.6 bp | +26.3 bp | 6/6 |
| Negative skew | +0.0362 | +2.31 | +21.4 bp | +29.0 bp | 6/6 |
| K_NET_SHARE | +0.0320 | +2.06 | +16.0 bp | +20.0 bp | 6/6 |

Flow control, -K_DS, and K_DB are not robust two-leg signals in the same 10-session family. Flow control is strongest over five sessions and mostly contributes to the long leg. -K_DS has stronger next-day and time-series evidence, but not the two-leg robustness record used by this strategy. K_DB is near zero in cross-sectional robustness testing.

### 4.4 Seat-purity evidence

Splitting the top-20 member book by the CROWD1 turnover threshold yields:

| Seat set | 10-day IC | Interpretation |
|---|---:|---|
| Sticky members | +0.0416 | Positive, stable, and passes the stated robustness checks |
| High-turnover active members | -0.0208 | Negative and unstable |

The sticky member set is used by the active strategy. The high-turnover set is not used as a separate factor.

## 5. Active Strategy

The current strategy is `STICKY_CORE_2X`.

It combines:

1. `BASIS_MOM_242`
2. `NEG_SKEW_242`
3. `STICKY_NET_SHARE`

The three factors are ranked independently inside the daily liquidity pool and averaged equally. No volatility inverse weighting, sector control, or extra member-flow factor is part of the active code.

Execution assumptions:

- Initial equity: RMB 10 million
- Target-weight multiplier: 2.0X input to orders
- Maximum name weight: 20%
- Account margin cap: 50%
- Flat margin: 20%
- Integer lots
- Historical commissions
- One-tick and two-tick slippage cases
- One tick per side is the report case
- Roll closes the old contract and opens the new contract
- Long and short gross weights are neutralized

## 6. Research-Period Backtest

Period: 2016-01-04 to 2021-12-31.

The following table uses one tick per side.

| Book | End equity, RMB 10k | Net annualized | Sharpe | Maximum drawdown | Daily turnover | Mean gross exposure | Margin peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal-weight benchmark | 1893.06 | 10.98% | 1.21 | -13.96% | 4.26% | 89.76% | 19.08% |
| STICKY_CORE_2X | 1991.26 | 12.06% | 1.06 | -18.72% | 18.67% | 193.16% | 40.09% |

The two-tick case preserves positive results:

| Book | Net annualized | Sharpe |
|---|---:|---:|
| STICKY_CORE_2X | 10.40% | 0.92 |

Annual returns:

| Year | Benchmark | STICKY_CORE_2X |
|---|---:|---:|
| 2016 | +36.73% | +20.03% |
| 2017 | +4.90% | +14.55% |
| 2018 | -3.61% | +12.64% |
| 2019 | +6.91% | -0.55% |
| 2020 | +7.66% | +28.57% |
| 2021 | +18.96% | +0.55% |

## 7. Out-of-Sample Backtest

Period: 2022-01-04 to 2025-12-31.

The OOS panel retains 2010-2021 history for trailing-window warm-up. It does not rewrite the factor signs, windows, rank composition, holding period, or 2X input multiplier.

The following table uses one tick per side.

| Book | End equity, RMB 10k | Net annualized | Sharpe | Maximum drawdown | Daily turnover | Mean gross exposure | Margin peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal-weight benchmark | 1009.07 | 0.50% | 0.07 | -15.07% | 4.62% | 75.72% | 16.36% |
| STICKY_CORE_2X | 1390.19 | 8.60% | 0.99 | -12.72% | 20.25% | 183.21% | 38.73% |

The two-tick case:

| Book | Net annualized | Sharpe |
|---|---:|---:|
| STICKY_CORE_2X | 7.29% | 0.84 |

Annual returns:

| Year | Benchmark | STICKY_CORE_2X |
|---|---:|---:|
| 2022 | +7.68% | +14.64% |
| 2023 | +4.24% | -6.95% |
| 2024 | -5.58% | +10.55% |
| 2025 | -4.79% | +17.89% |

## 8. Combined Result

The combined curve applies OOS results by scaling the 2022-01-04 OOS NAV to the 2021-12-31 research-period NAV, preserving both partitions as separate backtest specifications.

| Book | End equity, RMB 10k | Net annualized | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Equal-weight benchmark | 1910.23 | 6.80% | 0.81 | -15.07% |
| STICKY_CORE_2X | 2768.22 | 10.68% | 1.03 | -18.72% |

## 9. Output Files

- Research summary: [summary.csv](../runs/strategies/sticky_core/research/summary.csv)
- Research yearly returns: [yearly_returns.csv](../runs/strategies/sticky_core/research/yearly_returns.csv)
- Research selection summary: [selection_summary.csv](../runs/strategies/sticky_core/research/selection_summary.csv)
- Research performance chart: [performance.png](../runs/strategies/sticky_core/research/performance.png)
- OOS summary: [summary.csv](../runs/strategies/sticky_core/oos/summary.csv)
- OOS yearly returns: [yearly_returns.csv](../runs/strategies/sticky_core/oos/yearly_returns.csv)
- OOS selection summary: [selection_summary.csv](../runs/strategies/sticky_core/oos/selection_summary.csv)
- OOS performance chart: [performance.png](../runs/strategies/sticky_core/oos/performance.png)
- Combined summary: [summary.csv](../runs/strategies/sticky_core/combined/summary.csv)
- Combined yearly returns: [yearly_returns.csv](../runs/strategies/sticky_core/combined/yearly_returns.csv)
- Combined equity curve: [equity_curves.csv](../runs/strategies/sticky_core/combined/equity_curves.csv)
- Combined performance chart: [performance.png](../runs/strategies/sticky_core/combined/performance.png)

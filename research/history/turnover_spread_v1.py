"""探索性会员研究：低换手净多减高换手净多，2010–2021。

2026-11-25 事前冻结本次规格（想法来自旧样本结果，不是独立样本验证）：
原值 1:1 相减，不调参；5/10/20 日标签，以 10 日为主；同品种日比较。
主族为完整期三因子 10 日 IC；全规格族为三因子 × 三预测期 × 三区间。
改进检验另设族：差值分别减粘性 IC、反向高换手 IC，主族两项，全族十八项。
区间为完整期、2010–2015、2016–2021。不得从结果选择权重、方向或窗口。
"""
from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from research.factors.member import market_inputs, member_tables, sticky_net_share
from research.ic.study import _bh, _rank_corr, forward_returns
from research.history.labels_v1 import return_matrix
SETTINGS = {
    "specification": "member_turnover_spread_v1", "start": "2010-01-04", "end": "2021-12-31",
    "primary_horizon": 10, "horizons": (5,10,20), "min_names": 8, "min_days": 120,
}
from src import config as C
from src import data as D

FACTORS = ("STICKY_NET_SHARE", "SPEC_NET_SHARE", "STICKY_MINUS_SPEC")
PERIODS = {
    "2010-2021": ("2010-01-04", "2021-12-31"),
    "2010-2015": ("2010-01-04", "2015-12-31"),
    "2016-2021": ("2016-01-04", "2021-12-31"),
}


def spread_scores(sticky: pd.DataFrame, spec: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """原占比相减；三因子均只保留两组有效的品种日，不把缺失组当成零。"""
    sticky, spec = sticky.align(spec)
    common = np.isfinite(sticky) & np.isfinite(spec)
    sticky, spec = sticky.where(common), spec.where(common)
    return dict(zip(FACTORS, (sticky, spec, sticky - spec)))


def cumulative_ic(ic: pd.Series) -> pd.Series:
    """有效 IC 加总，缺失日仍为空；下一有效日继续累计。"""
    return ic.cumsum()


def inference(ic: pd.Series, lags: int) -> dict:
    """Bartlett Newey–West 均值检验；缺失保留其交易日距离，双侧正态近似。"""
    ic = ic.where(np.isfinite(ic))
    n, mean = int(ic.count()), ic.mean()
    se = np.nan
    if n >= SETTINGS['min_days']:
        resid = (ic - mean).fillna(0).to_numpy(dtype=float)
        variance = resid @ resid
        for lag in range(1, min(lags, len(resid) - 1) + 1):
            variance += 2 * (1 - lag / (lags + 1)) * (resid[lag:] @ resid[:-lag])
        if variance > 0:
            se = sqrt(variance) / n
    t = mean / se if np.isfinite(se) and se > 0 else np.nan
    return dict(mean_ic=mean, hac_se=se, hac_t=t,
                p=erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
                ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se, days=n)


def _adjust(result: pd.DataFrame) -> pd.DataFrame:
    primary = result.period.eq('2010-2021') & result.horizon.eq(SETTINGS['primary_horizon'])
    # 不可估计的假设仍占预先声明的族名额，保守按 p=1 校正后恢复空值。
    result['q_primary'] = _bh(result.loc[primary, 'p'].fillna(1)).reindex(result.index).where(result.p.notna())
    result['q_all'] = _bh(result.p.fillna(1)).where(result.p.notna())
    return result


def evaluate(scores: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame]
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """共同样本截面 IC、逐日累计值、配对 IC 增量与同标签逐品种时序 IC。"""
    for frame in (*scores.values(), *labels.values()):
        C.assert_no_holdout_dates(frame.index, '会员换手差值研究输入')
    rows, daily, gains, time_series = [], [], [], []
    for horizon, label in labels.items():
        label = label.reindex_like(scores[FACTORS[0]])
        common = np.isfinite(label)
        for name in FACTORS:
            common &= np.isfinite(scores[name])
        matched = {name: scores[name].where(common) for name in FACTORS}
        outcome = label.where(common)
        ic = pd.DataFrame({name: _rank_corr(matched[name], outcome, SETTINGS['min_names'])[0]
                           for name in FACTORS})
        # 常量截面没有秩相关；同一比较中的所有因子共用有效日期。
        ic = ic.where(ic.notna().all(axis=1), axis=0)
        counts = common.sum(axis=1).where(ic.notna().all(axis=1))
        lags = max(20, 2 * horizon)
        for period, (start, end) in PERIODS.items():
            selected = ic.loc[start:end]
            for name in FACTORS:
                stats = inference(selected[name], lags)
                rows.append(dict(period=period, factor=name, horizon=horizon, hac_lags=lags,
                                 avg_names=counts.loc[start:end].mean(), **stats))
            for baseline, direction in ((FACTORS[0], 1), (FACTORS[1], -1)):
                delta = selected[FACTORS[2]] - direction * selected[baseline]
                stats = inference(delta, lags)
                stats['mean_ic_gain'] = stats.pop('mean_ic')
                gains.append(dict(period=period, horizon=horizon, baseline=baseline,
                                  baseline_sign=direction, hac_lags=lags, **stats))
            # 与截面研究共用因子、方向、配对标签和流动性资格，只改变相关方向。
            for name in FACTORS:
                x, y = matched[name].loc[start:end], outcome.loc[start:end]
                coeff, _ = _rank_corr(x.T, y.T, SETTINGS['min_days'])
                n = (np.isfinite(x) & np.isfinite(y)).sum()
                time_series.extend(dict(period=period, factor=name, horizon=horizon, symbol=symbol,
                                        ic=coeff[symbol], n_pairs=int(n[symbol])) for symbol in x.columns)
            for name in FACTORS:
                daily.append(pd.DataFrame(dict(date=selected.index, period=period, factor=name,
                                               horizon=horizon, ic=selected[name].to_numpy(),
                                               cumulative_ic=cumulative_ic(selected[name]).to_numpy(),
                                               names=counts.reindex(selected.index).to_numpy())))
    return (_adjust(pd.DataFrame(rows)), pd.concat(daily, ignore_index=True),
            _adjust(pd.DataFrame(gains)), pd.DataFrame(time_series))


def _symbol_scores(symbol: str, dates: pd.DatetimeIndex, panel: pd.DataFrame) -> pd.DataFrame:
    tables = member_tables(symbol, dates, top20=True, partition='research')
    return sticky_net_share(tables, market_inputs(panel, symbol).reindex(dates))


def build_inputs(jobs: int) -> tuple[dict, dict, pd.DataFrame]:
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel.date, '会员换手差值日面板')
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(['date', 'symbol'])
    eligible, dominant = D.member_universe(panel, pd.Timestamp(SETTINGS['start']), 'research')
    dates, names = eligible.index, list(eligible.columns)
    parts = Parallel(n_jobs=jobs, prefer='threads')(
        delayed(_symbol_scores)(symbol, dates, panel.loc[panel.symbol.eq(symbol)]) for symbol in names)
    low = pd.DataFrame({s: f[FACTORS[0]] for s, f in zip(names, parts)}, index=dates)
    high = pd.DataFrame({s: f[FACTORS[1]] for s, f in zip(names, parts)}, index=dates)
    valid_market = panel.assign(valid=panel.oi_total.gt(0) & panel.volume_total.gt(0)).pivot(
        index='date', columns='symbol', values='valid').reindex_like(eligible).eq(True)
    low, high = low.where(eligible & valid_market), high.where(eligible & valid_market)
    scores = {name: value.loc[SETTINGS['start']:SETTINGS['end']]
              for name, value in spread_scores(low, high).items()}
    coverage = pd.DataFrame(dict(eligible_names=eligible.sum(axis=1),
                                sticky_names=low.notna().sum(axis=1),
                                spec_names=high.notna().sum(axis=1),
                                common_names=(low.notna() & high.notna()).sum(axis=1)))
    print(f'席位分组完成：{len(names)} 个商品，读取研究期逐合约开盘价', flush=True)
    contracts = D.load_contracts('research')
    returns = return_matrix(dominant, contracts.set_index(['contract', 'date'])['open'])
    labels = {h: forward_returns(returns, h).reindex_like(scores[FACTORS[0]]) for h in SETTINGS['horizons']}
    return scores, labels, coverage


def plot_cumulative(daily: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ('#147D89', '#D18B38', '#6F50A4')
    labels = ('粘性净多', '高换手净多（原方向）', '粘性 − 高换手')
    with plt.rc_context({'font.sans-serif': ['PingFang SC', 'Heiti SC', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex='row')
        for col, horizon in enumerate(SETTINGS['horizons']):
            table = daily.loc[daily.period.eq('2010-2021') & daily.horizon.eq(horizon)]
            for name, label, color in zip(FACTORS, labels, colors):
                part = table.loc[table.factor.eq(name)]
                axes[0, col].plot(part.date, part.cumulative_ic, label=label, color=color, lw=1.4)
                rolling = part.set_index('date').ic.rolling(120, min_periods=120).mean()
                axes[1, col].plot(rolling.index, rolling, color=color, lw=1.1)
            axes[0, col].set_title(f'{horizon} 日预测' + ('（主检验）' if horizon == 10 else ''))
            for row in range(2):
                ax = axes[row, col]
                ax.axhline(0, color='#8d9ba6', lw=.7)
                ax.axvline(pd.Timestamp('2016-01-04'), color='#a8b3bd', ls='--', lw=.8)
                ax.grid(alpha=.16)
                ax.spines[['top', 'right']].set_visible(False)
        axes[0, 0].set_ylabel('累计 IC（逐日相加）')
        axes[1, 0].set_ylabel('过去 120 个交易日平均 IC')
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc='lower center', ncol=3, frameon=False)
        fig.suptitle('会员换手分组与差值因子 · 2010–2021', x=.06, ha='left', fontsize=15)
        fig.text(.06, .923, '共同有效品种与日期；虚线为旧研究起点；累计 IC 不代表投资收益', color='#596a77')
        fig.tight_layout(rect=(.02, .06, 1, .9))
        fig.savefig(output, dpi=180, facecolor='white')
        plt.close(fig)


def run(out: Path, jobs: int = 4) -> pd.DataFrame:
    C.assert_research_only(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(SETTINGS, exploratory=True, factors=list(FACTORS), periods=PERIODS,
                    primary_family='3 factors, full period, horizon=10',
                    full_family='3 factors x 3 horizons x 3 periods = 27',
                    gain_primary_family='2 paired IC gains, full period, horizon=10',
                    gain_full_family='2 baselines x 3 horizons x 3 periods = 18',
                    min_liquidity_days=C.UNIVERSE['turnover_window_days'],
                    formula='STICKY_NET_SHARE - SPEC_NET_SHARE',
                    gains='spread IC - sticky IC; spread IC + raw spec IC',
                    missing='drop missing pairs; neutral zeros retained; no fabricated holdings')
    # 声明先落盘，再读取研究样本；输出是独立规格，旧文件不覆盖。
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    scores, labels, coverage = build_inputs(jobs)
    summary, daily, gains, ts = evaluate(scores, labels)
    summary.to_csv(out / 'factor_ic.csv', index=False)
    gains.to_csv(out / 'paired_ic_gain.csv', index=False)
    daily.to_csv(out / 'daily_ic.csv', index=False)
    ts.to_csv(out / 'timeseries_ic.csv', index=False)
    coverage.to_csv(out / 'coverage.csv', index_label='date')
    pd.concat(scores, axis=1).to_parquet(out / 'scores.parquet')
    yearly = daily.loc[daily.period.eq('2010-2021')].assign(year=lambda d: d.date.dt.year).groupby(
        ['factor', 'horizon', 'year'], as_index=False).agg(mean_ic=('ic', 'mean'), days=('ic', 'count'))
    yearly.to_csv(out / 'yearly_ic.csv', index=False)
    plot_cumulative(daily, out / 'cumulative_ic.png')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    result = run(C.study_dir(SETTINGS['specification']), args.jobs)
    print(result.loc[result.horizon.eq(10)].round(5).to_string(index=False))


if __name__ == '__main__':
    main()

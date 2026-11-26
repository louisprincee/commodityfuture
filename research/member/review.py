"""当前会员研究：共同样本的 0/0.5/0.8/1.0 截尾插补与持仓流独立信息。

保留组画像重算描述性统计，不在筛选后缩小检验族重做校正。不构造策略或账户。
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src import config as C
from src import data as D
from research.factors.member import (
    FACTOR_META, REDEFINITION_META, market_inputs, member_tables, profile_factors, redefinition_factors,
    sticky_composition,
)
from research.ic.statistics import fixed_bh, mean_inference, rank_time_series, rank_corr as _rank_corr, rank_residual
from research.ic.study import forward_returns, return_matrix
from research.settings import MEMBER_REVIEW as SETTINGS

START, END = SETTINGS['start'], SETTINGS['end']
HORIZONS = SETTINGS['horizons']
PERIODS = {'2011-2021': (START, END), '2011-2015': (START, '2015-12-31'),
           '2016-2021': ('2016-01-04', END)}
MIN_NAMES, MIN_DAYS = SETTINGS['min_names'], SETTINGS['min_days']
OUT_NAME = 'member_review'
FRACTIONS = ('00', '50', '80', '100')


def _correct_families(summary: pd.DataFrame, ts: pd.DataFrame) -> None:
    summary['q_primary'] = np.nan
    summary['q_all'] = np.nan
    ts['q_all'] = np.nan
    for family, subset in summary.groupby('family', sort=False):
        primary = subset.period.eq('2011-2021') & (
            subset.horizon.isin((1, 5)) if family == 'crowding' else subset.horizon.eq(10))
        summary.loc[subset.index[primary], 'q_primary'] = fixed_bh(subset.loc[primary, 'p'])
        members = ts.loc[ts.family.eq(family)]
        p = pd.concat([subset.p.reset_index(drop=True), members.p.reset_index(drop=True)], ignore_index=True)
        adjusted = fixed_bh(p)
        summary.loc[subset.index, 'q_all'] = adjusted.iloc[:len(subset)].to_numpy()
        ts.loc[members.index, 'q_all'] = adjusted.iloc[len(subset):].to_numpy()

def evaluate_factors(scores: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame],
                     metadata: dict = FACTOR_META) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, daily, per_name = [], [], []
    for value in (*scores.values(), *labels.values()):
        C.assert_no_holdout_dates(value.index, '会员扩展研究输入')
    for name, raw in scores.items():
        info = metadata[name]
        score = raw * info['sign']
        for horizon, label in labels.items():
            if info['family'] == 'crowding' and horizon not in (1, 5):
                continue
            outcome = label.reindex_like(score)
            ic, count = _rank_corr(score, outcome, MIN_NAMES)
            lags = max(20, 2 * horizon)
            for period, (start, end) in PERIODS.items():
                values, names = ic.loc[start:end], count.loc[start:end]
                stats = mean_inference(values, lags, MIN_DAYS)
                rows.append(dict(factor=name, family=info['family'], sign=info['sign'], horizon=horizon,
                                 period=period, raw_mean_ic=stats['mean_ic'] * info['sign'],
                                 avg_names=names.mean(), hac_lags=lags, **stats))
                daily.append(pd.DataFrame(dict(date=values.index, factor=name, family=info['family'],
                                               horizon=horizon, period=period, ic=values.to_numpy(),
                                               raw_ic=(values * info['sign']).to_numpy(),
                                               cumulative_ic=values.cumsum().to_numpy(),
                                               names=names.to_numpy())))
                x, y = score.loc[start:end], outcome.loc[start:end]
                for symbol in score.columns:
                    stats_ts = rank_time_series(x[symbol], y[symbol], lags, MIN_DAYS)
                    per_name.append(dict(factor=name, family=info['family'], sign=info['sign'],
                                         period=period, horizon=horizon, symbol=symbol,
                                         raw_ic=stats_ts['ic'] * info['sign'], **stats_ts))
    result, ts = pd.DataFrame(rows), pd.DataFrame(per_name)
    _correct_families(result, ts)
    return result, pd.concat(daily, ignore_index=True), ts

def matched_imputations(scores):
    result = {}
    for group in ('STICKY', 'SPEC'):
        names = [f'IMP{pct}_{group}' for pct in FRACTIONS]
        common = np.isfinite(scores[names[0]])
        for name in names[1:]: common &= np.isfinite(scores[name])
        result.update({name: scores[name].where(common) for name in names})
    return result


def build_inputs(jobs):
    panel = D.load_panel()
    C.assert_no_holdout_dates(panel.date, '会员研究面板')
    panel = panel.loc[panel.symbol.isin(C.COMMODITY_SYMBOLS)].sort_values(['date', 'symbol'])
    eligible, dominant = D.member_universe(panel, pd.Timestamp(START), 'research')
    def one(symbol):
        t = member_tables(symbol, eligible.index, top20=True, partition='research')
        mkt = market_inputs(panel, symbol).reindex(eligible.index)
        return profile_factors(t, mkt), redefinition_factors(t, mkt), sticky_composition(t, mkt)
    symbols = list(eligible.columns)
    parts = Parallel(n_jobs=jobs, prefer='threads')(delayed(one)(s) for s in symbols)

    def matrices(position, names):
        return {name: pd.DataFrame({s: part[position][name] for s, part in zip(symbols, parts)},
                                   index=eligible.index).where(eligible).loc[START:END] for name in names}
    scores = matrices(0, FACTOR_META)
    redefs = matrices(1, REDEFINITION_META)
    composition = matrices(2, ('UNLISTED_STICKY_SHARE', 'STICKY_GROSS_SHARE'))
    bars = D.load_contracts('research', for_execution=True)
    returns = return_matrix(dominant, bars.set_index(['contract', 'date'])['open'])
    labels = {h: forward_returns(returns, h).reindex_like(scores['STICKY_NET_SHARE']) for h in HORIZONS}
    return scores, redefs, labels, composition


def paired_comparisons(daily):
    rows = []
    for (group, horizon, period), part in daily.assign(group=daily.factor.str.rsplit('_', n=1).str[-1]).groupby(
            ['group', 'horizon', 'period']):
        matrix = part.pivot(index='date', columns='factor', values='ic')
        for left, right in itertools.combinations(FRACTIONS, 2):
            a, b = f'IMP{left}_{group}', f'IMP{right}_{group}'
            stats = mean_inference(matrix[b] - matrix[a], max(20, 2 * horizon))
            stats['mean_ic_gain'] = stats.pop('mean_ic')
            rows.append(dict(group=group, lower_fraction=int(left)/100, higher_fraction=int(right)/100,
                             horizon=horizon, period=period, **stats))
    result = pd.DataFrame(rows)
    primary = result.horizon.eq(10) & result.period.eq('2011-2021')
    result['q_primary'] = fixed_bh(result.loc[primary, 'p']).reindex(result.index)
    result['q_all'] = fixed_bh(result.p)
    return result


def flow_incremental(scores, labels):
    rows = []
    control = scores['STICKY_NET_SHARE']
    for name in ('FLOW_CONTROL', 'FLOW_CHANGE5'):
        score = scores[name]
        residual = rank_residual(score, control, MIN_NAMES)
        for h, label in labels.items():
            ic, n = _rank_corr(residual, label.reindex_like(residual), MIN_NAMES)
            for period, (start, end) in PERIODS.items():
                rows.append(dict(factor=name, control='STICKY_NET_SHARE', horizon=h, period=period,
                                 avg_names=n.loc[start:end].mean(), **mean_inference(ic.loc[start:end], max(20, 2*h))))
    result = pd.DataFrame(rows)
    primary = result.horizon.eq(10) & result.period.eq('2011-2021')
    result['q_primary'] = fixed_bh(result.loc[primary, 'p']).reindex(result.index)
    result['q_all'] = fixed_bh(result.p)
    return result


_FONTS = {'font.family': 'sans-serif', 'axes.unicode_minus': False,
          'font.sans-serif': ['Arial Unicode MS', 'Hiragino Sans GB', 'PingFang SC', 'Microsoft YaHei', 'Heiti SC', 'SimHei', 'DejaVu Sans']}
_INK, _MUTED, _GRID, _SURFACE = '#1f2d38', '#52636f', '#e6ecf0', '#fcfcfb'
_CUM_SERIES = (('IMP00_STICKY', '粘性席位净多', '#2a78d6'), ('IMP00_SPEC', '高换手席位净多', '#eb6834'))


def plot_comparison(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({**_FONTS, 'text.color': _INK, 'axes.labelcolor': _MUTED,
                         'xtick.color': _MUTED, 'ytick.color': _MUTED}):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), facecolor=_SURFACE)
        for ax, group, title, color in zip(
            axes, ('STICKY', 'SPEC'), ('粘性席位（低换手组）', '高换手席位（投机活跃组）'), ('#147D89', '#C27D4E')
        ):
            ax.set_facecolor(_SURFACE)
            part = summary.loc[summary.period.eq('2011-2021') & summary.horizon.eq(10)].set_index('factor')
            rows = part.loc[[f'IMP{pct}_{group}' for pct in FRACTIONS]]
            y = np.arange(4)
            ax.axvline(0, color='#9aa8b2', lw=.9, zorder=1)
            ax.errorbar(rows.mean_ic, y, xerr=1.96 * rows.hac_se, fmt='o', markersize=6.5,
                        elinewidth=1.8, capsize=4, color=color, ecolor=color, zorder=3)
            for idx, r in enumerate(rows.itertuples()):
                ax.annotate(f'{r.mean_ic:+.4f}  (q={r.q_primary:.3f})', (r.mean_ic, idx),
                            xytext=(0, 10), textcoords='offset points', ha='center', va='bottom',
                            fontsize=8.6, color=_INK)
            ax.set_yticks(y, ['0.0（默认）', '0.5', '0.8', '1.0'])
            ax.set_ylim(-0.6, 3.7)
            ax.set_ylabel('未上榜一侧填补比例（× 当日末位持仓）', fontsize=9.5)
            ax.set_xlabel('10 日截面平均 IC 与 95% HAC 置信区间', fontsize=9.5)
            ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', color=_INK, pad=10)
            ax.grid(axis='x', color=_GRID, lw=.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#d7e0e6')
            ax.tick_params(length=0, labelsize=9.2)
        fig.text(.022, .945, '未披露一侧末位比例插补敏感性 · 2011–2021', fontsize=14.5, fontweight='bold')
        days = int(summary.loc[summary.period.eq('2011-2021') & summary.horizon.eq(10)
                               & summary.factor.eq('IMP00_STICKY'), 'days'].iloc[0])
        fig.text(.022, .898, f'原高低换手分组固定、共同有效样本（{days:,} 交易日）；粘性组四档差异均不显著。',
                 fontsize=9.5, color=_MUTED)
        fig.tight_layout(rect=(0, 0, 1, .87), w_pad=3.0)
        fig.savefig(output, dpi=180, facecolor=_SURFACE)
        plt.close(fig)


def plot_cumulative_ic(daily, output, split='2016-01-04'):
    """逐日截面 IC 直接加总（不复利、缺失日不填补），按预测期分面；两段均值标在各面板内。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    part = daily.loc[daily.period.eq('2011-2021')]
    horizons = sorted(part.horizon.unique())
    cut = pd.Timestamp(split)
    with plt.rc_context({**_FONTS, 'text.color': _INK, 'axes.labelcolor': _MUTED,
                         'xtick.color': _MUTED, 'ytick.color': _MUTED}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True, facecolor=_SURFACE)
        for ax, horizon in zip(axes.flat, horizons):
            ax.set_facecolor(_SURFACE)
            ax.axvspan(part.date.min(), cut, color='#f1efea', zorder=0, lw=0)
            ax.axhline(0, color='#b9b7b0', lw=.8, zorder=1)
            notes = []
            for factor, label, color in _CUM_SERIES:
                ic = part.loc[part.factor.eq(factor) & part.horizon.eq(horizon)].set_index('date')['ic']
                cum = ic.dropna().cumsum()
                ax.plot(cum.index, cum, color=color, lw=2, solid_capstyle='round', zorder=3)
                ax.annotate(label, (cum.index[-1], cum.iloc[-1]), xytext=(6, 0), textcoords='offset points',
                            va='center', fontsize=9, color=_INK)
                early, late = ic.loc[:cut - pd.Timedelta(days=1)].mean(), ic.loc[cut:].mean()
                notes.append(f'{label}  前 {early:+.4f} / 后 {late:+.4f}')
            ax.text(.015, .97, '\n'.join(notes), transform=ax.transAxes, va='top', fontsize=8.6, color=_MUTED,
                    linespacing=1.6)
            ax.set_title(f'{horizon} 日标签', loc='left', fontsize=11.5, fontweight='bold', color=_INK, pad=8)
            ax.grid(axis='y', color=_GRID, lw=.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#d7d5cf')
            ax.tick_params(length=0, labelsize=9)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.margins(x=.0)
            ax.set_xlim(part.date.min(), part.date.max() + pd.Timedelta(days=420))
        for ax in axes[:, 0]:
            ax.set_ylabel('累计 IC（逐日截面 IC 之和）', fontsize=9.5)
        handles = [plt.Line2D([], [], color=color, lw=2) for _, _, color in _CUM_SERIES]
        fig.legend(handles, [label for _, label, _ in _CUM_SERIES], loc='upper right', ncol=2, frameon=False,
                   fontsize=10, bbox_to_anchor=(.985, .955))
        fig.text(.012, .955, '粘性与高换手席位净多的累计截面 IC · 2011–2021', fontsize=15, fontweight='bold')
        fig.text(.012, .918, '共同有效样本、未上榜一侧记 0；灰底为 2011–2015。上升 = 该段排序方向与后续收益一致，'
                 '斜率 ≈ 当段平均 IC。纵轴不是收益率。', fontsize=9.5, color=_MUTED)
        fig.tight_layout(rect=(0, 0, 1, .9), h_pad=2.2, w_pad=2.5)
        fig.savefig(output, dpi=180, facecolor=_SURFACE)
        plt.close(fig)


def evaluate_redefinitions(redefs: dict[str, pd.DataFrame], labels: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """检验事前冻结的新粘性席位定义及其相对原基线的配对 IC 增量。"""
    candidates = [k for k in REDEFINITION_META if k != 'STICKY_NET_SHARE']
    rows, daily_list, gains = [], [], []
    baseline = 'STICKY_NET_SHARE'
    for name in REDEFINITION_META:
        score = redefs[name]
        for h, label in labels.items():
            ic, count = _rank_corr(score, label.reindex_like(score), MIN_NAMES)
            lags = max(20, 2 * h)
            for period, (start, end) in PERIODS.items():
                values, names = ic.loc[start:end], count.loc[start:end]
                stats = mean_inference(values, lags, MIN_DAYS)
                rows.append(dict(factor=name, family=REDEFINITION_META[name]['family'], horizon=h,
                                 period=period, avg_names=names.mean(), hac_lags=lags, **stats))
                daily_list.append(pd.DataFrame(dict(date=values.index, factor=name, horizon=h,
                                                    period=period, ic=values.to_numpy())))
    summary = pd.DataFrame(rows)
    daily = pd.concat(daily_list, ignore_index=True)
    primary = summary.factor.isin(candidates) & summary.horizon.eq(10) & summary.period.eq('2011-2021')
    summary['q_primary'] = fixed_bh(summary.loc[primary, 'p']).reindex(summary.index)
    full = summary.factor.isin(candidates)
    summary['q_all'] = fixed_bh(summary.loc[full, 'p']).reindex(summary.index)

    for h in labels:
        for period in PERIODS:
            mat = daily.loc[daily.horizon.eq(h) & daily.period.eq(period)].pivot(index='date', columns='factor', values='ic')
            for name in candidates:
                stats = mean_inference(mat[name] - mat[baseline], max(20, 2 * h), MIN_DAYS)
                stats['mean_ic_gain'] = stats.pop('mean_ic')
                gains.append(dict(factor=name, control=baseline, horizon=h, period=period, **stats))
    gains_df = pd.DataFrame(gains)
    p_mask = gains_df.horizon.eq(10) & gains_df.period.eq('2011-2021')
    gains_df['q_primary'] = fixed_bh(gains_df.loc[p_mask, 'p']).reindex(gains_df.index)
    gains_df['q_all'] = fixed_bh(gains_df.p)
    return summary, daily, gains_df


REDEFINITION_SERIES = (
    ('STICKY_NET_SHARE', '原粘性净多（基线）', '#7a8a96'),
    ('STICKY_PERSIST', '持久同向粘性（20日>=15日）', '#147D89'),
    ('STICKY_LOW_ADJUST', '低换仓强度粘性（5日调仓<=中位）', '#C27D4E'),
    ('STICKY_UNILATERAL', '去双边对锁粘性', '#5B8FA8'),
    ('STICKY_PERSIST_PURE', '持久+去对锁粘性', '#6E9C7B'),
)


def plot_redefinition_ic(daily: pd.DataFrame, summary: pd.DataFrame, output: Path, split: str = '2016-01-04',
                         series=REDEFINITION_SERIES,
                         title: str = '粘性席位新定义探索 · 2011–2021（不含成交榜截尾的替代口径）',
                         bar_legend: str = 'lower right') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    part10 = daily.loc[daily.horizon.eq(10) & daily.period.eq('2011-2021')]
    cut = pd.Timestamp(split)
    with plt.rc_context({**_FONTS, 'text.color': _INK, 'axes.labelcolor': _MUTED,
                         'xtick.color': _MUTED, 'ytick.color': _MUTED}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), facecolor=_SURFACE,
                                       gridspec_kw={'width_ratios': [1.35, 1.0]})
        ax1.set_facecolor(_SURFACE)
        ax1.axvspan(part10.date.min(), cut, color='#f1efea', zorder=0, lw=0)
        ax1.axhline(0, color='#b9b7b0', lw=.8, zorder=1)
        for factor, label, color in series:
            ic = part10.loc[part10.factor.eq(factor)].set_index('date')['ic']
            cum = ic.dropna().cumsum()
            lw = 1.6 if factor == 'STICKY_NET_SHARE' else 2.0
            ls = '--' if factor == 'STICKY_NET_SHARE' else '-'
            ax1.plot(cum.index, cum, color=color, lw=lw, ls=ls, label=label, solid_capstyle='round')
        ax1.set_title('10 日累计截面 IC · 灰底为 2011–2015', loc='left', fontsize=11.5, fontweight='bold', pad=10)
        ax1.set_ylabel('累计 IC（逐日均值之和）', fontsize=9.5)
        ax1.legend(loc='upper left', frameon=False, fontsize=8.8)
        ax1.grid(axis='y', color=_GRID, lw=.7)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.spines[['left', 'bottom']].set_color('#d7e0e6')
        ax1.tick_params(length=0, labelsize=9)
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        ax2.set_facecolor(_SURFACE)
        sub = summary.loc[summary.horizon.eq(10) & summary.period.isin(['2011-2015', '2016-2021'])]
        y = np.arange(len(series))
        h_bar = 0.34
        for idx, (factor, label, _) in enumerate(series):
            early = float(sub.loc[(sub.factor == factor) & (sub.period == '2011-2015'), 'mean_ic'].iloc[0])
            late = float(sub.loc[(sub.factor == factor) & (sub.period == '2016-2021'), 'mean_ic'].iloc[0])
            ax2.barh(idx + h_bar / 2, early, height=h_bar, color='#8D83B5', label='2011–2015' if idx == 0 else None)
            ax2.barh(idx - h_bar / 2, late, height=h_bar, color='#147D89', label='2016–2021' if idx == 0 else None)
            for val, off in ((early, h_bar / 2), (late, -h_bar / 2)):
                ax2.annotate(f'{val:+.4f}', (val, idx + off), xytext=(4 if val >= 0 else -4, 0),
                             textcoords='offset points', ha='left' if val >= 0 else 'right', va='center', fontsize=8.4)
        ax2.axvline(0, color='#9aa8b2', lw=.8)
        ax2.set_yticks(y, [label for _, label, _ in series])
        ax2.invert_yaxis()
        low_x = min(-0.015, float(sub.mean_ic.min()) - 0.018)
        high_x = max(0.055, float(sub.mean_ic.max()) + 0.022)
        ax2.set_xlim(low_x, high_x)
        ax2.set_title('前后半期 10 日平均 IC 对比', loc='left', fontsize=11.5, fontweight='bold', pad=10)
        ax2.set_xlabel('平均截面 Spearman IC', fontsize=9.5)
        ax2.legend(loc=bar_legend, frameon=False, fontsize=9)
        ax2.grid(axis='x', color=_GRID, lw=.7)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.spines[['left', 'bottom']].set_color('#d7e0e6')
        ax2.tick_params(length=0, labelsize=9)
        fig.text(.018, .95, title, fontsize=14.5, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2.8)
        fig.savefig(output, dpi=180, facecolor=_SURFACE)
        plt.close(fig)


def composition_by_year(composition, daily):
    """逐年平均：粘性组构成（先取当日品种均值）与粘性、高换手两组的 10 日截面 IC。"""
    frame = pd.DataFrame({name: value.mean(axis=1) for name, value in composition.items()})
    frame['symbols'] = composition['UNLISTED_STICKY_SHARE'].notna().sum(axis=1)
    ic = daily.loc[daily.period.eq('2011-2021') & daily.horizon.eq(10)].pivot(index='date', columns='factor', values='ic')
    frame = frame.join(ic[['IMP00_STICKY', 'IMP00_SPEC']].rename(
        columns={'IMP00_STICKY': 'STICKY_IC10', 'IMP00_SPEC': 'SPEC_IC10'}))
    return frame.groupby(frame.index.year).mean().rename_axis('year').reset_index()


def plot_composition(yearly, output, split_year=2016):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    years = yearly.year.to_numpy()
    x = np.arange(len(years))
    early = x[years < split_year]
    with plt.rc_context({**_FONTS, 'text.color': _INK, 'axes.labelcolor': _MUTED,
                         'xtick.color': _MUTED, 'ytick.color': _MUTED}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True, facecolor=_SURFACE,
                                       gridspec_kw={'height_ratios': [1.15, 1]})
        width = .38
        for offset, column, (_, label, color) in ((-width / 2, 'STICKY_IC10', _CUM_SERIES[0]),
                                                  (width / 2, 'SPEC_IC10', _CUM_SERIES[1])):
            values = yearly[column].to_numpy()
            ax1.bar(x + offset, values, width * .9, color=color, label=label)
            for xi, value in zip(x + offset, values):
                ax1.annotate(f'{value:+.3f}', (xi, value), xytext=(0, 3 if value >= 0 else -3),
                             textcoords='offset points', ha='center', va='bottom' if value >= 0 else 'top',
                             fontsize=7.8, color=_MUTED)
        ax1.axhline(0, color='#9aa8b2', lw=.8)
        ax1.set_ylabel('10 日截面 IC 年均值', fontsize=9.5)
        ax1.set_title('粘性与高换手席位净多的逐年预测力', loc='left', fontsize=11.5, fontweight='bold', pad=8)
        ax1.legend(loc='upper left', frameon=False, fontsize=9.5, ncol=2)
        share = yearly.UNLISTED_STICKY_SHARE.to_numpy()
        ax2.bar(x, share, .55, color='#8D83B5', label='粘性组持仓中未上成交榜席位的占比')
        for xi, value in zip(x, share):
            ax2.annotate(f'{value:.0%}', (xi, value), xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8.2, color=_MUTED)
        ax2.plot(x, yearly.STICKY_GROSS_SHARE, color='#147D89', lw=2, marker='o', ms=4,
                 label='粘性组持仓占全部上榜持仓的比例')
        ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('品种日均值的年均', fontsize=9.5)
        ax2.set_title('粘性组的构成', loc='left', fontsize=11.5, fontweight='bold', pad=8)
        ax2.legend(loc='upper right', frameon=False, fontsize=9.5, ncol=2)
        ax2.set_xticks(x, [str(year) for year in years])
        for ax in (ax1, ax2):
            ax.set_facecolor(_SURFACE)
            if len(early):
                ax.axvspan(early.min() - .5, early.max() + .5, color='#f1efea', zorder=0, lw=0)
            ax.grid(axis='y', color=_GRID, lw=.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#d7e0e6')
            ax.tick_params(length=0, labelsize=9)
        fig.text(.012, .955, '2016 年前后符号切换的诊断 · 2011–2021', fontsize=15, fontweight='bold')
        fig.text(.012, .918, '灰底为 2011–2015。IC 取原定义两组的共同有效样本；构成只用持仓与成交榜，不涉及收益。',
                 fontsize=9.5, color=_MUTED)
        fig.tight_layout(rect=(0, 0, 1, .9), h_pad=2.0)
        fig.savefig(output, dpi=180, facecolor=_SURFACE)
        plt.close(fig)


def retained_profile_summary(scores, labels):
    """已按旧结果筛选的两组只重算描述性统计，不缩小原检验族重做 BH。"""
    rows = []
    for name in ("RUN_LONG", "DIR_STABLE"):
        for horizon, label in labels.items():
            ic, count = _rank_corr(scores[name], label.reindex_like(scores[name]), MIN_NAMES)
            for period, (left, right) in PERIODS.items():
                stats = mean_inference(ic.loc[left:right], max(20, 2 * horizon), MIN_DAYS)
                rows.append(dict(factor=name, family="profile_selected_descriptive", sign=1,
                                 horizon=horizon, period=period, avg_names=count.loc[left:right].mean(),
                                 q_primary=np.nan, q_all=np.nan,
                                 inference_role="post_selection_descriptive_only", **stats))
    return pd.DataFrame(rows)


def run(out, jobs=4):
    C.assert_research_only(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(start=START,end=END,warmup='2010',fractions=[0,.5,.8,1],fixed_original_seat_groups=True,
                    price_source='actual contract opens; old-code contracts retained after successor listing',
                    primary='8 full-period h10 XS tests',full='96 XS + all per-symbol TS over 3 periods and 4 horizons',
                    paired_primary='12 full-period h10 paired differences',paired_full='144 paired differences',
                    flow_primary='2 residual IC tests at h10',flow_full='24 residual IC tests',
                    redefinition_primary='4 new sticky seat definitions at h10 (2011-2021)',
                    exploratory=True, profile_selection='RUN_LONG,DIR_STABLE selected from prior results; current descriptive statistics only; no resized BH family or stale historical q')
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    scores, redefs, labels, composition = build_inputs(jobs)
    matched = matched_imputations(scores)
    metadata = {name:FACTOR_META[name] for name in matched}
    summary, daily, ts = evaluate_factors(matched, labels, metadata)
    summary.to_csv(out/'factor_ic.csv',index=False)
    paired_comparisons(daily).to_csv(out/'paired_differences.csv',index=False)
    ts.to_csv(out/'timeseries_ic.csv',index=False)
    flow_incremental(scores,labels).to_csv(out/'flow_incremental.csv',index=False)
    daily.loc[daily.period.eq('2011-2021')].to_parquet(out/'daily_ic.parquet',index=False)
    plot_comparison(summary,out/'imputation_comparison.png')
    plot_cumulative_ic(daily,out/'cumulative_ic.png')
    yearly = composition_by_year(composition, daily)
    yearly.to_csv(out / 'sticky_composition_by_year.csv', index=False)
    plot_composition(yearly, out / 'sticky_composition.png')
    redef_summary, redef_daily, redef_gains = evaluate_redefinitions(redefs, labels)
    redef_summary.to_csv(out / 'seat_redefinition_ic.csv', index=False)
    redef_gains.to_csv(out / 'seat_redefinition_gains.csv', index=False)
    plot_redefinition_ic(redef_daily, redef_summary, out / 'seat_redefinition_ic.png')
    retained_profile_summary(scores, labels).to_csv(out / 'retained_profile_evidence.csv', index=False)
    print(summary.loc[summary.period.eq('2011-2021') & summary.horizon.eq(10),
                      ['factor','mean_ic','p','q_primary','q_all','days']].round(6).to_string(index=False))
    print("\n--- 粘性席位新定义 (2011-2021, 10日) ---")
    print(redef_summary.loc[redef_summary.period.eq('2011-2021') & redef_summary.horizon.eq(10),
                            ['factor','mean_ic','p','q_primary','q_all','days']].round(6).to_string(index=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    run(C.study_dir(OUT_NAME),args.jobs)


if __name__=='__main__':
    main()

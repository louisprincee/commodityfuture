"""当前会员研究：共同样本的 0/0.5/0.8/1.0 截尾插补与持仓流独立信息。

保留既有候选原校正值，不在筛选后重画检验族。不构造策略或账户。
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
from research.factors.member import FACTOR_META, market_inputs, member_tables, profile_factors
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
        return profile_factors(t, market_inputs(panel, symbol).reindex(eligible.index))
    symbols = list(eligible.columns)
    parts = Parallel(n_jobs=jobs, prefer='threads')(delayed(one)(s) for s in symbols)
    scores = {name: pd.DataFrame({s: part[name] for s, part in zip(symbols, parts)}, index=eligible.index)
              .where(eligible).loc[START:END] for name in FACTOR_META}
    bars = D.load_contracts('research', for_execution=True)
    returns = return_matrix(dominant, bars.set_index(['contract', 'date'])['open'])
    labels = {h: forward_returns(returns, h).reindex_like(scores['STICKY_NET_SHARE']) for h in HORIZONS}
    return scores, labels


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


def evaluate_net_return_gains(returns, parts, control, alternatives):
    """固定一跳主族/全族的逐日净收益配对检验；只接受数据，不构造仓位。"""
    rows = []
    for book in alternatives:
        gain = returns[book] - returns[control]
        part_gain = parts[book] - parts[control]
        for period, (left, right) in PERIODS.items():
            stats = mean_inference(gain.loc[left:right], 20)
            stats["mean_daily_gain"] = stats.pop("mean_ic")
            rows.append(dict(book=book, control=control, period=period, ticks=1,
                             ann_gain=stats["mean_daily_gain"] * C.TRADING_DAYS_PER_YEAR,
                             ann_gross_gain=part_gain.gross.loc[left:right].mean() * C.TRADING_DAYS_PER_YEAR,
                             ann_cost_increase=part_gain.cost.loc[left:right].mean() * C.TRADING_DAYS_PER_YEAR,
                             ann_ci_low=stats["ci_low"] * C.TRADING_DAYS_PER_YEAR, ann_ci_high=stats["ci_high"] * C.TRADING_DAYS_PER_YEAR, **stats))
    result = pd.DataFrame(rows)
    primary = result.period.eq("2011-2021")
    result["q_primary"] = fixed_bh(result.loc[primary, "p"]).reindex(result.index)
    result["q_all"] = fixed_bh(result.p)
    return result


def plot_comparison(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.sans-serif':['PingFang SC','Heiti SC','DejaVu Sans'], 'axes.unicode_minus':False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, group, title in zip(axes, ('STICKY','SPEC'), ('粘性席位','高换手席位')):
            part = summary.loc[summary.period.eq('2011-2021') & summary.horizon.eq(10)].set_index('factor')
            rows = part.loc[[f'IMP{pct}_{group}' for pct in FRACTIONS]]
            ax.errorbar(rows.mean_ic, np.arange(4), xerr=1.96*rows.hac_se, fmt='o', capsize=4, color='#147D89')
            ax.set_yticks(np.arange(4), ['0','0.5','0.8','1.0'])
            ax.set_ylabel('未上榜持仓 / 末位持仓')
            ax.set_xlabel('10 日截面平均 IC 与 95% HAC 区间')
            ax.set_title(title)
            ax.axvline(0, color='#8b98a1', lw=.8)
            ax.grid(axis='x',alpha=.15)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('四档截尾插补 · 2011–2021 · 原分组固定、共同有效样本')
        fig.tight_layout()
        fig.savefig(output,dpi=180,facecolor='white')
        plt.close(fig)


_FONTS = {'font.family': 'sans-serif', 'axes.unicode_minus': False,
          'font.sans-serif': ['Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'SimHei', 'DejaVu Sans']}
_INK, _MUTED, _GRID, _SURFACE = '#0b0b0b', '#52514e', '#e6e4df', '#fcfcfb'
_CUM_SERIES = (('IMP00_STICKY', '粘性席位净多', '#2a78d6'), ('IMP00_SPEC', '高换手席位净多', '#eb6834'))


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


def run(out, jobs=4):
    C.assert_research_only(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(start=START,end=END,warmup='2010',fractions=[0,.5,.8,1],fixed_original_seat_groups=True,
                    price_source='actual contract opens; old-code contracts retained after successor listing',
                    primary='8 full-period h10 XS tests',full='96 XS + all per-symbol TS over 3 periods and 4 horizons',
                    paired_primary='12 full-period h10 paired differences',paired_full='144 paired differences',
                    flow_primary='2 residual IC tests at h10',flow_full='24 residual IC tests',
                    exploratory=True, profile_selection='RUN_LONG,DIR_STABLE selected from prior results; retain prior 12-test q')
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    scores, labels = build_inputs(jobs)
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
    prior = out / 'legacy_factor_ic.csv'
    if prior.is_file():
        old = pd.read_csv(prior)
        old.loc[old.factor.isin(['RUN_LONG', 'DIR_STABLE'])].to_csv(out / 'retained_profile_evidence.csv', index=False)
    print(summary.loc[summary.period.eq('2011-2021') & summary.horizon.eq(10),
                      ['factor','mean_ic','p','q_primary','q_all','days']].round(6).to_string(index=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    run(C.study_dir(OUT_NAME),args.jobs)


if __name__=='__main__':
    main()

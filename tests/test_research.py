"""现行因子、标签和跨品种残差的合成数据测试。"""
import numpy as np
import pandas as pd
import pytest


def test_forward_return_uses_signal_day_contract_at_next_two_opens():
    from research.ic.study import forward_open_returns, return_matrix, forward_returns

    days = pd.DatetimeIndex(["2021-12-24", "2021-12-27", "2021-12-28", "2021-12-29"])
    dominant = pd.Series(["OLD", "NEW", "LATER", "LATER"], index=days)
    opens = pd.Series([100.0, 108.0, 400.0, 420.0], index=pd.MultiIndex.from_tuples(
        [("OLD", days[1]), ("OLD", days[2]), ("NEW", days[2]), ("NEW", days[3])]))
    labels = forward_open_returns(dominant, opens)
    assert labels.at[days[0], "entry_contract"] == "OLD"
    assert labels.at[days[0], "forward_return"] == pytest.approx(.08)
    assert labels.iloc[-2:]["forward_return"].isna().all()
    forward = forward_returns(return_matrix(dominant.to_frame("A"), opens), 1)
    assert forward.iloc[0, 0] == pytest.approx(.08)
    assert forward.iloc[1, 0] == pytest.approx(.05)
    assert forward.iloc[-2:].isna().all().all()


def test_multi_day_forward_return_compounds_only_complete_windows():
    from research.ic.study import forward_returns

    dates = pd.bdate_range("2021-12-01", periods=10)
    returns = pd.DataFrame({"A": [0.9, .01, .02, .03, .04, .05, .06, .07, .08, .09]}, index=dates)
    forward = forward_returns(returns, 5)
    assert forward.iloc[0, 0] == pytest.approx(np.prod([1.01, 1.02, 1.03, 1.04, 1.05]) - 1)
    assert forward.iloc[-5:].isna().all().all()
    returns.iloc[4, 0] = np.nan
    assert pd.isna(forward_returns(returns, 5).iloc[0, 0])


def test_basis_momentum_needs_full_window_on_both_contracts():
    from research.factors.price import basis_momentum

    days = pd.bdate_range("2020-01-01", periods=5)
    ret1 = pd.DataFrame({"A": [.01, .01, .01, .01, .01]}, index=days)
    ret2 = pd.DataFrame({"A": [0.0, 0.0, np.nan, 0.0, 0.0]}, index=days)
    out = basis_momentum(ret1, ret2, window=2)
    assert np.isnan(out.iloc[0, 0]) and np.isnan(out.iloc[2, 0]) and np.isnan(out.iloc[3, 0])
    assert out.iloc[1, 0] == pytest.approx(1.01 ** 2 - 1)
    assert out.iloc[4, 0] == pytest.approx(1.01 ** 2 - 1)


def test_sticky_net_share_keeps_low_turnover_and_one_sided_seats():
    from research.factors.member import sticky_net_share

    days = pd.bdate_range("2020-01-01", periods=2)
    cols = ["STICKY_LONG", "ONE_SIDED_SHORT", "CHURNER"]
    market = pd.DataFrame({"volume": [100.0, 100.0], "oi": [200.0, 200.0]}, index=days)
    tables = {
        "L": pd.DataFrame([[40.0, np.nan, 20.0], [np.nan, np.nan, np.nan]], index=days, columns=cols),
        "S": pd.DataFrame([[10.0, 30.0, 10.0], [np.nan, np.nan, np.nan]], index=days, columns=cols),
        "V": pd.DataFrame([[10.0, np.nan, 50.0], [np.nan, np.nan, np.nan]], index=days, columns=cols),
        "ranked": pd.Series([True, False], index=days),
    }
    out = sticky_net_share(tables, market)
    assert out.loc[days[0], "STICKY_NET_SHARE"] == pytest.approx(0.0)
    assert out.loc[days[0], "SPEC_NET_SHARE"] == pytest.approx(10.0 / 30.0)
    assert np.isnan(out.loc[days[1], "STICKY_NET_SHARE"])


def test_chain_share_keeps_only_same_direction_sticky_seats():
    from research.member.chain import chain_consistent_share

    day = pd.DatetimeIndex(["2020-01-02"])
    seats = ["same", "opposite", "only_a"]
    net = {
        "A": pd.DataFrame([[6., -4., 2.]], index=day, columns=seats),
        "B": pd.DataFrame([[2., 5., 0.]], index=day, columns=seats),
    }
    gross = {
        "A": pd.DataFrame([[10., 10., 10.]], index=day, columns=seats),
        "B": pd.DataFrame([[10., 10., 0.]], index=day, columns=seats),
    }
    sticky = {symbol: holdings.gt(0) for symbol, holdings in gross.items()}
    oi = pd.DataFrame({"A": [100.], "B": [200.]}, index=day)
    result = chain_consistent_share(net, gross, sticky, oi, ["A", "B"])
    assert result["A"].iloc[0] == pytest.approx(0.6)
    assert result["B"].iloc[0] == pytest.approx(0.2)


def rank_tables():
    dates = pd.bdate_range('2010-01-04', periods=2)
    columns = ['a', 'b', 'offlist']
    values = {'L': [40., 20., np.nan], 'S': [10., 15., np.nan],
              'V': [20., 30., 50.], 'dL': [5., 2., np.nan], 'dS': [-1., -1., np.nan]}
    tables = {key: pd.DataFrame([row, row], index=dates, columns=columns)
              for key, row in values.items()}
    tables['ranked'] = pd.Series(True, index=dates)
    market = pd.DataFrame({'oi': 100., 'volume': 100.}, index=dates)
    return tables, market


@pytest.mark.parametrize('side', ['L', 'S'])
def test_partial_missing_disclosed_changes_poison_the_side_and_signal(side):
    from research.factors.member import daily_position_flow, spider_signal, disclosed_change_sum, profile_factors

    tables, market = rank_tables()
    tables[f'd{side}'].iloc[0, 1] = np.nan
    assert pd.isna(daily_position_flow(tables, market.oi).iloc[0])
    assert pd.isna(spider_signal(tables, active_only=False).iloc[0])
    assert pd.isna(spider_signal(tables, active_only=True).iloc[0])
    profiles = profile_factors(tables, market)
    assert pd.isna(profiles.FLOW_CONTROL.iloc[0])
    assert pd.isna(disclosed_change_sum(tables, side).iloc[0])
    other = 'S' if side == 'L' else 'L'
    assert disclosed_change_sum(tables, other).iloc[0] == pytest.approx(-2. if side == 'L' else 7.)


def test_structural_offlist_missing_does_not_poison_and_stray_changes_are_ignored():
    from research.factors.member import daily_position_flow, spider_signal

    tables, market = rank_tables()
    assert daily_position_flow(tables, market.oi).iloc[0] == pytest.approx(.09)
    assert spider_signal(tables, active_only=False).iloc[0] == 1
    tables['dL'].loc[:, 'offlist'] = -1000.
    tables['dS'].loc[:, 'offlist'] = 1000.
    assert daily_position_flow(tables, market.oi).iloc[0] == pytest.approx(.09)
    assert spider_signal(tables, active_only=False).iloc[0] == 1

    tables, market = rank_tables()
    tables['dL'].iloc[:, :2] = 0.
    tables['dS'].iloc[:, :2] = 0.
    assert daily_position_flow(tables, market.oi).iloc[0] == 0
    assert spider_signal(tables, active_only=False).iloc[0] == 0
    tables['L'].iloc[:] = np.nan
    assert pd.isna(daily_position_flow(tables, market.oi).iloc[0])
    assert pd.isna(spider_signal(tables, active_only=False).iloc[0])


def test_active_intersection_ignores_inactive_unknown_changes_but_needs_each_side():
    from research.factors.member import spider_signal

    tables, _ = rank_tables()
    tables['V'].loc[:, 'b'] = np.nan
    tables['dL'].loc[:, 'b'] = np.nan
    assert pd.isna(spider_signal(tables, active_only=False).iloc[0])
    assert spider_signal(tables, active_only=True).iloc[0] == 1
    tables['S'].loc[:, 'a'] = np.nan
    assert pd.isna(spider_signal(tables, active_only=True).iloc[0])


def contract_ranks():
    dates = pd.bdate_range('2010-01-04', periods=2)
    ranks = pd.DataFrame([
        {'trading_date': date, 'contract': 'A1001', 'side': side,
         'member_name': member, 'volume': volume, 'volume_change': change}
        for date in dates
        for side, member, volume, change in [
            ('long', 'a', 40., 5.), ('long', 'b', 20., 2.),
            ('short', 'a', 10., -1.), ('short', 'b', 15., -1.)]
    ])
    contracts = pd.DataFrame({'date': dates, 'contract': 'A1001', 'open_interest': 100.})
    return ranks, contracts, dates


def test_contract_partial_missing_change_keeps_net_share_but_no_direction():
    from research.factors.member import kaiyuan

    ranks, contracts, dates = contract_ranks()
    ranks.loc[1, 'volume_change'] = np.nan
    out = kaiyuan(ranks, contracts, dates)
    assert pd.isna(out.K_DB.iloc[0])
    assert out.K_DS.iloc[0] == -2
    assert pd.isna(out.SPIDER_FREE.iloc[0])
    assert pd.isna(out.SPIDER_HOLIDAY.iloc[0])
    assert out.K_NET_SHARE.iloc[0] == pytest.approx(.35)
    assert out.SPIDER_FREE.iloc[1] == 1

    ranks, contracts, dates = contract_ranks()
    ranks.loc[1, 'volume'] = np.nan
    ranks['volume_change'] = 0.
    out = kaiyuan(ranks, contracts, dates)
    assert pd.isna(out.K_NET_SHARE.iloc[0])
    assert out.K_DB.iloc[0] == out.K_DS.iloc[0] == 0
    assert out.SPIDER_FREE.iloc[0] == 0


def fixture(days=65):
    dates = pd.bdate_range('2010-01-04', periods=days)
    columns = ['a', 'b', 'c']
    values = {'L': [90., 10., 60.], 'S': [10., 90., 40.],
              'V': [10., 500., np.nan], 'dL': [1., 4., 3.], 'dS': [1., 2., 1.]}
    tables = {key: pd.DataFrame(np.tile(row, (days, 1)), index=dates, columns=columns)
              for key, row in values.items()}
    tables['ranked'] = pd.Series(True, index=dates)
    market = pd.DataFrame({'oi': 1000., 'volume': 1000.}, index=dates)
    return tables, market


def test_temporal_profiles_require_full_history_and_reset_on_absence():
    from research.factors.member import profile_factors

    tables, market = fixture(85)
    # b alternates direction; a and c stay directionally stable.
    tables['L'].loc[tables['L'].index[1::2], 'b'] = 90.
    tables['S'].loc[tables['S'].index[1::2], 'b'] = 10.
    out = profile_factors(tables, market)
    assert out.RUN_LONG.iloc[:19].isna().all()
    assert out.RUN_LONG.iloc[19] == pytest.approx(.6)
    assert out.DIR_STABLE.iloc[:19].isna().all()
    assert out.DIR_STABLE.iloc[19] == pytest.approx(.5)
    tables['L'].iloc[25, :] = np.nan
    tables['S'].iloc[25, :] = np.nan
    out = profile_factors(tables, market)
    assert pd.isna(out.RUN_LONG.iloc[44])
    assert out.RUN_LONG.iloc[45] == pytest.approx(.6)
    assert pd.isna(out.DIR_STABLE.iloc[44])
    assert out.DIR_STABLE.iloc[45] == pytest.approx(.5)
    tables['ranked'].iloc[60] = False
    out = profile_factors(tables, market)
    assert out.iloc[60].isna().all()
    assert pd.isna(out.RUN_LONG.iloc[79])
    assert out.RUN_LONG.iloc[80] == pytest.approx(1 / 15)


def test_imputation_changes_values_with_original_membership_fixed():
    from research.factors.member import profile_factors

    tables, market = fixture(1)
    tables['L'].iloc[0] = [90., 10., np.nan]
    tables['S'].iloc[0] = [np.nan, 90., np.nan]
    tables['V'].iloc[0] = [100., 500., 10000.]
    out = profile_factors(tables, market)
    for fraction in ('00', '50', '80', '100'):
        assert pd.isna(out[f'IMP{fraction}_STICKY'].iloc[0])
        # a and b remain spec; invisible c must never enter the denominator.
        denominator = {'00': 190, '50': 235, '80': 262, '100': 280}[fraction]
        numerator = {'00': 10, '50': -35, '80': -62, '100': -80}[fraction]
        assert out[f'IMP{fraction}_SPEC'].iloc[0] == pytest.approx(numerator / denominator)
    tables['S'].iloc[:] = np.nan
    out = profile_factors(tables, market)
    assert out.IMP00_SPEC.iloc[0] == out.SPEC_NET_SHARE.iloc[0] == 1
    for pct in ('50', '80', '100'):
        assert pd.isna(out[f'IMP{pct}_SPEC'].iloc[0])


def test_flows_require_disclosure_and_keep_full_five_day_warmup():
    from research.factors.member import profile_factors

    tables, market = fixture(8)
    out = profile_factors(tables, market)
    assert out.FLOW_CONTROL.iloc[0] == pytest.approx(.004)
    assert out.NEG_SHORT_FLOW.iloc[0] == pytest.approx(-.004)
    assert out.FLOW_MA5.iloc[:4].isna().all()
    assert out.FLOW_MA5.iloc[4] == pytest.approx(.004)
    assert out.FLOW_CHANGE5.iloc[:5].isna().all()
    assert out.FLOW_CHANGE5.iloc[5] == 0
    tables['dS'].iloc[1, 2] = np.nan
    out = profile_factors(tables, market)
    assert pd.isna(out.FLOW_CONTROL.iloc[1])
    assert pd.isna(out.NEG_SHORT_FLOW.iloc[1])
    assert pd.isna(out.FLOW_MA5.iloc[5])
    assert pd.isna(out.FLOW_CHANGE5.iloc[6])
    tables['ranked'].iloc[2] = False
    out = profile_factors(tables, market)
    assert pd.isna(out.FLOW_MA5.iloc[6])
    assert out.FLOW_MA5.iloc[7] == pytest.approx(.004)


def test_prefix_matches_full_history_and_empty_ranks_remain_unknown():
    from research.factors.member import profile_factors

    tables, market = fixture(70)
    full = profile_factors(tables, market)
    prefix = profile_factors({key: frame.iloc[:60].copy() for key, frame in tables.items()},
                             market.iloc[:60].copy())
    pd.testing.assert_frame_equal(full.iloc[:60], prefix)
    for name in ('L', 'S', 'V', 'dL', 'dS'):
        tables[name].iloc[:] = np.nan
    assert profile_factors(tables, market).isna().all().all()


def test_rank_time_series_has_same_spearman_estimate_and_hac_inference():
    from research.ic.statistics import rank_time_series
    from scipy.stats import spearmanr

    rng = np.random.default_rng(9)
    dates = pd.bdate_range('2011-01-04', periods=200)
    x = pd.Series(rng.normal(size=200), index=dates)
    y = x + rng.normal(size=200)
    x.iloc[10] = np.nan
    result = rank_time_series(x, y, 20)
    assert result['ic'] == pytest.approx(spearmanr(x, y, nan_policy='omit').statistic)
    assert result['n_pairs'] == 199
    assert result['ci_low'] < result['ic'] < result['ci_high']
    assert result['p'] < .05
    assert np.isnan(rank_time_series(x.iloc[:30], y.iloc[:30], 20)['ic'])


def test_fixed_bh_preserves_unavailable_hypotheses():
    from research.ic.statistics import fixed_bh

    out = fixed_bh(pd.Series([.01, np.nan, .04]))
    assert out.iloc[0] == pytest.approx(.03)
    assert out.iloc[2] == pytest.approx(.06)
    assert pd.isna(out.iloc[1])


def test_time_series_and_cross_sectional_curves_only_use_matured_labels():
    from research.ic.plot import rolling_ts_ic, matured_xs_ic

    dates = pd.bdate_range('2011-01-04', periods=135)
    x = pd.DataFrame({'A': np.tile([0., 2., 1.], 45)}, index=dates)
    labels = x.copy()
    before = rolling_ts_ic(x, labels, horizon=5, window=120)
    assert before.iloc[:125].isna().all()
    assert before.iloc[125] == pytest.approx(1.)
    labels.iloc[130:] *= -100
    after = rolling_ts_ic(x, labels, horizon=5, window=120)
    pd.testing.assert_series_equal(before.iloc[:135], after.iloc[:135])
    # Cross-sectional IC also waits for the return label to mature.
    dates = pd.bdate_range('2011-01-04', periods=10)
    x = pd.Series(np.arange(10.) / 10, index=dates)
    before = matured_xs_ic(x, 5)
    assert before.iloc[:6].isna().all()
    assert before.iloc[6] == 0
    changed = x.copy()
    changed.iloc[4:] = 100
    pd.testing.assert_series_equal(before, matured_xs_ic(changed, 5))


def test_imputation_comparison_uses_same_pairs_within_each_group():
    from research.member.review import matched_imputations, evaluate_factors

    dates = pd.bdate_range('2011-01-04', periods=150)
    rng = np.random.default_rng(18)
    x = pd.DataFrame(rng.normal(size=(150, 8)), index=dates)
    scores = {f'IMP{pct}_{group}':x.copy() for group in ('STICKY','SPEC') for pct in ('00','50','80','100')}
    scores['IMP80_STICKY'].iloc[0,0] = np.nan
    matched = matched_imputations(scores)
    labels = {10:x}
    meta = {name:{'sign':1,'family':'imputation','description':'synthetic'} for name in scores}
    summary, daily, _ = evaluate_factors(matched, labels, meta)
    full = summary.loc[summary.period.eq('2011-2021')]
    assert full.loc[full.factor.str.endswith('STICKY'),'days'].eq(149).all()
    assert full.loc[full.factor.str.endswith('SPEC'),'days'].eq(150).all()
    assert daily.loc[daily.factor.str.endswith('STICKY') & daily.date.eq(dates[0]),'ic'].isna().all()

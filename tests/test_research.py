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


def test_top20_peer_threshold_and_sticky_flow_disclosure_rule():
    from research.factors.member import extension_factors

    days = pd.bdate_range("2020-01-02", periods=2)
    seats = ["a", "b", "c", "d"]
    frame = lambda row: pd.DataFrame([row, row], index=days, columns=seats)
    tables = {
        "L": frame([20., 30., 10., 40.]), "S": frame([10., 10., 30., np.nan]),
        "V": frame([3., 20., 40., np.nan]),
        "dL": frame([1., 2., 3., 4.]), "dS": frame([0., 1., 1., np.nan]),
        "ranked": pd.Series(True, index=days),
    }
    tables["dS"].iloc[1, 0] = np.nan
    market = pd.DataFrame({"oi": 100., "volume": 1000.}, index=days)
    out = extension_factors(tables, market, pd.Series(0., index=days))
    # V/(L+S): a .1, b .5, c 1.0, d off the volume list; median .5 marks b, c as high turnover.
    assert out.loc[days[0], "STICKY_TOP20"] == pytest.approx((10. + 40.) / (30. + 40.))
    # Market turnover 10 leaves every seat sticky; a's missing dS on day 2 poisons the flow.
    assert out.loc[days[0], "STICKY_FLOW"] == pytest.approx((10. - 2.) / 100.)
    assert np.isnan(out.loc[days[1], "STICKY_FLOW"])


def test_extension_factors_use_only_information_known_at_the_close():
    from research.factors.member import extension_factors

    rng = np.random.default_rng(3)
    days = pd.bdate_range("2015-01-02", periods=320)
    seats = [f"m{i}" for i in range(8)]
    draw = lambda: pd.DataFrame(rng.uniform(1, 50, (320, 8)), index=days, columns=seats)
    tables = {"L": draw(), "S": draw(), "V": draw(), "dL": draw() - 25, "dS": draw() - 25,
              "ranked": pd.Series(True, index=days)}
    market = pd.DataFrame({"oi": rng.uniform(400, 600, 320), "volume": rng.uniform(2000, 3000, 320)}, index=days)
    returns = pd.Series(rng.normal(0, .01, 320), index=days)
    base = extension_factors(tables, market, returns)
    cut = 300
    later = {key: value.copy() for key, value in tables.items()}
    for key in ("L", "S", "V", "dL", "dS"):
        later[key].iloc[cut + 1:] = rng.uniform(1, 50, later[key].iloc[cut + 1:].shape)
    moved_market = market.copy()
    moved_market.iloc[cut + 1:] *= 3
    moved_returns = returns.copy()
    moved_returns.iloc[cut:] = .5
    shifted = extension_factors(later, moved_market, moved_returns)
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], shifted.iloc[:cut + 1])
    assert base[["STICKY_Z250", "STICKY_IDENTITY", "SEAT_SKILL_NET"]].iloc[cut].notna().all()
    pd.testing.assert_series_equal(base.STICKY_FLOW5, base.STICKY_FLOW.rolling(5, min_periods=5).mean(),
                                   check_names=False)
    pd.testing.assert_series_equal(base.STICKY_FLOW10, base.STICKY_FLOW.rolling(10, min_periods=10).mean(),
                                   check_names=False)


def test_persist_identity_requires_both_seat_conditions():
    from research.factors.member import extension_factors

    days = pd.bdate_range("2015-01-02", periods=200)
    seats = ["persist_sticky", "persist_active", "flipper_sticky"]
    flip = np.where(np.arange(200) % 2 == 0, 1., -1.)
    tables = {
        "L": pd.DataFrame({"persist_sticky": 30., "persist_active": 20., "flipper_sticky": 10. + 5 * flip}, index=days),
        "S": pd.DataFrame({"persist_sticky": 10., "persist_active": 5., "flipper_sticky": 10. - 5 * flip}, index=days),
        "V": pd.DataFrame({"persist_sticky": 1., "persist_active": 500., "flipper_sticky": 1.}, index=days),
        "dL": pd.DataFrame(0., index=days, columns=seats), "dS": pd.DataFrame(0., index=days, columns=seats),
        "ranked": pd.Series(True, index=days),
    }
    market = pd.DataFrame({"oi": 1000., "volume": 1000.}, index=days)
    out = extension_factors(tables, market)
    # Only persist_sticky is both persistent in direction and sticky over its listed history.
    assert out.STICKY_PERSIST_IDENTITY.iloc[-1] == pytest.approx(20. / 40.)
    assert out.SEAT_SKILL_NET.isna().all()
    # Preserve the evaluated implementation: the direction rolling mean needs
    # 20 valid listed days, despite the separate 15-of-20 presence threshold.
    for side in ("L", "S"):
        tables[side].loc[days[-3], "persist_sticky"] = np.nan
    assert np.isnan(extension_factors(tables, market).STICKY_PERSIST_IDENTITY.iloc[-1])


def test_sticky_composition_counts_unlisted_holdings_inside_the_sticky_group():
    from research.factors.member import sticky_composition

    days = pd.bdate_range("2015-01-02", periods=2)
    seats = ["listed_slow", "unlisted", "fast"]
    tables = {
        "L": pd.DataFrame({"listed_slow": 30., "unlisted": 10., "fast": 40.}, index=days),
        "S": pd.DataFrame({"listed_slow": 10., "unlisted": np.nan, "fast": 20.}, index=days),
        "V": pd.DataFrame({"listed_slow": 1., "unlisted": np.nan, "fast": 500.}, index=days),
        "dL": pd.DataFrame(0., index=days, columns=seats), "dS": pd.DataFrame(0., index=days, columns=seats),
        "ranked": pd.Series([True, False], index=days),
    }
    market = pd.DataFrame({"oi": 1000., "volume": 1000.}, index=days)
    out = sticky_composition(tables, market)
    assert out.UNLISTED_STICKY_SHARE.iloc[0] == pytest.approx(10. / 50.)
    assert out.STICKY_GROSS_SHARE.iloc[0] == pytest.approx(50. / 110.)
    assert out.iloc[1].isna().all()


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


def test_fixed_bh_preserves_unavailable_hypotheses(tmp_path):
    from research.ic.statistics import fixed_bh

    out = fixed_bh(pd.Series([.01, np.nan, .04]))
    assert out.iloc[0] == pytest.approx(.03)
    assert out.iloc[2] == pytest.approx(.06)
    assert pd.isna(out.iloc[1])
    from research.member.review import save_factor_view
    full = pd.DataFrame({"factor": ["retained", "retired"], "q_primary": [.06, .001]})
    view = save_factor_view(full, tmp_path, 'ic.csv', {'retained'})
    assert view.factor.tolist() == ['retained']
    assert view.q_primary.tolist() == [.06]
    assert pd.read_csv(tmp_path / 'ic.csv').factor.tolist() == ['retained']
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'history/ic.csv'), full)


def test_imputation_comparison_uses_same_pairs_within_each_group():
    from research.member.review import matched_imputations, evaluate_factors, retained_profile_summary

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
    profiles = retained_profile_summary({"RUN_LONG": x, "DIR_STABLE": x}, labels)
    assert profiles[["q_primary", "q_all"]].isna().all().all()
    assert profiles.inference_role.eq("post_selection_descriptive_only").all()
    assert profiles.loc[profiles.period.eq("2011-2021"), "days"].eq(150).all()
    assert full.loc[full.factor.str.endswith('SPEC'),'days'].eq(150).all()
    assert daily.loc[daily.factor.str.endswith('STICKY') & daily.date.eq(dates[0]),'ic'].isna().all()


def test_redefinition_factors_separate_persistence_purity_and_adjust():
    from research.factors.member import redefinition_factors

    dates = pd.bdate_range('2011-01-04', periods=25)
    tables = {
        'L': pd.DataFrame({'PERSIST_PURE': 40., 'BALANCED': 50., 'CHURNER': 10.}, index=dates),
        'S': pd.DataFrame({'PERSIST_PURE': 5., 'BALANCED': 40., 'CHURNER': 10.}, index=dates),
        'V': pd.DataFrame({'PERSIST_PURE': np.nan, 'BALANCED': np.nan, 'CHURNER': 100.}, index=dates),
        'dL': pd.DataFrame({'PERSIST_PURE': 1., 'BALANCED': 10., 'CHURNER': 8.}, index=dates),
        'dS': pd.DataFrame({'PERSIST_PURE': 0., 'BALANCED': 10., 'CHURNER': 8.}, index=dates),
        'ranked': pd.Series(True, index=dates),
    }
    market = pd.DataFrame({'oi': 200., 'volume': 200.}, index=dates)
    out = redefinition_factors(tables, market).iloc[-1]
    assert np.isfinite(out['STICKY_PERSIST'])
    assert out['STICKY_PERSIST_PURE'] == pytest.approx(35. / 45.)
    assert out['STICKY_UNILATERAL'] == pytest.approx(35. / 45.)
    assert out['STICKY_LOW_ADJUST'] == pytest.approx(45. / 135.)


def test_cumulative_ic_chart_is_written(tmp_path):
    from research.member.review import plot_cumulative_ic

    dates = pd.bdate_range('2014-01-02', periods=600)
    rng = np.random.default_rng(3)
    rows = [pd.DataFrame(dict(date=dates, factor=factor, horizon=h, period='2011-2021',
                              ic=np.where(rng.random(len(dates)) < .05, np.nan, rng.normal(.02, .1, len(dates)))))
            for factor in ('IMP00_STICKY', 'IMP00_SPEC') for h in (1, 5, 10, 20)]
    out = tmp_path / 'cumulative_ic.png'
    plot_cumulative_ic(pd.concat(rows, ignore_index=True), out)
    assert out.is_file() and out.stat().st_size > 0


def test_rank_composite_averages_percentile_ranks_on_common_names():
    from research.factors.member import rank_composite

    a = pd.DataFrame([[1., 2., 3., 4.]], columns=list("wxyz"))
    b = pd.DataFrame([[4., 3., np.nan, 1.]], columns=list("wxyz"))
    combined = rank_composite(a, b)
    assert np.isnan(combined.loc[0, "y"])
    assert combined.loc[0, ["w", "x", "z"]].tolist() == pytest.approx([(1 / 3 + 1) / 2, (2 / 3 + 2 / 3) / 2, (1 + 1 / 3) / 2])

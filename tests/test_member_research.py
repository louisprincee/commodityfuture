import numpy as np
import pandas as pd
import pytest


def test_group_contributions_share_one_denominator_and_keep_missing_days():
    from research.member_smoothing import common_seat_shares

    tables = {
        "L": pd.DataFrame([[80, 50], [80, 50], [np.nan, np.nan]]),
        "S": pd.DataFrame([[20, 150], [20, 150], [np.nan, np.nan]]),
        "V": pd.DataFrame([[np.nan, 300], [np.nan, np.nan], [np.nan, np.nan]]),
        "ranked": pd.Series([True, True, False]),
    }
    market = pd.DataFrame({"volume": [100, 100, 100], "oi": [100, 100, 100]})
    result = common_seat_shares(tables, market)

    assert list(result.iloc[0]) == pytest.approx([-40 / 300, 60 / 300, -100 / 300])
    assert list(result.iloc[1]) == pytest.approx([-40 / 300, -40 / 300, 0])
    assert result.iloc[2].isna().all()


def test_smoothing_uses_five_consecutive_observations_and_matches_baseline():
    from research.member_smoothing import matched_raw_and_ma5

    flow = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0]})
    matched = matched_raw_and_ma5({"FLOW": flow}, flow.notna())

    assert matched["FLOW_RAW"]["x"].dropna().to_dict() == {4: 5.0}
    assert matched["FLOW_MA5"]["x"].dropna().to_dict() == {4: 3.0}


def test_segment_starts_after_signal_and_compounds_only_its_days():
    from research.member_smoothing import window_returns

    returns = pd.DataFrame({"x": [0.99, 0.50, 0.10, -0.05, 0.02, 0.03, np.nan]})

    assert window_returns(returns, 2, 5).loc[0, "x"] == pytest.approx(0.097877)
    assert pd.isna(window_returns(returns, 2, 5).loc[1, "x"])


def test_oos_flow_uses_four_warmup_days_and_respects_daily_eligibility(monkeypatch):
    from research import member_smoothing as M

    dates = pd.to_datetime(["2021-12-27", "2021-12-28", "2021-12-29", "2021-12-30",
                            "2021-12-31", "2022-01-04", "2022-01-05", "2022-01-06"])
    eligible = pd.DataFrame({"RB": [False] * 5 + [True, False, True]}, index=dates)
    oi = pd.DataFrame({"RB": 100.0}, index=dates)

    def synthetic_tables(symbol, days, *, top20, partition):
        assert symbol == "RB" and partition == "oos" and not top20
        return {"dL": pd.DataFrame({"seat": 6.0}, index=days),
                "dS": pd.DataFrame({"seat": 2.0}, index=days)}

    monkeypatch.setattr(M, "member_tables", synthetic_tables)
    flow = M.load_oos_daily_flow(eligible, oi, pd.Timestamp("2022-01-04"), jobs=1)

    assert flow["RB"].dropna().index.equals(dates[[1, 2, 3, 4, 5, 7]])
    assert flow["RB"].rolling(5).mean().loc["2022-01-04"] == pytest.approx(.04)
    assert pd.isna(flow.loc["2022-01-05", "RB"])


def test_conditional_splits_stay_inside_each_sticky_tail():
    from research.member_complementarity import asymmetric_selections, sticky_quintiles

    names = list(range(22))
    sticky = pd.DataFrame([names, [None] * 22], columns=names)
    factor = pd.DataFrame([names, names], columns=names)
    buckets = sticky_quintiles(sticky)
    assert buckets.iloc[0].value_counts().sort_index().tolist() == [5, 4, 4, 4, 5]
    assert buckets.iloc[1].isna().all()

    long, short = asymmetric_selections(buckets, factor, -factor)
    assert long.columns[long.iloc[0]].tolist() == [20, 21]
    assert short.columns[short.iloc[0]].tolist() == [3, 4]
    assert not (long | short).iloc[1].any()


def test_primary_contrasts_have_the_expected_long_and_short_signs():
    from research.member_complementarity import daily_contrasts, sticky_quintiles

    scores = pd.DataFrame([list(range(20))], dtype=float)
    buckets = sticky_quintiles(scores)
    labels = scores / 100
    daily = daily_contrasts(buckets, scores, scores, labels)

    assert daily.at[0, "flow_q1"] == pytest.approx(0.02)
    assert daily.at[0, "flow_q5"] == pytest.approx(0.02)
    assert daily.at[0, "flow_confirmation"] == pytest.approx(0.04)
    assert daily.at[0, "size_amplification"] == pytest.approx(0)


def test_factor_specific_universe_does_not_require_other_signals():
    from research.member_complementarity import _mask_inputs

    eligible = pd.DataFrame({"A": [True], "B": [True]})
    sticky = pd.DataFrame({"A": [0.1], "B": [0.2]})
    flow = pd.DataFrame({"A": [0.3], "B": [None]})
    size = pd.DataFrame({"A": [0.4], "B": [0.5]})

    sticky_only, _, _ = _mask_inputs(eligible, sticky, flow, size, common_only=False)
    common_sticky, _, _ = _mask_inputs(eligible, sticky, flow, size, common_only=True)

    assert sticky_only.loc[0, "B"] == pytest.approx(0.2)
    assert pd.isna(common_sticky.loc[0, "B"])


def test_member_ic_label_uses_entry_contract_on_both_future_trading_dates():
    from research.ic_study import forward_open_returns

    days = pd.DatetimeIndex(["2021-12-24", "2021-12-27", "2021-12-28", "2021-12-29"])
    dominant = pd.Series(["OLD", "NEW", "LATER", "LATER"], index=days)
    opens = pd.Series([100.0, 108.0, 400.0, 420.0], index=pd.MultiIndex.from_tuples(
        [("NEW", days[1]), ("NEW", days[2]), ("LATER", days[2]), ("LATER", days[3])]))
    labels = forward_open_returns(dominant, opens)
    assert labels.at[days[0], "entry_date"] == days[1]
    assert labels.at[days[0], "exit_date"] == days[2]
    assert labels.at[days[0], "entry_contract"] == "NEW"
    assert labels.at[days[0], "forward_return"] == pytest.approx(.08)
    assert labels.at[days[1], "forward_return"] == pytest.approx(.05)
    assert labels.iloc[-2:]["forward_return"].isna().all()


def test_member_ic_pairs_only_dates_with_factor_and_label_and_keeps_zero_signals():
    from research.ic_study import summarize_symbol

    data = pd.DataFrame({"forward_return": [1.0, 2.0, 3.0, 4.0, np.nan],
                         "CROWD1": [3.0, 2.0, 1.0, np.nan, 0.0],
                         "SPIDER_ACTIVE": [0.0, 0.0, 1.0, np.nan, 1.0],
                         "C_LONG_20": [0.0, 0.0, 0.0, np.nan, 2.0]})
    result = summarize_symbol("RB", data, ["CROWD1", "SPIDER_ACTIVE", "C_LONG_20"]).set_index("factor")
    assert result.at["CROWD1", "n_pairs"] == 3
    assert result.at["CROWD1", "ic"] == pytest.approx(-1.0)
    assert result.at["CROWD1", "signed_ic"] == pytest.approx(1.0)
    assert result.at["SPIDER_ACTIVE", "n_pairs"] == 3
    assert result.at["SPIDER_ACTIVE", "n_nonzero"] == 1
    assert pd.isna(result.at["C_LONG_20", "ic"])


def test_multi_day_forward_return_compounds_only_complete_future_windows():
    from research.ic_study import HORIZONS, forward_returns

    dates = pd.bdate_range("2021-12-01", periods=10)
    returns = pd.DataFrame({"A": [0.9, .01, .02, .03, .04, .05, .06, .07, .08, .09]}, index=dates)
    assert HORIZONS == {"d1": 1, "d5": 5}
    forward = forward_returns(returns, 5)
    assert forward.iloc[0, 0] == pytest.approx(np.prod([1.01, 1.02, 1.03, 1.04, 1.05]) - 1)
    assert forward.iloc[-5:].isna().all().all()
    returns.iloc[4, 0] = np.nan
    assert pd.isna(forward_returns(returns, 5).iloc[0, 0])


def test_active_spider_requires_the_same_contract_and_day():
    from research.factors import kaiyuan_active

    day = pd.Timestamp("2020-01-06")
    ranks = pd.DataFrame({
        "trading_date": [day] * 6,
        "contract": ["A2001"] * 3 + ["A2005"] * 3,
        "side": ["long", "short", "volume"] * 2,
        "member_name": ["a", "a", "b", "c", "c", "a"],
        "volume_change": [5, -7, 0, 3, -4, 0],
    })
    result = kaiyuan_active(ranks, pd.DatetimeIndex([day]))
    assert result.at[day, "SPIDER_SAME_COVERAGE"] == 1
    assert result.at[day, "SPIDER_ACTIVE"] == 0
    ranks.loc[ranks.side.eq("volume"), "member_name"] = ["a", "c"]
    ranks.loc[ranks.index[0], "volume_change"] = np.nan
    result = kaiyuan_active(ranks, pd.DatetimeIndex([day]))
    assert pd.isna(result.at[day, "SPIDER_ACTIVE"])
    assert pd.isna(result.at[day, "SPIDER_SAME_COVERAGE"])


def test_paper_crowding_absolute_changes_are_taken_before_aggregation():
    from research.factors import crowding

    days = pd.bdate_range("2020-01-01", periods=4)
    tables = {
        "L": pd.DataFrame({"a": [50, 60, 50, 60], "b": [10] * 4}, index=days),
        "S": pd.DataFrame({"a": [50, 40, 50, 40], "b": [10] * 4}, index=days),
        "V": pd.DataFrame({"a": [20] * 4, "b": [20] * 4}, index=days),
    }
    market = pd.DataFrame({"oi": [100, 110, 100, 110], "volume": [20] * 4}, index=days)
    result = crowding(tables, market)
    assert result["CROWD2"].iloc[-1] == pytest.approx(0.2)
    assert result["CROWD3"].iloc[-1] == pytest.approx(0.2)


def test_paper_spider_aggregates_contracts_and_uses_disclosed_contract_oi():
    from research.factors import kaiyuan

    days = pd.bdate_range("2020-01-06", periods=2)
    ranks = pd.DataFrame({
        "trading_date": [days[0]] * 5,
        "contract": ["A2001", "A2001", "A2005", "A2005", "A2009"],
        "side": ["long", "short", "long", "short", "long"],
        "volume": [40, 30, 30, 20, 100],
        "volume_change": [-2, 1, 8, -5, 100],
    })
    contracts = pd.DataFrame({
        "date": [days[0]] * 3, "contract": ["A2001", "A2005", "A2009"],
        "open_interest": [100, 100, 200],
    })
    result = kaiyuan(ranks, contracts, days)
    assert result["SPIDER_FREE"].iloc[0] == 1
    assert result["SPIDER_HOLIDAY"].iloc[0] == 1
    assert result["SPIDER_FREE"].iloc[1] != result["SPIDER_FREE"].iloc[1]
    assert result["K_NET_SHARE"].iloc[0] == pytest.approx(0.1)


def _member_tables(dates, long_pos, short_pos, volume, d_long=None, d_short=None):
    tables = {
        "L": long_pos, "S": short_pos, "V": volume,
        "dL": d_long if d_long is not None else long_pos.diff(),
        "dS": d_short if d_short is not None else short_pos.diff(),
    }
    tables["ranked"] = pd.Series(True, index=dates)
    return tables


def test_member_quantile_bounds_exclude_today():
    from research import factors as MS

    series = pd.Series([1.0, 2.0, 3.0, 100.0])
    high, _low = MS.bounds(series, 3, 0.9, 0.1)
    assert np.isnan(high.iloc[2])
    assert high.iloc[3] == pytest.approx(np.quantile([1, 2, 3], 0.9))


def test_member_crowding1_uses_speculative_seats_only():
    from research import factors as MS

    dates = pd.bdate_range("2020-01-01", periods=2)
    long_pos = pd.DataFrame({"hedger": [1000.0, 1000.0], "spec": [100.0, 100.0]}, index=dates)
    short_pos = pd.DataFrame({"hedger": [3000.0, 3000.0], "spec": [10.0, 10.0]}, index=dates)
    volume = pd.DataFrame({"hedger": [10.0, 10.0], "spec": [500.0, 500.0]}, index=dates)
    market = pd.DataFrame({"volume": [1000.0, 1000.0], "oi": [2000.0, 2000.0], "d_oi": [0.0, 0.0]}, index=dates)
    out = MS.crowding_indicators(_member_tables(dates, long_pos, short_pos, volume), market)
    assert out["CROWD1"].iloc[0] == pytest.approx((100 - 10) / (100 + 10))


def test_member_crowding_signal_is_reversal_with_confirmation():
    from research import factors as MS

    n = 30
    series = pd.Series(np.linspace(0, 1, n))
    ret = pd.Series(0.01, index=series.index)
    signal = MS.crowding_signal(series, ret, 20)
    assert signal.iloc[-1] == -1.0
    signal_short = MS.crowding_signal(-series, -ret, 20)
    assert signal_short.iloc[-1] == 1.0
    broken = ret.where(ret.index != n - 2, -0.01)
    assert MS.crowding_signal(series, broken, 20).iloc[-1] == 0.0
    missing = ret.where(ret.index != n - 2)
    assert pd.isna(MS.crowding_signal(series, missing, 20).iloc[-1])


def test_member_consistency_camps_and_slope():
    from research import factors as MS

    dates = pd.bdate_range("2020-01-01", periods=10)
    trend = np.arange(10, dtype=float)
    noise = np.random.default_rng(0).normal(0, 1, (10, 6))
    long_pos = pd.DataFrame({
        f"m{i}": 1000 + 50 * i + (30 * trend if i >= 6 else noise[:, i % 6] * 5) for i in range(12)
    }, index=dates)
    short_pos = pd.DataFrame({f"m{i}": 900.0 for i in range(12)}, index=dates)
    out = MS.consistency_indicators(_member_tables(dates, long_pos, short_pos, long_pos), 10)
    row = out.iloc[-1]
    assert MS.CS["camp_size"] == 5
    assert row["C_LONG"] == pytest.approx(1.0)
    assert row["SLOPE_LONG"] == pytest.approx(5 * 30.0)
    assert np.isnan(out["C_LONG"].iloc[8])


def test_member_imputed_net_bounds_missing_side():
    from research import factors as MS

    dates = pd.bdate_range("2020-01-01", periods=1)
    long_pos = pd.DataFrame({"a": [100.0], "b": [40.0], "c": [np.nan], "d": [np.nan]}, index=dates)
    short_pos = pd.DataFrame({"a": [30.0], "b": [np.nan], "c": [60.0], "d": [np.nan]}, index=dates)
    net = MS.imputed_net({"L": long_pos, "S": short_pos}).iloc[0]
    assert net["a"] == 70.0
    assert net["b"] == 40.0 - 30.0 / 2
    assert net["c"] == 40.0 / 2 - 60.0
    assert np.isnan(net["d"])


def test_member_spider_signal():
    from research import factors as MS

    dates = pd.bdate_range("2020-01-01", periods=4)
    d_long = pd.DataFrame({"a": [5.0, -5.0, 5.0, 1.0], "b": [1.0, -1.0, 1.0, 1.0]}, index=dates)
    d_short = pd.DataFrame({"a": [-5.0, 5.0, 5.0, -9.0], "b": [-1.0, 1.0, 1.0, 1.0]}, index=dates)
    level = pd.DataFrame(1.0, index=dates, columns=["a", "b"])
    signal = MS.spider_signal(_member_tables(dates, level, level, level, d_long, d_short), active_only=False)
    assert signal.tolist() == [1.0, -1.0, 0.0, 1.0]


def test_economic_candidate_factors_exclude_recent_month_and_lag_inventory():
    from research.economic_ic import momentum_12_1, warehouse_draw_20

    dates = pd.bdate_range("2015-01-01", periods=284)
    returns = pd.Series(0.0, index=dates)
    returns.loc[dates[242:263]] = 0.01
    momentum = momentum_12_1(returns)
    assert momentum.loc[dates[262]] == pytest.approx(0.0)
    assert momentum.loc[dates[283]] == pytest.approx((1.01 ** 21) - 1)

    stock = pd.Series(100.0, index=dates)
    stock.loc[dates[21]] = 90.0
    oi = pd.Series(200.0, index=dates)
    inventory_draw = warehouse_draw_20(stock, oi)
    assert inventory_draw.loc[dates[21]] == pytest.approx(0.0)
    assert inventory_draw.loc[dates[22]] == pytest.approx(0.05)


def test_candidate_factor_matrix_aligns_roll_yield_momentum_and_warehouse(tmp_path):
    from research.economic_ic import candidate_factors

    dates = pd.bdate_range("2015-01-01", periods=284)
    returns = np.zeros(len(dates))
    returns[242:263] = 0.01
    panel = pd.DataFrame({
        "date": dates,
        "symbol": "CU",
        "ret1": returns,
        "oi_total": 200.0,
        "roll_yield_ann": 0.02,
    })
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    stock = np.full(len(dates), 100.0)
    stock[21] = 90.0
    warehouse = pd.DataFrame(
        {"on_warrant": stock},
        index=pd.MultiIndex.from_arrays(
            [dates.strftime("%Y%m%d"), np.repeat("CU", len(dates))],
            names=["date", "underlying_symbol"],
        ),
    )
    warehouse.to_pickle(warehouse_dir / "CU.pkl")

    factors = candidate_factors(panel, warehouse_dir)

    assert factors.at[dates[100], ("ROLL_YIELD", "CU")] == pytest.approx(0.02)
    assert factors.at[dates[262], ("MOM_12_1", "CU")] == pytest.approx(0.0)
    assert factors.at[dates[283], ("MOM_12_1", "CU")] == pytest.approx((1.01 ** 21) - 1)
    assert factors.at[dates[22], ("WAREHOUSE_DRAW_20", "CU")] == pytest.approx(0.05)


def test_basis_momentum_needs_full_window_on_both_contracts():
    from research.xs_robustness import basis_momentum

    days = pd.bdate_range("2020-01-01", periods=5)
    ret1 = pd.DataFrame({"A": [.01, .01, .01, .01, .01]}, index=days)
    ret2 = pd.DataFrame({"A": [0.0, 0.0, np.nan, 0.0, 0.0]}, index=days)
    out = basis_momentum(ret1, ret2, window=2)
    assert np.isnan(out.iloc[0, 0]) and np.isnan(out.iloc[2, 0]) and np.isnan(out.iloc[3, 0])
    assert out.iloc[1, 0] == pytest.approx(1.01 ** 2 - 1)
    assert out.iloc[4, 0] == pytest.approx(1.01 ** 2 - 1)


def test_leg_returns_split_top_and_bottom_against_pool_mean():
    from research.xs_robustness import leg_returns

    day = pd.DatetimeIndex(["2020-01-02"])
    names = list("ABCDEFGHIJ")
    score = pd.DataFrame([np.arange(10.0)], index=day, columns=names)
    label = pd.DataFrame([np.arange(10.0) / 100], index=day, columns=names)
    score.iloc[0, 0] = np.nan
    legs = leg_returns(score, label, tail=0.2).iloc[0]
    assert legs["top"] == pytest.approx((.09 + .08) / 2)
    assert legs["bottom"] == pytest.approx((.01 + .02) / 2)
    assert legs["pool"] == pytest.approx(np.arange(1, 10).mean() / 100)


def test_active_flow_uses_volume_board_members_and_blanks_missing_open_interest():
    from research.factors import active_flow

    day = pd.Timestamp("2020-01-06")
    ranks = pd.DataFrame({
        "trading_date": [day] * 10,
        "contract": ["A2001"] * 4 + ["A2005"] * 4 + ["A2009"] * 2,
        "side": ["long", "short", "volume", "volume"] * 2 + ["long", "short"],
        "member_name": ["a", "b", "a", "b", "c", "c", "c", "d", "e", "e"],
        "volume_change": [4.0, -3.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 100.0, -100.0],
    })
    contracts = pd.DataFrame({
        "date": [day, day],
        "contract": ["A2001", "A2005"],
        "open_interest": [100.0, 50.0],
    })
    result = active_flow(ranks, contracts, pd.DatetimeIndex([day]))
    assert result.at[day, "d_long"] == pytest.approx(5)
    assert result.at[day, "d_short"] == pytest.approx(-5)
    assert result.at[day, "disclosed_oi"] == pytest.approx(150)
    assert result.at[day, "flow"] == pytest.approx(10 / 150)

    off_board = ranks.copy()
    off_board.loc[off_board.member_name.eq("a") & off_board.side.eq("volume"), "member_name"] = "z"
    result = active_flow(off_board, contracts, pd.DatetimeIndex([day]))
    assert result.at[day, "d_long"] == pytest.approx(1)
    assert result.at[day, "d_short"] == pytest.approx(-5)

    missing_change = ranks.copy()
    missing_change.loc[missing_change.member_name.eq("c") & missing_change.side.eq("long"), "volume_change"] = np.nan
    result = active_flow(missing_change, contracts, pd.DatetimeIndex([day]))
    assert pd.isna(result.at[day, "d_long"])
    assert pd.isna(result.at[day, "flow"])

    missing_oi = contracts.copy()
    missing_oi.loc[missing_oi.contract.eq("A2005"), "open_interest"] = np.nan
    result = active_flow(ranks, missing_oi, pd.DatetimeIndex([day]))
    assert pd.isna(result.at[day, "disclosed_oi"])
    assert pd.isna(result.at[day, "flow"])


def test_sticky_net_share_separates_sticky_and_one_sided_seats_from_high_turnover():
    from research.factors import sticky_net_share

    days = pd.bdate_range("2020-01-01", periods=2)
    cols = ["STICKY_LONG", "ONE_SIDED_SHORT", "CHURNER"]
    # Market turnover = volume / oi = 100 / 200 = 0.50 on day 0
    mkt = pd.DataFrame({"volume": [100.0, 100.0], "oi": [200.0, 200.0]}, index=days)
    # STICKY_LONG: L=40, S=10, V=10 -> V/(L+S)=0.20 < 0.50 -> kept in sticky
    # ONE_SIDED_SHORT: L=NaN, S=30, V=NaN -> not high-turnover -> kept in sticky
    # CHURNER: L=20, S=10, V=50 -> V/(L+S)=1.67 >= 0.50 -> moved to SPEC_NET_SHARE
    tables = {
        "L": pd.DataFrame([[40.0, np.nan, 20.0], [np.nan, np.nan, np.nan]], index=days, columns=cols),
        "S": pd.DataFrame([[10.0, 30.0, 10.0], [np.nan, np.nan, np.nan]], index=days, columns=cols),
        "V": pd.DataFrame([[10.0, np.nan, 50.0], [np.nan, np.nan, np.nan]], index=days, columns=cols),
        "ranked": pd.Series([True, False], index=days),
    }
    out = sticky_net_share(tables, mkt)
    # Day 0 sticky: net=(40-10)+(0-30)=0, gross=50+30=80 -> 0.0; spec: net=10, gross=30 -> 1/3
    assert out.loc[days[0], "STICKY_NET_SHARE"] == pytest.approx(0.0)
    assert out.loc[days[0], "SPEC_NET_SHARE"] == pytest.approx(10.0 / 30.0)
    assert np.isnan(out.loc[days[1], "STICKY_NET_SHARE"])
    assert np.isnan(out.loc[days[1], "SPEC_NET_SHARE"])


def test_residual_scores_remove_the_rank_component_of_the_control():
    from research.member_behavior import residual_scores

    day = pd.DatetimeIndex(["2020-01-02"])
    cols = list("abcdefgh")
    control = pd.DataFrame([np.arange(8.0)], index=day, columns=cols)
    residual = residual_scores(control * 2, control)
    assert np.allclose(residual.to_numpy(), 0.0)


def test_chain_share_uses_only_sticky_members_with_same_direction_on_another_product():
    from research.member_behavior import chain_consistent_share

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

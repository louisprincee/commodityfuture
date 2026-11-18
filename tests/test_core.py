"""合成数据测试：样本守卫、品种池、权重、成本与会员持仓因子。不读取真实数据。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
from research import portfolio as P


def test_member_ic_label_uses_entry_contract_on_both_future_trading_dates():
    from research.member_ic import forward_open_returns

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
    from research.member_ic import summarize_symbol

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


def test_independent_timing_keeps_direction_until_its_own_exit():
    from research.experiments.timing_study import timing_directions

    z = pd.DataFrame({"A": [1.2, .2, np.nan, -1.2, -.3, .1],
                      "B": [-1.2, -.2, .2, 1.6, .6, -.2]})
    levels = z.copy()
    levels.loc[4, "B"] = -.02
    np.testing.assert_array_equal(
        timing_directions(z, levels).to_numpy(),
        [[1, -1], [1, -1], [0, 0], [-1, 1], [-1, 0], [0, 0]],
    )


def test_multi_day_forward_return_compounds_only_complete_future_windows():
    from research.experiments.hold_study import forward_returns

    dates = pd.bdate_range("2021-12-01", periods=10)
    returns = pd.DataFrame({"A": [0.9, .01, .02, .03, .04, .05, .06, .07, .08, .09]}, index=dates)
    forward = forward_returns(returns, 5, 0)
    assert forward.iloc[0, 0] == pytest.approx(np.prod([1.01, 1.02, 1.03, 1.04, 1.05]) - 1)
    assert forward_returns(returns, 5, 1).iloc[0, 0] == pytest.approx(np.prod([1.02, 1.03, 1.04, 1.05, 1.06]) - 1)
    assert forward.iloc[-5:].isna().all().all()
    returns.iloc[4, 0] = np.nan
    assert pd.isna(forward_returns(returns, 5, 0).iloc[0, 0])


def test_fixed_rebalance_holds_integer_lots_between_schedule_dates():
    from research.experiments.execution import Quote, simulate

    dates = pd.bdate_range("2021-12-01", periods=9)
    contracts = pd.DataFrame({"A": ["A2201"] * 4 + ["A2205"] * 5}, index=dates)
    weights = pd.DataFrame({"A": [.10, -.10, -.10, -.10, -.10, -.10, .10, .10, .10]}, index=dates)
    def quote(price):
        return Quote(price, price, 10, 1, .15, .15, True, 0, 0, 1, 10000, 10000)
    market = {day: {"A2201": quote(100 + step), "A2205": quote(100 + step)}
              for step, day in enumerate(dates)}
    daily, fills = simulate(weights, contracts, market, dates[0], capital=100_000,
                            neutral=False, slippage_ticks=0, rebalance_every=5)
    assert pd.DatetimeIndex(fills.date.unique()).sort_values().equals(dates[[1, 4, 6, 8]])
    assert (daily.open_lots.iloc[1:6] == 9).all()
    assert fills.loc[fills.date.eq(dates[6]), "trade_lots"].lt(0).all()


def test_lot_targets_round_to_zero_and_leave_margin_reserve():
    from research.experiments.execution import lot_targets

    result = lot_targets(np.array([1.0, -1.0, 0.01]), np.array([90, 80, 40]),
                         np.array([0.5, 0.5, 0.5]), np.full(3, 100), 1000, 0.5)
    np.testing.assert_array_equal(result, [5, -6, 0])


def test_execution_roll_closes_old_and_opens_new_with_separate_fees():
    from research.experiments.execution import Quote, simulate

    dates = pd.date_range("2021-12-27", periods=4)
    calls = {"A": "A2201", "B": "A2205"}
    contracts = pd.DataFrame({"A": [calls["A"], calls["A"], calls["B"], calls["B"]]}, index=dates)
    weights = pd.DataFrame({"A": [.1, .1, .1, .1]}, index=dates)
    def quote(price, open_fee, close_fee):
        return Quote(price, price, 10, 1, .15, .15, True, open_fee, close_fee, 1, 10000, 10000)
    market = {
        day: {calls["A"]: quote(100 + step, 2, 3), calls["B"]: quote(110 + step, 5, 7)}
        for step, day in enumerate(dates)
    }
    daily, orders = simulate(weights, contracts, market, dates[0], delay=0, capital=100_000, neutral=False)
    roll = orders[orders.date.eq(dates[2])].set_index("contract")
    assert roll.loc[calls["A"], "fee"] == 27
    assert roll.loc[calls["B"], "fee"] == 40
    assert daily.iloc[-1]["open_contracts"] == 0


def test_execution_carries_old_contract_if_a_sell_is_blocked_at_limit_down():
    from research.experiments.execution import Quote, simulate

    days = pd.date_range("2021-12-27", periods=4)
    principal = pd.DataFrame({"A": ["A2201", "A2201", "A2205", "A2205"]}, index=days)
    weights = pd.DataFrame({"A": [0.1] * 4}, index=days)
    normal = Quote(100, 100, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000)
    locked = Quote(90, 90, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000,
                   limit_up=110, limit_down=90)
    market = {day: {"A2201": normal, "A2205": normal} for day in days}
    market[days[2]]["A2201"] = locked
    daily, fills = simulate(weights, principal, market, days[0], capital=100_000, neutral=False)
    assert daily.at[days[2], "limit_rejections"] == 1
    assert daily.at[days[2], "blocked_rolls"] == 1
    assert not fills.date.eq(days[2]).any()
    assert daily.iloc[-1]["open_contracts"] == 0


def test_active_spider_requires_the_same_contract_and_day():
    from research.paper_factors import kaiyuan_active

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
    from research.paper_factors import crowding

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
    from research.paper_factors import kaiyuan

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


def test_paper_rank_buffer_keeps_incumbent_until_middle():
    from research.experiments.paper_ic import buffered_tails

    scores = pd.DataFrame([[5, 4, 3, 2, 1], [4, 5, 3, 2, 1], [2, 5, 4, 3, 1]])
    regular = buffered_tails(scores, min_names=5, buffered=False)
    buffered = buffered_tails(scores, min_names=5, buffered=True)
    assert regular.iloc[1, 0] == 0
    assert buffered.iloc[1, 0] > 0
    assert buffered.iloc[2, 0] == 0
    np.testing.assert_allclose(buffered.sum(axis=1), 0)


def synthetic_panel(n_days: int = 400, symbols=("AA", "BB", "CC", "DD"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    rows = []
    for k, symbol in enumerate(symbols):
        ret = rng.normal(0.0002 * k, 0.012, n_days)
        rows.append(pd.DataFrame({
            "date": dates, "symbol": symbol, "sector": "x",
            "ret1": ret, "turnover_total": 1e9, "c1_volume": 1000.0,
            "c1_contract": f"{symbol}0001", "c1_prev_close": 100.0,
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel.loc[panel.groupby("symbol").head(1).index, "ret1"] = np.nan
    return panel


def test_research_guard_rejects_oos_paths():
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.daily_dir("oos") / "contracts" / "CU2201.pkl")
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.minute_dir("oos") / "CU.parquet")
    C.assert_research_only(C.daily_dir("research") / "contracts" / "CU2101.pkl")


def test_partition_layout_seals_complete_oos_and_archive_roots():
    assert C.daily_dir("research") == C.DATA_ROOT / "research" / "daily"
    assert C.minute_dir("oos") == C.DATA_ROOT / "oos" / "minute"
    assert C.RESEARCH_FEE_PATH == C.DATA_ROOT / "research" / "execution" / "fee_history.csv"
    first, last = pd.Timestamp("2021-12-31").date(), pd.Timestamp("2022-01-04").date()
    assert C.split_period(first, last) == [("research", first, C.RESEARCH_END), ("oos", C.OOS_START, last)]
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.DATA_ROOT / "oos" / "future_dataset" / "A.parquet")
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.MINUTE_MONOLITH)


def test_minute_merge_uses_trading_dates_and_preserves_legacy_columns(tmp_path):
    from download.getMinutes import _missing_edges, write_shard

    path = tmp_path / "A.parquet"
    index = pd.DatetimeIndex(["2021-12-30 21:00"], name="datetime")
    old = pd.DataFrame({"trading_date": pd.to_datetime(["2021-12-31"]), "close": [100],
                        "legacy": [2]}, index=index)
    old.to_parquet(path)
    new = pd.DataFrame({"trading_date": pd.to_datetime(["2021-12-31"]), "close": [101],
                        "high": [102]}, index=index)
    write_shard(path, "research", new)
    result = pd.read_parquet(path)
    assert result.loc[index[0], ["close", "high", "legacy"]].tolist() == [101, 102, 2]
    days = pd.DatetimeIndex(["2021-12-30", "2021-12-31"])
    assert _missing_edges(path, "research", days) == [(days[0].date(), days[0].date())]
    assert _missing_edges(path, "research", days[1:]) == []
    lock = path.with_suffix(".parquet.lock")
    lock.touch()
    with pytest.raises(FileExistsError):
        write_shard(path, "research", new)
    lock.unlink()
    cross = new.set_axis(pd.DatetimeIndex(["2021-12-31 21:00"], name="datetime")).assign(
        trading_date=pd.Timestamp("2022-01-04")
    )
    with pytest.raises(C.HoldoutViolation):
        write_shard(path, "research", cross)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result)
    oos_path = tmp_path / "oos" / "A.parquet"
    write_shard(oos_path, "oos", cross)
    assert pd.read_parquet(oos_path).loc[cross.index[0], "trading_date"] == pd.Timestamp("2022-01-04")


def test_holdout_date_guard():
    C.assert_no_holdout_dates(pd.to_datetime(["2021-12-31"]))
    with pytest.raises(C.HoldoutViolation):
        C.assert_no_holdout_dates(pd.to_datetime(["2022-01-04"]))


def test_eligibility_ignores_future_turnover():
    panel = synthetic_panel()
    before = P.eligibility(panel)
    changed = panel.copy()
    cut = panel["date"].iloc[250]
    changed.loc[changed["date"] > cut, "turnover_total"] = 0.0
    after = P.eligibility(changed)
    past = panel["date"] <= cut
    pd.testing.assert_series_equal(before[past], after[past])
    assert before[past].iloc[-1]


def test_past_percentile_excludes_today_and_future():
    from research.experiments.xs_position import past_pct

    series = pd.Series([1.0, 3.0, 2.0, 4.0, 0.0])
    pct = past_pct(series, 3)
    assert pct.iloc[:3].isna().all()
    assert pct.iloc[3] == pytest.approx(1.0)
    assert pct.iloc[4] == pytest.approx(0.0)
    revised = series.copy()
    revised.iloc[-1] = 100.0
    pd.testing.assert_series_equal(pct.iloc[:-1], past_pct(revised, 3).iloc[:-1])


def test_past_zscore_uses_only_prior_observations():
    from research.experiments.xs_position import past_zscore

    series = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    score = past_zscore(series, window=3, min_periods=3)
    assert score.iloc[:3].isna().all()
    assert score.iloc[3] == pytest.approx(2.0)  # 过去 1,2,3 的均值 2、标准差 1，当日 4
    revised = series.copy()
    revised.iloc[-1] = 0.0
    pd.testing.assert_series_equal(score.iloc[:-1], past_zscore(revised, 3, 3).iloc[:-1])
    flat = pd.Series([0.0, 0.0, 0.0, 0.0, 5.0])
    assert np.isnan(past_zscore(flat, 3, 3).iloc[-1])  # 当日 5 不进入标准差


def test_tail_weights_leave_the_middle_flat():
    from research.experiments.xs_position import tail_weights

    score = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0]], columns=list("edcba"))
    weights, _count = tail_weights(score, tail=0.2, min_names=5)
    row = weights.iloc[0]
    assert row["a"] == pytest.approx(0.5) and row["b"] == pytest.approx(0.5)
    assert row["e"] == pytest.approx(-1.0)
    assert row["c"] == 0.0 and row["d"] == 0.0


def test_band_weights_keep_side_until_rank_crosses():
    from research.experiments.xs_position import band_weights

    score = pd.DataFrame({
        "a": [4.0, 3.0, 0.0],
        "b": [3.0, 4.0, 1.0],
        "c": [2.0, 2.0, 2.0],
        "d": [1.0, 1.0, 3.0],
        "e": [0.0, 0.0, 4.0],
    })
    weights, _count = band_weights(score, band=0.10, min_names=5)
    assert weights.loc[0, "a"] == pytest.approx(1 / 3)
    assert weights.loc[1, "a"] == pytest.approx(1 / 3)
    assert weights.loc[2, "a"] == pytest.approx(-0.5)
    assert weights.loc[0, "c"] == pytest.approx(1 / 3)
    assert weights.loc[2, "c"] == pytest.approx(1 / 3)


def test_signal_book_excludes_names_before_first_signal():
    from research.experiments.xs_position import signal_book

    signal = pd.DataFrame({"a": [np.nan, 1.0, 0.0], "b": [1.0, 0.0, -1.0]})
    weights = signal_book(signal, "reset", min_names=1)
    assert weights.loc[0, "b"] == pytest.approx(1.0)
    assert weights.loc[0, "a"] == 0.0
    assert weights.loc[2, "a"] == 0.0
    assert weights.loc[2, "b"] == pytest.approx(-0.5)


def test_crowding_cross_section_shorts_the_crowded_name():
    score = pd.DataFrame([[0.2, 0.8]], columns=["quiet", "crowded"]) * -1
    weights = P.score_to_weights(score).iloc[0]
    assert weights["quiet"] > 0 > weights["crowded"]


def test_weights_legs_and_monotone():
    score = pd.DataFrame([[0.1, 0.4, 0.6, 0.9, np.nan]], columns=list("abcde"))
    weights = P.score_to_weights(score).iloc[0]
    assert weights[weights > 0].sum() == pytest.approx(1.0)
    assert weights[weights < 0].sum() == pytest.approx(-1.0)
    assert weights["e"] == 0
    assert weights["a"] < weights["b"] < 0 < weights["c"] < weights["d"]


def test_backtest_timing_and_roll_cost():
    dates = pd.bdate_range("2020-01-01", periods=5)
    weights = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
    returns = pd.DataFrame({"x": [0.0, 0.01, 0.02, 0.03, 0.04]}, index=dates)
    rolls = pd.DataFrame({"x": [False, False, True, False, False]}, index=dates)
    same_day = P.backtest(weights, returns, rolls, delay=0, cost_bps=0)
    assert same_day["gross"].iloc[0] == pytest.approx(0.01)
    delayed = P.backtest(weights, returns, rolls, delay=1, cost_bps=0)
    assert delayed["gross"].iloc[0] == pytest.approx(0.02)
    charged = P.backtest(weights, returns, rolls, delay=0, cost_bps=5)
    assert charged.loc[dates[2], "turnover"] == pytest.approx(2.0)
    assert charged.loc[dates[1], "turnover"] == pytest.approx(1.0)
    assert charged.loc[dates[1], "cost"] == pytest.approx(0.0005)
    assert charged.loc[dates[3], "turnover"] == pytest.approx(0.0)


def test_per_symbol_cost_rates_and_missing_costs():
    dates = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [1.0, 0.0, 0.0]}, index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=["x", "y"])
    rolls = pd.DataFrame(False, index=dates, columns=["x", "y"])
    rates = pd.DataFrame({"x": [1.0] * 3, "y": [3.0] * 3}, index=dates)

    result = P.backtest(weights, returns, rolls, cost_bps=rates)
    assert result.loc[dates[1], "cost"] == pytest.approx(0.0004)
    assert result.loc[dates[2], "cost"] == pytest.approx(0.0003)

    rates.loc[dates[1], "x"] = np.nan
    with pytest.raises(ValueError, match="缺少成本数据"):
        P.backtest(weights, returns, rolls, cost_bps=rates)


def test_one_way_fee_and_tick_cost_conversion():
    rows = pd.DataFrame({
        "commission_type": ["by_money", "by_volume"],
        "open_commission": [0.0001, 5.0],
        "close_commission": [0.0002, 3.0],
        "contract_multiplier": [10.0, 10.0],
        "tick_size": [1.0, 1.0],
        "c1_prev_close": [1000.0, 1000.0],
    })
    costs = P.one_way_cost_components(rows, slippage_ticks=1.0)
    assert costs.loc[0, "commission_bps"] == pytest.approx(2.0)
    assert costs.loc[1, "commission_bps"] == pytest.approx(5.0)
    assert costs["slippage_bps"].tolist() == pytest.approx([10.0, 10.0])


def test_cost_matrix_uses_only_same_contract_past_fee_schedule():
    dates = pd.bdate_range("2020-01-01", periods=4)
    rows = pd.DataFrame({
        "date": dates, "symbol": "AA", "c1_contract": "AA0001", "c1_prev_close": 100.0,
    })
    fees = pd.DataFrame({
        "symbol": ["AA", "AA"], "trading_date": [dates[1], dates[3]],
        "contract": ["AA0001", "AA0001"], "commission_type": ["by_money", "by_money"],
        "open_commission": [0.0001, 0.0002], "close_commission": [0.0001, 0.0002],
        "contract_multiplier": [10.0, 10.0], "tick_size": [1.0, 1.0],
    })
    rates = P.cost_rate_matrix(rows, fees, slippage_ticks=1.0)["AA"]
    assert rates.iloc[0] != rates.iloc[0]
    assert rates.iloc[1] == pytest.approx(101.0)
    assert rates.iloc[2] == pytest.approx(101.0)
    assert rates.iloc[3] == pytest.approx(102.0)


def test_hold_cohorts_average_and_ramp():
    dates = pd.bdate_range("2020-01-01", periods=4)
    weights = pd.DataFrame({"a": [1.0, -1.0, 1.0, 1.0], "b": [-1.0, 1.0, -1.0, -1.0]}, index=dates)
    held = P.hold_cohorts(weights, 2)
    assert held.loc[dates[0], "a"] == pytest.approx(0.5)
    assert held.loc[dates[1], "a"] == pytest.approx(0.0)
    assert held.loc[dates[3], "a"] == pytest.approx(1.0)


def test_to_single_sided_halves_only_before_2020():
    import data as D

    frame = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-31", "2020-01-02"]),
        "close": [100.0, 100.0], "volume": [200, 100], "total_turnover": [4e6, 2e6],
        "open_interest": [1000, 500],
    })
    out = D.to_single_sided(frame)
    assert out["volume"].tolist() == [100.0, 100.0]
    assert out["open_interest"].tolist() == [500.0, 500.0]
    assert out["total_turnover"].tolist() == [2e6, 2e6]
    assert out["close"].tolist() == [100.0, 100.0]
    assert frame["volume"].tolist() == [200, 100]


def test_member_rank_daily_uses_ranked_contract_oi(monkeypatch):
    import data as D

    day = pd.Timestamp("2021-06-15")
    ranks = pd.DataFrame({
        "trading_date": [day] * 4, "symbol": "XX", "side": ["long", "long", "short", "volume"],
        "rank": [1, 2, 1, 1], "member_name": ["a", "b", "a", "a"],
        "volume": [300.0, 200.0, 400.0, 9e9], "volume_change": [0.0] * 4,
    })
    by_contract = pd.DataFrame({
        "trading_date": [day, day], "contract": ["XX2110", "XX2110"],
        "side": ["long", "short"], "member_name": ["a", "a"],
        "volume": [300.0, 400.0], "volume_change": [0.0, 0.0],
    })
    contracts = pd.DataFrame({
        "date": [day] * 3, "symbol": "XX",
        "contract": ["XX2110", "XX2201", "XX2205"],
        "open_interest": [1000.0, 500.0, 100.0],
    })
    monkeypatch.setattr(D, "load_member_rank", lambda symbol, partition="research": ranks)
    monkeypatch.setattr(D, "load_member_rank_contracts", lambda symbol, partition="research": by_contract)
    out = D.member_rank_daily("XX", contracts).iloc[0]
    assert out["top20_long"] == 500.0 and out["top20_short"] == 400.0
    assert out["ranked_oi"] == 1000.0 and out["n_ranked_contracts"] == 1
    assert out["total_oi"] == 1600.0


def _member_tables(dates, long_pos, short_pos, volume, d_long=None, d_short=None):
    tables = {
        "L": long_pos, "S": short_pos, "V": volume,
        "dL": d_long if d_long is not None else long_pos.diff(),
        "dS": d_short if d_short is not None else short_pos.diff(),
    }
    tables["ranked"] = pd.Series(True, index=dates)
    return tables


def test_member_quantile_bounds_exclude_today():
    from research import member_study as MS

    series = pd.Series([1.0, 2.0, 3.0, 100.0])
    high, _low = MS.bounds(series, 3, 0.9, 0.1)
    assert np.isnan(high.iloc[2])
    assert high.iloc[3] == pytest.approx(np.quantile([1, 2, 3], 0.9))


def test_member_crowding1_uses_speculative_seats_only():
    from research import member_study as MS

    dates = pd.bdate_range("2020-01-01", periods=2)
    long_pos = pd.DataFrame({"hedger": [1000.0, 1000.0], "spec": [100.0, 100.0]}, index=dates)
    short_pos = pd.DataFrame({"hedger": [3000.0, 3000.0], "spec": [10.0, 10.0]}, index=dates)
    volume = pd.DataFrame({"hedger": [10.0, 10.0], "spec": [500.0, 500.0]}, index=dates)
    market = pd.DataFrame({"volume": [1000.0, 1000.0], "oi": [2000.0, 2000.0], "d_oi": [0.0, 0.0]}, index=dates)
    out = MS.crowding_indicators(_member_tables(dates, long_pos, short_pos, volume), market)
    assert out["CROWD1"].iloc[0] == pytest.approx((100 - 10) / (100 + 10))


def test_member_crowding_signal_is_reversal_with_confirmation():
    from research import member_study as MS

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
    from research import member_study as MS

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
    from research import member_study as MS

    dates = pd.bdate_range("2020-01-01", periods=1)
    long_pos = pd.DataFrame({"a": [100.0], "b": [40.0], "c": [np.nan], "d": [np.nan]}, index=dates)
    short_pos = pd.DataFrame({"a": [30.0], "b": [np.nan], "c": [60.0], "d": [np.nan]}, index=dates)
    net = MS.imputed_net({"L": long_pos, "S": short_pos}).iloc[0]
    assert net["a"] == 70.0
    assert net["b"] == 40.0 - 30.0 / 2
    assert net["c"] == 40.0 / 2 - 60.0
    assert np.isnan(net["d"])


def test_member_spider_signal():
    from research import member_study as MS

    dates = pd.bdate_range("2020-01-01", periods=4)
    d_long = pd.DataFrame({"a": [5.0, -5.0, 5.0, 1.0], "b": [1.0, -1.0, 1.0, 1.0]}, index=dates)
    d_short = pd.DataFrame({"a": [-5.0, 5.0, 5.0, -9.0], "b": [-1.0, 1.0, 1.0, 1.0]}, index=dates)
    level = pd.DataFrame(1.0, index=dates, columns=["a", "b"])
    signal = MS.spider_signal(_member_tables(dates, level, level, level, d_long, d_short), active_only=False)
    assert signal.tolist() == [1.0, -1.0, 0.0, 1.0]


def test_download_rejects_financial_futures():
    from download import getMinutes

    with pytest.raises(SystemExit):
        getMinutes.main(["-t", "20240101-20240102", "--symbols", "IF"])


def test_short_shock_uses_past_quantiles_and_missing_stays_inactive():
    from research.experiments.event_timing import short_shock

    values = pd.Series([-30., -20., -10., 100., np.nan, -200.])
    signal, strength = short_shock(values, window=3, minimum=2)
    assert signal.iloc[:3].tolist() == [0., 0., 1.]
    assert signal.iloc[3:].tolist() == [1., 0., -1.]
    assert strength.iloc[3] > 0 and strength.iloc[5] < 0


def test_short_shock_keeps_crossings_after_flat_history():
    from research.experiments.event_timing import short_shock

    signal, strength = short_shock(pd.Series([0., 0., 0., 1.]), window=3, minimum=3)
    assert signal.iloc[-1] == 1
    assert strength.iloc[-1] > 0


def test_event_requires_capacity_for_a_minimum_order_roundtrip():
    from dataclasses import replace

    from research.experiments.event_timing import _returns_and_costs
    from research.experiments.execution import Quote

    quote = Quote(100, 100, 10, 1, .15, .15, True, 2, 3, 2, 100, 100)
    assert _returns_and_costs(quote, quote, 1)[-1]
    assert not _returns_and_costs(replace(quote, limit=0), quote, 1)[-1]
    assert not _returns_and_costs(quote, replace(quote, max_order=1), 1)[-1]


def test_event_day_price_rejects_mismatched_main_series():
    from research.experiments.event_timing import session_prices

    minutes = pd.DataFrame(
        {"trading_date": pd.to_datetime(["2021-12-29", "2021-12-29", "2021-12-29", "2021-12-30"]),
         "open": [100., 102., 103., 115.], "close": [101., 103., 104., 116.]},
        index=pd.to_datetime(["2021-12-28 21:01", "2021-12-29 09:01", "2021-12-29 15:00", "2021-12-30 09:01"]),
    )
    day = pd.DataFrame({"contract": ["A1", "A2"], "open": [100., 110.], "close": [104., 112.]},
                       index=pd.to_datetime(["2021-12-29", "2021-12-30"]))
    observed = session_prices(minutes, day)
    assert observed.iloc[0].night and observed.iloc[0].day_open == 102
    assert observed.iloc[0].contract == "A1"
    assert not observed.iloc[1].night and pd.isna(observed.iloc[1].day_open)


def test_session_minute_guard_accepts_final_research_day_and_rejects_oos():
    from research.experiments.event_timing import session_prices

    dates = pd.DatetimeIndex(["2021-12-31"])
    daily = pd.DataFrame({"open": [100], "close": [100], "contract": ["A1"]}, index=dates)
    minutes = pd.DataFrame({"trading_date": dates, "open": [100], "close": [100]},
                           index=pd.DatetimeIndex(["2021-12-31 09:01"]))
    assert session_prices(minutes, daily).day_open.iloc[0] == 100
    with pytest.raises(C.HoldoutViolation):
        session_prices(minutes.assign(trading_date=pd.Timestamp("2022-01-04")), daily)


def test_execution_retains_same_direction_lots_and_trade_costs_reconcile():
    from research.experiments.execution import Quote, simulate
    from research.experiments.event_timing import roundtrips

    dates = pd.bdate_range("2021-12-01", periods=4)
    contracts = pd.DataFrame({"A": "A2201"}, index=dates)
    weights = pd.DataFrame({"A": [.05, .05, 0., 0.]}, index=dates)
    market = {date: {"A2201": Quote(price, price, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000)}
              for date, price in zip(dates, [100, 100, 122, 125])}
    daily, fills = simulate(weights, contracts, market, dates[0], capital=100_000,
                            neutral=False, keep_same_lots=True)
    assert fills.trade_lots.tolist() == [5, -5]
    trades = roundtrips(fills)
    for field, ledger in (("gross_pnl", "gross_pnl"), ("fees", "fees"), ("slippage", "slippage")):
        assert trades[field].sum() == pytest.approx(daily[ledger].sum())

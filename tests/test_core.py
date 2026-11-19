"""合成数据测试：样本守卫、品种池、权重、成本与会员持仓因子。不读取真实数据。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
import data as D
from backtest import engine as P
from strategies import rules as R


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


def test_independent_timing_keeps_direction_until_its_own_exit():
    from strategies.rules import timing_directions

    z = pd.DataFrame({"A": [1.2, .2, np.nan, -1.2, -.3, .1],
                      "B": [-1.2, -.2, .2, 1.6, .6, -.2]})
    levels = z.copy()
    levels.loc[4, "B"] = -.02
    np.testing.assert_array_equal(
        timing_directions(z, levels).to_numpy(),
        [[1, -1], [1, -1], [0, 0], [-1, 1], [-1, 0], [0, 0]],
    )


def test_multi_day_forward_return_compounds_only_complete_future_windows():
    from research.ic_study import forward_returns

    dates = pd.bdate_range("2021-12-01", periods=10)
    returns = pd.DataFrame({"A": [0.9, .01, .02, .03, .04, .05, .06, .07, .08, .09]}, index=dates)
    forward = forward_returns(returns, 5, 0)
    assert forward.iloc[0, 0] == pytest.approx(np.prod([1.01, 1.02, 1.03, 1.04, 1.05]) - 1)
    assert forward_returns(returns, 5, 1).iloc[0, 0] == pytest.approx(np.prod([1.02, 1.03, 1.04, 1.05, 1.06]) - 1)
    assert forward.iloc[-5:].isna().all().all()
    returns.iloc[4, 0] = np.nan
    assert pd.isna(forward_returns(returns, 5, 0).iloc[0, 0])


def test_fixed_rebalance_holds_integer_lots_between_schedule_dates():
    from backtest.engine import Quote, simulate

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
    from backtest.engine import lot_targets

    result = lot_targets(np.array([1.0, -1.0, 0.01]), np.array([90, 80, 40]),
                         np.array([0.5, 0.5, 0.5]), np.full(3, 100), 1000, 0.5)
    np.testing.assert_array_equal(result, [5, -6, 0])


def test_strategy_summary_calmar_uses_annual_return_and_absolute_drawdown():
    from strategies.power_kds import _eval_book

    returns = pd.Series([0.10, -0.05, 0.02], index=pd.bdate_range("2020-01-01", periods=3))
    equity = P.CAPITAL_CNY * (1 + returns).cumprod()
    daily = pd.DataFrame({
        "return": returns,
        "turnover": 0.0,
        "equity": equity,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "slippage": 0.0,
        "gross_notional": 0.0,
        "net_notional": 0.0,
        "margin_fraction": 0.0,
        "open_contracts": 0,
        "stale_marks": 0,
    })
    benchmark = pd.Series([0.0, 0.01, -0.01], index=returns.index)

    result = _eval_book("TEST", "combo", 0, 1, daily, benchmark)

    assert result["max_drawdown"] == pytest.approx(-0.05)
    cagr = (equity.iloc[-1] / P.CAPITAL_CNY) ** (P.TD / len(returns)) - 1
    assert result["calmar"] == pytest.approx(cagr / 0.05)
    assert result["sortino"] == pytest.approx(P.metrics(returns)["sortino"])


def test_execution_roll_closes_old_and_opens_new_with_separate_fees():
    from backtest.engine import Quote, simulate

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


def test_expired_contract_leaves_the_book_but_a_one_day_gap_does_not():
    from backtest.engine import Quote, simulate

    dates = pd.bdate_range("2021-12-01", periods=5)
    contracts = pd.DataFrame({"A": ["A2201"] * 2 + ["A2205"] * 3}, index=dates)
    weights = pd.DataFrame({"A": [.2, .2, 0, 0, 0]}, index=dates)

    def quote(price):
        return Quote(price, price, 10, 1, .1, .1, True, 0, 0, 1, 100, 100)

    expired = {
        dates[0]: {"A2201": quote(100)},
        dates[1]: {"A2201": quote(110)},
        dates[2]: {},
        dates[3]: {},
        dates[4]: {},
    }
    daily, _ = simulate(weights, contracts, expired, dates[0], capital=100_000, neutral=False, slippage_ticks=0)
    assert daily.loc[dates[1], "open_contracts"] == 1
    assert daily.loc[dates[2], "stale_marks"] == 1
    assert daily.loc[dates[2], "open_contracts"] == 0
    assert daily.loc[dates[3:], "stale_marks"].eq(0).all()
    assert daily.loc[dates[2], "equity"] == pytest.approx(daily.loc[dates[1], "equity"])

    same = pd.DataFrame({"A": ["A2201"] * 5}, index=dates)
    held = pd.DataFrame({"A": [.2, .2, .2, .2, .2]}, index=dates)
    gapped = {
        dates[0]: {"A2201": quote(100)},
        dates[1]: {"A2201": quote(110)},
        dates[2]: {},
        dates[3]: {"A2201": quote(110)},
        dates[4]: {"A2201": quote(112)},
    }
    daily, _ = simulate(held, same, gapped, dates[0], capital=100_000, neutral=False, slippage_ticks=0)
    assert daily.loc[dates[2], "stale_marks"] == 1
    assert daily.loc[dates[2], "open_contracts"] == 1
    assert daily.loc[dates[3], "open_contracts"] == 1
    assert daily.loc[dates[3], "stale_marks"] == 0


def test_zero_client_limit_forces_full_expiry_exit():
    from backtest.engine import Quote, simulate

    dates = pd.bdate_range("2022-05-09", periods=4)
    contracts = pd.DataFrame(
        {"ZC": ["ZC2205", "ZC2205", "ZC2205", "ZC2209"]}, index=dates
    )
    weights = pd.DataFrame({"ZC": [0.05] * len(dates)}, index=dates)

    def quote(price, client_limit, max_order=100):
        return Quote(
            price, price, 10, 1, 0.1, 0.1, True, 1, 1, 1, max_order,
            client_limit, open_cap=100, exit_cap=1,
        )

    market = {
        dates[0]: {"ZC2205": quote(100, 100)},
        dates[1]: {"ZC2205": quote(100, 100)},
        dates[2]: {"ZC2205": quote(90, 0, max_order=3)},
        dates[3]: {"ZC2209": quote(90, 100)},
    }

    daily, fills = simulate(
        weights, contracts, market, dates[0], capital=100_000,
        neutral=False, slippage_ticks=1,
    )

    expiry_fills = fills.loc[fills["date"].eq(dates[2])]
    assert expiry_fills["trade_lots"].sum() == -5
    assert expiry_fills["action"].eq("close").all()
    assert expiry_fills["notional"].sum() == pytest.approx(4_500)
    assert expiry_fills["fee"].sum() == pytest.approx(5)
    assert expiry_fills["slippage"].sum() == pytest.approx(50)
    assert daily.loc[dates[2], "turnover"] > 0
    assert daily.loc[dates[2], "stale_marks"] == 0
    assert daily.loc[dates[3], "open_contracts"] == 0


def test_execution_carries_old_contract_if_a_sell_is_blocked_at_limit_down():
    from backtest.engine import Quote, simulate

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
    before = D.eligibility(panel)
    changed = panel.copy()
    cut = panel["date"].iloc[250]
    changed.loc[changed["date"] > cut, "turnover_total"] = 0.0
    after = D.eligibility(changed)
    past = panel["date"] <= cut
    pd.testing.assert_series_equal(before[past], after[past])
    assert before[past].iloc[-1]


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
    held = R.hold_cohorts(weights, 2)
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


def test_member_rank_download_rejects_empty_and_invalidates_empty_cache(tmp_path):
    from download.getExternal import CoverageCache
    import datetime as dt

    root = tmp_path / "daily"
    cache_dir = root / "member_rank"
    cache_dir.mkdir(parents=True)
    empty_path = cache_dir / "A_long.pkl"
    empty_path.write_bytes(b"")
    with empty_path.open("wb") as stream:
        pd.to_pickle(pd.DataFrame(), stream)

    cache = CoverageCache(root)
    cache._entry("member_rank", "A_long")["coverage"] = [["20220101", "20251231"]]
    cache.flush()
    cache.invalidate_empty("member_rank")
    assert not empty_path.exists()
    assert cache.manifest["items"]["member_rank/A_long"]["coverage"] == []

    cache.fetch(
        "member_rank", "A_long",
        dt.date(2022, 1, 1), dt.date(2022, 1, 2),
        lambda left, right: pd.DataFrame(),
        require_nonempty=True,
    )
    assert not empty_path.exists()
    assert cache.manifest["items"]["member_rank/A_long"]["coverage"] == []


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


def test_download_rejects_financial_futures():
    from download import getMinutes

    with pytest.raises(SystemExit):
        getMinutes.main(["-t", "20240101-20240102", "--symbols", "IF"])


def test_short_shock_uses_past_quantiles_and_missing_stays_inactive():
    from strategies.rules import short_shock

    values = pd.Series([-30., -20., -10., 100., np.nan, -200.])
    signal, strength = short_shock(values, window=3, minimum=2)
    assert signal.iloc[:3].tolist() == [0., 0., 1.]
    assert signal.iloc[3:].tolist() == [1., 0., -1.]
    assert strength.iloc[3] > 0 and strength.iloc[5] < 0


def test_short_shock_keeps_crossings_after_flat_history():
    from strategies.rules import short_shock

    signal, strength = short_shock(pd.Series([0., 0., 0., 1.]), window=3, minimum=3)
    assert signal.iloc[-1] == 1
    assert strength.iloc[-1] > 0


def test_execution_retains_same_direction_lots_and_reconciles_equity():
    from backtest.engine import Quote, simulate

    dates = pd.bdate_range("2021-12-01", periods=4)
    contracts = pd.DataFrame({"A": "A2201"}, index=dates)
    weights = pd.DataFrame({"A": [.05, .05, 0., 0.]}, index=dates)
    market = {date: {"A2201": Quote(price, price, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000)}
              for date, price in zip(dates, [100, 100, 122, 125])}
    daily, fills = simulate(weights, contracts, market, dates[0], capital=100_000,
                            neutral=False, keep_same_lots=True)
    assert fills.trade_lots.tolist() == [5, -5]
    assert daily.equity.iloc[-1] == pytest.approx(
        100_000 + daily.gross_pnl.sum() - daily.fees.sum() - daily.slippage.sum()
    )


def test_active_weights_respects_min_slots():
    from strategies.rules import active_weights

    dates = pd.bdate_range("2021-01-04", periods=2)
    dirs = pd.DataFrame(0.0, index=dates, columns=["A", "B", "C", "D"])
    dirs.iloc[0, 0] = 1.0
    dirs.iloc[1, :3] = [1.0, 1.0, -1.0]
    w = active_weights(dirs, min_slots=2)
    assert w.iloc[0].tolist() == [0.5, 0.0, 0.0, 0.0]
    assert w.iloc[1].tolist() == pytest.approx([1 / 3, 1 / 3, -1 / 3, 0.0])


def test_execution_parameters_partition_guards():
    from download import getExecutionParameters as gep
    import config as C

    assert gep.PARTITION_CACHE["oos"] == C.OOS_EXECUTION_PARAMETERS
    gep._check_dates(pd.to_datetime(["2021-12-31"]), "research", "test")
    gep._check_dates(pd.to_datetime(["2022-01-04", "2025-12-31"]), "oos", "test")
    with pytest.raises(C.HoldoutViolation):
        gep._check_dates(pd.to_datetime(["2022-01-04"]), "research", "test")
    with pytest.raises(C.HoldoutViolation):
        gep._check_dates(pd.to_datetime(["2021-12-31"]), "oos", "test")


def test_load_market_accepts_different_datetime_resolutions(tmp_path):
    from backtest.engine import load_market

    dates = pd.to_datetime(["2021-12-29", "2021-12-30"]).astype("datetime64[us]")
    contract = "AA2201"
    root = tmp_path / "parameters" / "contracts"
    root.mkdir(parents=True)
    parameters = pd.DataFrame({
        "long_margin_ratio": [.1, .1],
        "short_margin_ratio": [.1, .1],
        "commission_type": ["by_volume", "by_volume"],
        "open_commission": [1., 1.],
        "close_commission": [1., 1.],
        "max_order_quantity": [100, 100],
        "min_order_quantity": [1, 1],
        "client_limit": [100, 100],
    }, index=dates)
    parameters.to_pickle(root / f"{contract}.pkl")
    price_limits = pd.DataFrame({"limit_up": [120., 120.], "limit_down": [80., 80.]}, index=dates)
    price_limit_root = root.parent / "price_limits"
    price_limit_root.mkdir()
    price_limits.to_pickle(price_limit_root / f"{contract}.pkl")
    bars = pd.DataFrame({
        "date": dates,
        "contract": contract,
        "open": [100., 101.],
        "settlement": [100., 101.],
        "volume": [1000., 1000.],
    })
    fees = pd.DataFrame({
        "trading_date": pd.to_datetime(["2021-12-29"]).astype("datetime64[ns]"),
        "contract": [contract],
        "commission_type": ["by_volume"],
        "open_commission": [1.],
        "close_commission": [1.],
        "contract_multiplier": [10.],
        "tick_size": [1.],
    })

    market = load_market(bars, fees, {contract}, dates[0], root=root)

    assert market[pd.Timestamp("2021-12-29")][contract].multiplier == 10


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


def test_simulate_margin_cap_and_power_kds_weights():
    from backtest.engine import Quote, simulate
    from strategies.power_kds import power_kds_weights

    dates = pd.bdate_range("2021-12-01", periods=3)
    contracts = pd.DataFrame({"A": "A2201"}, index=dates)
    weights = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=dates)
    # 100 * 10 = 1000 名义，80% 保证金 -> 每手保证金 800；权重 1.0 目标 100 手(80% 保证金)，50% 上限=62 手，100% 上限=100 手
    market = {d: {"A2201": Quote(100, 100, 10, 1, 0.80, 0.80, True, 0, 0, 1, 10000, 10000)} for d in dates}
    _, fills_50 = simulate(weights, contracts, market, dates[0], capital=100_000,
                           neutral=False, slippage_ticks=0, name_cap=1.0, margin_cap=0.50)
    _, fills_100 = simulate(weights, contracts, market, dates[0], capital=100_000,
                            neutral=False, slippage_ticks=0, name_cap=1.0, margin_cap=1.00)
    assert fills_50.iloc[0]["trade_lots"] == 62
    assert fills_100.iloc[0]["trade_lots"] == 100
    with pytest.raises(ValueError):
        simulate(weights, contracts, market, dates[0], margin_cap=1.2)

    elig = pd.DataFrame(True, index=dates, columns=["A", "B"])
    z = pd.DataFrame({"A": [1.5, 1.2, 0.5], "B": [0.0, -1.5, -0.5]}, index=dates)
    raw = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, -1.0, -1.0]}, index=dates)
    shock = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [-1.0, 0.0, 0.0]}, index=dates)
    wc, ws, w50 = power_kds_weights(z, raw, shock, elig, horizon=2)
    np.testing.assert_allclose(w50.to_numpy(), (0.5 * wc + 0.5 * ws).to_numpy())


def test_flow_control_ablation_preserves_fixed_leg_budgets():
    from strategies.power_kds import flow_ablation_weights, power_kds_weights

    dates = pd.bdate_range("2021-12-01", periods=3)
    eligible = pd.DataFrame(True, index=dates, columns=["A", "B"])
    z_power = pd.DataFrame({"A": [1.5, 1.2, 0.5], "B": [-1.5, -1.2, -0.5]}, index=dates)
    raw_power = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=dates)
    shock = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates)
    z_flow = pd.DataFrame({"A": [0.0, 1.5, 0.0], "B": [0.0, -1.5, 0.0]}, index=dates)
    raw_flow = pd.DataFrame({"A": [0.0, 1.0, 0.0], "B": [0.0, -1.0, 0.0]}, index=dates)

    w_core, w_shock, _ = power_kds_weights(z_power, raw_power, shock, eligible, horizon=2)
    flow_direction = R.timing_directions(z_flow, raw_flow)
    w_flow = R.hold_cohorts(R.active_weights(flow_direction), 2)
    ablations = flow_ablation_weights(
        z_power, raw_power, shock, z_flow, raw_flow, eligible, horizon=2
    )

    np.testing.assert_allclose(
        ablations["POWER_KDS_FLOW_REPLACES_SAT_2X"].to_numpy(),
        (0.5 * w_core + 0.5 * w_flow).to_numpy(),
    )
    np.testing.assert_allclose(
        ablations["POWER_KDS_FLOW_502525_2X"].to_numpy(),
        (0.5 * w_core + 0.25 * w_shock + 0.25 * w_flow).to_numpy(),
    )


def test_power_kds_weights_can_normalize_by_actual_active_count():
    from strategies.power_kds import power_kds_weights

    dates = pd.bdate_range("2022-01-03", periods=1)
    eligible = pd.DataFrame(True, index=dates, columns=["A", "B"])
    z_power = pd.DataFrame({"A": [1.5], "B": [0.0]}, index=dates)
    raw_power = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates)
    shock = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates)

    core_default, _, combined_default = power_kds_weights(
        z_power, raw_power, shock, eligible, horizon=1,
    )
    core_actual, _, combined_actual = power_kds_weights(
        z_power, raw_power, shock, eligible, horizon=1, min_slots=1,
    )

    assert core_default.loc[dates[0], "A"] == pytest.approx(0.1)
    assert core_actual.loc[dates[0], "A"] == pytest.approx(1.0)
    assert combined_default.loc[dates[0], "A"] == pytest.approx(0.1)
    assert combined_actual.loc[dates[0], "A"] == pytest.approx(1.0)


def test_equity_drawdown_plot_compares_strategy_with_equal_weight_on_log_scale(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    dates = pd.bdate_range("2021-12-27", periods=4)
    equity = pd.DataFrame({
        "strategy": [10_000_000, 12_400_000, 6_000_000, 10_500_000],
        "benchmark": [10_000_000, 10_500_000, 10_300_000, 11_000_000],
    }, index=dates)
    closed_figures = []
    close = plt.close

    def capture_figure(*args, **kwargs):
        closed_figures.append(plt.gcf())
        close(*args, **kwargs)

    monkeypatch.setattr(plt, "close", capture_figure)

    path = P.plot_equity_drawdown(
        equity, tmp_path / "performance.png", strategy="strategy", benchmark="benchmark",
        period_label="样本外",
    )

    assert path.is_file()
    assert path.stat().st_size > 0
    assert len(closed_figures[-1].axes) == 3
    assert closed_figures[-1].axes[0].get_yscale() == "log"
    assert any("样本外" in text.get_text() for text in closed_figures[-1].texts)
    closed_figures[-1].canvas.draw()
    nav_tick_labels = {
        tick.get_text() for tick in closed_figures[-1].axes[0].get_yticklabels()
    }
    assert {"0.75×", "1×", "1.25×"} <= nav_tick_labels


def test_legacy_contracts_stop_at_successor_listing_date():
    frame = pd.DataFrame({
        "symbol": ["ME", "ME", "MA", "RB"],
        "date": pd.to_datetime(["2014-06-16", "2014-06-17", "2014-06-17", "2014-06-16"]),
        "contract": ["ME1501", "ME1501", "MA1506", "RB1410"],
    })
    out = D.apply_legacy_contracts(frame)
    assert out["symbol"].tolist() == ["MA", "RB", "MA"]
    assert out["contract"].tolist() == ["MA1506", "RB1410", "ME1501"]


def test_legacy_dominant_does_not_overwrite_successor_days():
    current = pd.Series(["MA1506", "MA1506"], index=pd.to_datetime(["2014-06-17", "2014-06-18"]), name="rank1")
    legacy = pd.Series(["ME1505", "ME1505", "ME1505"], index=pd.to_datetime(["2014-06-16", "2014-06-17", "2014-06-18"]))
    out = D.stitch_legacy_dominant(current, legacy, pd.Timestamp("2014-06-17"))
    assert out.tolist() == ["ME1505", "MA1506", "MA1506"]
    assert str(out.index.min().date()) == "2014-06-16"


def test_calendar_year_return_uses_prior_year_end_equity():
    from strategies.power_kds import calendar_year_returns

    equity = pd.DataFrame({
        "A": [100.0, 110.0, 121.0, 100.0],
    }, index=pd.to_datetime(["2016-01-04", "2016-12-30", "2017-01-03", "2017-12-29"]))
    out = calendar_year_returns(equity).set_index(["book", "year"])["return"]
    assert out[("A", 2016)] == pytest.approx(0.10)
    assert out[("A", 2017)] == pytest.approx(100.0 / 110.0 - 1.0)

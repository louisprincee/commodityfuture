"""整数手执行、账户和绩效图的合成数据测试。"""

import numpy as np
import pandas as pd
import pytest
from backtest import engine as P


def test_only_three_active_strategies_are_registered_and_continuous_results_use_date_folder():
    assert set(P.MODULES) == {"sticky_net", "persist_identity", "sticky_basis_skew"}
    assert P.COMPARED == ["sticky_net", "persist_identity"]
    assert P.partition_directory_name("combined") == "2011-2025"
    assert P.partition_directory_name("research") == "research"


def test_each_strategy_writes_its_own_partition_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(P.C, "RUNS_DIR", tmp_path)
    assert P.book_dir("sticky_net") == tmp_path / "strategies" / "sticky_net"
    price = "sticky_basis_skew"
    assert P.book_dir(price) == tmp_path / "strategies" / price
    assert P.holding_days(price) == P.HORIZON
    with monkeypatch.context() as patch:
        patch.setattr(P.MODULES[price], "HOLDING_DAYS", 20)
        assert P.holding_days(price) == 20
        assert P.book_dir(price) == tmp_path / "strategies" / price / "holding_20"
        assert P.book_dir("sticky_net") == tmp_path / "strategies" / "sticky_net"
        patch.setattr(P.MODULES[price], "HOLDING_DAYS", 0)
        with pytest.raises(ValueError, match="正整数"):
            P.book_dir(price)
    assert P.comparison_dir() == tmp_path / "strategies" / "comparison"

    dates = pd.bdate_range("2022-01-03", periods=5)
    equity = pd.DataFrame({"STICKY_NET": np.linspace(1.0, 1.04, 5) * P.CAPITAL_CNY,
                           P.MODULES[price].MAIN_BOOK: -100.,
                           P.BENCHMARK_BOOK: np.linspace(1.0, 0.98, 5) * P.CAPITAL_CNY}, index=dates)
    for name, book in (("sticky_net", "STICKY_NET"), ("benchmark", P.BENCHMARK_BOOK)):
        folder = P.C.strategy_dir(name) / "oos"
        folder.mkdir(parents=True)
        pd.DataFrame({"book": [book], "ticks": [1]}).to_csv(folder / "summary.csv", index=False)
    out = P.write_comparison("oos", ["sticky_net", price], equity)
    assert out == tmp_path / "strategies" / "comparison" / "oos"
    assert pd.read_csv(out / "summary.csv")["book"].tolist() == ["STICKY_NET", P.BENCHMARK_BOOK]
    assert {"equity_curves.csv", "yearly_returns.csv", "overview.png"} <= {p.name for p in out.iterdir()}
    assert not (out / "paired_gains.csv").exists()


def test_historical_fee_supplement_stays_inside_sourced_interval():
    from src.costs import fill_historical_fees, carry_forward_fees

    dates = pd.to_datetime(['2011-07-29', '2011-08-01', '2011-08-02', '2012-06-01'])
    index = pd.MultiIndex.from_arrays([['CU1112'] * 4, dates], names=['contract', 'date'])
    bars = pd.DataFrame({'commission_type': ['by_money'] * 4,
                         'open_commission': [np.nan, np.nan, .0002, np.nan],
                         'close_commission': [np.nan, np.nan, .0002, np.nan]}, index=index)
    rules = pd.DataFrame([dict(symbol='CU', start='2011-08-01', end='2012-05-31',
                               commission_type='by_money', open_commission=.0001, close_commission=.0001)])
    result = fill_historical_fees(bars, rules)
    assert result.open_commission.isna().tolist() == [True, False, False, True]
    assert result.open_commission.iloc[1] == .0001
    assert result.open_commission.iloc[2] == .0002
    carried = carry_forward_fees(result)
    assert pd.isna(carried.open_commission.iloc[0])
    assert carried.open_commission.iloc[-1] == .0002
    assert carried.fee_source_date.iloc[-1] == dates[2]
    assert carried.fee_carried.iloc[-1]
    # A new contract can use only a strictly earlier, unambiguous schedule of
    # the same underlying; its own prior record continues to take precedence.
    later_days = pd.to_datetime(["2011-08-02", "2011-08-03", "2011-08-03"])
    later_index = pd.MultiIndex.from_arrays([["CU1205", "CU1205", "AL1205"], later_days],
                                             names=["contract", "date"])
    new_contracts = pd.DataFrame(np.nan, index=later_index, columns=result.columns)
    expanded = pd.concat([result, new_contracts])
    carried = carry_forward_fees(expanded)
    assert carried.loc[("CU1205", later_days[0]), "open_commission"] == .0001
    assert carried.loc[("CU1205", later_days[1]), "open_commission"] == .0002
    assert carried.loc[("CU1205", later_days[1]), "fee_source_contract"] == "CU1112"
    assert carried.loc[("CU1205", later_days[1]), "fee_cross_contract"]
    assert pd.isna(carried.loc[("AL1205", later_days[2]), "open_commission"])
    conflict = result.iloc[[2]].copy()
    conflict.index = pd.MultiIndex.from_tuples([("CU1212", dates[2])], names=result.index.names)
    conflict[["open_commission", "close_commission"]] = .0003
    carried = carry_forward_fees(pd.concat([expanded, conflict]))
    assert carried.loc[("CU1205", later_days[1]), "open_commission"] == .0001
    assert carried.loc[("CU1112", dates[-1]), "open_commission"] == .0002

def test_early_account_starts_flat_and_trades_close_signal_at_next_open():
    from costs import Quote

    dates = pd.to_datetime(["2010-12-31", "2011-01-04", "2011-01-05", "2011-01-06"])
    weights = pd.DataFrame({"A": [.9, .1, .1, .1]}, index=dates)
    contracts = pd.DataFrame({"A": ["A1105"] * 4}, index=dates)
    market = {day: {"A1105": Quote(100+i, 100+i, 10, 1, .2, .2, True, 1, 1, 1, 999, 999)}
              for i, day in enumerate(dates)}
    # Account receives only cohorts started in its declared evaluation span.
    ledger, fills = P.simulate(weights.loc[dates[1]:], contracts.loc[dates[1]:], market, dates[1],
                               neutral=False, capital=100_000, slippage_ticks=0, non_delivery=False)
    assert ledger.index.min() == dates[1]
    assert ledger.loc[dates[1], "open_lots"] == 0
    assert fills.iloc[0]["date"] == dates[2]
    assert fills.iloc[0]["fill"] == 102


def test_summary_uses_equity_drawdown_and_backtest_sortino():
    returns = pd.Series([0.10, -0.05, 0.02], index=pd.bdate_range("2020-01-01", periods=3))
    equity = P.CAPITAL_CNY * (1 + returns).cumprod()
    frame = equity.to_frame("synthetic")
    assert P._period_sharpe(frame)["synthetic"] == pytest.approx(P.metrics(returns)["sharpe"])
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

    result = P._book_summary("TEST", 1, daily)

    assert result["max_drawdown"] == pytest.approx(-0.05)
    assert result["sortino"] == pytest.approx(P.metrics(returns)["sortino"])


def test_execution_roll_closes_old_and_opens_new_with_separate_fees():
    from backtest.engine import simulate
    from costs import Quote

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
    daily, orders = simulate(weights, contracts, market, dates[0], capital=100_000, neutral=False, non_delivery=False)
    roll = orders[orders.date.eq(dates[2])].set_index("contract")
    assert roll.loc[calls["A"], "fee"] == 27
    assert roll.loc[calls["B"], "fee"] == 40
    assert daily.iloc[-1]["open_contracts"] == 0


def test_missing_quotes_retain_unresolved_positions_without_inventing_a_close():
    from backtest.engine import simulate
    from costs import Quote

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
    daily, _ = simulate(weights, contracts, expired, dates[0], capital=100_000, neutral=False, slippage_ticks=0, non_delivery=False)
    assert daily.loc[dates[1], "open_contracts"] == 1
    assert daily.loc[dates[2], "stale_marks"] == 1
    assert daily.loc[dates[2], "open_contracts"] == 1
    assert daily.loc[dates[3:], "stale_marks"].eq(1).all()
    assert daily.iloc[-1]["open_contracts"] == 1
    assert daily.loc[dates[2]:, "traded_notional"].eq(0).all()
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
    daily, _ = simulate(held, same, gapped, dates[0], capital=100_000, neutral=False, slippage_ticks=0, non_delivery=False)
    assert daily.loc[dates[2], "stale_marks"] == 1
    assert daily.loc[dates[2], "open_contracts"] == 1
    assert daily.loc[dates[3], "open_contracts"] == 1
    assert daily.loc[dates[3], "stale_marks"] == 0


def test_zero_client_limit_stops_opening_without_bypassing_close_capacity():
    from backtest.engine import simulate
    from costs import Quote

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
        dates[3]: {"ZC2205": quote(90, 0), "ZC2209": quote(90, 100)},
    }

    daily, fills = simulate(
        weights, contracts, market, dates[0], capital=100_000,
        neutral=False, slippage_ticks=1,
        non_delivery=False,
    )

    expiry_fills = fills.loc[fills["date"].eq(dates[2])]
    assert expiry_fills["trade_lots"].sum() == -1
    assert expiry_fills["action"].eq("close").all()
    assert expiry_fills["notional"].sum() == pytest.approx(900)
    assert expiry_fills["fee"].sum() == pytest.approx(1)
    assert expiry_fills["slippage"].sum() == pytest.approx(10)
    assert daily.loc[dates[2], "turnover"] > 0
    assert daily.loc[dates[2], "stale_marks"] == 0
    assert daily.loc[dates[3], "open_lots"] == 3


def test_execution_carries_old_contract_if_a_sell_is_blocked_at_limit_down():
    from backtest.engine import simulate
    from costs import Quote

    days = pd.date_range("2021-12-27", periods=4)
    principal = pd.DataFrame({"A": ["A2201", "A2201", "A2205", "A2205"]}, index=days)
    weights = pd.DataFrame({"A": [0.1] * 4}, index=days)
    normal = Quote(100, 100, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000)
    locked = Quote(90, 90, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000,
                   limit_up=110, limit_down=90)
    market = {day: {"A2201": normal, "A2205": normal} for day in days}
    market[days[2]]["A2201"] = locked
    daily, fills = simulate(weights, principal, market, days[0], capital=100_000, neutral=False, non_delivery=False)
    assert daily.at[days[2], "limit_rejections"] == 1
    assert daily.at[days[2], "blocked_rolls"] == 1
    assert not fills.date.eq(days[2]).any()
    assert daily.iloc[-1]["open_contracts"] == 0


def test_execution_rebalances_integer_lots_and_reconciles_equity():
    from backtest.engine import simulate
    from costs import Quote

    dates = pd.bdate_range("2021-12-01", periods=4)
    contracts = pd.DataFrame({"A": "A2201"}, index=dates)
    weights = pd.DataFrame({"A": [.05, .05, 0., 0.]}, index=dates)
    market = {date: {"A2201": Quote(price, price, 10, 1, .15, .15, True, 2, 3, 1, 10000, 10000)}
              for date, price in zip(dates, [100, 100, 122, 125])}
    daily, fills = simulate(weights, contracts, market, dates[0], capital=100_000,
                            neutral=False, non_delivery=False)
    assert fills.trade_lots.tolist() == [5, -1, -4]
    assert daily.equity.iloc[-1] == pytest.approx(
        100_000 + daily.gross_pnl.sum() - daily.fees.sum() - daily.slippage.sum()
    )


@pytest.mark.parametrize("limit_state", ["available", "missing", "nonpositive"])
def test_load_market_datetime_resolution_and_missing_limit_entry_guard(tmp_path, monkeypatch, limit_state):
    from costs import load_market

    monkeypatch.setattr(P.C, "CONFIG_DIR", tmp_path)

    dates = pd.bdate_range("2021-12-20", periods=8).astype("datetime64[us]")
    contract = "AA2201"
    root = tmp_path / "parameters" / "contracts"
    root.mkdir(parents=True)
    parameters = pd.DataFrame({
        "long_margin_ratio": .1,
        "short_margin_ratio": .1,
        "commission_type": "by_volume",
        "open_commission": 1.,
        "close_commission": 1.,
        "max_order_quantity": 100,
        "min_order_quantity": 1,
        "client_limit": 100,
    }, index=dates)
    parameters.to_pickle(root / f"{contract}.pkl")
    price_limits = pd.DataFrame({"limit_up": 120., "limit_down": 80.}, index=dates)
    if limit_state == "missing":
        price_limits = price_limits.drop(dates[-1])
    elif limit_state == "nonpositive":
        price_limits.loc[dates[-1], "limit_down"] = 0.
    price_limit_root = root.parent / "price_limits"
    price_limit_root.mkdir()
    price_limits.to_pickle(price_limit_root / f"{contract}.pkl")
    bars = pd.DataFrame({
        "date": dates,
        "contract": contract,
        "open": 100.,
        "settlement": 100.,
        "volume": 1000.,
    })
    fees = pd.DataFrame({
        "trading_date": pd.DatetimeIndex([dates[0]]).astype("datetime64[ns]"),
        "contract": [contract],
        "commission_type": ["by_volume"],
        "open_commission": [1.],
        "close_commission": [1.],
        "contract_multiplier": [10.],
        "tick_size": [1.],
    })
    (root.parent / "specifications").mkdir()
    pd.DataFrame({"contract_multiplier": 10., "tick_size": 1.,
                  "last_trading_date": pd.Timestamp("2022-01-20"),
                  "start_delivery_date": pd.Timestamp("2022-01-01")}, index=dates).to_pickle(
                      root.parent / "specifications" / f"{contract}.pkl")

    market = load_market(bars, fees, {contract}, dates[0], root=root)

    assert market[dates[0]][contract].multiplier == 10
    final = market[dates[-1]][contract]
    assert final.open_cap == (10 if limit_state == "available" else 0)
    assert final.exit_cap == 10


def test_simulate_margin_cap():
    from backtest.engine import simulate
    from costs import Quote

    dates = pd.bdate_range("2021-12-01", periods=3)
    contracts = pd.DataFrame({"A": "A2201"}, index=dates)
    weights = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=dates)
    # 100 * 10 = 1000 名义，80% 保证金 -> 每手保证金 800；权重 1.0 目标 100 手(80% 保证金)，50% 上限=62 手，100% 上限=100 手
    market = {d: {"A2201": Quote(100, 100, 10, 1, 0.80, 0.80, True, 0, 0, 1, 10000, 10000)} for d in dates}
    _, fills_50 = simulate(weights, contracts, market, dates[0], capital=100_000,
                           neutral=False, slippage_ticks=0, name_cap=1.0, margin_cap=0.50, non_delivery=False)
    _, fills_100 = simulate(weights, contracts, market, dates[0], capital=100_000,
                            neutral=False, slippage_ticks=0, name_cap=1.0, margin_cap=1.00, non_delivery=False)
    assert fills_50.iloc[0]["trade_lots"] == 62
    assert fills_100.iloc[0]["trade_lots"] == 100
    with pytest.raises(ValueError):
        simulate(weights, contracts, market, dates[0], margin_cap=1.2, non_delivery=False)


def test_oos_fee_fallback_reaches_back_to_research_rows():
    research = pd.DataFrame({"contract": ["RR2202"], "trading_date": pd.to_datetime(["2021-12-31"]),
                             "open_commission": [1.0]})
    oos = pd.DataFrame({"contract": ["RR2203"], "trading_date": pd.to_datetime(["2022-01-04"]),
                        "open_commission": [2.0]})
    fallback = P._oos_fee_fallback(research, oos).sort_values("trading_date")
    query = pd.DataFrame({"contract": ["RR2202", "RR2203"], "date": pd.to_datetime(["2022-01-04"] * 2)})
    matched = pd.merge_asof(query, fallback.rename(columns={"trading_date": "date"}),
                            on="date", by="contract", direction="backward")
    assert matched["open_commission"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("blocked", [None, "price_limit", "missing_limits"])
def test_non_delivery_exit_respects_capacity_and_fails_if_untradeable(blocked):
    from costs import Quote

    dates = pd.bdate_range("2016-03-01", "2016-04-08")
    contract = "AA1604"
    contracts = pd.DataFrame({"AA": contract}, index=dates)
    weights = pd.DataFrame({"AA": 1.}, index=dates)
    market = {}
    for day in dates:
        in_window = pd.Timestamp("2016-03-25") <= day <= pd.Timestamp("2016-03-31")
        price = 80. if blocked == "price_limit" and in_window else 100.
        market[day] = {contract: Quote(
            price, price, 10, 1, .1, .1, True, 1, 2, 1, 100, 1000,
            open_cap=2, exit_cap=2,
            limit_down=np.nan if blocked == "missing_limits" and in_window else 80., limit_up=120.,
            last_trading_date=pd.Timestamp("2016-04-15"),
            start_delivery_date=pd.Timestamp("2016-04-01"))}
    if blocked:
        with pytest.raises(P.ExecutionFailure, match="AA1604"):
            P.simulate(weights, contracts, market, dates[0], capital=100_000,
                       neutral=False, slippage_ticks=0, name_cap=1.)
    else:
        daily, fills = P.simulate(weights, contracts, market, dates[0], capital=100_000,
                                  neutral=False, slippage_ticks=0, name_cap=1.)
        assert daily.open_lots.max() == 6  # 3-day target capacity leaves safety buffer inside 5-day exit window
        assert fills.trade_lots.abs().le(2).all()
        assert not fills.loc[fills.date.ge("2016-03-25"), "action"].eq("open").any()
        assert daily.loc["2016-03-31":, "open_contracts"].eq(0).all()
        assert daily.stale_marks.eq(0).all()
        assert daily.equity.iloc[-1] == pytest.approx(100_000 - fills.fee.sum())


def test_contract_specifications_do_not_depend_on_commission_history(tmp_path, monkeypatch):
    from costs import load_market

    monkeypatch.setattr(P.C, "CONFIG_DIR", tmp_path)

    days = pd.bdate_range("2011-01-04", periods=8)
    contract = "AA1105"
    root = tmp_path / "parameters" / "contracts"
    root.mkdir(parents=True)
    pd.DataFrame({"long_margin_ratio": .1, "short_margin_ratio": .1,
                  "commission_type": float("nan"), "open_commission": float("nan"),
                  "close_commission": float("nan"), "min_order_quantity": 1,
                  "max_order_quantity": 100, "client_limit": 100}, index=days).to_pickle(root / f"{contract}.pkl")
    (root.parent / "price_limits").mkdir()
    pd.DataFrame({"limit_up": 120., "limit_down": 80.}, index=days).to_pickle(root.parent / "price_limits" / f"{contract}.pkl")
    (root.parent / "specifications").mkdir()
    # Exact-day observations only: no future specification may leak backwards.
    pd.DataFrame({"contract_multiplier": [20.], "tick_size": [.2],
                  "last_trading_date": [pd.Timestamp("2011-05-17")],
                  "start_delivery_date": [pd.Timestamp("2011-05-01")]}, index=days[1:2]).to_pickle(root.parent / "specifications" / f"{contract}.pkl")
    bars = pd.DataFrame({"date": days, "contract": contract, "open": 100.,
                         "settlement": 100., "volume": 1000.})
    fees = pd.DataFrame({"trading_date": [days[-1]], "contract": [contract],
                         "commission_type": ["by_volume"], "open_commission": [1.],
                         "close_commission": [1.], "contract_multiplier": [99.], "tick_size": [5.]})
    market = load_market(bars, fees, {contract}, days[0], root=root)
    assert market[days[1]][contract].multiplier == 20
    assert market[days[1]][contract].tick == .2
    assert market[days[1]][contract].last_trading_date == pd.Timestamp("2011-05-17")
    assert pd.isna(market[days[0]][contract].last_trading_date)
    assert not market[days[1]][contract].can_trade
    assert pd.isna(market[days[0]][contract].multiplier)
    assert pd.isna(market[days[2]][contract].multiplier)
    # Prior observations before the account start still seed fee carry; an
    # incomplete new charging unit must not combine with an older unit's rate.
    parameters = pd.read_pickle(root / f"{contract}.pkl")
    parameters["commission_type"] = parameters.commission_type.astype(object)
    columns = ["commission_type", "open_commission", "close_commission"]
    parameters.loc[days[2], columns] = ["by_volume", 2., 3.]
    parameters.loc[days[4], columns] = ["by_money", np.nan, .0002]
    parameters.to_pickle(root / f"{contract}.pkl")
    later = load_market(bars, fees, {contract}, days[3], root=root)
    assert later[days[3]][contract].open_fee == 2.
    assert later[days[4]][contract].by_lot
    assert later[days[4]][contract].close_fee == 3.
    assert later[days[4]][contract].fee_source_date == days[2]
    assert later[days[4]][contract].fee_carried


def test_engine_research_window_uses_2010_warmup_and_preserves_legacy(monkeypatch):
    dates = pd.bdate_range("2010-01-04", "2016-01-10")
    panel = pd.DataFrame({"date": dates, "symbol": "AA", "c1_contract": "AA1105"})
    seen = {}

    monkeypatch.setattr(P.D, "load_panel", lambda: panel)
    monkeypatch.setattr(P.D, "load_contracts_span", lambda part, **kw: seen.update(span=part, kw=kw) or pd.DataFrame())
    monkeypatch.setattr(P.C, "assert_research_only", lambda p: None)
    monkeypatch.setattr(P.pd, "read_csv", lambda *a, **kw: pd.DataFrame({"trading_date": dates[:2]}))

    got_panel, _, _, start, _, data_partition, _ = P._partition_inputs("research")
    assert start == P.BACKTEST_START == pd.Timestamp("2011-01-04")
    assert data_partition == "research"
    assert seen["span"] == "research" and seen["kw"] == {"for_execution": True}
    assert got_panel["date"].min() == P.WARMUP_START

    _, _, _, legacy_start, _, legacy_part, _ = P._partition_inputs("legacy_research")
    assert legacy_start == P.LEGACY_START == pd.Timestamp("2016-01-04")
    assert legacy_part == "research"

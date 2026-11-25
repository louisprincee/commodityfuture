"""整数手执行、账户和绩效图的合成数据测试。"""

import numpy as np
import pandas as pd
import pytest

from backtest import engine as P


def test_lot_targets_round_to_zero_and_leave_margin_reserve():
    from backtest.engine import lot_targets

    result = lot_targets(np.array([1.0, -1.0, 0.01]), np.array([90, 80, 40]),
                         np.array([0.5, 0.5, 0.5]), np.full(3, 100), 1000, 0.5)
    np.testing.assert_array_equal(result, [5, -6, 0])


def test_sticky_core_summary_uses_equity_drawdown_and_backtest_sortino():
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
    daily, orders = simulate(weights, contracts, market, dates[0], capital=100_000, neutral=False)
    roll = orders[orders.date.eq(dates[2])].set_index("contract")
    assert roll.loc[calls["A"], "fee"] == 27
    assert roll.loc[calls["B"], "fee"] == 40
    assert daily.iloc[-1]["open_contracts"] == 0


def test_expired_contract_leaves_the_book_but_a_one_day_gap_does_not():
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
    daily, fills = simulate(weights, principal, market, days[0], capital=100_000, neutral=False)
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
                            neutral=False)
    assert fills.trade_lots.tolist() == [5, -1, -4]
    assert daily.equity.iloc[-1] == pytest.approx(
        100_000 + daily.gross_pnl.sum() - daily.fees.sum() - daily.slippage.sum()
    )


def test_load_market_accepts_different_datetime_resolutions(tmp_path):
    from costs import load_market

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


def test_simulate_margin_cap():
    from backtest.engine import simulate
    from costs import Quote

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


def test_continuous_dates_cover_both_partitions():
    dates = pd.bdate_range("2016-01-04", "2025-12-31")
    P._assert_continuous_dates(dates, "账本")
    with pytest.raises(ValueError):
        P._assert_continuous_dates(pd.bdate_range("2016-01-04", "2021-12-31"), "账本")


def test_overview_sharpe_can_come_from_separate_runs(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    dates = pd.bdate_range("2021-12-01", "2022-02-01")
    equity = pd.DataFrame({
        "strategy": np.linspace(P.CAPITAL_CNY, P.CAPITAL_CNY * 1.1, len(dates)),
        "benchmark": np.linspace(P.CAPITAL_CNY, P.CAPITAL_CNY * 1.05, len(dates)),
    }, index=dates)
    closed = []
    close = plt.close

    def capture(*args, **kwargs):
        closed.append(plt.gcf())
        close(*args, **kwargs)

    monkeypatch.setattr(plt, "close", capture)
    P.plot_equity_drawdown(
        equity, tmp_path / "overview.png", strategies=["strategy"], benchmark="benchmark",
        research_sharpe=pd.Series({"strategy": 1.5, "benchmark": 0.2}),
        oos_sharpe=pd.Series({"strategy": 0.4, "benchmark": 0.1}),
    )
    texts = [text.get_text() for text in closed[-1].texts]
    assert "1.500" in texts
    assert "0.400" in texts


def test_combined_price_factors_accept_dates_around_the_split():
    from research.factors.price import load_price_factors

    dates = pd.bdate_range("2021-11-01", periods=40)
    panel = pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["RB"] * len(dates) + ["HC"] * len(dates),
        "ret1": 0.001,
        "ret2": 0.0,
    })
    index = dates[dates >= "2021-12-15"]
    basis, skew = load_price_factors(index, pd.Index(["RB", "HC"]), partition="combined", panel=panel)
    assert basis.index.equals(pd.DatetimeIndex(index))
    assert list(skew.columns) == ["RB", "HC"]


def test_span_market_reads_contract_parameters_and_warms_oos_volume(monkeypatch):
    calls = []

    def fake_load_market(bars, fees, ids, start, *, root, flat_margin, partition):
        calls.append(dict(first=bars["date"].min(), last=bars["date"].max(), start=start,
                          root=root, partition=partition))
        return {pd.Timestamp(start): {}}

    monkeypatch.setattr(P, "load_market", fake_load_market)
    bars = pd.DataFrame({"date": pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-04"])})
    market = P._span_market(bars, pd.DataFrame(), pd.DataFrame(), {"RB2205"}, P.BACKTEST_START)
    research, oos = calls
    assert research["root"].name == oos["root"].name == "contracts"
    assert research["last"] == pd.Timestamp("2021-12-31")
    assert oos["partition"] == "oos" and oos["start"] == pd.Timestamp(P.C.OOS_START)
    assert oos["first"] == pd.Timestamp("2021-12-30")
    assert len(market) == 2


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

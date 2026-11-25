"""现行因子、标签和跨品种残差的合成数据测试。"""
import numpy as np
import pandas as pd
import pytest


def test_forward_return_uses_the_next_two_opens_of_the_entry_contract():
    from research.ic.study import forward_open_returns

    days = pd.DatetimeIndex(["2021-12-24", "2021-12-27", "2021-12-28", "2021-12-29"])
    dominant = pd.Series(["OLD", "NEW", "LATER", "LATER"], index=days)
    opens = pd.Series([100.0, 108.0, 400.0, 420.0], index=pd.MultiIndex.from_tuples(
        [("NEW", days[1]), ("NEW", days[2]), ("LATER", days[2]), ("LATER", days[3])]))
    labels = forward_open_returns(dominant, opens)
    assert labels.at[days[0], "entry_contract"] == "NEW"
    assert labels.at[days[0], "forward_return"] == pytest.approx(.08)
    assert labels.iloc[-2:]["forward_return"].isna().all()


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


def test_residual_scores_remove_the_rank_component_of_the_control():
    from research.member.chain import residual_scores

    day = pd.DatetimeIndex(["2020-01-02"])
    cols = list("abcdefgh")
    control = pd.DataFrame([np.arange(8.0)], index=day, columns=cols)
    residual = residual_scores(control * 2, control)
    assert np.allclose(residual.to_numpy(), 0.0)


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

import numpy as np
import pandas as pd
import pytest

from research import member_extensions as X


def test_top_share_needs_enough_names_and_uses_top_k():
    days = pd.bdate_range("2020-01-01", periods=2)
    first = list(range(1, 11))
    holdings = pd.DataFrame([first, first[:9] + [np.nan]], index=days)
    share = X.top_share(holdings, k=5, min_names=10)
    assert share.iloc[0] == pytest.approx((10 + 9 + 8 + 7 + 6) / 55)
    assert np.isnan(share.iloc[1])


def test_range_position_maps_window_extremes_to_plus_and_minus_one():
    days = pd.bdate_range("2020-01-01", periods=6)
    sticky = pd.Series([0.0, 1.0, np.nan, 2.0, -1.0, 0.5], index=days)
    position = X.range_position(sticky, window=3)
    assert position.iloc[:3].isna().all()
    assert position.iloc[3] == pytest.approx(1.0)
    assert position.iloc[4] == pytest.approx(-1.0)
    assert position.iloc[5] == pytest.approx(2 * (0.5 + 1) / 3 - 1)


def test_position_signs_treat_the_absent_side_as_zero_but_missing_change_as_missing():
    day = pd.DatetimeIndex(["2020-01-02"])
    tables = {
        "L": pd.DataFrame([[10.0, np.nan, 5.0]], index=day, columns=list("abc")),
        "S": pd.DataFrame([[4.0, 3.0, np.nan]], index=day, columns=list("abc")),
        "dL": pd.DataFrame([[1.0, np.nan, np.nan]], index=day, columns=list("abc")),
        "dS": pd.DataFrame([[3.0, -2.0, np.nan]], index=day, columns=list("abc")),
        "ranked": pd.Series([True], index=day),
    }
    signs = X.position_signs(tables).iloc[0]
    assert signs["a"] == -1 and signs["b"] == 1 and np.isnan(signs["c"])


def test_member_skill_only_uses_labels_realized_before_the_signal_day():
    days = pd.bdate_range("2020-01-01", periods=8)
    sign = pd.DataFrame({"m": [1.0] * 8}, index=days)
    label = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.5]}, index=days)
    eligible = pd.DataFrame({"A": [True] * 8}, index=days)
    skill = X.member_skill({"A": sign}, label, eligible, window=3, lag=2, min_obs=2)["m"]
    assert skill.iloc[:3].isna().all()
    assert skill.iloc[3] == pytest.approx(-0.005)
    assert skill.iloc[4] == pytest.approx(0.02 / 3)
    assert skill.iloc[7] == pytest.approx(0.0)


def test_term_spread_compares_sticky_seats_off_and_on_the_dominant_contract():
    day = pd.Timestamp("2020-01-02")
    ranks = pd.DataFrame({
        "trading_date": [day] * 5,
        "member_name": ["a", "a", "b", "b", "c"],
        "contract": ["X01", "X05", "X01", "X05", "X05"],
        "side": ["long", "long", "short", "long", "long"],
        "volume": [10.0, 30.0, 10.0, 10.0, 100.0],
        "rank": [1] * 5,
    })
    sticky = pd.DataFrame({"a": [True], "b": [True], "c": [False]}, index=[day])
    dominant = pd.Series(["X01"], index=[day])
    spread = X.term_spread(ranks, sticky, dominant)
    assert spread.loc[day] == pytest.approx(1.0 - 0.0)


def test_residual_scores_remove_the_rank_component_of_the_control():
    day = pd.DatetimeIndex(["2020-01-02"])
    cols = list("abcdefgh")
    control = pd.DataFrame([np.arange(8.0)], index=day, columns=cols)
    residual = X.residual_scores(control * 2, control)
    assert np.allclose(residual.to_numpy(), 0.0)

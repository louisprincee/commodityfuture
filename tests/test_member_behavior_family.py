import numpy as np
import pandas as pd
import pytest

from research import member_behavior_family as F


def _tables(long, short, d_long, d_short, ranked=None):
    days = pd.bdate_range("2020-01-01", periods=len(long))
    cols = list(long[0].keys()) if isinstance(long[0], dict) else None
    frame = lambda rows: pd.DataFrame(rows, index=days, columns=cols)
    return {
        "L": frame(long), "S": frame(short), "dL": frame(d_long), "dS": frame(d_short),
        "ranked": pd.Series(ranked if ranked is not None else [True] * len(long), index=days),
    }


def test_sticky_flow_counts_only_sticky_seats_and_absent_side_as_zero():
    tables = _tables(
        long=[{"a": 10.0, "b": 5.0, "c": np.nan}],
        short=[{"a": np.nan, "b": 2.0, "c": 4.0}],
        d_long=[{"a": 3.0, "b": 1.0, "c": np.nan}],
        d_short=[{"a": np.nan, "b": 2.0, "c": -1.0}],
    )
    sticky = pd.DataFrame([{"a": True, "b": False, "c": True}], index=tables["L"].index)
    assert F.sticky_flow(tables, sticky).iloc[0] == pytest.approx(3.0 + 1.0)


def test_contra_flow_keeps_only_flow_against_the_day_return():
    days = pd.bdate_range("2020-01-01", periods=4)
    flow = pd.Series([2.0, 2.0, -1.0, 1.0], index=days)
    gross = pd.Series([10.0] * 4, index=days)
    ret = pd.Series([-0.01, 0.01, 0.02, 0.0], index=days)
    contra = F.contra_flow(flow, gross, ret, window=4, min_days=4)
    assert contra.iloc[:3].isna().all()
    assert contra.iloc[3] == pytest.approx(0.2 + 0.0 - 0.1 + 0.0)


def test_board_stability_is_jaccard_overlap_with_the_lagged_board():
    tables = _tables(
        long=[{"a": 1.0, "b": 1.0, "c": np.nan}, {"a": 1.0, "b": np.nan, "c": 1.0}],
        short=[{"a": np.nan, "b": np.nan, "c": np.nan}] * 2,
        d_long=[{"a": 0.0, "b": 0.0, "c": 0.0}] * 2,
        d_short=[{"a": 0.0, "b": 0.0, "c": 0.0}] * 2,
    )
    stability = F.board_stability(tables, lag=1)
    assert np.isnan(stability.iloc[0])
    assert stability.iloc[1] == pytest.approx(1 / 3)


def test_split_halves_drops_the_middle_name_and_needs_enough_names():
    day = pd.DatetimeIndex(["2020-01-02"])
    condition = pd.DataFrame([np.arange(5.0)], index=day, columns=list("abcde"))
    valid = condition.notna()
    high, low = F.split_halves(condition, valid, min_half=2)
    assert high.iloc[0].tolist() == [False, False, False, True, True]
    assert low.iloc[0].tolist() == [True, True, False, False, False]
    high, low = F.split_halves(condition, valid, min_half=3)
    assert not high.iloc[0].any() and not low.iloc[0].any()


def test_declared_family_has_three_primary_nine_full_and_two_increment_tests():
    days = pd.bdate_range("2016-01-04", periods=260)
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(24)]
    frame = lambda: pd.DataFrame(rng.normal(size=(len(days), len(cols))), index=days, columns=cols)
    factors = {name: frame() for name in ("STICKY_NET_SHARE", *F.FACTORS, F.CONDITION)}
    labels = {h: frame() for h in F.HORIZONS}
    result = F.evaluate(factors, labels)
    assert result["q_primary"].notna().sum() == 3
    assert result["q_full"].notna().sum() == 9
    assert result["q_increment"].notna().sum() == 2
    stability_ic = result[(result["test"] == "ic") & (result["factor"] == F.CONDITION)]
    assert stability_ic[["q_primary", "q_full", "q_increment"]].isna().all().all()

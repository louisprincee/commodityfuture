"""合成数据测试：样本守卫、特征无前视、标签清洗、权重与成交时点。不读取真实数据。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
from research import features as F
from research import model as M
from research import portfolio as P


def synthetic_panel(n_days: int = 900, symbols=("AA", "BB", "CC", "DD"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    rows = []
    for k, s in enumerate(symbols):
        r = rng.normal(0.0002 * k, 0.012, n_days)
        rows.append(pd.DataFrame({
            "date": dates, "symbol": s, "sector": "x",
            "ret1": r, "ret2": r + rng.normal(0, 0.002, n_days),
            "roll_yield_ann": rng.normal(0, 0.05, n_days),
            "oi_total": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))),
            "turnover_total": 1e9, "c1_volume": 1000.0,
            "roll": False, "c1_contract": f"{s}0001",
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel.loc[panel.groupby("symbol").head(1).index, "ret1"] = np.nan
    return panel


def test_research_guard_rejects_oos_paths():
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.EXTERNAL_OOS_DIR / "contracts" / "CU2201.pkl")
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.OOS_SHARD_DIR / "CU.parquet")
    C.assert_research_only(C.EXTERNAL_DIR / "contracts" / "CU2101.pkl")


def test_holdout_date_guard():
    C.assert_no_holdout_dates(pd.to_datetime(["2021-12-31"]))
    with pytest.raises(C.HoldoutViolation):
        C.assert_no_holdout_dates(pd.to_datetime(["2022-01-04"]))


def test_feature_grid_matches_paper():
    assert len(F.FEATURES) == 126
    assert F.HORIZONS == [5, 10, 22, 44, 66, 88, 100, 200]


def test_features_do_not_use_future_data():
    panel = synthetic_panel()
    ds, raw = F.build_dataset(panel)
    cut = panel["date"].iloc[700]
    changed = panel.copy()
    future = changed["date"] > cut
    changed.loc[future, ["ret1", "ret2", "roll_yield_ann", "oi_total"]] *= 3.0
    ds2, raw2 = F.build_dataset(changed)
    past = ds["date"] <= cut
    pd.testing.assert_frame_equal(ds.loc[past, F.FEATURES], ds2.loc[past, F.FEATURES])


def test_forward_return_starts_next_day():
    panel = synthetic_panel(n_days=300, symbols=("AA",))
    fwd = F.forward_returns(panel)
    r = panel["ret1"].to_numpy()
    t = 100
    expected = np.prod(1 + r[t + 1:t + 6]) - 1
    assert fwd["fwd5"].iloc[t] == pytest.approx(expected)
    assert fwd["fwd5"].iloc[-5:].isna().all()


def test_split_rows_purges_label_overlap():
    panel = synthetic_panel(n_days=1200)
    ds, _ = F.build_dataset(panel)
    cal = M.calendar(ds)
    for h in (5, 200):
        rows = M.split_rows(ds, cal, h, 2014)
        pos = ds["date"].map(cal)
        end = cal[cal.index[cal.index.year < 2014][-1]]
        val_start = pos[rows["val"]].min()
        assert (pos[rows["train"]] + h < val_start).all()
        assert (pos[rows["full"]] + h <= end).all()
        assert rows["val"].sum() > 0
        assert (ds.loc[rows["test"], "date"].dt.year == 2014).all()
        assert not (rows["full"] & rows["test"]).any()


def test_weights_legs_and_monotone():
    score = pd.DataFrame([[0.1, 0.4, 0.6, 0.9, np.nan]], columns=list("abcde"))
    w = P.score_to_weights(score).iloc[0]
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert w[w < 0].sum() == pytest.approx(-1.0)
    assert w["e"] == 0
    assert w["a"] < w["b"] < 0 < w["c"] < w["d"]


def test_backtest_timing_and_roll_cost():
    dates = pd.bdate_range("2020-01-01", periods=5)
    w = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
    r = pd.DataFrame({"x": [0.0, 0.01, 0.02, 0.03, 0.04]}, index=dates)
    rolls = pd.DataFrame({"x": [False, False, True, False, False]}, index=dates)
    bt0 = P.backtest(w, r, rolls, delay=0, cost_bps=0)
    assert bt0["gross"].iloc[0] == pytest.approx(0.01)  # t 日权重取得 t+1 日收益
    bt1 = P.backtest(w, r, rolls, delay=1, cost_bps=0)
    assert bt1["gross"].iloc[0] == pytest.approx(0.02)
    bt = P.backtest(w, r, rolls, delay=0, cost_bps=5)
    assert bt.loc[dates[2], "turnover"] == pytest.approx(2.0)
    assert bt.loc[dates[1], "turnover"] == pytest.approx(1.0)  # 首次建仓
    assert bt.loc[dates[3], "turnover"] == pytest.approx(0.0)


def test_hold_cohorts_average_and_ramp():
    dates = pd.bdate_range("2020-01-01", periods=4)
    w = pd.DataFrame({"a": [1.0, -1.0, 1.0, 1.0], "b": [-1.0, 1.0, -1.0, -1.0]}, index=dates)
    held = P.hold_cohorts(w, 2)
    assert held.loc[dates[0], "a"] == pytest.approx(0.5)  # 只有一个组合，另一半为现金
    assert held.loc[dates[1], "a"] == pytest.approx(0.0)
    assert held.loc[dates[3], "a"] == pytest.approx(1.0)
    norm = P.normalize_legs(held)
    assert norm.loc[dates[0], "a"] == pytest.approx(1.0)
    assert norm.loc[dates[1]].abs().sum() == 0


def test_download_rejects_financial_futures():
    from download import getMinutes

    with pytest.raises(SystemExit):
        getMinutes.main(["-t", "20240101-20240102", "--symbols", "IF"])

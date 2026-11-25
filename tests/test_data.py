"""数据分区、流动性和历史合约的合成数据测试。"""

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import data as D


def test_partition_guards_seal_paths_and_dates():
    assert C.daily_dir("research") == C.DATA_ROOT / "research" / "daily"
    assert C.minute_dir("oos") == C.DATA_ROOT / "oos" / "minute"
    assert C.RESEARCH_FEE_PATH == C.DATA_ROOT / "research" / "execution" / "fee_history.csv"
    first, last = pd.Timestamp("2021-12-31").date(), pd.Timestamp("2022-01-04").date()
    assert C.split_period(first, last) == [("research", first, C.RESEARCH_END), ("oos", C.OOS_START, last)]
    C.assert_research_only(C.daily_dir("research") / "contracts" / "CU2101.pkl")
    C.assert_no_holdout_dates(pd.to_datetime(["2021-12-31"]))
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.daily_dir("oos") / "contracts" / "CU2201.pkl")
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.minute_dir("oos") / "CU.parquet")
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.DATA_ROOT / "oos" / "future_dataset" / "A.parquet")
    with pytest.raises(C.HoldoutViolation):
        C.assert_research_only(C.ARCHIVE_DIR / "minute" / "A.parquet")
    with pytest.raises(C.HoldoutViolation):
        C.assert_no_holdout_dates(pd.to_datetime(["2022-01-04"]))


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


def test_partial_fee_refresh_keeps_dates_the_vendor_did_not_return(tmp_path):
    from download.getFeeHistory import update_history

    path = tmp_path / "fee_history.csv"
    dates = pd.to_datetime(["2021-12-29", "2021-12-30", "2021-12-31"])
    existing = pd.DataFrame({"symbol": ["A", "A", "B"], "trading_date": dates, "open_commission": [1., 2., 3.]})
    existing.to_csv(path, index=False)
    incoming = existing.iloc[[1]].assign(open_commission=4.)

    update_history(path, "research", incoming)

    expected = existing.copy()
    expected.loc[1, "open_commission"] = 4.
    pd.testing.assert_frame_equal(pd.read_csv(path, parse_dates=["trading_date"]), expected)


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


def test_eligibility_requires_full_window_and_ignores_future_turnover():
    panel = synthetic_panel()
    before = D.eligibility(panel)
    changed = panel.copy()
    cut = panel["date"].iloc[250]
    changed.loc[changed["date"] > cut, "turnover_total"] = 0.0
    after = D.eligibility(changed)
    past = panel["date"] <= cut
    pd.testing.assert_series_equal(before[past], after[past])
    assert before[past].iloc[-1]

    dates = pd.bdate_range("2020-01-01", periods=242)
    panel = pd.DataFrame({
        "date": dates,
        "symbol": ["A"] * len(dates),
        "turnover_total": [500_000_000.0] * len(dates),
        "c1_volume": [1.0] * len(dates),
        "ret1": [0.01] * len(dates),
    })

    eligible = D.eligibility(panel)

    assert eligible.iloc[:-1].eq(False).all()
    assert eligible.iloc[-1]


def test_to_single_sided_halves_only_before_2020():
    from src import data as D

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


def test_download_rejects_financial_futures():
    from download import getMinutes

    with pytest.raises(SystemExit):
        getMinutes.main(["-t", "20240101-20240102", "--symbols", "IF"])


def test_legacy_contracts_stop_at_successor_listing_date():
    frame = pd.DataFrame({
        "symbol": ["ME", "ME", "MA", "RB"],
        "date": pd.to_datetime(["2014-06-16", "2014-06-17", "2014-06-17", "2014-06-16"]),
        "contract": ["ME1501", "ME1501", "MA1506", "RB1410"],
    })
    out = D.apply_legacy_contracts(frame)
    assert out["symbol"].tolist() == ["MA", "RB", "MA"]
    assert out["contract"].tolist() == ["MA1506", "RB1410", "ME1501"]
    execution = D.apply_legacy_contracts(frame, for_execution=True)
    old = execution.loc[execution.contract.eq("ME1501")]
    assert len(old) == 2
    assert old.symbol.eq("MA").all()
    assert old.date.max() == pd.Timestamp("2014-06-17")


def test_legacy_dominant_does_not_overwrite_successor_days():
    current = pd.Series(["MA1506", "MA1506"], index=pd.to_datetime(["2014-06-17", "2014-06-18"]), name="rank1")
    legacy = pd.Series(["ME1505", "ME1505", "ME1505"], index=pd.to_datetime(["2014-06-16", "2014-06-17", "2014-06-18"]))
    out = D.stitch_legacy_dominant(current, legacy, pd.Timestamp("2014-06-17"))
    assert out.tolist() == ["ME1505", "MA1506", "MA1506"]
    assert str(out.index.min().date()) == "2014-06-16"


def test_build_daily_panel_oos_carries_research_prior_close_across_boundary(monkeypatch):
    res_dates = pd.to_datetime(["2021-12-30", "2021-12-31"])
    oos_dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
    res_df = pd.DataFrame({
        "contract": ["RB2205", "RB2205", "RB2210", "RB2210"],
        "date": list(res_dates) * 2,
        "symbol": ["RB"] * 4,
        "close": [4000.0, 4050.0, 3900.0, 3920.0],
        "total_turnover": [1e9] * 4,
        "open_interest": [1e5] * 4,
        "volume": [1e5] * 4,
    })
    oos_df = pd.DataFrame({
        "contract": ["RB2205", "RB2205", "RB2210", "RB2210"],
        "date": list(oos_dates) * 2,
        "symbol": ["RB"] * 4,
        "close": [4090.5, 4131.405, 3959.2, 3998.792],
        "total_turnover": [1e9] * 4,
        "open_interest": [1e5] * 4,
        "volume": [1e5] * 4,
    })
    monkeypatch.setattr(D, "load_contracts", lambda part="research", **kwargs: res_df if part == "research" else oos_df)
    monkeypatch.setattr(D, "symbols_with_data", lambda part="research": ["RB"])
    monkeypatch.setattr(D, "load_dominant", lambda sym, rank=1, part="research": pd.Series(
        ["RB2205" if rank == 1 else "RB2210"] * 2, index=oos_dates, name=f"rank{rank}",
    ))
    monkeypatch.setattr(D, "load_roll_yield", lambda sym, part="research": pd.DataFrame())

    out = D.build_daily_panel("oos")
    assert out["date"].tolist() == list(oos_dates)
    assert out.loc[0, "c1_prev_close"] == pytest.approx(4050.0)
    assert out.loc[0, "c2_prev_close"] == pytest.approx(3920.0)
    assert out.loc[0, "ret1"] == pytest.approx(4090.5 / 4050.0 - 1.0)
    assert out.loc[0, "ret2"] == pytest.approx(3959.2 / 3920.0 - 1.0)

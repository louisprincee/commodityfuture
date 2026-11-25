import numpy as np
import pandas as pd
import pytest


def test_sticky_flow_uses_common_samples_equal_slots_and_two_stage_selection(monkeypatch):
    from strategies import sticky_flow as S
    from research.ic.statistics import rank_residual

    dates = pd.bdate_range("2011-01-04", periods=3)
    names = [f"A{i:02d}" for i in range(20)]
    sticky = pd.DataFrame(np.tile(np.arange(20.), (3, 1)), index=dates, columns=names)
    flow = sticky.copy()
    flow.iloc[0, -1] = np.nan
    flow.iloc[1, 11:] = np.nan
    flow.iloc[2, 7:] = np.nan
    residual = sticky.copy()
    residual.iloc[0, [0, 2, 16, 18]] = [-90., -100., 100., 90.]
    dominant = pd.DataFrame("AA1105", index=dates, columns=names)
    monkeypatch.setattr(S, "load_sticky_flow_inputs", lambda *args: (sticky, flow, residual, dominant))
    books = S.build_books(pd.DataFrame(), dates[0], 1)
    matched = [books[name] for name in (S.STICKY_BOOK, S.FLOW_BOOK, S.COMBINED_BOOK)]
    for target in matched:
        assert target.selection.candidates.tolist() == [19, 11, 7]
        assert target.selection.long_slots.tolist() == [2, 2, 0]
        assert target.selection.short_slots.tolist() == [2, 2, 0]
        pd.testing.assert_series_equal(target.selection.sample_hash, matched[0].selection.sample_hash)
    first = books[S.COMBINED_BOOK].weights.iloc[0]
    assert first[first.gt(0)].index.tolist() == ["A16", "A18"]
    assert first[first.lt(0)].index.tolist() == ["A00", "A02"]
    assert books[S.REFERENCE_BOOK].selection.candidates.eq(20).all()
    assert books[S.COMBINED_BOOK].weights.iloc[2].abs().sum() > 0  # prior cohorts survive a missing fresh sample
    x = pd.DataFrame([np.cos(np.arange(20.))], columns=names)
    z = sticky.iloc[:1].reset_index(drop=True)
    r = rank_residual(x, z)
    centered = z.rank(axis=1).sub(z.rank(axis=1).mean(axis=1), axis=0)
    assert r.sum(axis=1).abs().lt(1e-10).all()
    assert (r * centered).sum(axis=1).abs().lt(1e-10).all()
    assert rank_residual(x, z * 0).isna().all().all()


def test_cohorts_average_opposing_signals_and_expire_at_both_horizons():
    from strategies.common import hold_cohorts

    for horizon in (10, 20):
        days = pd.bdate_range("2020-01-01", periods=horizon + 1)
        weights = pd.DataFrame({
            "A": [0.5, -0.5] + [0.0] * (horizon - 1),
            "B": [1.0] + [0.0] * horizon,
        }, index=days)
        held = hold_cohorts(weights, horizon)
        assert held["A"].iloc[0] == pytest.approx(0.5 / horizon)
        assert held["A"].iloc[1:horizon].eq(0).all()
        assert held["A"].iloc[horizon] == pytest.approx(-0.5 / horizon)
        assert held["B"].iloc[:horizon].eq(1 / horizon).all()
        assert held["B"].iloc[horizon] == 0


def test_early_sticky_targets_keep_full_2010_feature_warmup_but_no_prior_cohorts(monkeypatch):
    from research.factors import member as M
    from research.member import complementarity as legacy
    from strategies import sticky_only
    from strategies.common import HORIZON

    dates = pd.bdate_range("2010-01-04", "2016-01-07")
    names = M.C.COMMODITY_SYMBOLS[:10]
    start = pd.Timestamp("2011-01-04")
    panel = pd.MultiIndex.from_product([dates, names], names=["date", "symbol"]).to_frame(index=False)
    panel = panel.assign(volume_total=100, oi_total=100, turnover_total=1e9,
                         c1_volume=1, ret1=0., c1_contract=panel.symbol + "1105")
    monkeypatch.setattr(M.D, "symbols_with_member_rank", lambda partition: set(names))

    def tables(name, requested_dates, *, top20, partition):
        assert requested_dates.min() == pd.Timestamp("2010-01-04")
        assert len(requested_dates) > 242
        assert top20 and partition == "research"
        value = names.index(name)
        return {"L": pd.DataFrame({"seat": value + 1.}, index=requested_dates),
                "S": pd.DataFrame({"seat": 11. - value}, index=requested_dates),
                "V": pd.DataFrame({"seat": 0.}, index=requested_dates),
                "ranked": pd.Series(True, index=requested_dates)}

    monkeypatch.setattr(M, "member_tables", tables)
    monkeypatch.setattr(legacy, "member_tables", tables)
    targets = sticky_only.build_targets(panel, pd.DataFrame(), start, "research", 1)
    assert targets.weights.index.min() == start
    assert targets.weights.loc[start].abs().sum() == pytest.approx(1 / HORIZON)
    historical, _, _ = legacy.load_seat_inputs(panel, 1, partition="research")
    assert historical.index.min() == pd.Timestamp("2016-01-04")


def test_sticky_factor_loader_matches_historical_inputs_and_rejects_sealed_dates(monkeypatch):
    from research.factors import member as M
    from research.member import complementarity as legacy

    dates = pd.bdate_range("2010-12-30", periods=6)
    panel = pd.DataFrame({"date": dates, "symbol": "AL", "volume_total": [100., 100., 0., 100., 100., 100.],
                          "oi_total": [100., 100., 100., np.nan, 100., 100.],
                          "c1_contract": "AL1105"})
    monkeypatch.setattr(M.D, "eligibility", lambda p: pd.Series([False, True, True, True, True, True], index=p.index))
    monkeypatch.setattr(M.D, "symbols_with_member_rank", lambda partition: {"AL"})

    def tables(symbol, requested_dates, *, top20, partition):
        assert symbol == "AL" and requested_dates.equals(dates)
        assert top20 and partition == "research"
        return {"L": pd.DataFrame({"slow": 9., "fast": 2., "one_side": 1.}, index=dates),
                "S": pd.DataFrame({"slow": 3., "fast": 2., "one_side": np.nan}, index=dates),
                "V": pd.DataFrame({"slow": 1., "fast": 100., "one_side": np.nan}, index=dates),
                "ranked": pd.Series([True, True, True, True, False, True], index=dates)}

    monkeypatch.setattr(M, "member_tables", tables)
    monkeypatch.setattr(legacy, "member_tables", tables)
    start = dates[1]
    sticky, dominant = M.load_sticky_inputs(panel, start, "research", 1)
    old_sticky, _, old_dominant = legacy.load_seat_inputs(panel, 1, partition="research", start=start)
    pd.testing.assert_frame_equal(sticky, old_sticky, check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(dominant, old_dominant)
    assert sticky.index.min() == start
    assert sticky.iloc[:, 0].tolist()[0] == pytest.approx(7 / 13)
    assert sticky.iloc[1:4, 0].isna().all()
    assert sticky.iloc[4, 0] == pytest.approx(7 / 13)
    with pytest.raises(M.C.HoldoutViolation):
        M.load_sticky_inputs(panel.assign(date=pd.date_range("2022-01-04", periods=6)), start, "research", 1)

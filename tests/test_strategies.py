import numpy as np
import pandas as pd
import pytest


def test_identity_requires_only_its_own_finite_score(monkeypatch):
    from strategies.member import persist_identity as K
    from strategies.common import hold_cohorts, tail_weights

    dates = pd.bdate_range("2011-01-04", periods=2)
    names = [f"A{i:02d}" for i in range(20)]
    ramp = pd.DataFrame(np.tile(np.arange(20.), (2, 1)), index=dates, columns=names)
    ramp.iloc[0, 0] = np.inf
    signals = {"STICKY_PERSIST_IDENTITY": ramp}
    dominant = pd.DataFrame("AA1105", index=dates, columns=names)
    def load_inputs(*args, **kwargs):
        assert kwargs["partition"] == "oos"
        assert kwargs["names"] == ("STICKY_PERSIST_IDENTITY",)
        return signals, dominant
    monkeypatch.setattr(K, "load_extension_inputs", load_inputs)
    target = K.build_targets(pd.DataFrame(), pd.DataFrame(), dates[0], "oos", 1)
    common = np.isfinite(ramp)
    expected = hold_cohorts(tail_weights(ramp.where(common)))
    pd.testing.assert_frame_equal(target.weights, expected)
    assert target.selection.candidates.tolist() == [19, 20]
    assert target.weights.iloc[0].loc[lambda r: r.gt(0)].index.tolist() == ["A18", "A19"]


def test_cohorts_average_opposing_signals_and_expire_at_both_horizons(monkeypatch):
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

    from strategies.member_price import sticky_basis_skew as S
    dates = pd.bdate_range("2020-01-01", periods=22)
    score = pd.DataFrame(np.tile(np.arange(20.), (22, 1)), index=dates)
    dominant = pd.DataFrame("AA2005", index=dates, columns=score.columns)
    monkeypatch.setattr(S, "load_sticky_inputs", lambda *a, **kw: (score, dominant))
    monkeypatch.setattr(S, "load_price_factors", lambda *a, **kw: (score, score))
    assert S.HOLDING_DAYS == 10
    default = S.build_targets(pd.DataFrame(), pd.DataFrame(), dates[0], "research", 1)
    monkeypatch.setattr(S, "HOLDING_DAYS", 20)
    twenty = S.build_targets(pd.DataFrame(), pd.DataFrame(), dates[0], "research", 1)
    assert default.weights.abs().sum(axis=1).iloc[9] == pytest.approx(1.)
    assert twenty.weights.abs().sum(axis=1).iloc[9] == pytest.approx(.5)
    assert twenty.weights.abs().sum(axis=1).iloc[19] == pytest.approx(1.)
    monkeypatch.setattr(S, "HOLDING_DAYS", 0)
    with pytest.raises(ValueError, match="正整数"):
        S.build_targets(pd.DataFrame(), pd.DataFrame(), dates[0], "research", 1)


def test_early_sticky_targets_keep_full_2010_feature_warmup_but_no_prior_cohorts(monkeypatch):
    from research.factors import member as M
    from research.member import complementarity as legacy
    from strategies.member import sticky_net
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
    targets = sticky_net.build_targets(panel, pd.DataFrame(), start, "research", 1)
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

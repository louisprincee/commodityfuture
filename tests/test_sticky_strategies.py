import numpy as np
import pandas as pd
import pytest


def test_active_strategy_families_and_equal_weight_composite():
    from backtest import engine as E
    from strategies import sticky_chain, sticky_core, sticky_only
    from strategies.common import composite_score

    assert E.FAMILIES == {
        "member": (sticky_only, sticky_chain),
        "member_price": (sticky_core,),
    }
    assert {
        module.STRATEGY_ID for family in E.FAMILIES.values() for module in family
    } == {"sticky_only", "sticky_chain", "sticky_core"}

    candidates = pd.DataFrame([[True, True, True, False]], columns=list("ABCD"))
    factors = {
        "basis": pd.DataFrame([[1.0, 2.0, 3.0, 100.0]], columns=list("ABCD")),
        "skew": pd.DataFrame([[3.0, 2.0, 1.0, 100.0]], columns=list("ABCD")),
        "sticky": pd.DataFrame([[1.0, 3.0, 2.0, 100.0]], columns=list("ABCD")),
    }
    score = composite_score(factors, candidates, ("basis", "skew", "sticky")).iloc[0]

    assert score["A"] == pytest.approx(5 / 9)
    assert score["B"] == pytest.approx(7 / 9)
    assert score["C"] == pytest.approx(2 / 3)
    assert np.isnan(score["D"])


def test_sticky_chain_targets_only_need_net_share_and_chain_score(monkeypatch):
    from strategies import sticky_chain
    from strategies.common import HORIZON

    names = list(range(20))
    dates = pd.date_range("2016-01-04", periods=2)
    sticky = pd.DataFrame([names, names], index=dates, dtype=float)
    chain = -sticky.pow(2)
    chain.iloc[1, 0] = np.nan
    dominant = pd.DataFrame("contract", index=dates, columns=names)

    def load_inputs(panel, jobs, *, partition):
        assert panel.empty and jobs == 1 and partition == "research"
        return sticky, chain, dominant

    monkeypatch.setattr(sticky_chain, "load_chain_inputs", load_inputs)

    targets = sticky_chain.build_targets(pd.DataFrame(), pd.DataFrame(), dates[0], "research", 1)

    assert targets.weights.iloc[0].abs().sum() * HORIZON == pytest.approx(1)
    assert targets.weights.iloc[0].gt(0).sum() == 2
    assert targets.weights.iloc[0].lt(0).sum() == 2
    assert targets.weights.iloc[1].equals(targets.weights.iloc[0])
    assert targets.dominant is dominant


@pytest.mark.parametrize("partition,start", [("research", "2016-01-04"), ("oos", "2022-01-01")])
def test_sticky_only_targets_need_no_flow_file(monkeypatch, tmp_path, partition, start):
    from research import member_complementarity as M
    from strategies import sticky_only
    from strategies.common import HORIZON

    start = pd.Timestamp(start)
    dates = pd.bdate_range(start, periods=2)
    names = sorted(M.C.COMMODITY_SYMBOLS[:10])
    panel = pd.MultiIndex.from_product([dates, names], names=["date", "symbol"]).to_frame(index=False)
    panel = panel.assign(volume_total=100, oi_total=100, c1_contract=panel.symbol + "0001")
    rank_dir = tmp_path / "member_rank"
    rank_dir.mkdir()
    for name in names:
        (rank_dir / f"{name}_long.pkl").touch()

    monkeypatch.setattr(M.C, "daily_dir",
                        lambda requested: tmp_path if requested == partition else pytest.fail(requested))
    monkeypatch.setattr(M.C, "study_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(M.D, "eligibility", lambda p: p.symbol.ne(names[4]))
    def seat_state(name, requested_dates, market, requested_partition):
        assert requested_dates.equals(dates)
        assert market.index.equals(dates)
        assert requested_partition == partition
        return pd.DataFrame({"sticky": names.index(name), "size": 0.5}, index=dates)

    monkeypatch.setattr(M, "_seat_state", seat_state)

    targets = sticky_only.build_targets(panel, pd.DataFrame(), start, partition, 1)
    fresh = targets.weights.iloc[0] * HORIZON
    assert fresh[fresh.gt(0)].index.tolist() == names[-1:]
    assert fresh[fresh.lt(0)].index.tolist() == names[:1]
    assert fresh.abs().sum() == pytest.approx(1.0)
    assert targets.dominant.loc[dates[0], names[-1]] == names[-1] + "0001"


def test_ten_day_cohorts_average_and_expire():
    from strategies.common import HORIZON, hold_cohorts

    days = pd.bdate_range("2020-01-01", periods=HORIZON + 1)
    weights = pd.DataFrame({
        "A": [0.5, -0.5] + [0.0] * (HORIZON - 1),
        "B": [1.0] + [0.0] * HORIZON,
    }, index=days)
    held = hold_cohorts(weights)
    assert held["A"].iloc[0] == pytest.approx(0.05)
    assert held["A"].iloc[1:HORIZON].eq(0).all()
    assert held["A"].iloc[HORIZON] == pytest.approx(-0.05)
    assert held["B"].iloc[:HORIZON].eq(0.1).all()
    assert held["B"].iloc[HORIZON] == 0

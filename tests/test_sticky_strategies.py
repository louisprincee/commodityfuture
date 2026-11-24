from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _save_equity(base, partition, module, columns):
    dates = {"research": ["2016-01-04", "2016-01-05"], "oos": ["2022-01-04", "2022-01-05"]}
    from backtest.engine import FAMILY_OF

    path = base / FAMILY_OF[module.STRATEGY_ID] / module.STRATEGY_ID / partition / module.EQUITY_CURVE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns, index=pd.to_datetime(dates[partition])).to_csv(path, index_label="date")


def test_blend_nets_opposing_sleeves_without_restoring_target_gross():
    from strategies.sticky_price_blend import blend_weights

    names = list(range(20))
    sticky = pd.DataFrame([names], columns=names)
    weights = blend_weights(sticky, sticky, sticky, -sticky, -sticky, sticky.notna())

    assert weights.iloc[0].sum() == pytest.approx(0)
    assert weights.iloc[0].abs().sum() == pytest.approx(0.5)
    assert weights.loc[0, 18] == pytest.approx(.0625)
    assert weights.loc[0, 0] == pytest.approx(-.0625)


@pytest.mark.parametrize("invalid_factor", ["chain", "basis", "neg_skew"])
def test_price_blend_requires_only_its_signals_and_rechecks_pool_size(invalid_factor):
    from strategies.sticky_price_blend import daily_target_weights

    names = list(range(20))
    sticky = pd.DataFrame([names, names], dtype=float)
    chain = pd.DataFrame([[10, 9, 0, 1, *range(4, 9), 2, 3, *range(11, 16), 18, 19, 16, 17]] * 2, dtype=float)
    factors = dict(basis=sticky.copy(), neg_skew=sticky.copy(), chain=chain)
    factors[invalid_factor].loc[1, 0] = np.nan
    weights = daily_target_weights(sticky, **factors)

    assert weights.loc[0, 16] == pytest.approx(.1875)
    assert weights.loc[0, 17] == pytest.approx(.1875)
    assert weights.loc[0, 2] == pytest.approx(-.1875)
    assert weights.loc[0, 3] == pytest.approx(-.1875)
    assert weights.loc[1].eq(0).all()


def test_sticky_chain_targets_only_need_net_share_and_chain_score(monkeypatch):
    from strategies import sticky_chain

    names = list(range(20))
    dates = pd.date_range("2016-01-04", periods=2)
    sticky = pd.DataFrame([names, names], index=dates, dtype=float)
    chain = -sticky.pow(2)
    chain.iloc[1, 0] = np.nan
    dominant = pd.DataFrame("contract", index=dates, columns=names)
    monkeypatch.setattr(sticky_chain, "load_chain_inputs", lambda *args, **kwargs: (sticky, chain, dominant))

    targets = sticky_chain.build_targets(pd.DataFrame(), pd.DataFrame(), dates[0], "research", 1, horizon=1)

    assert targets.weights.iloc[0].abs().sum() == pytest.approx(1)
    assert targets.weights.iloc[1].eq(0).all()
    assert targets.dominant is dominant


@pytest.mark.parametrize("partition,loader,fee,root", [
    ("research", "load_panel", "RESEARCH_FEE_PATH", "RESEARCH_EXECUTION_PARAMETERS"),
    ("oos", "load_evaluation_panel", "OOS_FEE_PATH", "OOS_EXECUTION_PARAMETERS"),
])
def test_single_engine_routes_blend_to_the_requested_partition(monkeypatch, tmp_path, partition, loader, fee, root):
    from backtest import engine as E
    from strategies import Targets, sticky_core as core, sticky_price_blend as blend

    start = E.BACKTEST_START if partition == "research" else pd.Timestamp(E.C.OOS_START)
    dates = pd.bdate_range(start if partition == "research" else "2022-01-04", periods=2)
    panel = pd.DataFrame({"date": dates})
    bars = object()
    fee_path = tmp_path / "fees.csv"
    pd.DataFrame({"trading_date": dates}).to_csv(fee_path, index=False)
    calls = []
    monkeypatch.setattr(E.D, loader, lambda: panel)
    monkeypatch.setattr(E.D, "load_contracts_span", lambda part: bars)
    monkeypatch.setattr(E.C, fee, fee_path)
    monkeypatch.setattr(E.C, root, tmp_path / partition)
    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    _save_equity(tmp_path, partition, core, {core.MAIN_BOOK: [1e7, 1.2e7], E.BENCHMARK_BOOK: [1e7, 1.1e7]})

    def build_targets(data, daily_bars, lower, part, jobs, *, horizon):
        assert horizon == blend.HORIZON
        assert data is panel
        assert daily_bars is bars
        assert (lower, part, jobs) == (start, partition, 2)
        return Targets(pd.DataFrame({"A": [0., 0.]}, index=dates),
                       pd.DataFrame({"A": ["A2201", "A2201"]}, index=dates))

    def load_market(daily_bars, fees, ids, lower, *, root, flat_margin, partition):
        calls.append((daily_bars, tuple(fees["trading_date"]), ids, lower, root, flat_margin, partition))
        return {}

    monkeypatch.setattr(blend, "build_targets", build_targets)
    monkeypatch.setattr(E, "load_market", load_market)
    assert E.main(["sticky_price_blend", f"--{partition}", "--jobs", "2"]) == 0
    assert calls == [(bars, tuple(dates), {"A2201"}, start, tmp_path / partition / "contracts", 0.20, partition)]


def test_combined_replaces_old_blend_and_plots_both_strategies_with_one_benchmark(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    from backtest import engine as E
    from strategies import sticky_core as core, sticky_price_blend as blend

    for partition in ("research", "oos"):
        _save_equity(tmp_path, partition, core, {core.MAIN_BOOK: [1e7, 1.1e7], E.BENCHMARK_BOOK: [1e7, 1.05e7]})
    _save_equity(tmp_path, "research", blend, {blend.MAIN_BOOK: [1e7, 1.2e7]})
    _save_equity(tmp_path, "oos", blend, {"STICKY_PRICE_BLEND_SPLIT": [1e7, 1.4e7]})
    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(E.D, "load_evaluation_panel", lambda: pd.DataFrame())
    calls = []
    figures = {}
    savefig = Figure.savefig

    def capture(figure, path, *args, **kwargs):
        figures[path.relative_to(tmp_path)] = (
            [line.get_label() for line in figure.axes[0].lines],
            [text.get_text() for text in figure.texts],
        )
        savefig(figure, path, *args, **kwargs)

    def run(module, partition, out, jobs, horizon=None):
        calls.append((module, partition))
        _save_equity(tmp_path, partition, module, {module.MAIN_BOOK: [1e7, .9e7]})

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(E, "_run_partition", run)
    assert E.main(["--family", "member_price", "--start", "2016", "--end", "2025"]) == 0

    out = tmp_path / "member_price" / blend.STRATEGY_ID / "combined"
    curves = pd.read_csv(out / "equity_curves.csv")
    assert calls == [(blend, "oos")]
    assert curves[blend.MAIN_BOOK].tolist() == pytest.approx([1e7, 1.2e7, 1.2e7, 1.08e7])
    assert sorted(p.name for p in (out / "yearly").glob("*.png")) == ["2016.png", "2022.png"]
    comparison = tmp_path / "member_price" / "comparison" / "2016-2022"
    overview = comparison / "plots" / "overview.png"
    assert overview.is_file()
    titles = figures[overview.relative_to(tmp_path)][1]
    assert {"会员持仓因子 · 价格因子 · 商品期货截面策略", core.MAIN_BOOK, blend.MAIN_BOOK, E.BENCHMARK_LABEL} <= set(titles)
    returns = pd.read_csv(comparison / "daily_returns.csv")
    assert returns.columns.tolist() == ["date", core.MAIN_BOOK, blend.MAIN_BOOK, E.BENCHMARK_BOOK]
    assert returns[E.BENCHMARK_BOOK].tolist() == pytest.approx([0, .05, 0, .05])
    for path in (out / "yearly" / "2022.png", out.parent / "oos" / "performance.png"):
        assert figures[path.relative_to(tmp_path)][0] == [blend.MAIN_BOOK, E.BENCHMARK_LABEL]


def test_member_family_runs_oos_and_mirrors_the_price_family_layout(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    from backtest import engine as E
    from strategies import sticky_core as core, sticky_only as only, sticky_chain as chain

    for partition in ("research", "oos"):
        _save_equity(tmp_path, partition, core, {core.MAIN_BOOK: [1e7, 1.1e7], E.BENCHMARK_BOOK: [1e7, 1.05e7]})
    for module in (only, chain):
        _save_equity(tmp_path, "research", module, {module.MAIN_BOOK: [1e7, 1.2e7]})
    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    calls, titles = [], {}
    savefig = Figure.savefig

    def capture(figure, path, *args, **kwargs):
        titles[path.relative_to(tmp_path)] = [text.get_text() for text in figure.texts]
        savefig(figure, path, *args, **kwargs)

    def run(module, partition, out, jobs, horizon=None):
        calls.append((module, partition))
        _save_equity(tmp_path, partition, module, {module.MAIN_BOOK: [1e7, .9e7]})

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr(E, "_run_partition", run)
    assert E.main(["--family", "member", "--start", "2016", "--end", "2025"]) == 0

    assert calls == [(only, "oos"), (chain, "oos")]
    assert "sticky_split" not in E.MODULES
    family = tmp_path / "member"
    for module in (only, chain):
        for partition in ("research", "oos", "combined"):
            assert (family / module.STRATEGY_ID / partition / "performance.png").is_file()
    for folder in ("research", "oos", "2016-2022"):
        assert (family / "comparison" / folder / "daily_returns.csv").is_file()
    overview = Path("member") / "comparison" / "2016-2022" / "plots" / "overview.png"
    assert {"会员持仓因子 · 商品期货截面策略", only.MAIN_BOOK, chain.MAIN_BOOK, E.BENCHMARK_LABEL} <= set(titles[overview])
    assert not (tmp_path / "member_price" / "comparison").exists()


def test_combined_single_strategy_leaves_family_outputs_alone(monkeypatch, tmp_path):
    from backtest import engine as E
    from strategies import sticky_core as core, sticky_only as only

    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(E, "_plot_book_figures", lambda *args: None)
    for period in E.C.PARTITIONS:
        _save_equity(tmp_path, period, core, {core.MAIN_BOOK: [1e7, 1.1e7], E.BENCHMARK_BOOK: [1e7, 1.05e7]})
        _save_equity(tmp_path, period, only, {only.MAIN_BOOK: [1e7, 1.2e7]})

    def unexpected_run(*args):
        pytest.fail("A different strategy was scheduled")

    monkeypatch.setattr(E, "_run_partition", unexpected_run)
    assert E.main(["sticky_only", "--partition", "combined"]) == 0
    assert (tmp_path / "member" / "sticky_only" / "combined" / "summary.csv").is_file()
    assert not (tmp_path / "member" / "comparison").exists()


def test_all_books_generate_one_overview_and_summary(monkeypatch, tmp_path):
    from backtest import engine as E

    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(E, "_plot_book_figures", lambda *args: None)
    monkeypatch.setattr(E, "_plot_performance", lambda *args, **kwargs: None)
    monkeypatch.setattr(E, "_plot_yearly_performance", lambda *args: None)
    books = []
    for modules in E.FAMILIES.values():
        for module in modules:
            books.append(module.MAIN_BOOK)
            curves = {module.MAIN_BOOK: [1e7, 1.1e7]}
            if module.STRATEGY_ID == E.BENCHMARK_SOURCE:
                curves[E.BENCHMARK_BOOK] = [1e7, 1.05e7]
            for period in E.C.PARTITIONS:
                _save_equity(tmp_path, period, module, curves)

    figures = {}
    monkeypatch.setattr(E, "plot_equity_drawdown",
                        lambda equity, path, **kwargs: figures.update({path: equity.columns.tolist()}))
    assert E.main(["--partition", "combined"]) == 0
    root = tmp_path / "comparison" / "2016-2022"
    assert figures[root / "plots" / "overview.png"] == [*books, E.BENCHMARK_BOOK]
    assert pd.read_csv(root / "daily_returns.csv").columns.tolist() == ["date", *books, E.BENCHMARK_BOOK]
    table = pd.read_csv(tmp_path / "summary.csv")
    assert set(zip(table.book, table.phase)) == {
        (book, period) for book in books for period in ("research", "oos", "combined")
    }
    assert table.query("phase == 'oos'").net_ann.tolist() == pytest.approx([12.1] * len(books))


def test_longer_horizon_is_a_separate_book_beside_the_current_one(monkeypatch, tmp_path):
    from backtest import engine as E
    from strategies import sticky_core as core, sticky_only as only, sticky_chain as chain

    assert E.book_label("sticky_only", 20) == "STICKY_ONLY_20D"
    assert E.book_label("sticky_core", 20) == "STICKY_CORE_20D"
    assert E.book_label("sticky_core", core.HORIZON) == core.MAIN_BOOK
    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    _save_equity(tmp_path, "research", core, {core.MAIN_BOOK: [1e7, 1.1e7], E.BENCHMARK_BOOK: [1e7, 1.05e7]})
    calls = []

    def run(module, partition, out, jobs, horizon):
        calls.append((module, partition, out.name, horizon))
        path = out / partition / module.EQUITY_CURVE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        dates = pd.to_datetime(["2016-01-04", "2016-01-05"])
        pd.DataFrame({E.book_label(module.STRATEGY_ID, horizon): [1e7, 1.3e7]}, index=dates).to_csv(
            path, index_label="date")

    monkeypatch.setattr(E, "_run_partition", run)
    assert E.main(["--family", "member", "--horizon", "20", "--research"]) == 0

    assert calls == [(only, "research", "sticky_only_hold20", 20), (chain, "research", "sticky_chain_hold20", 20)]
    returns = pd.read_csv(tmp_path / "member" / "comparison_hold20" / "research" / "daily_returns.csv")
    assert returns.columns.tolist() == ["date", "STICKY_ONLY_20D", chain.MAIN_BOOK + "_20D", E.BENCHMARK_BOOK]
    assert not (tmp_path / "member" / "comparison").exists()


def test_price_core_is_a_research_only_control_scored_without_sticky():
    from backtest import engine as E
    from strategies import price_core, sticky_core as core

    assert price_core.SCORE_FACTORS == tuple(f for f in core.FACTORS if f != "STICKY_NET_SHARE")
    assert (price_core.HORIZON, price_core.TARGET_MULTIPLIER, price_core.NAME_CAP) == (
        core.HORIZON, core.TARGET_MULTIPLIER, core.NAME_CAP)
    assert price_core.STRATEGY_ID not in {m.STRATEGY_ID for books in E.FAMILIES.values() for m in books}
    with pytest.raises(SystemExit):
        E.main(["price_core", "--oos"])

    day = pd.DatetimeIndex(["2020-01-02"])
    cols = list("ABC")
    candidates = pd.DataFrame([[True, True, True]], index=day, columns=cols)
    values = {
        "BASIS_MOM_242": pd.DataFrame([[1.0, 2.0, 3.0]], index=day, columns=cols),
        "NEG_SKEW_242": pd.DataFrame([[1.0, 2.0, 3.0]], index=day, columns=cols),
        "STICKY_NET_SHARE": pd.DataFrame([[3.0, 2.0, 1.0]], index=day, columns=cols),
    }
    score = core.composite_score(values, candidates, price_core.SCORE_FACTORS).iloc[0]
    assert score.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_sticky_core_keeps_only_two_times_book():
    from strategies.sticky_core import MAIN_BOOK, NAME_CAP, TARGET_MULTIPLIER

    assert (TARGET_MULTIPLIER, NAME_CAP) == (2.0, 0.20)
    assert MAIN_BOOK == "STICKY_CORE"


def test_sticky_core_cohorts_average_daily_cross_sections():
    from strategies.sticky_core import hold_cohorts

    dates = pd.bdate_range("2021-12-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, -0.5, 0.0]}, index=dates)
    held = hold_cohorts(weights, 2)
    assert held["A"].tolist() == pytest.approx([0.25, 0.0, -0.25])


def test_sticky_core_tails_allocate_equal_half_gross_per_leg():
    from strategies.sticky_core import tail_weights

    days = pd.bdate_range("2020-01-01", periods=2)
    names = [f"S{i}" for i in range(10)]
    score = pd.DataFrame(np.tile(np.arange(10.0), (2, 1)), index=days, columns=names)
    score.iloc[1, 3:] = np.nan
    weights = tail_weights(score)
    row = weights.loc[days[0]]
    assert row["S9"] == pytest.approx(0.25) and row["S8"] == pytest.approx(0.25)
    assert row["S0"] == pytest.approx(-0.25) and row["S1"] == pytest.approx(-0.25)
    assert row.sum() == pytest.approx(0.0)
    assert row.abs().sum() == pytest.approx(1.0)
    assert weights.loc[days[1]].eq(0).all()


def test_sticky_core_score_ranks_only_candidates():
    from strategies.sticky_core import FACTORS, composite_score

    day = pd.DatetimeIndex(["2020-01-02"])
    cols = list("ABCD")
    rising = pd.DataFrame([[1.0, 2.0, 3.0, 100.0]], index=day, columns=cols)
    falling = pd.DataFrame([[3.0, 2.0, 1.0, 100.0]], index=day, columns=cols)
    candidates = pd.DataFrame([[True, True, True, False]], index=day, columns=cols)
    values = {FACTORS[0]: rising, FACTORS[1]: falling, FACTORS[2]: rising}
    score = composite_score(values, candidates).iloc[0]
    assert score["A"] == pytest.approx((1 / 3 + 1 + 1 / 3) / 3)
    assert score["B"] == pytest.approx(2 / 3)
    assert np.isnan(score["D"])


def test_sticky_core_cohorts_hold_ten_batches():
    from strategies.sticky_core import HORIZON, hold_cohorts

    days = pd.bdate_range("2020-01-01", periods=HORIZON + 1)
    weights = pd.DataFrame({"A": [1.0] + [0.0] * HORIZON}, index=days)
    held = hold_cohorts(weights)["A"]
    assert held.iloc[: HORIZON].tolist() == pytest.approx([0.1] * HORIZON)
    assert held.iloc[-1] == 0

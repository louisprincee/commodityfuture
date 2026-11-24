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
    weights = blend_weights(sticky, sticky, -sticky, -sticky)

    assert weights.iloc[0].sum() == pytest.approx(0)
    assert weights.iloc[0].abs().sum() == pytest.approx(0)


@pytest.mark.parametrize("invalid_factor", ["chain", "basis", "neg_skew"])
def test_price_blend_requires_only_its_signals_and_rechecks_pool_size(invalid_factor):
    from strategies.sticky_price_blend import daily_target_weights

    names = list(range(20))
    sticky = pd.DataFrame([names, names], dtype=float)
    chain = pd.DataFrame([[10, 9, 0, 1, *range(4, 9), 2, 3, *range(11, 16), 18, 19, 16, 17]] * 2, dtype=float)
    factors = dict(basis=sticky.copy(), neg_skew=sticky.copy(), chain=chain)
    factors[invalid_factor].loc[1, 0] = np.nan
    weights = daily_target_weights(sticky, **factors)

    assert weights.loc[0, 16] == pytest.approx(.125)
    assert weights.loc[0, 17] == pytest.approx(.125)
    assert weights.loc[0, 18] == pytest.approx(.125)
    assert weights.loc[0, 19] == pytest.approx(.125)
    assert weights.loc[0, 0] == pytest.approx(-.125)
    assert weights.loc[0, 1] == pytest.approx(-.125)
    assert weights.loc[0, 2] == pytest.approx(-.125)
    assert weights.loc[0, 3] == pytest.approx(-.125)
    assert weights.loc[1].eq(0).all()


def test_sticky_chain_targets_only_need_net_share_and_chain_score(monkeypatch):
    from strategies import sticky_chain
    from strategies.common import HORIZON

    names = list(range(20))
    dates = pd.date_range("2016-01-04", periods=2)
    sticky = pd.DataFrame([names, names], index=dates, dtype=float)
    chain = -sticky.pow(2)
    chain.iloc[1, 0] = np.nan
    dominant = pd.DataFrame("contract", index=dates, columns=names)
    monkeypatch.setattr(sticky_chain, "load_chain_inputs", lambda *args, **kwargs: (sticky, chain, dominant))

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

    monkeypatch.setattr(M.C, "daily_dir", lambda part: tmp_path)
    monkeypatch.setattr(M.C, "study_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(M.D, "eligibility", lambda p: p.symbol.ne(names[4]))
    monkeypatch.setattr(M, "_seat_state", lambda name, days, market, part:
                        pd.DataFrame({"sticky": names.index(name), "size": 0.5}, index=days))

    targets = sticky_only.build_targets(panel, pd.DataFrame(), start, partition, 1)
    fresh = targets.weights.iloc[0] * HORIZON
    assert fresh[fresh.gt(0)].index.tolist() == names[-1:]
    assert fresh[fresh.lt(0)].index.tolist() == names[:1]
    assert fresh.abs().sum() == pytest.approx(1.0)
    assert targets.dominant.loc[dates[0], names[-1]] == names[-1] + "0001"


def test_benchmark_does_not_run_any_unselected_strategy(monkeypatch, tmp_path):
    from backtest import engine as E
    from strategies import sticky_only

    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(E, "_plot_book_figures", lambda *args: None)
    _save_equity(tmp_path, "research", sticky_only, {sticky_only.MAIN_BOOK: [1e7, 1.2e7]})

    def run(module, partition, out, jobs):
        assert module is None
        path = out / partition / "equity_curves.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({E.BENCHMARK_BOOK: [1e7, 1.1e7]}, index=pd.to_datetime(
            ["2016-01-04", "2016-01-05"],
        )).to_csv(path, index_label="date")

    monkeypatch.setattr(E, "_run_partition", run)
    result = E._evaluate_partition([sticky_only.STRATEGY_ID], "research", 1, reuse=True)
    assert result.columns.tolist() == [sticky_only.MAIN_BOOK, E.BENCHMARK_BOOK]


@pytest.mark.parametrize("partition,loader,fee,root", [
    ("research", "load_panel", "RESEARCH_FEE_PATH", "RESEARCH_EXECUTION_PARAMETERS"),
    ("oos", "load_evaluation_panel", "OOS_FEE_PATH", "OOS_EXECUTION_PARAMETERS"),
])
def test_single_engine_routes_blend_to_the_requested_partition(monkeypatch, tmp_path, partition, loader, fee, root):
    from backtest import engine as E
    from strategies import Targets, sticky_price_blend as blend

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
    monkeypatch.setattr(E, "_benchmark_curve", lambda *args, **kwargs:
                        pd.Series([1e7, 1.1e7], index=dates, name=E.BENCHMARK_BOOK))

    def build_targets(data, daily_bars, lower, part, jobs):
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

    def run(module, partition, out, jobs):
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

    def run(module, partition, out, jobs):
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
    from strategies import sticky_core

    monkeypatch.setattr(E.C, "strategy_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(E, "_plot_book_figures", lambda *args: None)
    monkeypatch.setattr(E, "_plot_performance", lambda *args, **kwargs: None)
    monkeypatch.setattr(E, "_plot_yearly_performance", lambda *args: None)
    books = []
    for modules in E.FAMILIES.values():
        for module in modules:
            books.append(module.MAIN_BOOK)
            curves = {module.MAIN_BOOK: [1e7, 1.1e7]}
            if module is sticky_core:
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


def test_every_book_holds_ten_days_and_the_engine_has_no_horizon_option():
    from backtest import engine as E

    assert E.HORIZON == 10
    with pytest.raises(SystemExit):
        E.main(["--research", "--horizon", "20"])


def test_price_core_is_a_research_only_control_scored_without_sticky():
    from backtest import engine as E
    from strategies import price_core
    from strategies.common import RANKED_FACTORS, composite_score

    assert price_core.PRICE_FACTORS == tuple(f for f in RANKED_FACTORS if f != "STICKY_NET_SHARE")
    assert E.CONTROL_IDS == {price_core.STRATEGY_ID}
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
    score = composite_score(values, candidates, price_core.PRICE_FACTORS).iloc[0]
    assert score.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_sticky_core_keeps_only_two_times_book():
    from strategies.common import NAME_CAP, TARGET_MULTIPLIER
    from strategies.sticky_core import MAIN_BOOK

    assert (TARGET_MULTIPLIER, NAME_CAP) == (2.0, 0.20)
    assert MAIN_BOOK == "STICKY_CORE"


def test_cohorts_average_daily_cross_sections():
    from strategies.common import HORIZON, hold_cohorts

    dates = pd.bdate_range("2021-12-01", periods=HORIZON + 1)
    weights = pd.DataFrame({"A": [0.5, -0.5] + [0.0] * (HORIZON - 1)}, index=dates)
    held = hold_cohorts(weights)
    assert held["A"].tolist() == pytest.approx([0.05] + [0.0] * (HORIZON - 1) + [-0.05])


def test_tails_allocate_equal_half_gross_per_leg():
    from strategies.common import tail_weights

    days = pd.bdate_range("2020-01-01", periods=2)
    names = [f"S{i}" for i in range(10)]
    score = pd.DataFrame(np.tile(np.arange(10.0), (2, 1)), index=days, columns=names)
    score.iloc[1, 3:] = np.nan
    weights = tail_weights(score)
    row = weights.loc[days[0]]
    assert row["S9"] == pytest.approx(0.5)
    assert row["S0"] == pytest.approx(-0.5)
    assert row.sum() == pytest.approx(0.0)
    assert row.abs().sum() == pytest.approx(1.0)
    assert weights.loc[days[1]].eq(0).all()


def test_tail_weights_defaults_to_ten_percent_and_allows_overrides():
    from strategies.common import tail_weights

    score = pd.DataFrame([range(20)], columns=range(20), dtype=float)
    default = tail_weights(score).iloc[0]
    twenty_percent = tail_weights(score, tail_share=0.20).iloc[0]

    assert default.gt(0).sum() == 2
    assert default.lt(0).sum() == 2
    assert twenty_percent.gt(0).sum() == 4
    assert twenty_percent.lt(0).sum() == 4


def test_score_ranks_only_candidates():
    from strategies.common import RANKED_FACTORS, composite_score

    day = pd.DatetimeIndex(["2020-01-02"])
    cols = list("ABCD")
    rising = pd.DataFrame([[1.0, 2.0, 3.0, 100.0]], index=day, columns=cols)
    falling = pd.DataFrame([[3.0, 2.0, 1.0, 100.0]], index=day, columns=cols)
    candidates = pd.DataFrame([[True, True, True, False]], index=day, columns=cols)
    values = {RANKED_FACTORS[0]: rising, RANKED_FACTORS[1]: falling, RANKED_FACTORS[2]: rising}
    score = composite_score(values, candidates, RANKED_FACTORS).iloc[0]
    assert score["A"] == pytest.approx((1 / 3 + 1 + 1 / 3) / 3)
    assert score["B"] == pytest.approx(2 / 3)
    assert np.isnan(score["D"])


def test_cohorts_hold_ten_batches():
    from strategies.common import HORIZON, hold_cohorts

    days = pd.bdate_range("2020-01-01", periods=HORIZON + 1)
    weights = pd.DataFrame({"A": [1.0] + [0.0] * HORIZON}, index=days)
    held = hold_cohorts(weights)["A"]
    assert held.iloc[: HORIZON].tolist() == pytest.approx([0.1] * HORIZON)
    assert held.iloc[-1] == 0

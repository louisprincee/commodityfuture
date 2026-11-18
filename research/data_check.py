"""商品期货本地数据完整性与一致性检查。

    python -m research check

只检查结构、覆盖与跨来源一致性；样本外分区不计算任何收益统计。
结果写入 runs/research/data_check/。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
import data as D

OUT = C.study_dir("data_check")
WINDOWS = C.PARTITION_WINDOWS
FEE_PATHS = {"research": C.RESEARCH_FEE_PATH, "oos": C.OOS_FEE_PATH}
# 郑商所更名品种：更名前的历史在米筐使用旧代码，本地数据不含
CZCE_RENAMED = {"MA": "ME", "ZC": "TC", "OI": "RO", "WH": "WS", "RI": "ER"}
JUMP = 0.15


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, partition: str, check: str, ok: bool, detail: str, level: str | None = None):
        status = level or ("pass" if ok else "fail")
        self.rows.append({"partition": partition, "check": check, "status": status, "detail": detail})
        print(f"[{status:4}] {partition:8} {check}: {detail}", flush=True)


def check_manifest(rep: Report, partition: str) -> None:
    root = D.external_root(partition)
    path = root / "coverage.json"
    if partition == "research":
        C.assert_research_only(path)
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    lo, hi = (d.strftime("%Y%m%d") for d in WINDOWS[partition])
    empty, missing_file, short = [], [], []
    for key, entry in items.items():
        if not entry["coverage"]:
            empty.append(key)
            continue
        if not (root / entry["file"]).exists():
            missing_file.append(key)
        spans = sorted(entry["coverage"])
        if spans[0][0] > lo or spans[-1][1] < hi:
            short.append(key)
    datasets = pd.Series([k.split("/")[0] for k in items]).value_counts().to_dict()
    rep.add(partition, "coverage.json 数据项", True, json.dumps(datasets))
    rep.add(partition, "覆盖区间完整", not short and not missing_file,
            f"区间不足 {len(short)}，文件缺失 {len(missing_file)} {short[:5]}{missing_file[:5]}")
    only_spot_daily = all(k.startswith("spot/daily_") for k in empty)
    rep.add(partition, "空覆盖项", only_spot_daily,
            f"{len(empty)} 项，均为已作废的 spot/daily_* 现货接口" if only_spot_daily else str(empty[:10]),
            level="info" if only_spot_daily else None)


def check_contracts(rep: Report, partition: str) -> pd.DataFrame:
    c = D.load_contracts(partition)
    lo, hi = WINDOWS[partition]
    rep.add(partition, "合约日行情日期", c.date.min() >= pd.Timestamp(lo) and c.date.max() <= pd.Timestamp(hi),
            f"{len(c)} 行，{c.contract.nunique()} 合约，{c.symbol.nunique()} 品种，{c.date.min().date()}..{c.date.max().date()}")
    fin = sorted(set(c.symbol) & set(C.FINANCIAL_SYMBOLS))
    rep.add(partition, "无金融期货", not fin, f"金融期货: {fin}")
    traded = c[c.volume > 0]
    bad_hl = (traded.high < traded[["open", "close"]].max(axis=1) - 1e-9) | \
             (traded.low > traded[["open", "close"]].min(axis=1) + 1e-9)
    nonpos = (traded[["open", "high", "low", "close"]] <= 0).any(axis=1)
    bad = traded[bad_hl]
    rep.add(partition, "OHLC 一致性（有成交日）", nonpos.sum() == 0,
            f"高低价矛盾 {int(bad_hl.sum())} 行（成交量中位 {bad.volume.median():.0f} 手，"
            f"多为交割月合约收盘取结算价），非正价格 {int(nonpos.sum())}；主力日上的矛盾见下一项",
            level="warn" if bad_hl.any() and nonpos.sum() == 0 else None)
    on_dom = []
    for symbol in bad.symbol.unique():
        r1 = D.load_dominant(symbol, 1, partition)
        hit = bad[(bad.symbol == symbol) & bad.set_index(["contract", "date"]).index.isin(
            pd.MultiIndex.from_arrays([r1.values, r1.index]))]
        on_dom += hit[["contract", "date", "volume"]].astype(str).values.tolist()
    rep.add(partition, "主力日 OHLC 矛盾", len(on_dom) <= 5, f"{len(on_dom)} 行 {on_dom[:5]}",
            level="warn" if 0 < len(on_dom) <= 5 else None)
    neg = (c[["volume", "open_interest", "total_turnover"]] < 0).any(axis=1).sum()
    rep.add(partition, "量/持仓非负", neg == 0, f"负值行 {int(neg)}")
    na_close = c.close.isna().sum()
    rep.add(partition, "收盘价缺失", na_close == 0, f"{int(na_close)} 行")
    if partition == "research":
        check_units(rep)
    return c


def check_units(rep: Report) -> None:
    """原始日线在 2020-01-01 前为双边口径：量与持仓均为偶数，跨年处整体减半；折半后应连续。"""
    raw = D.load_contracts("research", raw_units=True)
    before = raw["date"] < pd.Timestamp(C.SINGLE_SIDED_FROM)
    odd = {}
    for col in ("open_interest", "volume"):
        nz = raw[col] > 0
        odd[col] = (float(((raw[col] % 2 != 0) & nz & before).sum() / max((nz & before).sum(), 1)),
                    float(((raw[col] % 2 != 0) & nz & ~before).sum() / max((nz & ~before).sum(), 1)))
    total = raw.groupby(["symbol", "date"])["open_interest"].sum().unstack(0)
    d0, d1 = pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-02")
    both = total.loc[[d0, d1]].dropna(axis=1)
    both = both.loc[:, both.loc[d0] > 0]
    ratio = (both.loc[d1] / both.loc[d0]).median()
    fixed = D.to_single_sided(raw)
    fixed_total = fixed.groupby(["symbol", "date"])["open_interest"].sum().unstack(0)
    fixed_ratio = (fixed_total.loc[d1, both.columns] / fixed_total.loc[d0, both.columns]).median()
    ok = odd["open_interest"][0] == 0 and odd["volume"][0] == 0 and 0.4 < ratio < 0.6 and 0.8 < fixed_ratio < 1.2
    rep.add("research", "持仓/成交量单双边口径", ok,
            f"2020 年前持仓、成交量奇数占比 {odd['open_interest'][0]:.3f} / {odd['volume'][0]:.3f}，"
            f"之后 {odd['open_interest'][1]:.3f} / {odd['volume'][1]:.3f}；"
            f"{len(both.columns)} 个品种 2020-01-02 对 2019-12-31 总持仓比中位 {ratio:.2f}，折半后 {fixed_ratio:.2f}。"
            "研究读取统一折半为单边口径", level="info" if ok else None)


def check_dominant(rep: Report, partition: str, contracts: pd.DataFrame) -> pd.DataFrame:
    px = contracts.set_index(["contract", "date"])
    rows = []
    for symbol in C.COMMODITY_SYMBOLS:
        path = D.external_root(partition) / "dominant" / f"{symbol}_rank1.pkl"
        if not path.exists():
            rows.append({"symbol": symbol, "status": "no_file"})
            continue
        r1, r2 = D.load_dominant(symbol, 1, partition), D.load_dominant(symbol, 2, partition)
        row = {"symbol": symbol, "sector": C.SECTOR.get(symbol), "rank1_days": len(r1), "rank2_days": len(r2)}
        if r1.empty:
            row["status"] = "not_listed"
            rows.append(row)
            continue
        row["first"], row["last"] = r1.index.min().date(), r1.index.max().date()
        for k, r in (("rank1", r1), ("rank2", r2)):
            key = pd.MultiIndex.from_arrays([r.values, r.index])
            row[f"{k}_no_price"] = int(px["close"].reindex(key).isna().sum())
        k1 = pd.MultiIndex.from_arrays([r1.values, r1.index])
        row["rank1_zero_volume_days"] = int((px["volume"].reindex(k1) == 0).sum())
        both = r2.reindex(r1.index).dropna()
        row["rank1_eq_rank2"] = int((both == r1.reindex(both.index)).sum())
        row["rank2_missing_days"] = int(len(r1) - len(both))
        row["rank1_switches"] = int(r1.ne(r1.shift()).sum() - 1)
        # 主力在本品种已有交易日之间的空档
        cal = contracts.loc[contracts.symbol == symbol, "date"].drop_duplicates()
        cal = cal[(cal >= r1.index.min()) & (cal <= r1.index.max())]
        row["calendar_gaps"] = int(len(set(cal) - set(r1.index)))
        row["status"] = "ok"
        rows.append(row)
    table = pd.DataFrame(rows)
    ok = table[table.status == "ok"]
    rep.add(partition, "主力/次主力均有合约行情",
            int(ok[["rank1_no_price", "rank2_no_price"]].sum().sum()) == 0,
            f"{len(ok)} 品种；无行情天数 rank1={int(ok.rank1_no_price.sum())} rank2={int(ok.rank2_no_price.sum())}")
    same = ok.loc[ok.rank1_eq_rank2 > 0, ["symbol", "rank1_eq_rank2"]].values.tolist()
    rep.add(partition, "主力与次主力不同", int(ok.rank1_eq_rank2.sum()) <= 20,
            f"相同天数 {int(ok.rank1_eq_rank2.sum())} {same}；日面板中这些日的次主力置空",
            level="warn" if same and int(ok.rank1_eq_rank2.sum()) <= 20 else None)
    rep.add(partition, "次主力缺失天数", True,
            f"{int(ok.rank2_missing_days.sum())} 天，品种 "
            f"{ok.loc[ok.rank2_missing_days > 0, ['symbol', 'rank2_missing_days']].values.tolist()}", level="info")
    rep.add(partition, "主力日期相对品种日历的空档", int(ok.calendar_gaps.sum()) == 0,
            f"{ok.loc[ok.calendar_gaps > 0, ['symbol', 'calendar_gaps']].values.tolist()}")
    zero = ok.loc[ok.rank1_zero_volume_days > 0, ["symbol", "rank1_zero_volume_days"]]
    rep.add(partition, "主力零成交日", True, f"{zero.values.tolist()}", level="info")
    unlisted = table.loc[table.status == "not_listed", "symbol"].tolist()
    rep.add(partition, "分区内未上市品种", True, f"{unlisted}", level="info")
    renamed = {s: f"{CZCE_RENAMED[s]}→{s} 自 {ok.set_index('symbol').loc[s, 'first']}"
               for s in CZCE_RENAMED if s in set(ok.symbol)}
    if partition == "research":
        rep.add(partition, "郑商所更名前历史缺失", True,
                f"{renamed}；更名前旧代码合约未下载，这些品种的序列从更名日开始", level="warn")
    return table


def check_roll_yield(rep: Report, partition: str) -> None:
    match_from, match_to, n = 0, 0, 0
    for symbol in D.symbols_with_data(partition):
        ry = D.load_roll_yield(symbol, partition)
        if ry.empty:
            continue
        r1 = D.load_dominant(symbol, 1, partition).reindex(ry.index)
        r2 = D.load_dominant(symbol, 2, partition).reindex(ry.index)
        match_from += int((ry.from_contract == r1).sum())
        match_to += int((ry.to_contract == r2).sum())
        n += len(ry)
    rep.add(partition, "展期收益 from/to = 主力/次主力", match_from == n and match_to == n,
            f"from 匹配 {match_from}/{n}，to 匹配 {match_to}/{n}")


def check_shards(rep: Report, partition: str, contracts: pd.DataFrame) -> None:
    root = C.minute_dir(partition)
    lo, hi = WINDOWS[partition]
    px = contracts.set_index(["contract", "date"])
    stats = []
    for path in sorted(root.glob("*.parquet")):
        symbol = path.stem
        if partition == "research":
            C.assert_research_only(path)
        m = pd.read_parquet(path, columns=["close", "trading_date"])
        td = pd.to_datetime(m.trading_date)
        if partition == "research":
            C.assert_no_holdout_dates(td, f"{symbol} 研究期分钟")
        r1 = D.load_dominant(symbol, 1, partition)
        last = m.groupby("trading_date")["close"].last()
        last.index = pd.to_datetime(last.index)
        key = pd.MultiIndex.from_arrays([r1.values, r1.index])
        daily = pd.Series(px["close"].reindex(key).values, index=r1.index)
        settle = pd.Series(px["settlement"].reindex(key).values, index=r1.index)
        j = pd.concat([last, daily, settle], axis=1, keys=["m", "c", "s"]).dropna()
        eq = np.isclose(j.m, j.c)
        stats.append({
            "symbol": symbol,
            "dup_index": int(m.index.duplicated().sum()),
            "monotonic": bool(m.index.is_monotonic_increasing),
            "in_window": bool(td.min() >= pd.Timestamp(lo) and td.max() <= pd.Timestamp(hi)),
            "days_minute_not_dominant": len(set(last.index) - set(r1.index)),
            "days_dominant_not_minute": len(set(r1.index) - set(last.index)),
            "close_match": float(eq.mean()) if len(j) else np.nan,
            "mismatch_eq_settlement": float(np.isclose(j.m[~eq], j.s[~eq]).mean()) if (~eq).any() else np.nan,
        })
    s = pd.DataFrame(stats)
    s.to_csv(OUT / f"minute_vs_daily_{partition}.csv", index=False)
    rep.add(partition, "分钟分片结构", int(s.dup_index.sum()) == 0 and s.monotonic.all() and s.in_window.all(),
            f"{len(s)} 品种，重复时间戳 {int(s.dup_index.sum())}，非单调 {int((~s.monotonic).sum())}，越窗 {int((~s.in_window).sum())}")
    rep.add(partition, "分钟交易日 = 主力交易日",
            int(s.days_minute_not_dominant.sum() + s.days_dominant_not_minute.sum()) == 0,
            f"分钟多 {int(s.days_minute_not_dominant.sum())}，主力多 {int(s.days_dominant_not_minute.sum())}")
    rep.add(partition, "88 分钟末根收盘 = 主力日收盘", True,
            f"中位匹配率 {s.close_match.median():.3f}，最低 {s.close_match.min():.3f}（{s.loc[s.close_match.idxmin(), 'symbol']}）；"
            f"不一致日中末根收盘等于结算价的比例中位 {s.mismatch_eq_settlement.median():.2f}。"
            "差异来自分钟末根收盘取值口径，日频研究以合约日行情为准", level="info")


def check_fees(rep: Report, partition: str, dominant: pd.DataFrame) -> None:
    path = FEE_PATHS[partition]
    if partition == "research":
        C.assert_research_only(path)
    if not path.exists():
        rep.add(partition, "手续费表", False, f"缺失 {path}")
        return
    fee = pd.read_csv(path, parse_dates=["trading_date"])
    if partition == "research":
        C.assert_no_holdout_dates(fee.trading_date, "研究期手续费")
    lo, hi = WINDOWS[partition]
    rep.add(partition, "手续费表日期", fee.trading_date.min() >= pd.Timestamp(lo) and fee.trading_date.max() <= pd.Timestamp(hi),
            f"{len(fee)} 行，{fee.symbol.nunique()} 品种，{fee.trading_date.min().date()}..{fee.trading_date.max().date()}")
    spec_fields = {"contract_multiplier", "tick_size"}
    missing_specs = spec_fields - set(fee.columns)
    if missing_specs:
        rep.add(partition, "手续费成本规格字段", False, f"缺少 {sorted(missing_specs)}",
                level="warn")
    else:
        multiplier = pd.to_numeric(fee["contract_multiplier"], errors="coerce")
        tick = pd.to_numeric(fee["tick_size"], errors="coerce")
        bad_specs = int((multiplier.isna() | multiplier.le(0) | tick.isna() | tick.le(0)).sum())
        rep.add(partition, "手续费成本规格有效", bad_specs == 0,
                f"无效合约乘数或 tick 行数 {bad_specs}", level="warn" if bad_specs else None)
    missing = []
    for symbol in dominant.loc[dominant.status == "ok", "symbol"]:
        days = D.load_dominant(symbol, 1, partition).index
        have = set(fee.loc[fee.symbol == symbol, "trading_date"])
        gap = len(set(days) - have)
        if gap:
            missing.append((symbol, gap))
    rep.add(partition, "主力日手续费覆盖", not missing, f"缺失 (品种, 天数): {missing}",
            level="warn" if missing else None)


def check_research_jumps(rep: Report, contracts: pd.DataFrame) -> None:
    """研究期同合约日收益异常值（只在研究期做）。"""
    c = contracts.copy()
    c["ret"] = c.groupby("contract")["close"].pct_change()
    rows = []
    for symbol in D.symbols_with_data("research"):
        r1 = D.load_dominant(symbol, 1, "research")
        key = pd.MultiIndex.from_arrays([r1.values, r1.index])
        ret = c.set_index(["contract", "date"])["ret"].reindex(key)
        big = ret[ret.abs() > JUMP]
        rows += [{"symbol": symbol, "contract": k[0], "date": k[1].date(), "ret": v} for k, v in big.items()]
    t = pd.DataFrame(rows, columns=["symbol", "contract", "date", "ret"])
    t.to_csv(OUT / "research_dominant_jumps.csv", index=False)
    rep.add("research", f"主力同合约日收益 |r|>{JUMP:.0%}", True,
            f"{len(t)} 次：{t.head(8).to_dict('records')}", level="info" if len(t) < 30 else "warn")


def check_member_rank(rep: Report, contracts: pd.DataFrame) -> None:
    """研究期会员排名：覆盖、入榜人数、占比上限、品种与合约级一致性、持仓变化字段与名称清单。"""
    root = C.daily_dir("research") / "member_rank"
    if not root.exists():
        rep.add("research", "会员排名", False, f"缺失 {root}")
        return
    cov, daily_parts, names, consistency, change = [], [], [], [], []
    for symbol in D.symbols_with_data("research"):
        r1 = D.load_dominant(symbol, 1, "research")
        if r1.empty:
            continue
        daily = D.member_rank_daily(symbol, contracts, "research")
        ranks = D.load_member_rank(symbol, "research")
        row = {"symbol": symbol, "exchange_days": len(r1), "first_dominant": r1.index.min().date()}
        if daily.empty:
            cov.append({**row, "ranked_days": 0})
            continue
        days = set(daily.trading_date)
        row.update({"ranked_days": int(len(days & set(r1.index))),
                    "first_ranked": daily.trading_date.min().date(),
                    "short_lists": int((daily[["n_long", "n_short"]].min(axis=1) < 20).sum()),
                    "no_contract_detail": int(daily["n_ranked_contracts"].isna().sum())
                    if "n_ranked_contracts" in daily else len(daily)})
        cov.append(row)
        daily_parts.append(daily)
        names.append(ranks[["member_name", "trading_date"]])
        # 品种级会员持仓 ≥ 该会员各入榜合约持仓之和（合约级只含前 20 名，可小于品种级）
        by_contract = D.load_member_rank_contracts(symbol, "research")
        if not by_contract.empty:
            c_sum = by_contract.groupby(["trading_date", "side", "member_name"])["volume"].sum()
            s = ranks[ranks.side.isin(["long", "short"])].set_index(["trading_date", "side", "member_name"])["volume"]
            j = pd.concat([s, c_sum.reindex(s.index)], axis=1, keys=["symbol", "contracts"]).dropna()
            consistency.append({"symbol": symbol, "pairs": len(j),
                                "equal": int(np.isclose(j.symbol, j.contracts).sum()),
                                "symbol_below_contracts": int((j.symbol < j.contracts - 0.5).sum())})
        # 持仓变化字段：连续两日在榜的会员，volume_t - volume_{t-1} 应等于 volume_change
        held = ranks[ranks.side.isin(["long", "short"])].sort_values(["side", "member_name", "trading_date"])
        prev_day = pd.Series(sorted(days)).shift(1)
        prev_map = dict(zip(sorted(days), prev_day))
        held = held.assign(prev_date=held.trading_date.map(prev_map))
        lag = held.groupby(["side", "member_name"])[["trading_date", "volume"]].shift(1)
        ok_pair = lag["trading_date"].eq(held["prev_date"])
        diff = held["volume"] - lag["volume"]
        change.append({"symbol": symbol, "pairs": int(ok_pair.sum()),
                       "match": int((ok_pair & np.isclose(diff, held["volume_change"])).sum())})
    cov = pd.DataFrame(cov)
    cov["coverage"] = cov["ranked_days"] / cov["exchange_days"]
    cov.to_csv(OUT / "member_rank_coverage.csv", index=False)
    daily = pd.concat(daily_parts, ignore_index=True)
    daily["long_share_ranked"] = daily["top20_long"] / daily["ranked_oi"]
    daily["short_share_ranked"] = daily["top20_short"] / daily["ranked_oi"]
    daily["long_share_total"] = daily["top20_long"] / daily["total_oi"]
    daily.to_csv(OUT / "member_rank_daily.csv", index=False)
    low = cov.loc[cov.coverage < 0.95, ["symbol", "first_dominant", "first_ranked", "coverage"]]
    rep.add("research", "会员排名覆盖（主力日）", True,
            f"{len(cov)} 品种，覆盖率中位 {cov.coverage.median():.3f}；低于 95% 的品种 "
            f"{low.round(3).astype(str).values.tolist()}", level="info" if len(low) else None)
    rep.add("research", "会员排名不足 20 名的日子", True,
            f"{int(cov.short_lists.sum())} 个品种日", level="info")
    total_share = daily[["top20_long", "top20_short"]].max(axis=1) / daily["total_oi"]
    over_total = int((total_share > 1.0 + 1e-9).sum())
    rep.add("research", "前 20 名持仓 / 全品种单边持仓 ≤ 1", over_total == 0,
            f"超过 1 的品种日 {over_total}，最大 {total_share.max():.3f}；多头占比中位 "
            f"{(daily.top20_long / daily.total_oi).median():.3f}，空头 {(daily.top20_short / daily.total_oi).median():.3f}")
    share = daily[["long_share_ranked", "short_share_ranked"]].max(axis=1)
    over = daily[share > 1.0 + 1e-9]
    rep.add("research", "合约级排名明细覆盖", True,
            f"以入榜合约持仓为分母时占比超过 1 的品种日 {len(over)}（{over.symbol.value_counts().head(6).to_dict()}），"
            "原因是米筐合约级排名缺少部分郑商所合约（含主力），分母偏小；品种级排名不受影响，研究使用品种级排名与全品种持仓",
            level="warn" if len(over) else "info")
    jump = daily.set_index("trading_date").groupby("symbol")["long_share_ranked"]
    d0, d1 = pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-02")
    ratios = [s.get(d1) / s.get(d0) for _, s in jump if d0 in s.index and d1 in s.index]
    rep.add("research", "跨 2020 口径切换的占比连续性", 0.8 < np.nanmedian(ratios) < 1.25,
            f"{len(ratios)} 个品种 2020-01-02 对 2019-12-31 多头占比之比中位 {np.nanmedian(ratios):.2f}")
    cons = pd.DataFrame(consistency)
    below = int(cons.symbol_below_contracts.sum())
    rep.add("research", "品种级会员持仓 ≥ 合约级之和", below <= 0.01 * cons.pairs.sum(),
            f"{int(cons.pairs.sum())} 个会员日，相等 {cons.equal.sum() / cons.pairs.sum():.3f}，"
            f"品种级小于合约级 {below}", level="warn" if below else None)
    ch = pd.DataFrame(change)
    rep.add("research", "持仓变化字段 = 相邻两日持仓差", True,
            f"{int(ch.pairs.sum())} 个连续在榜会员日，一致 {ch.match.sum() / max(ch.pairs.sum(), 1):.3f}", level="info")
    inv = pd.concat(names).groupby("member_name")["trading_date"].agg(["min", "max", "count"])
    inv.sort_values("count", ascending=False).to_csv(OUT / "member_names.csv", encoding="utf-8-sig")
    rep.add("research", "会员名称清单", True,
            f"{len(inv)} 个名称；更名与合并未处理，清单见 member_names.csv", level="info")


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = Report()
    raw = C.MINUTE_MONOLITH
    rep.add("all", "分钟原始单体", raw.exists(),
            f"{raw.name} {raw.stat().st_size / 1e9:.1f} GB；体积超过可用内存，按其研究/样本外分片检查" if raw.exists() else "缺失",
            level="info" if raw.exists() else None)
    for partition in D.PARTITIONS:
        check_manifest(rep, partition)
        contracts = check_contracts(rep, partition)
        dominant = check_dominant(rep, partition, contracts)
        dominant.to_csv(OUT / f"symbol_coverage_{partition}.csv", index=False)
        check_roll_yield(rep, partition)
        check_shards(rep, partition, contracts)
        check_fees(rep, partition, dominant)
        if partition == "research":
            check_research_jumps(rep, contracts)
            check_member_rank(rep, contracts)
        del contracts
    table = pd.DataFrame(rep.rows)
    table.to_csv(OUT / "checks.csv", index=False, encoding="utf-8-sig")
    fails = table[table.status == "fail"]
    print(f"\n{len(table)} 项检查，失败 {len(fails)}，警告 {int((table.status == 'warn').sum())}。结果: {OUT}")
    return 1 if len(fails) else 0

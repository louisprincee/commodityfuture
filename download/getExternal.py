"""下载商品期货逐合约日行情、主力/次主力映射、主次展期收益、仓单、现货基准与会员持仓排名。

    python -m download.getExternal --start-date 20100101 --end-date 20251231

研究期写入 data/research/daily/，2022–2025 写入 data/oos/daily/；
每个数据项在 coverage.json 记录已覆盖区间，重跑只补缺口。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config as C  # noqa: E402
from rq_auth import rqdata_credentials  # noqa: E402

DEFAULT_SYMBOLS = ["all"]
PRICE_FIELDS = [
    "open", "high", "low", "close", "settlement", "prev_settlement",
    "volume", "total_turnover", "open_interest",
]
DATASETS = ("contracts", "dominant", "roll_yield", "warehouse", "spot",
            "member_rank", "member_rank_contracts")
# 会员排名：品种级取多、空、成交三张榜；合约级只取多、空两张，用于确定每日入榜合约与分合约持仓
MEMBER_SIDES = ("long", "short", "volume")
MEMBER_CONTRACT_SIDES = ("long", "short")
SPOT_BENCHMARKS = ["AU9999.SGEX", "AG9999.SGEX"]
MANIFEST_FLUSH_EVERY = 100


def parse_day(value: str) -> date:
    if re.fullmatch(r"[0-9]{8}", value):
        return datetime.strptime(value, "%Y%m%d").date()
    return date.fromisoformat(value)


def missing_ranges(start: date, end: date,
                   covered: list[list[str]]) -> list[tuple[date, date]]:
    """返回已下载覆盖范围里还没有的日期区间。"""
    if start > end:
        return []
    intervals = sorted((parse_day(left), parse_day(right)) for left, right in covered)
    merged: list[list[date]] = []
    for left, right in intervals:
        if not merged or left > merged[-1][1] + timedelta(days=1):
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)

    gaps: list[tuple[date, date]] = []
    cursor = start
    for left, right in merged:
        if right < cursor:
            continue
        if left > end:
            break
        if left > cursor:
            gaps.append((cursor, min(end, left - timedelta(days=1))))
        cursor = max(cursor, right + timedelta(days=1))
        if cursor > end:
            break
    if cursor <= end:
        gaps.append((cursor, end))
    return gaps


def merge_frames(existing: pd.DataFrame | None,
                 incoming: pd.DataFrame | None) -> pd.DataFrame:
    if incoming is None:
        incoming = pd.DataFrame()
    if existing is None or existing.empty:
        result = incoming.copy()
    elif incoming.empty:
        result = existing.copy()
    else:
        result = pd.concat([existing, incoming], axis=0)
    if not result.empty:
        result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


class CoverageCache:
    def __init__(self, root: Path, dry_run: bool = False):
        self.root = Path(root)
        if not dry_run:
            self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "coverage.json"
        self.dry_run = dry_run
        self.failures: list[str] = []
        self._manifest_updates = 0
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"version": 1, "items": {}}

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    def _entry(self, dataset: str, key: str) -> dict:
        return self.manifest["items"].setdefault(
            f"{dataset}/{key}", {"file": f"{dataset}/{self._slug(key)}.pkl",
                                 "coverage": []})

    def _save_manifest(self, force: bool = False) -> None:
        if not force and self._manifest_updates < MANIFEST_FLUSH_EVERY:
            return
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)
        self._manifest_updates = 0

    def flush(self) -> None:
        self._save_manifest(force=True)

    def invalidate_empty(self, dataset: str, key_prefix: str = "") -> None:
        if self.dry_run:
            return
        changed = False
        for item_key, entry in self.manifest["items"].items():
            if not item_key.startswith(f"{dataset}/"):
                continue
            key = item_key.split("/", 1)[1]
            if not key.startswith(key_prefix) or not entry["coverage"]:
                continue
            path = self.root / entry["file"]
            if path.exists() and pd.read_pickle(path).empty:
                entry["coverage"] = []
                path.unlink()
                changed = True
        if changed:
            self._save_manifest(force=True)

    def fetch(self, dataset: str, key: str, start: date, end: date,
              request: Callable[[str, str], pd.DataFrame],
              require_nonempty: bool = False) -> None:
        entry = self._entry(dataset, key)
        path = self.root / entry["file"]
        if entry["coverage"] and not path.is_file():
            # 覆盖记录还在、文件却没了（例如只拷了 coverage.json）：按未下载处理
            entry["coverage"] = []
            self._manifest_updates += 1
        if require_nonempty and entry["coverage"]:
            cached = pd.read_pickle(path)
            if cached is None or cached.empty:
                entry["coverage"] = []
                if not self.dry_run:
                    path.unlink()
                    self._save_manifest()
        gaps = missing_ranges(start, end, entry["coverage"])
        if not gaps:
            print(f"已覆盖，跳过: {dataset}/{key}")
            return

        if self.dry_run:
            for left, right in gaps:
                print(f"待下载: {dataset}/{key} {left:%Y%m%d}-{right:%Y%m%d}")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_pickle(path) if path.exists() else None
        for left, right in gaps:
            left_s, right_s = left.strftime("%Y%m%d"), right.strftime("%Y%m%d")
            print(f"下载: {dataset}/{key} {left_s}-{right_s}")
            try:
                incoming = request(left_s, right_s)
                if incoming is None:
                    raise ValueError("接口返回 None")
                if require_nonempty and incoming.empty:
                    raise ValueError("接口返回空数据")
            except Exception as exc:
                failure = (f"{dataset}/{key} {left_s}-{right_s}: "
                           f"{type(exc).__name__}: {exc}")
                self.failures.append(failure)
                print(f"接口失败，保留为未覆盖: {failure}", file=sys.stderr)
                continue
            existing = merge_frames(existing, incoming)
            tmp = path.with_suffix(".pkl.tmp")
            with tmp.open("wb") as stream:
                pickle.dump(existing, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            entry["coverage"].append([left_s, right_s])
            self._manifest_updates += 1
            self._save_manifest()


def member_rank(rq, obj: str, side: str, left: str, right: str) -> pd.DataFrame:
    """前 20 名会员排名；无排名返回空表。索引为 (trading_date, member_name)，保证缓存合并时不被去重。"""
    frame = rq.futures.get_member_rank(obj, start_date=left, end_date=right, rank_by=side)
    columns = ["commodity_id", "rank", "volume", "volume_change"]
    if frame is None or len(frame) == 0:
        index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), pd.Index([], dtype=object)],
                                          names=["trading_date", "member_name"])
        return pd.DataFrame(columns=columns, index=index)
    out = frame.reset_index()
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.normalize()
    if out.duplicated(["trading_date", "member_name"]).any():
        raise ValueError(f"{obj} {side} 同日会员重复")
    return out.set_index(["trading_date", "member_name"])[columns].sort_index()


def commodity_instruments(rq, symbols: list[str] | None) -> tuple[list[str], pd.DataFrame]:
    instruments = rq.all_instruments(type="Future")
    if "product" not in instruments or "underlying_symbol" not in instruments:
        raise RuntimeError("all_instruments 返回缺少 product/underlying_symbol 字段")
    instruments = instruments[instruments["product"] == "Commodity"].copy()
    available = sorted(instruments["underlying_symbol"].dropna().astype(str).unique())
    if symbols is None:
        requested = set(C.COMMODITY_SYMBOLS)
        selected = sorted(requested.intersection(available))
        missing = sorted(requested - set(available))
        if missing:
            print(f"米筐合约清单缺少这些商品品种，跳过: {missing}")
    else:
        selected = sorted(set(symbols))
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"米筐未返回这些商品品种: {unknown}")

    listed = instruments.get("listed_date", pd.Series(index=instruments.index, dtype=str))
    instruments = instruments[listed.astype(str) != "0000-00-00"]
    instruments = instruments[instruments["underlying_symbol"].isin(selected)]
    instruments = instruments.drop_duplicates("order_book_id")
    return selected, instruments


def trading_in(instruments: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """上市区间与 [start, end] 有交集的合约；缺上市/退市日期的保留。"""
    def day(column: str, fallback: pd.Timestamp) -> pd.Series:
        raw = instruments.get(column, pd.Series(index=instruments.index, dtype=str))
        raw = raw.astype(str).replace("0000-00-00", "")
        return pd.to_datetime(raw, errors="coerce").fillna(fallback)

    listed = day("listed_date", pd.Timestamp.min)
    delisted = day("de_listed_date", pd.Timestamp.max)
    return instruments[(listed <= pd.Timestamp(end)) & (delisted >= pd.Timestamp(start))]


def run_download(rq, cache: CoverageCache, start: date, end: date,
                 symbols: list[str] | None, datasets: set[str],
                 contract_rank_sides: tuple[str, ...] = MEMBER_CONTRACT_SIDES) -> None:
    selected, instruments = commodity_instruments(rq, symbols)
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    if "member_rank" in datasets:
        cache.invalidate_empty("member_rank")
    if "member_rank_contracts" in datasets:
        cache.invalidate_empty("member_rank_contracts")

    if "contracts" in datasets:
        contracts = trading_in(instruments, start, end)["order_book_id"].astype(str).tolist()
        print(f"区间内交易的商品合约: {len(contracts)} 个（共 {len(instruments)} 个）")
        for contract in tqdm(contracts, desc="逐合约日行情"):
            cache.fetch("contracts", contract, start, end,
                        lambda left, right, contract=contract: rq.get_price(
                            contract, start_date=left, end_date=right,
                            frequency="1d", fields=PRICE_FIELDS,
                            adjust_type="none"))

    if "member_rank_contracts" in datasets:
        contracts = trading_in(instruments, start, end)["order_book_id"].astype(str).tolist()
        for contract in tqdm(contracts, desc="逐合约会员排名"):
            for side in contract_rank_sides:
                cache.fetch("member_rank_contracts", f"{contract}_{side}", start, end,
                            lambda left, right, contract=contract, side=side:
                            member_rank(rq, contract, side, left, right),
                            require_nonempty=True)

    for symbol in tqdm(selected, desc="商品品种"):
        if "member_rank" in datasets:
            for side in MEMBER_SIDES:
                cache.fetch("member_rank", f"{symbol}_{side}", start, end,
                            lambda left, right, symbol=symbol, side=side:
                            member_rank(rq, symbol, side, left, right),
                            require_nonempty=True)

        if "dominant" in datasets:
            for rank in (1, 2):
                cache.fetch(
                    "dominant", f"{symbol}_rank{rank}", start, end,
                    lambda left, right, symbol=symbol, rank=rank: rq.futures.get_dominant(
                        symbol, start_date=left, end_date=right, rank=rank),
                    require_nonempty=True)

        if "roll_yield" in datasets:
            cache.fetch(
                "roll_yield", f"{symbol}_main_sub", start, end,
                lambda left, right, symbol=symbol: rq.futures.get_roll_yield(
                    symbol, start_date=left, end_date=right, type="main_sub"),
                require_nonempty=True)

        if "warehouse" in datasets:
            cache.fetch(
                "warehouse", symbol, start, end,
                lambda left, right, symbol=symbol: rq.futures.get_warehouse_stocks(
                    symbol, start_date=left, end_date=right),
                require_nonempty=True)

    if "spot" in datasets:
        cache.invalidate_empty("spot", key_prefix="daily_")
        for order_book_id in SPOT_BENCHMARKS:
            cache.fetch(
                "spot", f"benchmark_{order_book_id}", start, end,
                lambda left, right, order_book_id=order_book_id:
                rq.get_spot_benchmark_price(
                    order_book_id, start_date=left, end_date=right),
                require_nonempty=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载米筐商品期货外部研究数据")
    parser.add_argument("--start-date", default="20100101", help="YYYYMMDD，默认 20100101")
    default_end = min(date.today() - timedelta(days=1), C.DEFAULT_OOS_END).strftime("%Y%m%d")
    parser.add_argument("--end-date", default=default_end,
                        help=f"YYYYMMDD，默认昨天 ({default_end})")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="商品品种代码，默认 all（使用米筐期货合约中的商品品种）")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
                        help="选择接口数据集，默认全部")
    parser.add_argument("--contract-rank-sides", nargs="+", choices=MEMBER_SIDES,
                        default=MEMBER_CONTRACT_SIDES, help="合约级榜单类型，默认 long short")
    parser.add_argument("--datapath", type=Path, default=C.DATA_ROOT,
                        help="数据根目录，分别写入 research/daily 和 oos/daily")
    parser.add_argument("--dry-run", action="store_true", help="查询合约清单并显示缺失区间，不下载数据或写文件")
    args = parser.parse_args()

    try:
        start, end = parse_day(args.start_date), parse_day(args.end_date)
    except (TypeError, ValueError):
        parser.error("日期必须是有效的 YYYYMMDD")
    if start > end:
        parser.error("开始日期不能晚于结束日期")
    if end > C.DEFAULT_OOS_END:
        parser.error(f"结束日期不能晚于已登记样本范围 {C.DEFAULT_OOS_END}")
    if end > date.today():
        parser.error("结束日期不能晚于今天")

    periods = C.split_period(start, end)
    if not periods:
        parser.error("日期窗口与已登记数据分区没有交集")
    requested_symbols = None if args.symbols == ["all"] else [s.upper() for s in args.symbols]
    credentials = rqdata_credentials()
    if credentials is None:
        parser.error("请在 config/rqdata.env 填写 RQDATAC_LICENSE，或填写用户名和密码")

    import rqdatac as rq

    rq.init(*credentials)
    output_root = args.datapath
    try:
        failures = []
        for partition, left, right in periods:
            print(f"\n时间分区: {partition} ({left:%Y%m%d}-{right:%Y%m%d})")
            partition_root = C.daily_dir(partition, output_root)
            if partition == "research":
                C.assert_research_only(partition_root)
            cache = CoverageCache(partition_root, dry_run=args.dry_run)
            try:
                run_download(rq, cache, left, right, requested_symbols, set(args.datasets),
                             tuple(dict.fromkeys(args.contract_rank_sides)))
            finally:
                if not args.dry_run:
                    cache.flush()
            failures.extend(cache.failures)
    except Exception as exc:
        print(f"下载中断: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if failures:
        print(f"完成但有 {len(failures)} 项失败；失败区间未标记覆盖，重跑可补拉。",
              file=sys.stderr)
        return 1
    print(f"完成。分区数据与覆盖记录保存在: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

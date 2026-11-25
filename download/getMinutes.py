"""下载商品期货 1 分钟行情，按样本分区与品种保存 Parquet。"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from src import config as C  # noqa: E402
from rq_auth import rqdata_credentials  # noqa: E402

PRICE = ["open", "high", "low", "close"]


def _series(rq, contract: str, start: date, end: date, fields: list[str]) -> pd.DataFrame:
    frame = rq.get_price(contract, start_date=start, end_date=end, frequency="1m", fields=fields)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], name="datetime"))
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index.get_level_values("datetime"), name="datetime")
    return frame.loc[~frame.index.duplicated(keep="last"), fields].sort_index()


def _trading_dates(frame: pd.DataFrame) -> pd.Series:
    if "trading_date" not in frame:
        raise ValueError("分钟分片必须包含交易日")
    raw = frame["trading_date"]
    if pd.api.types.is_numeric_dtype(raw):
        raw = raw.astype("Int64").astype("string")
    days = pd.to_datetime(raw).dt.normalize()
    if days.isna().any():
        raise ValueError("分钟分片存在空交易日")
    return days


def _check_partition(frame: pd.DataFrame, partition: str) -> None:
    days = _trading_dates(frame)
    if partition == "research":
        C.assert_no_holdout_dates(days, "分钟分片")
        if days.lt(pd.Timestamp(C.RESEARCH_START)).any():
            raise C.HoldoutViolation("分钟分片早于研究期起点")
    elif partition == "oos":
        C.assert_oos_dates(days, "分钟分片")
    else:
        raise C.HoldoutViolation(f"未知分区 {partition}")


def write_shard(path: Path, partition: str, incoming: pd.DataFrame) -> None:
    """保留历史列并以新观测更新重叠分钟，校验后原子替换。"""
    if partition == "research":
        C.assert_research_only(path)
    _check_partition(incoming, partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".parquet.lock")
    with lock.open("x"):
        pass
    try:
        _merge_and_replace(path, partition, incoming)
    finally:
        lock.unlink()


def _merge_and_replace(path: Path, partition: str, incoming: pd.DataFrame) -> None:
    merged = incoming.assign(trading_date=_trading_dates(incoming))
    if path.exists():
        previous = pd.read_parquet(path)
        _check_partition(previous, partition)
        previous = previous.assign(trading_date=_trading_dates(previous))
        # 同一时间戳应由交易所交易日唯一确定。
        common = merged.index.intersection(previous.index)
        if not merged.loc[common, "trading_date"].equals(previous.loc[common, "trading_date"]):
            raise ValueError(f"{path.name} 的重叠分钟交易日不一致")
        merged = merged.combine_first(previous)
    merged = merged.sort_index()
    if not merged.index.is_unique:
        raise ValueError(f"{path.name} 有重复分钟")
    with NamedTemporaryFile(dir=path.parent, prefix=path.stem, suffix=".parquet.tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        merged.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_symbol(rq, symbol: str, start: date, end: date) -> pd.DataFrame:
    parts = []
    for year in range(start.year, end.year + 1):
        left, right = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        base = _series(rq, symbol + "88", left, right,
                       [*PRICE, "volume", "total_turnover", "open_interest", "trading_date"])
        if base.empty:
            continue
        adjusted = _series(rq, symbol + "889", left, right, PRICE).add_suffix("w")
        aggregated = _series(rq, symbol + "99", left, right, ["open_interest", "volume"]).add_suffix("99")
        combined = base.join([adjusted, aggregated])
        day = _trading_dates(combined)
        parts.append(combined.loc[day.between(pd.Timestamp(left), pd.Timestamp(right))])
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts)
    return combined.loc[~combined.index.duplicated(keep="last")].sort_index()


def _missing_edges(path: Path, partition: str, days: pd.DatetimeIndex) -> list[tuple[date, date]]:
    """已有分片只补首尾区间，交易日以交易所日历为准。"""
    if days.empty:
        return []
    if partition == "research":
        C.assert_research_only(path)
    if not path.exists():
        return [(days[0].date(), days[-1].date())]
    previous = pd.read_parquet(path, columns=["trading_date"])
    if previous.empty:
        return [(days[0].date(), days[-1].date())]
    _check_partition(previous, partition)
    observed = _trading_dates(previous)
    gaps = (days[days < observed.min()], days[days > observed.max()])
    return [(gap[0].date(), gap[-1].date()) for gap in gaps if not gap.empty]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_end = min(date.today() - timedelta(days=1), C.DEFAULT_OOS_END)
    parser.add_argument("-t", "--time-period", default=f"20240101-{default_end:%Y%m%d}",
                        help="YYYYMMDD-YYYYMMDD")
    parser.add_argument("--datapath", type=Path, default=C.DATA_ROOT,
                        help="数据根目录，分别写入 research/minute 和 oos/minute")
    parser.add_argument("--symbols", nargs="+", default=None, help="默认全部商品品种")
    parser.add_argument("--standalone", action="store_true", help="写到 minute/segments 下的独立日期段")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}-\d{8}", args.time_period):
        parser.error("--time-period 格式必须为 YYYYMMDD-YYYYMMDD")
    try:
        start, end = (pd.Timestamp(value).date() for value in args.time_period.split("-"))
    except ValueError:
        parser.error("--time-period 包含无效日期")
    if start > end or end > default_end:
        parser.error("日期倒置或晚于已登记范围")
    periods = C.split_period(start, end)
    if not periods:
        parser.error("日期窗口与已登记数据分区没有交集")
    names = list(dict.fromkeys(s.upper() for s in (args.symbols or C.COMMODITY_SYMBOLS)))
    unknown = sorted(set(names) - set(C.COMMODITY_SYMBOLS))
    if unknown:
        parser.error(f"本仓库只下载已登记的商品期货品种: {unknown}")

    credentials = rqdata_credentials()
    if credentials is None:
        parser.error("请在 config/rqdata.env 配置米筐凭证")
    import rqdatac as rq

    rq.init(*credentials)
    failures = 0
    for partition, left, right in periods:
        directory = C.minute_dir(partition, args.datapath)
        if args.standalone:
            directory = directory / "segments" / f"{left:%Y%m%d}-{right:%Y%m%d}"
        days = pd.DatetimeIndex(rq.get_trading_dates(left, right)).sort_values()
        for name in tqdm(names, desc=f"分钟 {partition}"):
            try:
                path = directory / f"{name}.parquet"
                for start, end in _missing_edges(path, partition, days):
                    frame = _download_symbol(rq, name, start, end)
                    if not frame.empty:
                        write_shard(path, partition, frame)
            except Exception as exc:
                failures += 1
                print(f"{partition}/{name} 未更新: {type(exc).__name__}: {exc}", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())

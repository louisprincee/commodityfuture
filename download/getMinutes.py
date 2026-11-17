"""从米筐下载商品期货 1 分钟行情（88 原始主力、889 复权主力、99 指数持仓/成交量），合并为单体 pickle。

    python -m download.getMinutes -t 20100101-20251231

输出 data/data_min/future_all1mdata_<起>-<止>.txt（内容为 pickle 的 DataFrame，列为 (品种, 字段)）。
已有总表时只增量下载末日之后的区间。金融期货不在下载列表中。
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config as C  # noqa: E402
from rq_auth import rqdata_credentials  # noqa: E402

DEFAULT_END = C.DEFAULT_OOS_END
SERIES = ("88", "889", "99")
TIME_TYPE = "1m"
PRICE = ["open", "close", "high", "low"]
PRICE_W = ["openw", "closew", "highw", "loww"]


def find_and_load_all(prefix: str, directory: str) -> tuple[pd.DataFrame, list[str]]:
    """加载目录中同一前缀的全部分段文件，按时间去重排序。"""
    pattern = re.compile(rf"{prefix}(\d{{8}})-(\d{{8}})\.txt")
    paths = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
    frames = []
    for path in paths:
        with open(path, "rb") as f:
            frames.append(pickle.load(f))
        print(f"已加载: {os.path.basename(path)}")
    if not frames:
        return pd.DataFrame(), paths
    data = pd.concat(frames, axis=0)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data, paths


def save_and_clean(data: pd.DataFrame, file_path: str, clean: bool = True) -> None:
    """保存数据；只删除日期区间被新文件完全覆盖的同前缀旧文件。"""
    if data.empty:
        print("无数据可保存")
        return
    file_name = os.path.basename(file_path)
    prefix = "_".join(file_name.split("_")[:-1])
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"已保存: {file_name}")
    if not clean:
        return
    new_start, new_end = re.search(r"(\d{8})-(\d{8})\.txt$", file_name).groups()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8}})-(\d{{8}})\.txt$")
    out_dir = os.path.dirname(os.path.abspath(file_path))
    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m and f != file_name and new_start <= m.group(1) and m.group(2) <= new_end:
            os.remove(os.path.join(out_dir, f))
            csv = os.path.join(out_dir, f.replace(".txt", ".csv"))
            if os.path.exists(csv):
                os.remove(csv)
            print(f"已清理旧文件: {f}")


def file_span(path: str | None) -> tuple[str | None, str | None]:
    if not path:
        return None, None
    m = re.search(r"(\d{8})-(\d{8})", path)
    return (m.group(1), m.group(2)) if m else (None, None)


def download_series(rq, names: list[str], suffix: str, start: str, end: str) -> pd.DataFrame:
    """逐品种下载一个连续序列，拼成 (品种, 字段) 双层列。"""
    parts = []
    for name in tqdm(names, desc=f"下载 {suffix}"):
        frame = rq.get_price(name + suffix, start_date=start, end_date=end, frequency=TIME_TYPE)
        if frame is None or frame.empty:
            print(f"{name}{suffix} 无数据，跳过", flush=True)
            continue
        frame.index = frame.index.get_level_values("datetime")
        frame.columns = pd.MultiIndex.from_product([[name], frame.columns])
        parts.append(frame)
    if not parts:
        raise ValueError(f"{suffix} 未下载到任何品种")
    return pd.concat(parts, axis=1)


def assemble(data: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """88 原始价 + 889 复权价（openw/closew/highw/loww）+ 99 持仓量/成交量。"""
    parts = []
    for name in names:
        if any(name not in data[s].columns.get_level_values(0) for s in SERIES):
            continue
        w = data["889"][name][PRICE].rename(columns=dict(zip(PRICE, PRICE_W)))
        v = data["99"][name][["open_interest", "volume"]].rename(
            columns={"open_interest": "open_interest99", "volume": "volume99"})
        frame = pd.concat([data["88"][name], w, v], axis=1)
        frame.columns = pd.MultiIndex.from_product([[name], frame.columns])
        parts.append(frame)
    return pd.concat(parts, axis=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_end = min(date.today() - timedelta(days=1), DEFAULT_END)
    parser.add_argument("-t", "--time-period", dest="time_period",
                        default=f"20240101-{default_end:%Y%m%d}", help="YYYYMMDD-YYYYMMDD")
    parser.add_argument("--datapath", default=str(C.MINUTE_RAW_DIR))
    parser.add_argument("--symbols", nargs="+", default=None, help="默认全部商品品种")
    parser.add_argument("--standalone", action="store_true",
                        help="只保存本次区间，不与已有总表合并，也不删除已有文件")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}-\d{8}", args.time_period):
        parser.error("--time-period 格式必须为 YYYYMMDD-YYYYMMDD")
    target_start, target_end = args.time_period.split("-")
    if target_start > target_end:
        parser.error("开始日期不能晚于结束日期")
    names = [s.upper() for s in args.symbols] if args.symbols else list(C.COMMODITY_SYMBOLS)
    financial = sorted(set(names) & set(C.FINANCIAL_SYMBOLS))
    if financial:
        parser.error(f"本仓库只下载商品期货，拒绝金融期货: {financial}")

    data_dir = os.path.abspath(args.datapath)
    os.makedirs(data_dir, exist_ok=True)
    prefix = "future_all1mdata"
    existing = None
    if not args.standalone:
        monoliths = sorted(f for f in os.listdir(data_dir)
                           if re.fullmatch(rf"{prefix}_\d{{8}}-\d{{8}}\.txt", f))
        if len(monoliths) > 1:
            parser.error(f"目录里有多个总表 {monoliths}；请用 --standalone 下载新区间")
        existing = os.path.join(data_dir, monoliths[0]) if monoliths else None
    ex_start, ex_end = file_span(existing)
    period, full_period = args.time_period, args.time_period
    if existing is not None:
        if target_end <= ex_end:
            print(f"已有数据已覆盖 {args.time_period}，无需更新")
            return 0
        nxt = (pd.Timestamp(ex_end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        period = f"{nxt}-{target_end}"
        full_period = f"{min(ex_start, target_start)}-{target_end}"
        print(f"增量下载范围: {period}")

    credentials = rqdata_credentials()
    if credentials is None:
        parser.error("请在 config/rqdata.env 填写 RQDATAC_LICENSE，或填写用户名和密码")
    import rqdatac as rq

    rq.init(*credentials)
    start, end = period.split("-")
    fetched = {s: download_series(rq, names, s, start, end) for s in SERIES}
    data = assemble(fetched, names)
    if existing is not None:
        history, _ = find_and_load_all(f"{prefix}_", data_dir)
        data = pd.concat([history, data], axis=0)
        data = data[~data.index.duplicated(keep="last")].sort_index()
    out = os.path.join(data_dir, f"{prefix}_{full_period}.txt")
    save_and_clean(data, out, clean=not args.standalone)
    print(f"下载完成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

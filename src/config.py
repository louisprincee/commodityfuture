"""商品期货数据路径、样本分区、品种表与读取守卫。"""
from __future__ import annotations

import datetime as _dt
import os as _os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(_os.environ.get("CF_DATA_ROOT", PROJECT_ROOT / "data"))
RUNS_DIR = Path(_os.environ.get("CF_RUNS_ROOT", PROJECT_ROOT / "runs"))
CONFIG_DIR = Path(_os.environ.get("CF_CONFIG_DIR", PROJECT_ROOT / "config"))

PARTITIONS = ("research", "oos")
ARCHIVE_DIR = DATA_ROOT / "_archive"
MINUTE_MONOLITH = ARCHIVE_DIR / "minute_monolith" / "future_all1mdata_20100101-20251231.txt"


def partition_dir(partition: str, data_root: Path = DATA_ROOT) -> Path:
    if partition not in PARTITIONS:
        raise HoldoutViolation(f"未知分区 {partition}")
    return Path(data_root) / partition


def daily_dir(partition: str, data_root: Path = DATA_ROOT) -> Path:
    return partition_dir(partition, data_root) / "daily"


def minute_dir(partition: str, data_root: Path = DATA_ROOT) -> Path:
    return partition_dir(partition, data_root) / "minute"


def execution_dir(partition: str, data_root: Path = DATA_ROOT) -> Path:
    return partition_dir(partition, data_root) / "execution"


RESEARCH_FEE_PATH = execution_dir("research") / "fee_history.csv"
OOS_FEE_PATH = execution_dir("oos") / "fee_history.csv"
RESEARCH_EXECUTION_PARAMETERS = execution_dir("research") / "parameters"

# 2020-01-01 前商品期货成交量、成交额、持仓量为双边统计，之后为单边；米筐日线保留原口径
SINGLE_SIDED_FROM = _dt.date(2020, 1, 1)
DOUBLE_SIDED_FIELDS = ("volume", "total_turnover", "open_interest")

RESEARCH_START = _dt.date(2010, 1, 4)
RESEARCH_END = _dt.date(2021, 12, 31)
OOS_START = _dt.date(2022, 1, 1)
DEFAULT_OOS_END = _dt.date(2025, 12, 31)
TRADING_DAYS_PER_YEAR = 242
PARTITION_WINDOWS = {
    "research": (RESEARCH_START, RESEARCH_END),
    "oos": (OOS_START, DEFAULT_OOS_END),
}


def split_period(start: _dt.date, end: _dt.date) -> list[tuple[str, _dt.date, _dt.date]]:
    periods = []
    for partition, (lower, upper) in PARTITION_WINDOWS.items():
        left, right = max(start, lower), min(end, upper)
        if left <= right:
            periods.append((partition, left, right))
    return periods

ALL_SYMBOLS = [
    "AG", "AL", "AU", "BC", "BU", "CU", "FU", "HC", "LU", "NI", "NR", "PB", "RB",
    "RU", "SC", "SN", "SP", "SS", "WR", "ZN", "A", "B", "BB", "C", "CS", "EB",
    "EG", "FB", "I", "J", "JD", "JM", "L", "LH", "M", "P", "PG", "PP", "RR", "V",
    "Y", "AP", "CF", "CJ", "CY", "FG", "JR", "LR", "MA", "OI", "PF", "PK", "PM",
    "RI", "RM", "RS", "SA", "SF", "SM", "SR", "TA", "UR", "WH", "ZC", "LC",
    "AO", "EC", "SH", "AD", "BR", "LG", "PR", "PX", "SI", "PS",
]
# 金融期货不进商品池；保留此表只为下载与校验时显式排除
FINANCIAL_SYMBOLS = ["IC", "IF", "IH", "IM", "T", "TF", "TS", "TL"]
COMMODITY_SYMBOLS = [s for s in ALL_SYMBOLS if s not in FINANCIAL_SYMBOLS]

# 板块划分，对应论文 Exhibit 2 的二级分组；EC（集运指数）不属于实物商品，单列
SECTOR = {
    **dict.fromkeys(["CU", "AL", "ZN", "PB", "NI", "SN", "BC", "AO", "AD", "SI", "LC", "PS"], "base_metals"),
    **dict.fromkeys(["AU", "AG"], "precious_metals"),
    **dict.fromkeys(["RB", "HC", "I", "J", "JM", "SF", "SM", "WR", "SS", "ZC"], "ferrous"),
    **dict.fromkeys(["SC", "FU", "LU", "BU", "PG"], "energy"),
    **dict.fromkeys(["TA", "EG", "MA", "L", "PP", "V", "EB", "PF", "SA", "UR", "FG",
                     "SH", "BR", "PX", "PR", "SP"], "chemicals"),
    **dict.fromkeys(["RU", "NR"], "softs"),
    **dict.fromkeys(["A", "B", "C", "CS", "M", "Y", "P", "OI", "RM", "RS", "WH", "PM",
                     "RI", "LR", "JR", "RR", "PK"], "grains_oilseeds"),
    **dict.fromkeys(["CF", "CY", "SR", "AP", "CJ", "JD", "BB", "FB", "LG"], "softs"),
    **dict.fromkeys(["LH"], "livestock"),
    **dict.fromkeys(["EC"], "index"),
}


class HoldoutViolation(RuntimeError):
    """日期或路径越出声明的分区。"""


def assert_research_only(path) -> None:
    """研究读取不得打开样本外或归档路径。"""
    original = Path(_os.path.abspath(path))
    p = original.resolve()
    for root in (partition_dir("oos"), ARCHIVE_DIR):
        if original.is_relative_to(root.absolute()) or p.is_relative_to(root.resolve()):
            raise HoldoutViolation(f"研究阶段禁止读取样本外或归档数据: {p}")


def assert_no_holdout_dates(values, what: str = "data") -> None:
    """日期不得晚于研究期末。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(values))
    if len(ts) and ts.max() > pd.Timestamp(RESEARCH_END):
        raise HoldoutViolation(f"{what} 含样本外日期 {ts.max().date()}")


def assert_oos_dates(values, what: str = "data") -> None:
    """样本外日期必须落在 2022-01-01 至 2025-12-31。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(values))
    if len(ts) and (ts.min() < pd.Timestamp(OOS_START) or ts.max() > pd.Timestamp(DEFAULT_OOS_END)):
        raise HoldoutViolation(f"{what} 日期越出样本外窗口: {ts.min().date()}..{ts.max().date()}")


def study_dir(name: str) -> Path:
    """研究输出：runs/research/<名称>/。"""
    return RUNS_DIR / "research" / name

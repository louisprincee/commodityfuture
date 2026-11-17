"""商品期货数据路径、样本分区、品种表与读取守卫。"""
from __future__ import annotations

import datetime as _dt
import os as _os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(_os.environ.get("CF_DATA_ROOT", PROJECT_ROOT / "data"))
RUNS_DIR = Path(_os.environ.get("CF_RUNS_ROOT", PROJECT_ROOT / "runs"))
CONFIG_DIR = Path(_os.environ.get("CF_CONFIG_DIR", PROJECT_ROOT / "config"))

# 分钟单体 pickle（扩展名 .txt，内容为 pickle），由 download/getMinutes.py 生成
MINUTE_RAW_DIR = DATA_ROOT / "data_min"
MINUTE_MONOLITH = MINUTE_RAW_DIR / "future_all1mdata_20100101-20251231.txt"
SHARD_ROOT = DATA_ROOT / "minute_shards"
RESEARCH_SHARD_DIR = SHARD_ROOT / "research"
OOS_SHARD_DIR = SHARD_ROOT / "oos"
ROLL_DIR = DATA_ROOT / "roll_dates"
UNIVERSE_DIR = DATA_ROOT / "universe"

# 米筐日频外部数据：研究期在根目录，2022–2025 在 oos/ 子目录
EXTERNAL_DIR = DATA_ROOT / "external_rqdata"
EXTERNAL_OOS_DIR = EXTERNAL_DIR / "oos"
RESEARCH_FEE_PATH = DATA_ROOT / "research" / "fee_history.csv"
OOS_FEE_PATH = DATA_ROOT / "oos" / "fee_history.csv"

RESEARCH_START = _dt.date(2010, 1, 1)
RESEARCH_END = _dt.date(2021, 12, 31)
OOS_START = _dt.date(2022, 1, 1)
DEFAULT_OOS_END = _dt.date(2025, 12, 31)
TRADING_DAYS_PER_YEAR = 242

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


def _oos_roots() -> tuple[Path, ...]:
    return (OOS_SHARD_DIR, EXTERNAL_OOS_DIR, OOS_FEE_PATH.parent)


def assert_research_only(path) -> None:
    """研究读取不得打开任何样本外路径。"""
    p = Path(path).resolve()
    for root in _oos_roots():
        r = root.resolve()
        if p == r or r in p.parents:
            raise HoldoutViolation(f"研究阶段禁止读取样本外数据: {p}")


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

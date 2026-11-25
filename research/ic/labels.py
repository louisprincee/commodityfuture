"""与账户一致的信号日、次日开盘及同合约收益标签。"""
from functools import reduce
from operator import mul
import numpy as np
import pandas as pd

def forward_open_returns(dominant: pd.Series, opens: pd.Series) -> pd.DataFrame:
    """信号 t 使用当日主力映射，t+1 开盘入场、t+2 同合约开盘退出，与账户一致。"""
    dates = dominant.index.to_series()
    entry_day = dates.shift(-1)
    exit_day = dates.shift(-2)
    contract = dominant
    entry = opens.reindex(pd.MultiIndex.from_arrays([contract, entry_day])).to_numpy(dtype=float)
    exit_price = opens.reindex(pd.MultiIndex.from_arrays([contract, exit_day])).to_numpy(dtype=float)
    valid = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_price) & (exit_price > 0)
    returns = np.divide(exit_price, entry, out=np.full(len(dates), np.nan), where=valid) - 1
    return pd.DataFrame({
        "entry_date": entry_day, "exit_date": exit_day, "entry_contract": contract,
        "entry_open": entry, "exit_open": exit_price, "forward_return": returns,
    }, index=dominant.index)


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """信号日之后连续 horizon 段主力开盘收益连乘，任何单日缺失则整段缺失。"""
    if horizon < 1:
        raise ValueError("累计收益窗口必须为正")
    legs = (1 + returns.shift(-(1 + offset)) for offset in range(horizon))
    return reduce(mul, legs) - 1


def return_matrix(contract_map: pd.DataFrame, bar_open: pd.Series) -> pd.DataFrame:
    """交易日开盘使用前一收盘主力映射；同合约到下一开盘，缺失日不跨越。"""
    dates = contract_map.index
    ids = contract_map.shift(1).to_numpy().ravel()
    entry_date = dates.to_numpy().repeat(len(contract_map.columns))
    exit_date = dates.to_series().shift(-1).to_numpy().repeat(len(contract_map.columns))
    entry = bar_open.reindex(pd.MultiIndex.from_arrays([ids, entry_date])).to_numpy()
    exit_price = bar_open.reindex(pd.MultiIndex.from_arrays([ids, exit_date])).to_numpy()
    valid = (entry > 0) & (exit_price > 0)
    ret = np.divide(exit_price, entry, out=np.full(len(entry), np.nan), where=valid) - 1
    return pd.DataFrame(ret.reshape(contract_map.shape), index=dates, columns=contract_map.columns)

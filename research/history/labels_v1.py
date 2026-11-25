"""封存差值研究的原始主力映射口径，仅用于历史复现。"""
import numpy as np
import pandas as pd


def return_matrix(contract_map: pd.DataFrame, bar_open: pd.Series) -> pd.DataFrame:
    dates = contract_map.index
    ids = contract_map.to_numpy().ravel()
    entry_date = dates.to_numpy().repeat(len(contract_map.columns))
    exit_date = dates.to_series().shift(-1).to_numpy().repeat(len(contract_map.columns))
    entry = bar_open.reindex(pd.MultiIndex.from_arrays([ids, entry_date])).to_numpy()
    exit_price = bar_open.reindex(pd.MultiIndex.from_arrays([ids, exit_date])).to_numpy()
    valid = (entry > 0) & (exit_price > 0)
    ret = np.divide(exit_price, entry, out=np.full(len(entry), np.nan), where=valid) - 1
    return pd.DataFrame(ret.reshape(contract_map.shape), index=dates, columns=contract_map.columns)

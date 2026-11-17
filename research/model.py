"""逐年扩展窗口训练 XGBoost 回归树，按标签期限清洗训练样本，输出样本外截面预测与特征重要性。"""
from __future__ import annotations

import itertools
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from research import features as F

M = F.PROTOCOL["model"]
S = F.PROTOCOL["sample"]
GRID = M["grid"]
VAL_DAYS = 121
TEST_YEARS = list(range(S["first_test_year"], S["last_test_year"] + 1))


def calendar(ds: pd.DataFrame) -> pd.Series:
    """可交易日历：日期 → 序号。"""
    dates = np.sort(ds.loc[ds["eligible"], "date"].unique())
    return pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))


def split_rows(ds: pd.DataFrame, cal: pd.Series, horizon: int, year: int) -> dict[str, np.ndarray]:
    """返回训练前段、验证段、全窗口与测试年的行掩码。标签窗口不越过各段终点。"""
    pos = ds["date"].map(cal).to_numpy(dtype=float)
    in_window = cal.index[cal.index.year < year]
    if len(in_window) == 0:
        raise ValueError(f"{year} 之前没有训练数据")
    end = cal[in_window[-1]]
    # 验证段为标签已完全实现的最后 VAL_DAYS 个特征日；训练前段标签止于验证段之前
    val_last = end - horizon
    val_start = val_last - VAL_DAYS + 1
    usable = ds["eligible"].to_numpy() & ds[f"y{horizon}"].notna().to_numpy()
    label_end = pos + horizon
    return {
        "train": usable & (label_end < val_start),
        "val": usable & (pos >= val_start) & (pos <= val_last),
        "full": usable & (pos <= val_last),
        "test": ds["eligible"].to_numpy() & (ds["date"].dt.year == year).to_numpy(),
    }


def _params(cfg: dict, seed: int, nthread: int) -> dict:
    return {
        "objective": M["objective"], "tree_method": M["tree_method"], "lambda": M["reg_lambda"],
        "max_depth": cfg["max_depth"], "eta": cfg["learning_rate"],
        "colsample_bytree": cfg["colsample_bytree"], "colsample_bylevel": cfg["colsample_bylevel"],
        "seed": seed, "nthread": nthread, "verbosity": 0,
    }


def fit_task(ds: pd.DataFrame, cal: pd.Series, horizon: int, year: int, nthread: int = 2) -> dict:
    import xgboost as xgb

    rows = split_rows(ds, cal, horizon, year)
    X = ds[F.FEATURES].to_numpy(dtype=np.float32)
    y = ds[f"y{horizon}"].to_numpy(dtype=np.float32)
    mk = lambda m, label=True: xgb.DMatrix(X[m], label=y[m] if label else None,
                                           feature_names=F.FEATURES, missing=np.nan, nthread=nthread)
    dtrain, dval = mk(rows["train"]), mk(rows["val"])
    yval = y[rows["val"]]
    max_trees = max(GRID["n_estimators"])
    results = []
    for depth, lr, cbt, cbl in itertools.product(GRID["max_depth"], GRID["learning_rate"],
                                                 GRID["colsample_bytree"], GRID["colsample_bylevel"]):
        cfg = {"max_depth": depth, "learning_rate": lr, "colsample_bytree": cbt, "colsample_bylevel": cbl}
        booster = xgb.train(_params(cfg, 0, nthread), dtrain, num_boost_round=max_trees)
        for n in GRID["n_estimators"]:
            pred = booster.predict(dval, iteration_range=(0, n))
            results.append({**cfg, "n_estimators": n, "val_mse": float(np.mean((pred - yval) ** 2))})
    grid = pd.DataFrame(results)
    best = grid.loc[grid["val_mse"].idxmin()].to_dict()
    cfg = {k: best[k] for k in ("max_depth", "learning_rate", "colsample_bytree", "colsample_bylevel")}
    cfg["max_depth"] = int(cfg["max_depth"])
    n_trees = int(best["n_estimators"])

    dfull, dtest = mk(rows["full"]), mk(rows["test"], label=False)
    preds, imps = [], []
    for seed in range(M["seeds"]):
        booster = xgb.train(_params(cfg, seed, nthread), dfull, num_boost_round=n_trees)
        preds.append(booster.predict(dtest))
        gain = pd.Series(booster.get_score(importance_type="total_gain"), dtype=float)
        gain = gain.reindex(F.FEATURES).fillna(0.0)
        imps.append(gain / gain.sum() if gain.sum() > 0 else gain)
    test = ds.loc[rows["test"], ["date", "symbol"]].copy()
    test["pred"] = np.mean(preds, axis=0)
    test["horizon"] = horizon
    return {
        "horizon": horizon, "year": year,
        "predictions": test,
        "importance": pd.concat(imps, axis=1).mean(axis=1),
        "params": {**cfg, "n_estimators": n_trees, "val_mse": best["val_mse"],
                   "val_mse_constant": float(np.mean((yval - y[rows["train"]].mean()) ** 2)),
                   "n_train": int(rows["train"].sum()), "n_val": int(rows["val"].sum()),
                   "n_full": int(rows["full"].sum()), "n_test": int(rows["test"].sum()),
                   "full_end": str(ds.loc[rows["full"], "date"].max().date())},
    }


def _task(ds, cal, horizon, year, cache: Path, nthread: int) -> Path:
    path = cache / f"H{horizon}_Y{year}.pkl"
    if path.exists():
        return path
    t0 = time.time()
    out = fit_task(ds, cal, horizon, year, nthread=nthread)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f)
    tmp.replace(path)
    print(f"H={horizon:3d} Y={year} 参数={out['params']} 用时 {time.time() - t0:.0f}s", flush=True)
    return path


def run(ds: pd.DataFrame, horizons: list[int], out_dir: Path, jobs: int = 8) -> None:
    from joblib import Parallel, delayed

    C.assert_no_holdout_dates(ds["date"], "训练数据")
    cache = out_dir / "tasks"
    cache.mkdir(parents=True, exist_ok=True)
    cal = calendar(ds)
    cols = ["date", "symbol", "eligible", *F.FEATURES, *[f"y{h}" for h in horizons]]
    slim = ds[cols]
    tasks = [(h, y) for h in horizons for y in TEST_YEARS]
    t0 = time.time()
    Parallel(n_jobs=jobs, backend="loky")(
        delayed(_task)(slim, cal, h, y, cache, max(1, 20 // jobs)) for h, y in tasks)
    collect(out_dir, horizons)
    print(f"训练完成 {len(tasks)} 个任务，用时 {(time.time() - t0) / 60:.1f} 分钟")


def collect(out_dir: Path, horizons: list[int]) -> None:
    preds, imps, params = [], [], []
    for h in horizons:
        for y in TEST_YEARS:
            with (out_dir / "tasks" / f"H{h}_Y{y}.pkl").open("rb") as f:
                out = pickle.load(f)
            preds.append(out["predictions"])
            imps.append(out["importance"].rename((h, y)))
            params.append({"horizon": h, "year": y, **out["params"]})
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet", index=False)
    imp = pd.concat(imps, axis=1).T
    imp.index = pd.MultiIndex.from_tuples(imp.index, names=["horizon", "year"])
    imp.to_csv(out_dir / "feature_importance.csv")
    pd.DataFrame(params).to_csv(out_dir / "model_params.csv", index=False)

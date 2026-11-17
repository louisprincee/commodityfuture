# Repository instructions

Read before changes. If instructions conflict with code, establish the intended behavior from the current user request and update both together.

## Scope

Chinese commodity futures only. Financial futures (IF, IH, IC, IM, T, TF, TS, TL) are excluded from download, data checks and research; `src/config.py` lists them only so scripts can reject them. The current study is the replication of Guida (2025), "Machine Learning in Commodity Futures" (`docs/` holds the PDF locally). Nothing here is a proven deployable edge.

## Samples

Research data run from 2010-01-04 to 2021-12-31. Backtests and every reported statistic start 2016-01-04; 2010–2015 serve only as feature warm-up and training data. Named OOS is 2022-01-01 to 2025-12-31 and is stored separately under `data/external_rqdata/oos/`, `data/minute_shards/oos/` and `data/oos/`. Research loaders must call `assert_research_only` before opening a path and `assert_no_holdout_dates` on loaded dates; keep both and `HoldoutViolation` in `src/config.py`. `python -m research check` may read OOS files for structure and coverage only; it computes no return statistics on OOS. Reading OOS for model evaluation requires an explicit user decision and a new declared protocol.

## Research protocol

`config/research/replication.json` is the declared protocol for the first specification (daily rebalancing, paper construction); `config/research/holding_aligned.json` is a posthoc second specification that holds each daily book for H days, uses the saved predictions unchanged, and must be described as posthoc. replication.json fixes: universe rule, 126 features, eight horizons, model grid, purge, portfolio construction, costs, baselines and hypotheses. It was fixed before results. Do not change signs, thresholds, features or the grid after seeing returns; a changed rule needs a new protocol file and must be reported as a separate, later specification. Report every declared result, including null and adverse findings.

Features at close t use only data through close t. Labels are forward returns from close t to close t+H. Training rows must have their label window end inside the training window; validation rows are the last 121 feature dates whose labels are realized. Annual retraining predicts the following calendar year only. Portfolio weights from close t earn the return from close t to close t+1 (paper timing); the one-day-delay variant is always reported beside it. Turnover includes dominant-contract roll trades.

## Data

Raw minutes, contract pickles, panels and model caches remain in local `data/` and `runs/research/panel/`, excluded from Git, as is `config/rqdata.env`. Never print, commit or move RiceQuant credentials. Download scripts in `download/` write the existing layout: `getMinutes.py` (minute monolith), `getExternal.py` (per-contract daily bars, rank1/rank2 dominant maps, roll yield, warehouse, spot), `getFeeHistory.py` (exchange fees). Daily research uses per-contract daily bars; the 88 minute series is not used for daily returns because its last-bar close differs from the daily close on about 1% of days.

Known data limits, documented in `runs/research/data_check/checks.csv`: CZCE products renamed after 2012 (MA, ZC, OI, WH, RI) lack pre-rename history; SHFE and some CZCE fees are missing before 2013–2015; a few delivery-month rows have high/low inconsistent with close.

## Files and validation

`docs/research.md` is the single hand-maintained report. Its tables must match the saved CSVs in `runs/research/replication/`; update both together when results change. Match the existing concise Chinese comments and English identifiers. Docstrings, comments and documents use formal research wording; do not name the audience or the person requesting a change. No new dependencies without asking. Keep focused synthetic tests in `tests/test_core.py`; tests never read real data.

The repository tracks `origin` (github.com/louisprincee/commodityfuture). Do not create branches or commit/push unless explicitly requested; when asked, stage only in-scope files and never `data/`, `config/rqdata.env`, `docs/*.pdf` or model caches.

Use PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider. After checks, delete repository-local `__pycache__`, pytest caches and temporary test directories, verifying resolved paths stay inside the repository. Do not touch raw data or credentials during cleanup.

## Commands

The project runs in the `factor-mining` conda environment (Python 3.12, xgboost installed there).

```bash
python -m pytest tests -p no:cacheprovider
python -m research check
python -m research panel
python -m research train --jobs 10
python -m research report
python -m research holding
python -m download.getExternal --start-date 20100101 --end-date 20251231
python -m download.getFeeHistory --partition research
python -m download.getMinutes -t 20100101-20251231
```

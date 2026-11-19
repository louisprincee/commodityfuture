# Repository instructions

Read before changes. If instructions conflict with code, establish the intended behavior from the current user request and update both together.

## Scope

Chinese commodity futures only. Financial futures (IF, IH, IC, IM, T, TF, TS, TL) are excluded from download and research. `src/config.py` lists them only so scripts can reject them. The active study computes the Xingye crowding/consistency and Kaiyuan spider-web indicators by commodity and trading date (`factors.py`), evaluates unified time-series and cross-sectional rank IC (`ic_study.py`), and evaluates the combined `risk_budget.py` book. Obsolete earlier scripts (`data_check.py`, `xs_study.py`, `event_study.py`) have been pruned from `research/`. The earlier Guida machine-learning replication has been removed from the code.

## Samples

Research data run from 2010-01-04 to 2021-12-31 under `data/research/`. Backtests and every reported statistic start 2016-01-04. The 2010–2015 period serves only as feature warm-up and training data. Named OOS is 2022-01-01 to 2025-12-31 under `data/oos/`. Both partitions use `minute/`, `daily/`, and `execution/`. Research loaders must call `assert_research_only` before opening a path and `assert_no_holdout_dates` on loaded dates. Keep both and `HoldoutViolation` in `src/config.py`. The path guard excludes all of `data/oos/` and `data/_archive/`. Reading OOS for model evaluation requires an explicit user decision and a new declared protocol.

## Research protocol

The `member-ic` study uses the t+1 dominant contract's opening price on t+1 and the same contract's opening price on t+2. It is unified with `paper-ic` on universe (`eligible` liquidity filter + member rank availability), the full 80-factor set, factor signs, forward return labels, and the 120-day minimum observation requirement (`MIN_DAYS = 120`), differing only in time-series vs cross-sectional correlation slicing. Inputs through t determine the factor. Missing factor/label pairs are excluded, and neutral event signals retain their zero values.

`research/settings.py` `XS_POSITION` fixes the liquidity universe, factor signs, windows, composite members, cost and the 2016-01-04 evaluation start before results. Commodity crowding is a reversal; consistency, the change in top-20 net-long share, and member position flow are momentum. The later blocks in that dict — five-day overlap, within-symbol z-scores, the 20% tails, the rank band, and the 10- and 60-day overlaps — are separate specifications and are not retuned after their own results. Do not change signs, windows or the composite after seeing returns; a changed rule is a separate, later specification. Report every declared book, including null and adverse findings.

Factors at close t use only member ranks and prices through close t. Weights from close t earn the dominant contract's open-to-open return from the next session; the one-extra-day delay is always reported beside it. Turnover includes dominant-contract roll trades. 2010–2015 are warm-up for trailing percentiles only and are not used to choose factors.

## Data

Raw market files, panels and model caches remain in local `data/` and `runs/research/panel/`, excluded from Git, as is `config/rqdata.env`. Never print, commit or move RiceQuant credentials. In each partition, `getMinutes.py` writes per-symbol Parquet to `minute/`, `getExternal.py` writes daily bars, dominant maps and rankings to `daily/`, `getFeeHistory.py` writes exchange fees to `execution/`, and `getExecutionParameters.py` writes trading parameters and price limits to `execution/parameters/`. The mixed-period minute monolith belongs in `data/_archive/minute_monolith/`. Daily research uses per-contract daily bars. The 88 minute last-bar close differs from the daily close on about 1% of research days.

Known data limits: CZCE products renamed after 2012 (MA, ZC, OI, WH, RI) lack pre-rename history; SHFE and some CZCE fees are missing before 2013–2015; a few delivery-month rows have high/low inconsistent with close.

## Files and validation

`docs/research.md` is the single hand-maintained report. Its tables must match the saved CSVs in `runs/research/member_ic/` and `runs/research/xs_position/`. Update both together when results change. Match the existing concise Chinese comments and English identifiers. Docstrings, comments and documents use formal research wording and omit the audience or person requesting a change. No new dependencies without asking. Keep focused synthetic tests in `tests/test_core.py`. Tests never read real data.

The repository tracks `origin` (github.com/louisprincee/commodityfuture). Do not create branches or commit/push unless explicitly requested; when asked, stage only in-scope files and never `data/`, `config/rqdata.env`, `docs/*.pdf` or model caches.

Use PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider. After checks, delete repository-local `__pycache__`, pytest caches and temporary test directories, verifying resolved paths stay inside the repository. Do not touch raw data or credentials during cleanup.

## Commands

The project runs in the `factor-mining` conda environment (Python 3.12).

```bash
python -m pytest tests -p no:cacheprovider
python -m research panel
python -m research member-ic --jobs 8
python -m research paper-ic --jobs 8
python -m research hold-five --jobs 8
python -m research risk-budget --jobs 8
python -m download.getExternal --start-date 20100101 --end-date 20251231
python -m download.getFeeHistory --partition research
python -m download.getMinutes -t 20100101-20251231
```

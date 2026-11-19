# Repository instructions

Read before changes. These principles govern factor research, strategy development, and backtests. Paths, commands, factor names, and known data gaps live in the code and in `docs/research.md`.

## Scope

Chinese commodity futures only. Financial futures appear in configuration solely so download and research scripts can reject them.

## Layers

Research computes factors and runs tests. It builds no positions and imports nothing from `strategies/` or `backtest/`.

Strategies turn existing signals into position rules.

The backtest owns the account, execution, costs, and performance. The liquidity universe and the daily panel used by all three layers live in `src/`.

Keep one current strategy. Move a retired design out of the active code and record it in the history document.

## Timing

A factor on day t uses only member ranks and prices known at the close of t.

Weights formed at the close of t first trade at the next session's open.

## Samples

2010–2015 is warm-up for trailing windows only. It is not used to choose factors or parameters.

Reported statistics and backtests start on 2016-01-04.

2022–2025 is a sealed out-of-sample partition. Read it only after an explicit request and a declared evaluation protocol. After reading it, do not revise signs, windows, composites, slot rules, or leverage from those results.

## Rules are fixed before results

Fix factor signs, windows, composite members, the liquidity threshold, and cost assumptions before seeing returns.

A rule changed after seeing results is a new specification: name it, compute it, and report it on its own.

Report every book declared in advance, including null and adverse results.

A parameter grid inside the same evaluation window documents the chosen parameter. It is not out-of-sample evidence.

## Tests of predictive content

Time-series and cross-sectional studies share the universe, the factor set, the signs, the forward-return labels, and the minimum sample. They differ only in whether the correlation is taken through time or across names.

Drop a pair when the factor or the label is missing. Keep a zero on a neutral event signal.

Apply Benjamini–Hochberg within a family declared in advance. Keep the primary family and the full-specification family separate. Do not redraw the family boundary after seeing p-values.

## Backtests

Use integer lots and historical commissions. A roll closes the old contract and opens the new one, and both trades enter turnover.

The reported case is one tick per side. Save the two-tick case in the result files and leave it out of the report tables.

The target-weight multiplier is an input before orders. Do not describe it as leverage the account has already achieved.

## Data and records

Do not invent a record the vendor does not have, and do not rewrite a published price. Never print, commit, or move the RiceQuant credentials.

`docs/research.md` is the only active report. Change a table and its CSV together.

Add no dependency without asking. Automated tests use synthetic data only and do not read real market files.

## Temporary files and caches

Run Python with `PYTHONDONTWRITEBYTECODE=1` and pytest with `-p no:cacheprovider` where possible. After validation, immediately remove repository-local caches and test temporary directories created by the task, including `__pycache__`, `.pytest_cache`, and `.pytest_tmp`. Inspect exact paths before cleanup; do not touch raw data, `runs/`, credentials, or pre-existing user files.

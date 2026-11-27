# Repository instructions

Read before changes. These principles govern factor research, strategy development, and backtests. Paths, commands, factor names, and known data gaps live in the code and in `docs/research.md`.

## Scope

Chinese commodity futures only. Financial futures appear in configuration solely so download and research scripts can reject them.

## Layers

Research computes factors and runs tests. It builds no positions and imports nothing from `strategies/` or `backtest/`.

Strategies turn existing signals into position rules.

The backtest owns the account, execution, costs, and performance. The liquidity universe and the daily panel used by all three layers live in `src/`.

Keep the active strategies available: sticky_net, persist_identity, and sticky_basis_skew. Per explicit user instruction (2026-11-26), persist_identity requires only STICKY_PERSIST_IDENTITY to be finite, without FLOW5 or baseline validity filters; its account book is PERSIST_IDENTITY. The former sticky_core variants are consolidated into sticky_basis_skew (BASIS_MOM_242, NEG_SKEW_242, STICKY_NET_SHARE), with holding days set only by HOLDING_DAYS at the top of its strategy file, default 10; non-default results use a separate holding_<days>/ folder. member_price strategies keep their source and their own results/figures only; leave them out of report tables and comparison charts. The user-authorized research-only 5/10-day flow-smoothing experiment is not an active strategy and must not read OOS. Do not retire or delete strategies based on research results without an explicit user instruction. Record genuinely retired experiments in the history document. Propose obsolete files and results for deletion, and obtain explicit user approval before deleting them.

## Timing

A factor on day t uses only member ranks and prices known at the close of t.

Weights formed at the close of t first trade at the next session's open.

## Samples

The current member review and early account specification approved on 2026-11-25 evaluate 2011-01-04–2021-12-31, with 2010 data for full trailing-window warm-up. Never fabricate early observations or unavailable historical execution parameters. Report execution-data gaps explicitly.

Preserve legacy 2016–2021 outputs separately; their legacy entry points retain the original start for reproduction. The sticky-minus-spec factor is retired under research/history/, with previous results preserved. All newly proposed member groups and sensitivities are exploratory.

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

For missing commissions, the user approved using the previous known schedule (2026-11-25). Prefer the same actual contract's last complete fee schedule; otherwise use the most recent unambiguous earlier schedule for the same underlying code. Preserve its source contract and date, and flag both carry and cross-contract estimates. Never backfill from a future fee or invent a first observation; keep published vendor records unchanged. Save this cost specification separately from the prior no-carry results.

The reported case is one tick per side. Save the two-tick case in the result files and leave it out of the report tables.

The target-weight multiplier is an input before orders. Do not describe it as leverage the account has already achieved.

The current execution specification is a non-delivery trading account. Contractual maturity and delivery dates must come from actual-contract metadata, never the last available price. Cap target inventory at five days of lagged close capacity, reducing existing excess inventory as fills permit; stop adding and start liquidation during the last five sessions before the earliest contractual/delivery-month/account-end deadline. Close orders retain capacity and price-limit constraints. If a position remains at the deadline, raise an execution failure rather than fabricate delivery or keep reporting stale account performance. Keep this execution specification separate from older results, and apply it uniformly to all active strategies and the benchmark.

## Data and records

Do not invent a record the vendor does not have, and do not rewrite a published price. Never print, commit, or move the RiceQuant credentials.

`docs/research.md` is the only active report. Change a table and its CSV together.
Limit future report content to member-position research and the evaluation of member-position signals.

The active report, its CSV views, and plots show retained factors and active accounts only. Preserve complete predeclared test-family records in each research output's history/ directory, and select displayed rows only after computing the original family corrections. Do not reinsert retired-factor explanations or figures into the active report.

Add no dependency without asking. Automated tests use synthetic data only and do not read real market files.

## Temporary files and caches

Run Python with `PYTHONDONTWRITEBYTECODE=1` and pytest with `-p no:cacheprovider` where possible. After validation, immediately remove repository-local caches and test temporary directories created by the task, including `__pycache__`, `.pytest_cache`, and `.pytest_tmp`. Inspect exact paths before cleanup; do not touch raw data, `runs/`, credentials, or pre-existing user files.

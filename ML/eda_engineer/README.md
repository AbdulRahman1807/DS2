# EDA Engineer Workspace - Stage 01

## Mission
Identify noise floor variations across seasonal demand cycles, audit telemetry reporting drift, and establish empirical distributions for bottleneck thresholds.

## Key Files & Tasks
1. [`profiler.py`](profiler.py):
   - Cardinality (events, cases, containers, carriers), activity frequencies and case coverage.
   - Milestone timing relative to `DISCHARGE`: which activities are knowable before discharge.
   - Timestamp quality ledger (sentinel years, date-only precision, null starts, ordering, `Duration` artifacts).
   - Dwell percentiles by split, container type/size and customs flags; log-dwell mode and cutoff diagnostics; univariate feature signal.
   - **Status:** Complete.

2. [`drift_analysis.py`](drift_analysis.py):
   - Extract-window analysis: the log is selected by **gate-out date**, so call-date edges are truncated. Defines a fully observable "clean band" and a per-stay observable share.
   - Weekly dwell versus the fixed training cutoffs, label mix by split, label maturity, cold-start rows.
   - Call rhythm and gap analysis, discharge weekday/hour cycles.
   - Noise floor of delayed-share and volume series (indexed by gate-out date), compared with binomial sampling noise.
   - Reporting drift (activity coverage by week against sampling noise), PSI/KS train-to-val/test shift, customs and quarantine effects.
   - **Status:** Complete.

3. [`eda_runner.py`](eda_runner.py):
   - Runs both analyzers, writes CSV tables, 10 figures and a generated `findings.md` (every number is computed, none typed by hand).
   - **Status:** Complete.

## Execution

Run `ML/data_engineer/build_pipeline.py` first (EDA reads `data/processed/` and `data/interim/`), then from the project root:

```bash
python ML/eda_engineer/eda_runner.py            # writes reports/eda/
python ML/eda_engineer/eda_runner.py --no-figures
```

Outputs under `reports/eda/`: `findings.md` (the handoff for the ML and Evaluation Engineers), `tables/*.csv`, `figures/*.png`.

## Scope notes
- EDA reads the data engineer's outputs and reports problems as findings; it does not modify `ML/data_engineer/`.
- The event log covers one terminal over about three months, so weekly and monthly cycles are the longest that can be studied.

# EDA Engineer Workspace - Container Logistics & Dwell Bottleneck Triage

## Mission
Identify noise floor variations across seasonal demand cycles, audit telemetry reporting drift, establish empirical distributions for bottleneck thresholds, enforce strict information horizon contracts, and audit the new Master Dataset (`data/master/`).

---

## Key Files & Tasks

### 1. [`profiler.py`](profiler.py):
- **Cardinality & Coverage:** Entity counts (events, cases, containers, carriers), activity frequencies, case coverage, and typical durations.
- **Milestone Timing:** Timing of all terminal activities relative to `DISCHARGE` start ($t_0=0$), distinguishing pre-discharge stowage/berthing milestones from post-discharge in-yard events.
- **Ordered Trajectories:** Full trajectory transitions ($N=55,645$ complete ordered stays) across `BAPLIE -> DISCHARGE -> STACK -> CUSTOMS_DEL -> UNSTACK_TO_TRUCK -> TRUCK_OUT` with segment and cumulative duration quantiles.
- **Data Quality Ledger:** Comprehensive audit of source data defects (sentinel 1954 years, midnight date-only timestamps, uncomputed duration strings, administrative truck-out before unstack overlaps).
- **Strict Horizon Audit (Stage 01, 32 Columns):** Field-by-field audit of all 32 processed columns at $t_{\text{pred}} = t_{\text{discharge}}$, classifying features as `PASS`, `CONDITIONAL`, `EXCLUDE_POST_DISCHARGE`, `METADATA`, or `TARGET_ONLY`.
- **Master Dataset Horizon Audit (Stage 03, 55 Columns):** Comprehensive audit of all 55 features in `data/master/master_real.parquet`, auditing 19 milestone availability timestamps (`avail_*`), normalized gross weights, and external context.
- **Gross Weight Unit Normalization Audit:** Quantifies the 18.9% (18,038 stays) unit recording anomaly where `GROSS` was logged in kilograms (>100) instead of metric tonnes (<=100), validating `gross_t`.
- **Synthetic Data Fidelity Audit:** Comparative audit of 30,000 synthetic process-recombination stays against real in-cohort stays across class representation, dwell percentiles, and segment durations.
- **Univariate Predictive Signal:** Non-parametric association rankings (Spearman rho for numerics, Kruskal-Wallis $\epsilon^2$ for categoricals) with automated arrival-eligibility filtering.

### 2. [`drift_analysis.py`](drift_analysis.py):
- **Extract-Window Truncation Analysis:** Diagnoses gate-out date extract selection (`2021-07-01` to `2021-09-30`) and establishes the unbiased **Clean Band** (`2021-07-01` to `2021-09-12`) based on the 431.4h (99th percentile) dwell horizon.
- **Observable Share Calculation:** Quantifies per-stay observability across call weeks, proving that the Test split suffers from severe right-censoring (85.4% observable share).
- **Weekly Dwell Drift & Congestion Surge:** Maps the July 12–23 operational gridlock peak (weekly p85 dwell reaching ~260h and critical share >28%) against calm August/September regimes (~110h p85).
- **Rolling Dwell Dynamics:** Correlates 7-day lagged completed departure volume and delay velocity against daily cohort median/p85 dwell (Spearman $\rho = -0.47$).
- **Vessel Call Rhythms & Thursday Gaps:** Distinguishes genuine operational liner schedules (Thursday dips) from telemetry collection failures.
- **Operational Noise Floor:** Calculates the baseline operational variation ($\sigma = \pm 2.7\text{ pp}$) of trailing 7-day delay share compared to binomial sampling noise, establishing a minimum detectable shift of 5.4 pp.
- **Reporting Drift Detection:** Monitors week-to-week milestone coverage stability across 18 activities relative to sampling noise.
- **Distribution Shift (PSI & KS):** Computes Population Stability Index and Kolmogorov-Smirnov tests across training, validation, test splits, and synthetic data.

### 3. [`eda_runner.py`](eda_runner.py):
- **End-to-End Orchestrator:** Ingests both Stage 01 processed splits (`data/processed/`, `data/interim/`) and the new Master Dataset (`data/master/`), executes all profiling and drift diagnostics, and outputs:
  - [`findings.md`](../../reports/eda/findings.md): Comprehensive, fully reproducible report where every number is computed from code.
  - 14 diagnostic figures in `reports/eda/figures/`.
  - 34 structured tabular exports in `reports/eda/tables/`.

---

## Execution

Run from the project root:

```bash
# 1. Ensure Python virtual environment is activated
source .venv/bin/activate

# 2. Run full EDA pipeline (generates tables, figures, and findings.md)
python ML/eda_engineer/eda_runner.py

# 3. Fast execution without regenerating figures
python ML/eda_engineer/eda_runner.py --no-figures
```

Outputs are automatically written to `reports/eda/`.

---

## Scope & Methodological Safeguards
- **Anti-Leakage Guarantee:** Univariate correlation rankings must never bypass prediction-time contracts. Mid-stay milestones (`JOB_DEL_DOCTYPE`, `total_events_in_stay`, `has_behandle_inspection`) represent target leakage if included in arrival-time models.
- **Local Operational Context:** Target classes reflect local terminal operational turnover percentiles ($p_{50}=65.11\text{h}$, $p_{85}=160.90\text{h}$), not commercial demurrage or tariff penalties.
- **External Data Sparsity:** Weather and disruption feeds are constant zero in 2021 and are strictly excluded from baseline triage models.

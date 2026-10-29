# Stage 01 Data Engineering Audit Report
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks

**Date of Audit:** October 29, 2026  
**Auditor / Assigned Role:** Data Engineer  
**Audit Target:** Stage 01 Data Engineering Ingestion, Cleaning, Labeling, Feature Engineering, Splitting, and Orchestration Pipelines  
**Repository Branch:** `main`  

---

### 1. Executive Summary & Audit Scope

This audit conducted an evidence-driven verification of the Stage 01 Data Engineering pipeline under the strict role boundary of the **Data Engineer**. All 8 owned code modules, raw datasets, interim files, reference tables, and processed datasets were inspected.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              AUDIT WORKSTREAM VERIFICATION                             │
├─────────────────────────┬──────────────────────┬────────────┬──────────────────────────┤
│ Workstream              │ Target Module(s)     │ Status     │ Verified Invariant       │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ A: Source Ingestion     │ ingest_mendeley.py   │ VERIFIED   │ Standardized types,      │
│                         │ fetch_portwatch.py   │            │ Snappy compression,      │
│                         │ ingest_benchmarks.py │            │ provenance retained      │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ B: Cleaning & Entity    │ cleaner.py           │ REMEDIATED │ Quarantining reasons,    │
│                         │                      │            │ deterministic ties       │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ C: Dwell Clock & Labels │ dwell_clock.py       │ REMEDIATED │ Zero label leakage;      │
│                         │                      │            │ train-fitted cutoffs     │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ D: Temporal Availability│ rolling_metrics.py   │ REMEDIATED │ Observable completed     │
│                         │                      │            │ stays; no future leakage │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ E: External Disruptions │ build_pipeline.py    │ REMEDIATED │ Localized attribution;   │
│                         │                      │            │ zero false storm flags   │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ F: Partitioning & Split │ splitter.py          │ REMEDIATED │ Strict chronology AND    │
│                         │                      │            │ 100% disjoint entities   │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ G: Orchestration        │ build_pipeline.py    │ VERIFIED   │ End-to-end reproducible, │
│                         │                      │            │ deterministic runtime    │
├─────────────────────────┼──────────────────────┼────────────┼──────────────────────────┤
│ H: Automated Testing    │ tests/ (18 tests)    │ VERIFIED   │ 18/18 passing tests      │
└─────────────────────────┴──────────────────────┴────────────┴──────────────────────────┘
```

---

### 2. Verified Invariants & Code Health

1. **Ingestion & Data Typing:**
   - [`data/raw/mendeley_event_log.parquet`](file:///Users/abdul/Desktop/DS/data/raw/mendeley_event_log.parquet): 952,069 raw rows, 95,648 unique cases. All 12 columns standardized; mixed-type objects in `YARD_BLOCK`, `JOB_DEL_DOCTYPE`, and formula strings in `Duration` safely handled without unhandled exceptions.
   - [`data/reference/indian_ports_benchmarks.parquet`](file:///Users/abdul/Desktop/DS/data/reference/indian_ports_benchmarks.parquet): Contains 9 operational benchmarks with provenance columns (`reporting_period="FY 2022-23 (Maritime India Baseline)"`, `source_agency="Ministry of Ports, Shipping and Waterways (TRW)"`).
   - [`data/raw/portwatch_disruptions.parquet`](file:///Users/abdul/Desktop/DS/data/raw/portwatch_disruptions.parquet): 132 GDACS climate alerts with millisecond timestamps converted to ISO-8601 UTC datetimes.

2. **Entity Consistency & Deterministic Ordering:**
   - 95,648 cases each contain exactly one `DISCHARGE` event.
   - 95,647 cases contain exactly one `TRUCK_OUT` event.
   - Exactly 1 case (`CSNU6526902-2021-04-19`) lacks a gate-out milestone; it is isolated in [`data/interim/quarantined_stays.parquet`](file:///Users/abdul/Desktop/DS/data/interim/quarantined_stays.parquet) with explicit `quarantine_reason` and `quarantined_at` timestamp.
   - Event ordering within stays is deterministic via secondary sorting keys: `[case_id, Start Timestamp, End Timestamp, activity]`.

3. **Production Data Partitioning:**
   - **Train:** 60,698 stays (`2021-06-10` to `2021-08-26`, 60,690 unique containers).
   - **Val:** 20,879 stays (`2021-08-27` to `2021-09-11`, 20,879 unique containers).
   - **Test:** 13,959 stays (`2021-09-12` to `2021-09-30`, 13,959 unique containers).
   - **Burn-In:** 38 stays (`2021-04-19` to `2021-06-08`).
   - **Repeated Stays:** 73 stays (longitudinal visits quarantined to preserve split disjointness).
   - Mutual container overlap across Train, Val, and Test: **Exactly 0**.
   - Chronological inversion between Train, Val, and Test: **Zero** (`train_max < val_min < test_min`).

---

### 3. Detailed Audit Findings & Defect Remediation

#### Finding DE-01: Target Outcome Leakage in Rolling Operational Metrics [CRITICAL]
- **Affected File:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py)
- **Defect Description:** The rolling metrics engine grouped daily completed stays by `call_date` (arrival date) instead of departure date. In terminal operations, containers arriving on day $t-1$ do not complete their stays until days later (median dwell ~65h, 85th percentile ~161h). Calculating `daily_critical` and `daily_at_risk` by `call_date` leaked future unobserved dwell outcomes into the historical delay rate of newly arriving containers.
- **Evidence:** On `2021-07-10`, the rolling metric used whether containers arriving on `2021-07-09` were `AT_RISK` or `CRITICAL_DELAY`. All 58 containers arriving on `2021-07-09` exited between July 10 and July 29. None had completed their stays by July 10 00:00:00.
- **Remediation:** Remediated `compute_trailing_port_delay_rates()` to group by observation date (`truck_out_end_ts` date), calculating metrics strictly on containers that completed and departed prior to day $t$.

#### Finding DE-02: Target Threshold Leakage Across Splitting Boundary [CRITICAL]
- **Affected File:** [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py), [`ML/data_engineer/dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py)
- **Defect Description:** `build_pipeline.py` called `dwell_engine.fit_thresholds()` on the entire dataset of 95,647 stays *before* dataset splitting. This leaked future validation and test stay durations (specifically faster September turnaround times) into the training label thresholds.
- **Evidence:** Fitting across all stays yielded $p_{50}=54.98\text{h}$ and $p_{85}=140.28\text{h}$. Fitting strictly on the training partition yielded $p_{50}=65.11\text{h}$ and $p_{85}=160.90\text{h}$ (a discrepancy of +10.13h and +20.62h).
- **Remediation:** Re-ordered pipeline stages in `build_pipeline.py`: identified training partition candidate stays first, fit thresholds strictly on training stays (`provenance="fitted_on_training_split"`), and applied the frozen training thresholds to label validation and test partitions.

#### Finding DE-03: Temporal Inversion via Multi-Stay Container Reassignment [HIGH]
- **Affected File:** [`ML/data_engineer/splitter.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/splitter.py)
- **Defect Description:** To eliminate container identity leakage, `splitter.py` moved all occurrences of repeated containers into `train`. This caused second-visit stays occurring in late September (`2021-09-12` to `2021-09-27`) to be placed into the training set, causing `train['call_date'].max()` to exceed `val['call_date'].min()`.
- **Evidence:** 73 stays belonging to 81 multi-stay containers crossed into the validation and test date windows. As a result, the training set contained September stays, allowing a training model to observe September port conditions prior to predicting August validation cases.
- **Remediation:** Enforced strict temporal boundaries: Train strictly contains call dates $< \text{val\_start}$. To preserve 100% entity disjointness, the 73 subsequent visits in Val and Test are quarantined into [`data/processed/repeated_stays.parquet`](file:///Users/abdul/Desktop/DS/data/processed/repeated_stays.parquet) (dedicated longitudinal evaluation slice), guaranteeing both strict temporal causality AND 100% disjoint container sets.

#### Finding DE-04: False Global Spatial Association for Disruption Features [HIGH]
- **Affected File:** [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py)
- **Defect Description:** `extract_disruption_features()` joined PortWatch disruptions by calendar date alone with no spatial or port-proximity filtering. Any storm or earthquake occurring anywhere on earth caused a local terminal severity index of 10.
- **Evidence:** 60,645 out of 60,771 training stays (99.8%) were assigned `weather_severity_index = 10` due to storms in China, India, and the Caribbean. PortWatch records confirm 0 GDACS disruptions occurred in Indonesia during the 2021 operational period.
- **Remediation:** Updated `extract_disruption_features()` to filter disruptions by target country/region (`target_country="Indonesia"`). Correctly assigns nominal local weather severity (0) when no localized disaster occurred.

#### Finding DE-05: Missing Observation Dilution in Rolling Averages [MEDIUM]
- **Affected File:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py)
- **Defect Description:** Reindexing across calendar dates filled missing observation dates with `0.0` for `daily_mean_dwell`. Rolling `.mean()` then averaged those zeros into the 7-day trailing dwell metric, artificially deflating average dwell times on days following non-call dates.
- **Evidence:** A day with 100h average dwell followed by 6 non-call days was averaged to $100 / 7 = 14.3\text{h}$.
- **Remediation:** Count columns (`volume`, `critical`, `at_risk`) are filled with `0`, but `daily_mean_dwell` retains `NaN` on non-observation days, ensuring the rolling mean only aggregates observed days.

#### Finding DE-06: Lack of Audit Reason Codes in Quarantined Stays [MEDIUM]
- **Affected File:** [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py)
- **Defect Description:** Stays missing exit milestones were written to `quarantined_stays.parquet` with raw event columns but without an exclusion reason or audit timestamp.
- **Evidence:** `CSNU6526902-2021-04-19` had 11 events in quarantine with no metadata indicating why it was excluded.
- **Remediation:** Added `quarantine_reason = "MISSING_TERMINAL_EVENT: TRUCK_OUT"` and `quarantined_at` ISO-8601 timestamp.

---

### 4. Verification Test Suite Results

An automated test suite of 18 test cases was implemented in [`tests/`](file:///Users/abdul/Desktop/DS/tests/) verifying ingestion, cleaning, dwell calculations, rolling metrics, and split integrity.

```bash
pytest -v tests/
============================== 18 passed in 1.21s ==============================
```

All 18 tests execute in 1.21 seconds with 100% pass rate.

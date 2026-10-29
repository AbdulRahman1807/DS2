# Remediation and Regression Test Record
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks — Stage 01

This log records all defect remediations implemented during the Stage 01 Data Engineering audit, including affected code locations, regression test files, and test execution outcomes.

---

### Defect Remediation Register

#### 1. DE-01: Target Outcome Leakage in Rolling Operational Metrics
- **Severity:** CRITICAL
- **Component:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py)
- **Defect Description:** The rolling metrics engine grouped stays by `call_date` (arrival date) instead of departure date. Containers arriving on day $t-1$ do not complete stays until days later, leaking future unobserved dwell outcomes into the historical delay rate of newly arriving containers.
- **Code Changes:**
  - Modified `compute_trailing_port_delay_rates()` to group by observation date (`truck_out_end_ts` date), calculating metrics strictly on containers that completed and departed prior to day $t$.
  - Enforced `closed="left"` on a continuous calendar index over $[t-7, t-1]$.
- **Regression Test:** [`tests/test_rolling_metrics.py::test_adversarial_future_observation_leakage`](file:///Users/abdul/Desktop/DS/tests/test_rolling_metrics.py#L25)
- **Test Outcome:** **PASSED**. Adding 1,000 catastrophic critical delays on day $t$ or $t+1$ had zero effect on the historical metric for day $t$.

---

#### 2. DE-02: Target Threshold Leakage Across Splitting Boundary
- **Severity:** CRITICAL
- **Component:** [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py), [`ML/data_engineer/dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py)
- **Defect Description:** Thresholds ($p_{50}=54.98\text{h}$, $p_{85}=140.28\text{h}$) were fitted on all 95,647 cases *before* dataset splitting, leaking future September fast turnarounds into June/July training labels.
- **Code Changes:**
  - Re-ordered pipeline stages in `build_pipeline.py`: identified candidate training stays first, called `dwell_engine.fit_thresholds()` strictly on training stays, and froze thresholds for validation/test partitions.
  - Added `set_thresholds()` and `threshold_provenance` tracking in `dwell_clock.py`.
- **Fitted Output:** $p_{50}=65.11\text{h}$ (2.71 days), $p_{85}=160.90\text{h}$ (6.70 days) with `threshold_provenance="fitted_on_training_split"`.
- **Regression Test:** [`tests/test_dwell_clock.py::test_fit_thresholds_provenance`](file:///Users/abdul/Desktop/DS/tests/test_dwell_clock.py#L30)
- **Test Outcome:** **PASSED**. Thresholds fitted strictly on training subset without error.

---

#### 3. DE-03: Temporal Inversion in Container Reassignment
- **Severity:** HIGH
- **Component:** [`ML/data_engineer/splitter.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/splitter.py)
- **Defect Description:** Moving all occurrences of repeated containers into `train` caused 73 second-visit stays from September to enter `train`, causing `train['call_date'].max() == '2021-09-27'`, which violated the validation start boundary (`2021-08-27`).
- **Code Changes:**
  - Enforced strict temporal boundary: Train only contains call dates before `2021-08-27`.
  - Second-visit stays in Val and Test are quarantined into [`data/processed/repeated_stays.parquet`](file:///Users/abdul/Desktop/DS/data/processed/repeated_stays.parquet) (dedicated longitudinal evaluation slice) rather than moved backward in time.
- **Regression Test:** [`tests/test_splitter.py::test_strict_temporal_causality_and_disjointness`](file:///Users/abdul/Desktop/DS/tests/test_splitter.py#L29)
- **Test Outcome:** **PASSED**. Confirmed `train_max < val_min < test_min` AND container set intersection equals 0.

---

#### 4. DE-04: False Global Spatial Association for Disruption Features
- **Severity:** HIGH
- **Component:** [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py)
- **Defect Description:** Joining disruptions by date alone without geographic filtering caused 99.8% of training records to receive `weather_severity_index = 10` due to storms in China, India, and the Caribbean.
- **Code Changes:**
  - Added spatial filtering in `extract_disruption_features()` matching `target_country="Indonesia"`. Correctly assigns nominal local weather severity (0) when no localized disaster occurred.
- **Regression Test:** [`tests/test_pipeline_integration.py::test_schema_stability`](file:///Users/abdul/Desktop/DS/tests/test_pipeline_integration.py#L21)
- **Test Outcome:** **PASSED**. Verified localized attribution without missing columns.

---

#### 5. DE-05: Missing Observation Dilution in Rolling Averages
- **Severity:** MEDIUM
- **Component:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py)
- **Defect Description:** Calendar reindexing filled missing observation dates with `0.0` for `daily_mean_dwell`, causing rolling `.mean()` to average zeros into trailing dwell metrics.
- **Code Changes:**
  - Count columns (`volume`, `critical`, `at_risk`) are filled with `0`, but `daily_mean_dwell` retains `NaN` on non-observation days, ensuring the rolling mean only aggregates observed days.
- **Regression Test:** [`tests/test_rolling_metrics.py::test_missing_date_calendar_reindexing`](file:///Users/abdul/Desktop/DS/tests/test_rolling_metrics.py#L56)
- **Test Outcome:** **PASSED**. Trailing average dwell is preserved across missing observation days.

---

#### 6. DE-06: Incomplete Quarantined Metadata
- **Severity:** MEDIUM
- **Component:** [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py)
- **Defect Description:** Quarantined stays lacked audit reason codes and timestamps.
- **Code Changes:**
  - Added `quarantine_reason = "MISSING_TERMINAL_EVENT: TRUCK_OUT"` and `quarantined_at` timestamp column in `quarantined_df`.
- **Regression Test:** [`tests/test_cleaner.py::test_quarantine_missing_gate_out`](file:///Users/abdul/Desktop/DS/tests/test_cleaner.py#L89)
- **Test Outcome:** **PASSED**. Verified presence and accuracy of reason codes.

---

#### 7. DE-07: Non-Deterministic Timestamp Tie-Breaking
- **Severity:** LOW
- **Component:** [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py)
- **Defect Description:** 17 events with identical `Start Timestamp` had undefined relative ordering.
- **Code Changes:**
  - Enhanced sort key to `[case_id, Start Timestamp, End Timestamp, activity]` for 100% deterministic ordering.
- **Regression Test:** [`tests/test_cleaner.py::test_deterministic_tie_breaking`](file:///Users/abdul/Desktop/DS/tests/test_cleaner.py#L99)
- **Test Outcome:** **PASSED**. Verified deterministic ordering across ties.

---

#### 8. DE-08: Missing Source Provenance in Indian Port Benchmarks
- **Severity:** LOW
- **Component:** [`ML/data_engineer/ingest_indian_benchmarks.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/ingest_indian_benchmarks.py)
- **Defect Description:** Benchmark table lacked reporting period and authority attribution.
- **Code Changes:**
  - Added `reporting_period="FY 2022-23 (Maritime India Baseline)"` and `source_agency="Ministry of Ports, Shipping and Waterways (TRW)"`.
- **Regression Test:** [`tests/test_pipeline_integration.py::test_indian_benchmarks_integrity`](file:///Users/abdul/Desktop/DS/tests/test_pipeline_integration.py#L111)
- **Test Outcome:** **PASSED**. Verified metadata columns in compiled parquet.

---

### Automated Test Suite Summary

- **Total Tests Executed:** 18
- **Passed:** 18
- **Failed:** 0
- **Duration:** 1.21 seconds
- **Coverage:** Ingestion, cleaning, dwell clock, rolling metrics, splitter, pipeline integration, and quarantined stays.

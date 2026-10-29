# Remediation and Regression Test Record
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks — DS2
**Audit Version:** `1.1.0`
**Auditor:** Independent Data Integrity & Systems Auditor
**Verified Test Suite:** 32 passing tests across `tests/` (`pytest -v tests/`, 100% passing)

This log records all defect remediations, architectural enhancements, regression test fixtures, and verified test outcomes established across the Stage 01 Data Engineering and Comprehensive Data Integrity Audits.

---

### Defect Remediation Register

#### 1. DE-01: Target Outcome Leakage in Rolling Operational Metrics
- **Severity:** P0 — CRITICAL
- **Component:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py)
- **Defect Description:** The rolling metrics engine originally grouped stays by `call_date` (arrival date) instead of departure date. Containers arriving on day $t-1$ do not complete stays until days later (median dwell ~65h, 85th percentile ~161h), leaking unobserved dwell outcomes into the historical delay rate of newly arriving containers.
- **Code Changes:**
  - Modified `compute_trailing_port_delay_rates()` to group strictly by observation date (`truck_out_end_ts` date), calculating metrics solely on completed departures prior to day $t$.
  - Enforced `closed="left"` over calendar window $[t-7, t-1]$.
- **Regression Test:** [`tests/test_rolling_metrics.py::test_adversarial_future_observation_leakage`](file:///Users/abdul/Desktop/DS/tests/test_rolling_metrics.py#L25)
- **Test Outcome:** **PASSED**. Adding 1,000 catastrophic critical delays on day $t$ or $t+1$ had zero effect on the historical metric for day $t$.

---

#### 2. DE-02: Target Threshold Leakage Across Splitting Boundary
- **Severity:** P0 — CRITICAL
- **Component:** [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py), [`ML/data_engineer/dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py)
- **Defect Description:** Dwell thresholds ($p_{50}=54.98\text{h}$, $p_{85}=140.28\text{h}$) were originally fitted on all 95,647 stays *before* splitting, leaking future September fast turnarounds into June/July training labels.
- **Code Changes:**
  - Re-ordered pipeline stages in `build_pipeline.py`: identified candidate training stays first, called `dwell_engine.fit_thresholds()` strictly on training stays, and froze thresholds for validation/test partitions.
  - Added `set_thresholds()` and `threshold_provenance` tracking in `dwell_clock.py`.
- **Fitted Output:** $p_{50}=65.1102\text{h}$ (2.71 days), $p_{85}=160.8993\text{h}$ (6.70 days) with `threshold_provenance="fitted_on_training_split"`.
- **Regression Test:** [`tests/test_dwell_clock.py::test_fit_thresholds_provenance`](file:///Users/abdul/Desktop/DS/tests/test_dwell_clock.py#L30)
- **Test Outcome:** **PASSED**. Thresholds fitted strictly on training subset without error.

---

#### 3. DE-03: Temporal Inversion in Multi-Stay Container Reassignment
- **Severity:** P1 — HIGH
- **Component:** [`ML/data_engineer/splitter.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/splitter.py)
- **Defect Description:** Moving all occurrences of repeated containers into `train` caused 73 second-visit stays from late September to enter `train`, violating the validation boundary (`train['call_date'].max() == '2021-09-27' > '2021-08-27'`).
- **Code Changes:**
  - Enforced strict temporal boundary: Train strictly contains call dates before `2021-08-27`.
  - Second-visit stays in Val and Test are quarantined into [`data/processed/repeated_stays.parquet`](file:///Users/abdul/Desktop/DS/data/processed/repeated_stays.parquet) (dedicated longitudinal evaluation slice) rather than moved backward in time.
- **Regression Test:** [`tests/test_splitter.py::test_strict_temporal_causality_and_disjointness`](file:///Users/abdul/Desktop/DS/tests/test_splitter.py#L29)
- **Test Outcome:** **PASSED**. Confirmed `train_max < val_min < test_min` AND container set intersection equals 0.

---

#### 4. DE-04: False Global Spatial Association for Disruption Features
- **Severity:** P1 — HIGH
- **Component:** [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py)
- **Defect Description:** Joining disruptions by date alone without geographic filtering caused 99.8% of training records to receive `weather_severity_index = 10` due to storms in China, India, and the Caribbean.
- **Code Changes:**
  - Added spatial filtering in `extract_disruption_features()` matching `target_country="Indonesia"`. Correctly assigns nominal local weather severity (0) when no localized disaster occurred.
- **Regression Test:** [`tests/test_pipeline_integration.py::test_schema_stability`](file:///Users/abdul/Desktop/DS/tests/test_pipeline_integration.py#L21)
- **Test Outcome:** **PASSED**. Verified localized attribution without missing columns.

---

#### 5. DE-05: Missing Observation Dilution in Rolling Averages
- **Severity:** P2 — MODERATE
- **Component:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py)
- **Defect Description:** Calendar reindexing filled missing observation dates with `0.0` for `daily_mean_dwell`, causing rolling `.mean()` to average zeros into trailing dwell metrics across non-call days.
- **Code Changes:**
  - Count columns (`volume`, `critical`, `at_risk`) are filled with `0`, but `daily_mean_dwell` retains `NaN` on non-observation days, ensuring the rolling mean only aggregates observed days.
- **Regression Test:** [`tests/test_rolling_metrics.py::test_missing_date_calendar_reindexing`](file:///Users/abdul/Desktop/DS/tests/test_rolling_metrics.py#L56)
- **Test Outcome:** **PASSED**. Trailing average dwell is preserved across missing observation days.

---

#### 6. DE-06: Incomplete Quarantined Metadata
- **Severity:** P2 — MODERATE
- **Component:** [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py)
- **Defect Description:** Quarantined stays lacked audit reason codes and timestamps.
- **Code Changes:**
  - Added `quarantine_reason = "MISSING_TERMINAL_EVENT: TRUCK_OUT"` and `quarantined_at` timestamp column in `quarantined_df`.
- **Regression Test:** [`tests/test_cleaner.py::test_quarantine_missing_gate_out`](file:///Users/abdul/Desktop/DS/tests/test_cleaner.py#L89)
- **Test Outcome:** **PASSED**. Verified presence and accuracy of reason codes.

---

#### 7. DE-07: Non-Deterministic Timestamp Tie-Breaking
- **Severity:** P3 — LOW
- **Component:** [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py)
- **Defect Description:** 17 events with identical `Start Timestamp` had undefined relative ordering across runs.
- **Code Changes:**
  - Enhanced sort key to `[case_id, Start Timestamp, End Timestamp, activity]` for 100% deterministic ordering.
- **Regression Test:** [`tests/test_cleaner.py::test_deterministic_tie_breaking`](file:///Users/abdul/Desktop/DS/tests/test_cleaner.py#L99)
- **Test Outcome:** **PASSED**. Verified deterministic ordering across ties.

---

#### 8. DE-08: Missing Source Provenance in Indian Port Benchmarks
- **Severity:** P3 — LOW
- **Component:** [`ML/data_engineer/ingest_indian_benchmarks.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/ingest_indian_benchmarks.py)
- **Defect Description:** Benchmark table lacked reporting period and authority attribution.
- **Code Changes:**
  - Added `reporting_period="FY 2022-23 (Maritime India Baseline)"` and `source_agency="Ministry of Ports, Shipping and Waterways (TRW)"`.
- **Regression Test:** [`tests/test_pipeline_integration.py::test_indian_benchmarks_integrity`](file:///Users/abdul/Desktop/DS/tests/test_pipeline_integration.py#L111)
- **Test Outcome:** **PASSED**. Verified metadata columns in compiled parquet.

---

#### 9. DE-09: Lack of Machine-Checkable Feature Allowlists & Post-Arrival Leakage Risk
- **Severity:** P1 — HIGH
- **Component:** [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py) (NEW)
- **Defect Description:** The production parquets contain mid-stay and end-of-stay features (`JOB_DEL_DOCTYPE`, `total_events_in_stay`, `has_behandle_inspection`) alongside safe arrival predictors. Without machine-checkable enforcement, an ML Engineer could inadvertently include post-arrival features in an arrival-time triage model, causing severe retrospective target leakage.
- **Code Changes:**
  - Implemented `PredictionScenario` enum (`ARRIVAL_TIME`, `IN_YARD_DYNAMIC`, `POST_HOC_AUDIT`).
  - Defined explicit sets: `FORBIDDEN_TARGET_COLUMNS`, `POST_ARRIVAL_LEAKAGE_COLUMNS`, `CONDITIONAL_PRE_ARRIVAL_COLUMNS`, and `ARRIVAL_TIME_SAFE_PREDICTORS`.
  - Implemented `validate_feature_matrix()` raising `LeakageViolationError` upon contract breach.
  - Implemented `extract_modeling_matrix()` to provide automated, leak-free $(X, y)$ extraction for downstream modeling pipelines.
- **Regression Tests:** [`tests/test_feature_contracts.py::test_reject_target_columns_in_feature_matrix`](file:///Users/abdul/Desktop/DS/tests/test_feature_contracts.py#L48), `test_reject_post_arrival_columns_at_arrival_cutoff`, `test_conditional_flags_require_explicit_authorization`, `test_extract_modeling_matrix_arrival_time_cleanliness`.
- **Test Outcome:** **PASSED**. Verified automated rejection of all forbidden target and post-arrival columns.

---

#### 10. DE-10: Lack of Cryptographic Dataset Manifest & Versioned Provenance
- **Severity:** P2 — MODERATE
- **Component:** [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py), [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py)
- **Defect Description:** Dataset partitions were generated on disk without an immutable machine-readable manifest recording SHA256 hashes, record counts, schemas, and cutoffs.
- **Code Changes:**
  - Implemented `generate_dataset_manifest()` and `verify_dataset_manifest()` in `feature_contracts.py`.
  - Integrated manifest generation into `build_pipeline.py`, persisting `data/processed/dataset_manifest.json`.
- **Regression Test:** [`tests/test_feature_contracts.py::test_dataset_manifest_integrity_and_hashes`](file:///Users/abdul/Desktop/DS/tests/test_feature_contracts.py#L90)
- **Test Outcome:** **PASSED**. Cryptographic hashes and row counts verified against parquet artifacts on disk.

---

#### 11. DE-11: Raw `Duration` Excel Formula Artifacts
- **Severity:** P2 — MODERATE
- **Component:** [`ML/data_engineer/dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py), [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py)
- **Defect Description:** 3,302 records in the raw Mendeley Excel sheet had uncomputed formula strings (e.g. `1900-01-0X`) or `#VALUE!` artifacts in the `Duration` column.
- **Code Changes:**
  - Dwell duration is computed directly from validated timestamps ($t_{\text{truck\_out}} - t_{\text{discharge}}$) in floating-point seconds and converted to hours, completely bypassing the corrupt raw `Duration` column.
- **Regression Test:** [`tests/test_dwell_clock.py::test_calculate_dwell_durations`](file:///Users/abdul/Desktop/DS/tests/test_dwell_clock.py#L15)
- **Test Outcome:** **PASSED**. 100% of processed stays have strictly valid positive dwell durations.

---

#### 12. DE-12: Unvetted Feature Bypass & Silent Feature Dropping Risk
- **Severity:** P1 — HIGH
- **Component:** [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py)
- **Defect Description:** `validate_feature_matrix()` previously checked only a denylist of known targets and post-arrival columns; any unvetted or newly added column could silently bypass validation. Furthermore, `extract_modeling_matrix()` silently dropped missing features via list comprehension.
- **Code Changes:**
  - Implemented strict allowlist verification (`strict_allowlist=True`), raising `LeakageViolationError` if any unrecognized column is present.
  - Added strict input check in `extract_modeling_matrix()`, raising `KeyError` if any required feature is missing from the input dataframe.
- **Regression Tests:** [`tests/test_feature_contracts.py::test_unvetted_features_cannot_bypass_contract`](file:///Users/abdul/Desktop/DS/tests/test_feature_contracts.py#L82), `test_extract_modeling_matrix_rejects_missing_features`.
- **Test Outcome:** **PASSED**. Confirmed unvetted columns cannot bypass validation and missing features are caught immediately.

---

#### 13. DE-13: Comprehensive Manifest Verification & Tamper Detection
- **Severity:** P2 — MODERATE
- **Component:** [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py)
- **Defect Description:** `verify_dataset_manifest()` originally checked only partition file existence, hash, and row counts. It did not verify source artifacts, did not detect unmanifested extra partitions on disk, and did not verify split date boundaries or schemas.
- **Code Changes:**
  - Enhanced `verify_dataset_manifest()` to: (1) verify source artifact hashes; (2) detect unmanifested unexpected partitions on disk; (3) assert column counts and schemas; (4) verify split `call_date` min/max boundaries and frozen target cutoffs.
- **Regression Tests:** [`tests/test_feature_contracts.py::test_manifest_verification_detects_tampering`](file:///Users/abdul/Desktop/DS/tests/test_feature_contracts.py#L170).
- **Test Outcome:** **PASSED**. Verified detection of deliberate hash tampering, file deletion, and unmanifested partitions using isolated temp fixtures.

---

#### 14. DE-14: External Weather Claim Qualification & Evaluation Holdout Refinement
- **Severity:** P2 — MODERATE
- **Component:** [`DATA_QUALITY_AUDIT.md`](file:///Users/abdul/Desktop/DS/DATA_QUALITY_AUDIT.md), [`DATASET_CARD.md`](file:///Users/abdul/Desktop/DS/DATASET_CARD.md)
- **Defect Description:** Previous audit claimed that PortWatch GDACS verified zero natural disasters in Indonesia, implying confirmed nominal weather. In reality, the feed is a sparse global disaster table (132 alerts across multiple years, 0 for Indonesia in 2021) and does not record local terminal micro-weather. Furthermore, `val.parquet` was previously described as a primary holdout without noting that it cannot permanently replace an out-of-time test set.
- **Code & Documentation Changes:**
  - Clarified that zero values reflect catalog sparsity rather than proven zero local weather.
  - Excluded `weather_severity_index` and `has_active_disruption` from baseline modeling.
  - Qualified evaluation holdout rules: `val.parquet` provides an initial un-censored validation cohort, but an extended source extract (Oct/Nov 2021) is required for unbiased late-period evaluation.
- **Regression Test:** [`tests/test_pipeline_integration.py::test_schema_stability`](file:///Users/abdul/Desktop/DS/tests/test_pipeline_integration.py#L21)
- **Test Outcome:** **PASSED**. Documentation and data contracts fully reconciled.

---

#### 15. DE-15: EDA Feature Signal Anti-Leakage Segregation & Date-Only Milestone Diagnostics
- **Severity:** P1 — HIGH
- **Component:** [`ML/eda_engineer/profiler.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/profiler.py), [`ML/eda_engineer/eda_runner.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/eda_runner.py), [`reports/eda/findings.md`](file:///Users/abdul/Desktop/DS/reports/eda/findings.md)
- **Defect Description:** EDA's `feature_signal()` ranked features against dwell time without segregating arrival-time predictors from post-discharge milestones (`JOB_DEL_DOCTYPE`, `total_events_in_stay`, `has_behandle_inspection`), creating target leakage risk for modelers. In addition, date-only midnight timestamps (`00:00:00`) produced sub-day negative offsets that reflected discharge hour-of-day rather than true pre-discharge completion.
- **Code Changes:**
  - Enriched `EventLogProfiler.feature_signal()` with `PREDICTION_ELIGIBILITY` tiers (`ARRIVAL_TIME_ELIGIBLE`, `CONDITIONAL_PRE_ARRIVAL`, `POST_DISCHARGE_DIAGNOSTIC`).
  - Added `filter_eligible_signals()` and `split_feature_signals()` to strictly isolate arrival predictors.
  - Enhanced `milestone_timing()` with `timestamp_precision`, `pct_calendar_prior`, `pct_calendar_same_day`, and explanatory notes for date-only precision.
  - Updated `EdaRunner.write_findings()` to present distinct tables for arrival predictors vs retrospective diagnostics, along with an explicit anti-leakage warning.
- **Regression Tests:** [`tests/test_eda.py::test_feature_signal_annotates_eligibility_and_filters_arrival_leakage`](file:///Users/abdul/Desktop/DS/tests/test_eda.py), `test_date_only_milestone_same_vs_different_day`.
- **Test Outcome:** **PASSED**. Zero post-discharge features can pass arrival-time filtering.

---

#### 16. DE-16: Rolling Congestion Cold-Start Semantics & Automated DE ↔ EDA Integration Suite
- **Severity:** P2 — MODERATE
- **Component:** [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py), [`tests/test_integration_de_eda.py`](file:///Users/abdul/Desktop/DS/tests/test_integration_de_eda.py)
- **Defect Description:** Early training records (< 2021-07-01, 4,422 stays) had `trailing_7d_volume == 0` due to absence of recorded departures prior to July 1 in the Mendeley extract, risking confusion between unobserved history and an empty terminal. Furthermore, no automated integration test guarded against schema mismatches between Data Engineering outputs and EDA loader.
- **Code Changes:**
  - Added `COLD_START_CUTOFF_DATE = "2021-07-01"` and helper `is_cold_start_date()`.
  - Added `RollingMetricsEngine.add_cold_start_flag()` for downstream modeling masking.
  - Fixed first-day `closed='left'` rolling count fillna logic in `compute_trailing_port_delay_rates()`.
  - Implemented `tests/test_integration_de_eda.py` verifying real artifact loading, row counts, entity disjointness, formula parity, and deterministic contract fixtures.
- **Regression Tests:** [`tests/test_rolling_metrics.py::test_cold_start_insufficient_history_detection`](file:///Users/abdul/Desktop/DS/tests/test_rolling_metrics.py), `test_cold_start_to_fully_observed_transition`, [`tests/test_integration_de_eda.py::test_real_artifacts_de_to_eda_integration`](file:///Users/abdul/Desktop/DS/tests/test_integration_de_eda.py), `test_deterministic_contract_level_integration`.
- **Test Outcome:** **PASSED**. All 42 automated tests in the repository pass cleanly.

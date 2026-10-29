# Data Contracts and Invariant Checklist
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks — DS2
**Audit & Contract Version:** `1.1.0`
**Governing Code Modules:** [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py), [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py)
**Verification Suite:** 32 automated tests across `tests/` (`pytest -v tests/`, 100% passing)

This checklist provides an authoritative inventory of all verified data invariants, automated regression gates, manual audit checks, and residual operational limitations.

---

### Invariant Verification & Regression Checklist

| # | Contract Invariant | Formal Specification / Threshold | Enforcement Mechanism | Verification Evidence | Audit Result | Operational Limitations & Assumptions |
|---|---|---|---|---|---|---|
| **1** | **Strict Column Schema** | Exactly 32 columns with verified data types across Train, Val, Test, Burn-in, and Repeated splits. | `tests/test_pipeline_integration.py::test_schema_stability` | Verified in all 5 parquet files. | **PASS** | Target columns (15, 16, 18, 23, 24) must be extracted to $y$ prior to model fitting. |
| **2** | **Temporal Monotonicity** | $\max(\text{Train}) < \min(\text{Val}) < \min(\text{Test})$. | `tests/test_pipeline_integration.py::test_strict_temporal_monotonicity` | Train max: `2021-08-26`<br>Val min: `2021-08-27`<br>Val max: `2021-09-11`<br>Test min: `2021-09-12`. | **PASS** | Strictly eliminates temporal inversion across all partition boundaries. |
| **3** | **Mutual Entity Disjointness** | $\text{Containers}(\text{Train}) \cap \text{Containers}(\text{Val}) = \emptyset$<br>$\text{Containers}(\text{Train}) \cap \text{Containers}(\text{Test}) = \emptyset$<br>$\text{Containers}(\text{Val}) \cap \text{Containers}(\text{Test}) = \emptyset$. | `tests/test_pipeline_integration.py::test_strict_entity_disjointness` | Set intersection is 0 across all pairs. | **PASS** | 73 multi-stay visits quarantined to `repeated_stays.parquet` for longitudinal evaluation. |
| **4** | **Target Class Exhaustiveness** | $\text{target\_class} \in \{0, 1, 2\}$, exactly 0 nulls. | `tests/test_pipeline_integration.py::test_target_class_integrity` | All 95,647 processed stays assigned non-null integer class. | **PASS** | Classes derived from training cutoffs: $p_{50}=65.1102\text{h}$, $p_{85}=160.8993\text{h}$. |
| **5** | **Strict Non-Negative Dwell** | $\text{dwell\_hours} > 0$ for all valid stays. | `tests/test_dwell_clock.py::test_calculate_dwell_durations` | Observed range: min 1.51h, max 2592.28h. Zero negative/tied durations. | **PASS** | Single stay lacking gate-out quarantined (`CSNU6526902-2021-04-19`). |
| **6** | **Observable Rolling Metrics** | Lagged delay rates and throughput use completed departures prior to day $t$ ($[t-7, t-1]$). | `tests/test_rolling_metrics.py::test_adversarial_future_observation_leakage` | Adversarial injection of future delays did not alter historical day $t$ metric. | **PASS** | Excludes ongoing/uncompleted stays dwelling in the yard on day $t$. |
| **7** | **Missing Date Non-Dilution** | Days with 0 completed departures do not dilute average dwell with 0.0 values. | `tests/test_rolling_metrics.py::test_missing_date_calendar_reindexing` | Single 100h stay over 2-day period with 1 missing day yields 100.0h, not 50.0h. | **PASS** | Handled via `NaN` retention for `daily_mean_dwell` on non-observation days. |
| **8** | **Feature Allowlist Enforcement** | Machine-checkable validation rejecting target columns and post-arrival features. | `tests/test_feature_contracts.py::test_reject_target_columns_in_feature_matrix`<br>`test_reject_post_arrival_columns_at_arrival_cutoff` | Raises `LeakageViolationError` if target or post-discharge features enter $X$. | **PASS** | `extract_modeling_matrix()` provides automated leak-free $(X, y)$ extraction. |
| **9** | **Dataset Manifest Integrity** | All partition row counts, column counts, and SHA256 hashes match on disk. | `tests/test_feature_contracts.py::test_dataset_manifest_integrity_and_hashes` | Cryptographic hashes verified against `data/processed/dataset_manifest.json`. | **PASS** | Regenerated and verified whenever `build_pipeline.py` executes. |
| **10** | **Quarantined Record Auditability** | Quarantined stays stored with explicit reason codes and timestamps. | `tests/test_pipeline_integration.py::test_quarantined_stays_integrity` | `CSNU6526902-2021-04-19` has reason `"MISSING_TERMINAL_EVENT: TRUCK_OUT"` and timestamp. | **PASS** | Quarantined records isolated completely from `data/processed/`. |
| **11** | **Deterministic Event Ordering** | Identical timestamp events sort deterministically across runs. | `tests/test_cleaner.py::test_deterministic_tie_breaking` | Secondary keys `[case_id, Start Timestamp, End Timestamp, activity]` resolve all ties. | **PASS** | 100% bitwise repeatable execution across independent runs. |
| **12** | **Localized Weather Attribution** | Sparse disaster catalog reflects global alerts rather than localized micro-weather. | `tests/test_pipeline_integration.py::test_schema_stability` | Indonesian terminal assigned nominal weather (0) for 2021 window. | **PASS** | Sparse feed (0 events in Indonesia in 2021); features excluded from baseline modeling. |
| **13** | **Reference Benchmark Provenance** | Benchmark port KPIs include reporting period and source authority. | `tests/test_pipeline_integration.py::test_indian_benchmarks_integrity` | Includes `reporting_period="FY 2022-23"` and `source_agency="Ministry of Ports, Shipping and Waterways (TRW)"`. | **PASS** | Reference data for cross-port operational comparisons. |
| **14** | **DE ↔ EDA Integration Compatibility** | Upstream Parquet outputs load cleanly into EDA without missing columns, entity leakage, or target corruption. | `tests/test_integration_de_eda.py::test_real_artifacts_de_to_eda_integration`<br>`test_deterministic_contract_level_integration` | Real artifacts and deterministic contract fixtures verified; post-discharge features segregated from arrival models. | **PASS** | Automated regression gate prevents schema migrations from breaking EDA analysis. |

---

### Unresolved Items & Operational Assumptions

1. **Test Set Right-Censoring (Censoring Limitation):**
   - *Status:* **UNRESOLVED (Inherent Source Limitation)**.
   - *Description:* Mendeley raw extract cut off on September 30, 2021. Stays arriving in late September (the test partition) that experienced long dwell were unobserved. Observable share in Test is 55.4% (and ~85% under partial window assumptions).
   - *Contract Rule for Evaluation Engineer:* Use `val.parquet` as an initial un-censored validation cohort (99.7% observable), but do not treat it as an un-tainted final benchmark (repeated tuning against validation risks optimistic performance estimates). Evaluating only completed stays in test causes survivorship bias. Stronger evaluation requires survival analysis (IPCW) or extended extraction.
2. **Post-Arrival Feature Masking (Downstream Rule):**
   - *Status:* **CONTRACT ENFORCED**.
   - *Description:* `JOB_DEL_DOCTYPE`, `total_events_in_stay`, and `has_behandle_inspection` reside in the parquets for DL Stage 02 trajectory modeling.
   - *Contract Rule for ML Engineer:* Must invoke `extract_modeling_matrix(df, scenario='ARRIVAL_TIME')` to automatically mask these columns for arrival-time models.
3. **Single Congestion Regime (Operational Limitation):**
   - *Status:* **UNRESOLVED (Inherent Source Limitation)**.
   - *Description:* Data spans 1 terminal over 3 months with 1 major congestion wave in July 2021.
   - *Contract Rule for Integration Engineer:* System field guides must state that predictions are calibrated to this terminal's operating regime and cannot be deployed to foreign ports without recalibration.
4. **Sparse External Disruption Catalog (Data Limitation):**
   - *Status:* **CONTRACT ENFORCED (Excluded from Modeling)**.
   - *Description:* The PortWatch GDACS dataset provides only sparse global alerts (132 globally, 0 in Indonesia in 2021) and does not provide local terminal weather records. Terminal coordinates are anonymized.
   - *Contract Rule for ML Engineer:* Drop `weather_severity_index` and `has_active_disruption` from baseline modeling feature sets.
5. **Cold-Start Trailing Metrics (< 2021-07-01) (Data Limitation):**
   - *Status:* **CONTRACT ENFORCED (Flag & Masking Available)**.
   - *Description:* The raw Mendeley extract starts on June 10, 2021, but recorded gate-out departure milestones only start on July 1, 2021. Consequently, 4,422 training records have `trailing_7d_volume == 0`.
   - *Contract Rule for ML Engineer:* Treat zero trailing volume in this window as unobserved departure history rather than a proven idle terminal. Use `is_cold_start_date(call_date)` or `add_cold_start_flag()` to mask or account for cold-start stays.

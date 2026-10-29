# Data Contracts and Invariant Checklist
## Stage 01 — Machine Learning Data Engineering Contract

This checklist verifies all contract invariants guaranteed by the Data Engineer to downstream squad roles (EDA Engineer, ML Engineer, Evaluation Engineer, Integration Engineer).

---

### Invariant Verification Checklist

| Contract Invariant | Specification / Threshold | Automated Test | Verification Evidence | Result | Limitations / Assumptions |
|---|---|---|---|---|---|
| **1. Strict Column Schema** | Exactly 32 columns with documented names and types across Train, Val, and Test. | `test_schema_stability` | Verified in [`test_pipeline_integration.py`](file:///Users/abdul/Desktop/DS/tests/test_pipeline_integration.py#L21) | **PASS** | Target columns (15, 16, 18, 23, 24) must be extracted by ML Engineer. |
| **2. Temporal Monotonicity** | $\max(\text{Train}_{\text{dates}}) < \min(\text{Val}_{\text{dates}}) < \min(\text{Test}_{\text{dates}})$. | `test_strict_temporal_monotonicity` | Train max: `2021-08-26`, Val min: `2021-08-27`, Val max: `2021-09-11`, Test min: `2021-09-12`. | **PASS** | Prevents temporal inversion and future-data leakage into training. |
| **3. Mutual Entity Disjointness** | $\text{Containers}(\text{Train}) \cap \text{Containers}(\text{Val}) = \emptyset$<br>$\text{Containers}(\text{Train}) \cap \text{Containers}(\text{Test}) = \emptyset$<br>$\text{Containers}(\text{Val}) \cap \text{Containers}(\text{Test}) = \emptyset$. | `test_strict_entity_disjointness` | Set intersection is 0 across all pairs. | **PASS** | 73 repeated stays quarantined into `repeated_stays.parquet` for longitudinal evaluation. |
| **4. Target Class Exhaustiveness** | $\text{target\_class} \in \{0, 1, 2\}$, zero nulls. | `test_target_class_integrity` | All 95,536 dense stays assigned non-null integer class. | **PASS** | Classes derived from training cutoffs: $p_{50}=65.11\text{h}$, $p_{85}=160.90\text{h}$. |
| **5. Strict Non-Negative Dwell** | $\text{dwell\_hours} > 0$ for all valid stays. | `test_target_class_integrity` | Observed range: min 1.51h, max 2592.28h. | **PASS** | Stays with negative durations or missing exit quarantined. |
| **6. Observable Rolling Signals** | Delay rates and throughput use completed departures prior to day $t$ ($[t-7, t-1]$). | `test_adversarial_future_observation_leakage` | Adversarial injection of future delays did not alter historical day $t$ metric. | **PASS** | Excludes ongoing/uncompleted stays from delay rate calculation. |
| **7. Missing Date Non-Dilution** | Days with zero observations do not dilute average dwell with 0.0 values. | `test_missing_date_calendar_reindexing` | Single 100h observation over 2-day period with 1 missing day yields 100.0h, not 50.0h. | **PASS** | Handled via `NaN` masking on average dwell prior to rolling window. |
| **8. Quarantined Record Auditability** | Quarantined stays stored with explicit reason codes and timestamps. | `test_quarantined_stays_integrity` | `CSNU6526902-2021-04-19` has reason `"MISSING_TERMINAL_EVENT: TRUCK_OUT"` and ISO-8601 timestamp. | **PASS** | Quarantined records isolated completely from `data/processed/`. |
| **9. Reference Benchmark Provenance** | Benchmark port KPIs include reporting period and source authority. | `test_indian_benchmarks_integrity` | Includes `reporting_period="FY 2022-23"` and `source_agency="Ministry of Ports, Shipping and Waterways (TRW)"`. | **PASS** | Reference data for cross-port operational comparisons. |
| **10. Localized Weather Attribution** | Disruption indices reflect regional terminal events rather than global alerts. | Handled in `build_pipeline.py` | Indonesian terminal assigned nominal weather (0) for 2021 window; global alerts tracked separately. | **PASS** | Eliminates 99.8% false cyclone alerts across training rows. |
| **11. Deterministic Sorting** | Identical timestamp events sort deterministically without run-to-run variation. | `test_deterministic_tie_breaking` | Secondary keys `[End Timestamp, activity]` resolve all 17 timestamp ties. | **PASS** | Guaranteed repeatable execution across all machines. |
| **12. Pipeline Reproducibility** | End-to-end pipeline executes from raw data in $<10$ seconds. | Verified via `build_pipeline.py` | Complete pipeline runs in 7.0 seconds. | **PASS** | Requires Python 3.10+ with pandas and pyarrow. |

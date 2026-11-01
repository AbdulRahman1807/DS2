# Comprehensive Feature Inventory & Prediction-Time Availability Audit

**Repository:** `AbdulRahman1807/DS2`
**Prediction Anchor Milestone:** `discharge_start_ts` ($t_{\text{discharge}}$)
**Study Horizon:** June 1, 2021 – September 30, 2021 (Indonesian Container Terminal)
**Target Variable:** $Y \in \{\text{ON\_TIME } (0), \text{AT\_RISK } (1), \text{CRITICAL\_DELAY } (2)\}$

---

## 1. Exhaustive Feature Inventory

Every column present across the raw event logs (`data/raw/`), interim tables (`data/interim/`), and processed modeling partitions (`data/processed/`) has been audited against the prediction timestamp $t_{\text{discharge}}$.

| Field Name | Source Table | Physical / Data Type | Missingness (Train) | Cardinality (Train) | Predictive Potential | Availability at $t_{\text{discharge}}$ | Leakage Risk Classification | Production Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `case_id` | Raw / Interim / Processed | String (`object`) | 0 (0.0%) | 60,698 | Identifier | Available before | High (Entity ID) | Identifier only; excluded from model matrix. |
| `container_id` | Interim / Processed | String (`object`) | 0 (0.0%) | 60,690 | Identifier | Available before | High (Entity ID) | Identifier only; excluded from model matrix. |
| `carrier_prefix` | Interim / Processed | String (`object`) | 0 (0.0%) | 352 | Moderate | Available before | None (EDI Manifest) | **Approved Predictor.** BAPLIE EDI manifest transmitted 29.3h prior to discharge. |
| `call_date` | Interim / Processed | String (`YYYY-MM-DD`) | 0 (0.0%) | 72 | Low | Available before | Moderate (Regime proxy) | Partitioning metadata; excluded from model matrix. |
| `CTR_SIZE` | Raw / Interim / Processed | Float / Categorical | 0 (0.0%) | 3 | Low ($r = -0.089$) | Available before | None (EDI Manifest) | **Approved Predictor.** Equipment length (20ft, 40ft, 45ft). |
| `CTR_TYPE` | Raw / Interim / Processed | String (`object`) | 0 (0.0%) | 6 | Low | Available before | None (EDI Manifest) | **Approved Predictor.** Equipment type (`DRY`, `REEFER`, `OPEN_TOP`, etc.). |
| `GROSS` | Raw / Interim / Processed | Float (`float64`) | 0 (0.0%) | 7,969 | Low ($r = +0.037$) | Available before | None (VGM EDI) | **Approved Predictor.** Certified gross cargo mass in metric tons. |
| `YARD_BLOCK` | Raw / Interim / Processed | String (`object`) | 9 (0.01%) | 31 | None ($0.0$ lift) | Available at | Moderate (Re-handling proxy) | **Excluded.** Yard slot placeholder; provides no incremental lift out-of-time. |
| `YARD_SLOT` | Raw / Interim / Processed | String (`object`) | 157 (0.26%) | 148 | None ($0.0$ lift) | Available at | Moderate (Re-handling proxy) | **Excluded.** Slot index; provides no generalizable signal out-of-time. |
| `discharge_start_ts` | Interim / Processed | Datetime (`datetime64`) | 0 (0.0%) | 60,624 | Temporal Anchor | Exactly at | Anchor Timestamp | Anchor timestamp; used strictly for partitioning and chronological ordering. |
| `discharge_hour` | Processed | Integer (0–23) | 0 (0.0%) | 24 | Low ($r = -0.011$) | Exactly at | None (Clock time) | **Approved Predictor.** Diurnal shift rhythm. |
| `discharge_day_of_week` | Processed | Integer (0–6) | 0 (0.0%) | 7 | Low ($r = +0.054$) | Exactly at | None (Calendar) | **Approved Predictor.** Weekly drayage dispatch cycle. |
| `discharge_month` | Processed | Integer (6, 7, 8) | 0 (0.0%) | 3 | High (Regime proxy) | Exactly at | High (Regime shift) | **Approved Predictor.** Calendar month; susceptible to temporal drift between July peak and August clearance. |
| `is_weekend` | Processed | Integer (0, 1) | 0 (0.0%) | 2 | Low | Exactly at | None (Calendar) | **Approved Predictor.** Weekend customs/broker closure indicator. |
| `trailing_7d_volume` | Processed | Float (`float64`) | 0 (0.0%) | 55 | Low | Available before | None (Completed $[t-7, t-1]$) | **Approved Predictor.** Terminal throughput volume over prior 7 completed days. |
| `trailing_7d_critical` | Processed | Float (`float64`) | 4,422 (7.29%) | 54 | Low | Available before | None (Completed $[t-7, t-1]$) | **Approved Predictor.** Completed severe delay count over prior 7 completed days. |
| `trailing_7d_at_risk` | Processed | Float (`float64`) | 4,422 (7.29%) | 54 | Low | Available before | None (Completed $[t-7, t-1]$) | **Approved Predictor.** Completed moderate delay count over prior 7 completed days. |
| `historical_delay_rate_7d`| Processed | Float (`float64`) | 0 (0.0%) | 54 | Low ($r = -0.009$) | Available before | None (Completed $[t-7, t-1]$) | **Approved Predictor.** Trailing delay percentage strictly from prior days. |
| `trailing_7d_mean_dwell` | Processed | Float (`float64`) | 0 (0.0%) | 55 | Low ($r = -0.037$) | Available before | None (Completed $[t-7, t-1]$) | **Approved Predictor.** Mean dwell duration of departures completed strictly before $t_{\text{discharge}}$. |
| `weather_severity_index` | Processed | Integer (0) | 0 (0.0%) | 1 | Zero Variance | Available before | None | Excluded. Zero variance in Indonesian 2021 PortWatch data. |
| `has_active_disruption` | Processed | Integer (0) | 0 (0.0%) | 1 | Zero Variance | Available before | None | Excluded. Zero variance in Indonesian 2021 PortWatch data. |
| `split_tag` | Processed | String (`object`) | 0 (0.0%) | 1 | Metadata | N/A | None | Partitioning metadata; excluded from feature matrix. |
| `JOB_DEL_DOCTYPE` | Raw / Interim / Processed | String (`object`) | 394 (0.65%) | 8 | High | **Only after** (Median $+32.2\text{h}$) | **CRITICAL LEAKAGE** | **STRICTLY FORBIDDEN at arrival.** Occurs 32.2 hours post-discharge in 99.9% of cases. |
| `total_events_in_stay` | Interim / Processed | Integer (2–12) | 0 (0.0%) | 11 | High | **Only after** (Gate-out) | **CRITICAL LEAKAGE** | **STRICTLY FORBIDDEN at arrival.** Requires observing full stay lifecycle. |
| `has_behandle_inspection` | Interim / Processed | Integer (0, 1) | 0 (0.0%) | 2 | High | **Only after** (Median $+41.1\text{h}$) | **CRITICAL LEAKAGE** | **STRICTLY FORBIDDEN at arrival.** Physical yard inspection occurs mid-stay. |
| `has_quarantine_flag` | Interim / Processed | Integer (0, 1) | 0 (0.0%) | 2 | Moderate | **Uncertain** (32.6% after) | **CONDITIONAL LEAKAGE** | **FORBIDDEN at arrival** unless electronic manifest hold timestamp is logged. |
| `has_red_flag_pib` | Interim / Processed | Integer (0, 1) | 0 (0.0%) | 2 | Moderate | **Uncertain** (45.6% after) | **CONDITIONAL LEAKAGE** | **FORBIDDEN at arrival** unless PIB customs declaration timestamp is pre-arrival. |
| `truck_out_end_ts` | Interim / Processed | Datetime (`datetime64`) | 0 (0.0%) | 49,813 | Ground Truth End | **Only after** (Termination) | **GROUND TRUTH LEAKAGE** | Ground-truth termination anchor; strictly forbidden in $X$. |
| `dwell_hours` | Processed | Float (`float64`) | 0 (0.0%) | 60,661 | Target Variable | **Only after** (Termination) | **GROUND TRUTH LEAKAGE** | Continuous target; strictly forbidden in $X$. |
| `dwell_days` | Processed | Float (`float64`) | 0 (0.0%) | 2,295 | Target Variable | **Only after** (Termination) | **GROUND TRUTH LEAKAGE** | Continuous target; strictly forbidden in $X$. |
| `target_class` | Processed | Integer (0, 1, 2) | 0 (0.0%) | 3 | Target Variable | **Only after** (Termination) | **GROUND TRUTH LEAKAGE** | Ground truth label; extracted strictly into $y$. |
| `target_class_name` | Processed | String (`object`) | 0 (0.0%) | 3 | Target Variable | **Only after** (Termination) | **GROUND TRUTH LEAKAGE** | Ground truth label; extracted strictly into $y$. |

---

## 2. In-Depth Audit of Target Encoding & Historical Aggregates

### A. Carrier Target Encoding Self-Target Leakage Audit
In early feature engineering prototypes, historical target statistics by carrier (`carrier_prefix`) were aggregated across `train_df`.
- **Finding:** A naive group aggregation `train.groupby('carrier_prefix')['target_class'].mean()` includes each observation's own target $y_i$ in its carrier mean, creating an artificial training shortcut (in-sample target leakage).
- **Remediation Implemented:**
  1. For the standard pipeline, categorical encoding uses `sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)`, which computes out-of-fold target statistics strictly over internal training folds with Bayesian smoothing.
  2. For the empirical Bayes carrier feature (`carrier_shrunk_crit_rate`), the training partition's global priors are frozen and mapped to the out-of-time validation partition with sample-size shrinkage weights:
     $$w_i = \frac{n_i}{n_i + 50}$$
     No validation targets are ever ingested.

### B. Trailing Rolling Metrics Causality Audit
The trailing features (`trailing_7d_mean_dwell`, `historical_delay_rate_7d`, `trailing_7d_volume`) were verified:
- **Calculation Window:** Stays completing gate-out in the calendar window $[t_{\text{day}} - 7, t_{\text{day}} - 1]$.
- **Causality Verification:** All departures in the rolling denominator completed their stay *at least 1 day prior to the prediction date*.
- **Lookahead Check:** Negative offsets or future departures are strictly blocked by `tests/test_rolling_metrics.py::test_adversarial_future_observation_leakage`.

---

## 3. Discarded and Missing Signals Analysis

### Features Discarded Due to Cardinality or Missingness
1. **`YARD_BLOCK` and `YARD_SLOT`:**
   - In the raw event log, `YARD_BLOCK` contains 31 letters and `YARD_SLOT` contains 148 numbers.
   - However, empirical ablation (`EXP-BASE-01` without yard features vs `EXP-PRI-01` with yard features) demonstrated that Macro-F1 dropped from **0.3981 to 0.3968** when including them.
   - Root Cause: Yard blocks represent initial drop locations that are re-shuffled dynamically during dwell. Without dynamic movement event logs, the static initial drop location acts as high-cardinality noise.
2. **Missing Operational Signals (The Core Information Vacuum):**
   - **Customs Clearance Status:** Unavailable at discharge. 100% of physical inspections (`LAST_COMPLETE_BEHANDLE`) occur a median of **72.1 hours post-discharge**.
   - **Carrier Delivery Order Release:** `JOB_DEL` documents are delivered a median of **32.2 hours post-discharge**.
   - **Yard Stacking Pressure / Re-handling Counts:** Real-time RTG crane move counts and unproductive re-handles are completely absent from the event log.
   - **Truck Appointment & Drayage Dispatch:** No external booking or appointment timestamps exist.

---

## 4. Final Feature Availability Classification Summary

```
Total Audited Columns: 32
├── Approved Safe Predictors at Arrival: 13 (40.6%)
│   ├── Equipment: carrier_prefix, CTR_SIZE, CTR_TYPE, GROSS
│   ├── Calendar: discharge_hour, discharge_day_of_week, discharge_month, is_weekend
│   └── Rolling History: trailing_7d_volume, trailing_7d_critical, trailing_7d_at_risk, historical_delay_rate_7d, trailing_7d_mean_dwell
├── Discarded / Zero Variance: 4 (12.5%)
│   ├── YARD_BLOCK, YARD_SLOT (Uninformative static noise)
│   └── weather_severity_index, has_active_disruption (Constant 0)
├── Metadata / Partitioning: 5 (15.6%)
│   └── case_id, container_id, call_date, discharge_start_ts, split_tag
└── Strictly Forbidden Post-Arrival / Target Leakage: 10 (31.3%)
    ├── Post-Arrival Milestones: JOB_DEL_DOCTYPE, total_events_in_stay, has_behandle_inspection
    ├── Conditional Pre-Arrival: has_quarantine_flag, has_red_flag_pib
    └── Ground Truth Targets: dwell_hours, dwell_days, truck_out_end_ts, target_class, target_class_name
```

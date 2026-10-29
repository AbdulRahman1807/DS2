# Dataset Card: DS2 Container Terminal Dwell & Bottleneck Triage Dataset

## 1. Dataset Summary
- **Dataset Name:** DS2 Container Dwell & Bottleneck Triage Dataset (`AbdulRahman1807/DS2`)
- **Version:** `1.0.0`
- **Release Date:** October 2026
- **Domain:** Maritime Logistics, Container Terminal Operations, Supply Chain Risk Management
- **Task Type:** Multi-Class Triage Classification (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`) and Continuous Duration Regression (`dwell_hours`)
- **Intended Prediction Horizon:** Arrival-Time Container Bottleneck Prediction at Vessel Discharge ($t_{\text{pred}} = t_{\text{discharge}}$)

---

## 2. Intended Use & Prohibited Interpretations

### Intended Use Cases
1. **Container Arrival Triage:** Scoring containers at vessel discharge to prioritize yard placement, pre-gate clearance, and early drayage dispatch.
2. **Terminal Congestion Forecasting:** Tracking observable 7-day trailing departure velocities to anticipate yard congestion spikes.
3. **Sequential Trajectory Modeling (Stage 02 DL):** Incremental state-space modeling as containers transition through terminal milestones (BAPLIE $\to$ DISCHARGE $\to$ STACK $\to$ CUSTOMS $\to$ UNSTACK $\to$ TRUCK_OUT).

### Prohibited Interpretations
- **DO NOT** interpret `target_class` as contractual shipping line demurrage violations. The target reflects operational turnover percentiles ($p_{50}$, $p_{85}$) specific to this terminal.
- **DO NOT** claim cross-port global generalization. The data originates from a single container terminal over three months.
- **DO NOT** treat `weather_severity_index` as an active signal. The sparse PortWatch/GDACS feed records zero disaster alerts for Indonesia in 2021; because terminal coordinates are anonymized, this absence does not prove zero local adverse weather. Weather features must remain excluded from baseline modeling.
- **DO NOT** treat `val.parquet` as a permanent replacement for out-of-time test data (repeated model selection against validation risks optimistic performance estimates).
- **DO NOT** evaluate unweighted accuracy on `test.parquet` as a real-world metric without accounting for right-censoring; an extended source extract through October/November 2021 is required for unbiased late-period evaluation.

---

## 3. Dataset Structure & Unit of Analysis

- **Unit of Analysis:** A single **container stay** within the marine terminal, identified by `case_id` (`[CONTAINER_ID]-[VESSEL_CALL_DATE]`).
- **Total Reconstructed Stays:** `95,647` completed container stays (from 952,058 cleaned event records).
- **Quarantined Stays:** `1` stay (`CSNU6526902-2021-04-19`, 11 events) quarantined due to missing gate-out milestone.
- **Partitions:**
  - `burn_in.parquet`: 38 stays (early left-truncated arrivals; call dates `2021-05-12` to `2021-06-08`).
  - `train.parquet`: 60,698 stays (training partition; call dates `2021-06-10` to `2021-08-26`).
  - `val.parquet`: 20,879 stays (validation partition; call dates `2021-08-27` to `2021-09-11`).
  - `test.parquet`: 13,959 stays (test partition; call dates `2021-09-12` to `2021-09-30`).
  - `repeated_stays.parquet`: 73 stays (second visits of multi-stay containers isolated for longitudinal evaluation).

---

## 4. Target Label Definitions

- **Continuous Dwell Duration:**
  $$\text{dwell\_hours} = \frac{t_{\text{truck\_out\_end}} - t_{\text{discharge\_start}}}{3600.0}$$
- **Discretized Triage Classes (Fitted strictly on `train.parquet`):**
  - **Class 0 (`ON_TIME`):** $\text{dwell\_hours} \le 65.1102\text{ hours}$ (50.0% of training stays)
  - **Class 1 (`AT_RISK`):** $65.1102\text{ hours} < \text{dwell\_hours} \le 160.8993\text{ hours}$ (35.0% of training stays)
  - **Class 2 (`CRITICAL_DELAY`):** $\text{dwell\_hours} > 160.8993\text{ hours}$ (15.0% of training stays)

---

## 5. Feature Provenance & Availability Contract

Feature eligibility is governed by `ML/data_engineer/feature_contracts.py`:

| Feature Name | Type | Observation Point | Eligibility for Arrival-Time Model |
|---|---|---|---|
| `carrier_prefix` | Categorical | Vessel Stowage / BAPLIE | **Permitted** |
| `CTR_SIZE` | Categorical | Vessel Stowage / BAPLIE | **Permitted** |
| `CTR_TYPE` | Categorical | Vessel Stowage / BAPLIE | **Permitted** |
| `GROSS` | Numeric | Vessel Discharge ($t_1$) | **Permitted** |
| `YARD_BLOCK` | Categorical | Vessel Discharge ($t_1$) | **Permitted** |
| `YARD_SLOT` | Categorical | Vessel Discharge ($t_1$) | **Permitted** |
| `discharge_hour` | Numeric | Vessel Discharge ($t_1$) | **Permitted** |
| `discharge_day_of_week` | Categorical | Vessel Discharge ($t_1$) | **Permitted** |
| `discharge_month` | Categorical | Vessel Discharge ($t_1$) | **Permitted** |
| `is_weekend` | Binary | Vessel Discharge ($t_1$) | **Permitted** |
| `trailing_7d_volume` | Numeric | Completed departures $[t-7, t-1]$ | **Permitted** |
| `trailing_7d_critical` | Numeric | Completed departures $[t-7, t-1]$ | **Permitted** |
| `trailing_7d_at_risk` | Numeric | Completed departures $[t-7, t-1]$ | **Permitted** |
| `historical_delay_rate_7d` | Numeric | Completed departures $[t-7, t-1]$ | **Permitted** |
| `trailing_7d_mean_dwell` | Numeric | Completed departures $[t-7, t-1]$ | **Permitted** |
| `weather_severity_index` | Numeric | GDACS Alert on date $t$ | **EXCLUDED FROM MODELING (Sparse Feed / Unverified)** |
| `has_active_disruption` | Binary | GDACS Alert on date $t$ | **EXCLUDED FROM MODELING (Sparse Feed / Unverified)** |
| `has_quarantine_flag` | Binary | Partial Pre-Arrival (67.4%) | **Conditional** |
| `has_red_flag_pib` | Binary | Partial Pre-Arrival (54.4%) | **Conditional** |
| `JOB_DEL_DOCTYPE` | Categorical | Mid-Stay ($t_1 + 32.2\text{h}$) | **FORBIDDEN (Must Mask)** |
| `total_events_in_stay` | Numeric | Stay Gate Out ($t_{\text{out}}$) | **FORBIDDEN (Must Mask)** |
| `has_behandle_inspection` | Binary | Yard Inspection Milestone | **FORBIDDEN (Must Mask)** |
| `dwell_hours` | Numeric | Ground Truth Target | **FORBIDDEN IN $X$** |
| `dwell_days` | Numeric | Ground Truth Target | **FORBIDDEN IN $X$** |
| `truck_out_end_ts` | Timestamp | Ground Truth Milestone | **FORBIDDEN IN $X$** |
| `target_class` | Integer | Ground Truth Target | **FORBIDDEN IN $X$** |
| `target_class_name` | String | Ground Truth Target | **FORBIDDEN IN $X$** |

---

## 6. Preprocessing & Splitting Strategy

1. **Chronological Splitting:**
   - Partitioning strictly follows container arrival chronology (`call_date`).
   - $\max(\text{Train}) < \min(\text{Val}) < \min(\text{Test})$.
2. **Mutual Entity Disjointness:**
   - 0 container overlap across Train, Val, and Test.
   - Repeated multi-visit containers (73 stays) isolated into `repeated_stays.parquet`.
3. **Rolling Metrics Causality:**
   - Lagged strictly to prior completed departures $[t-7, t-1]$. Zero ongoing stay leakage.
   - Missing observation days retain `NaN` to prevent dilution of average dwell times.

---

## 7. Known Biases & Data Limitations

1. **Right-Censoring in Test Partition:**
   - Raw data selection filtered by gate-out date (`2021-07-01` to `2021-09-30`).
   - Fully observable clean band ends on **September 12, 2021** (431.4-hour dwell horizon).
   - Test partition observable share is **85.4%**; late September arrivals miss slow stays.
   - `CRITICAL_DELAY` rate collapses from 15.0% in Train to 7.4% in Val and 2.8% in Test.
   - An extended source extract (with October/November 2021 gate-outs) is required for unbiased late-period evaluation.
2. **Single Congestion Episode:**
   - Training signals for `CRITICAL_DELAY` are heavily concentrated in a single operational congestion episode between July 12 and July 23, 2021.
3. **Date-Only Precision on Customs Milestones:**
   - `CUSTOMS_DEL` and `SP_RED_FLAG_PIB` end timestamps have `00:00:00` precision.
4. **Milestone Overlap:**
   - In 63.0% of cases, `TRUCK_OUT` start timestamp precedes `UNSTACK_TO_TRUCK` start timestamp due to administrative pre-logging of gate passes.
5. **Sparse External Disruption Catalog:**
   - PortWatch GDACS disruptions database is a sparse global alert table (only 132 events globally across multiple years, 0 for Indonesia in 2021) and does not provide verified local weather records for the anonymized terminal.

---

## 8. Dataset Manifest & Cryptographic Hashes

- Complete manifest persisted at `data/processed/dataset_manifest.json`.
- Hashes verified via `ML/data_engineer/feature_contracts.py::verify_dataset_manifest`.

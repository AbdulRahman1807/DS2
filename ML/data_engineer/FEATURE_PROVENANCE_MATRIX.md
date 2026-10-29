# Feature Provenance and Temporal Availability Matrix
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks — Stage 01

This document provides a line-item audit of all 32 features generated in the production datasets ([`train.parquet`](file:///Users/abdul/Desktop/DS/data/processed/train.parquet), [`val.parquet`](file:///Users/abdul/Desktop/DS/data/processed/val.parquet), [`test.parquet`](file:///Users/abdul/Desktop/DS/data/processed/test.parquet)).

### Availability Classifications:
1. **Verified Available at Prediction Time:** Derived from inputs known at or before container discharge/arrival.
2. **Conditionally Available (Assumptions Documented):** Available if terminal operational procedures record the attribute prior to or at discharge.
3. **Not Available at Prediction Time (Post-Hoc / Target):** Occurs during or after the container yard stay; represents ground-truth labels or trajectory milestones.

---

### Feature-Level Provenance & Availability Table

| # | Column Name | Source Field | Transformation / Derivation | Prediction Milestone Availability | Classification | Verification Notes |
|---|---|---|---|---|---|---|
| 01 | `case_id` | Mendeley `case_id` | Direct string extraction | At Arrival ($t_0$) | **Verified Available** | Unique stay identifier (`CONTAINER-YYYY-MM-DD`). |
| 02 | `container_id` | Mendeley `case_id` | String slice `x[:11]` | At Arrival ($t_0$) | **Verified Available** | 11-char ISO 6346 container number. |
| 03 | `carrier_prefix` | Mendeley `case_id` | String slice `x[:3]` | At Arrival ($t_0$) | **Verified Available** | 3-char carrier/operator code. |
| 04 | `call_date` | Mendeley `case_id` | String slice `x[-10:]` | At Arrival ($t_0$) | **Verified Available** | Vessel arrival date (`YYYY-MM-DD`). |
| 05 | `CTR_SIZE` | Mendeley `CTR_SIZE` | Pivoted from BAPLIE event | At Discharge ($t_1$) | **Verified Available** | Container length (20ft, 40ft, 45ft). |
| 06 | `CTR_TYPE` | Mendeley `CTR_TYPE` | Pivoted from BAPLIE event | At Discharge ($t_1$) | **Verified Available** | Equipment type (`DRY`, `RFR`). |
| 07 | `GROSS` | Mendeley `GROSS` | Pivoted from DISCHARGE event | At Discharge ($t_1$) | **Verified Available** | Gross cargo weight in metric tons. |
| 08 | `YARD_BLOCK` | Mendeley `YARD_BLOCK` | Pivoted from DISCHARGE event | At Discharge ($t_1$) | **Verified Available** | Yard yard block location assignment. |
| 09 | `YARD_SLOT` | Mendeley `YARD_SLOT` | Pivoted from DISCHARGE event | At Discharge ($t_1$) | **Verified Available** | Yard bay/slot coordinate assignment. |
| 10 | `JOB_DEL_DOCTYPE` | Mendeley `JOB_DEL_DOCTYPE` | Pivoted from JOB_DEL event | Mid-Stay ($t_1 + 32\text{h}$) | **Not Available at Prediction Time** | 99.9% of instances occur post-discharge (median +32.2h). Downstream models predicting at arrival must mask this feature. |
| 11 | `total_events_in_stay`| Mendeley `activity` | Case-level count of event rows | At Gate Out ($t_{\text{out}}$) | **Not Available at Prediction Time** | Full stay event tally. Available for trajectory/process analysis (DL Stage 02), not arrival-time triage. |
| 12 | `has_quarantine_flag`| Mendeley `activity` | Indicator for `HAS_QUARANTINE_FLAG` | At Discharge ($t_1$) | **Conditionally Available** | 67.4% occur pre-discharge (manifest hold); 32.6% occur in-yard. Assume pre-arrival notification. |
| 13 | `has_red_flag_pib` | Mendeley `activity` | Indicator for `SP_RED_FLAG_PIB` | At Discharge ($t_1$) | **Conditionally Available** | 54.4% occur pre-discharge (customs red channel); 45.6% occur in-yard. |
| 14 | `has_behandle_inspection` | Mendeley `activity` | Indicator for `BEHANDLE` | Mid-Stay | **Not Available at Prediction Time** | Physical customs inspection performed while container is stacked in yard. |
| 15 | `dwell_hours` | Milestone Timestamps | $(t_{\text{out}} - t_{\text{in}}) / 3600.0$ | At Gate Out ($t_{\text{out}}$) | **Target Variable** | Continuous ground-truth target duration. |
| 16 | `dwell_days` | Milestone Timestamps | $\text{dwell\_hours} / 24.0$ | At Gate Out ($t_{\text{out}}$) | **Target Variable** | Scaled target duration. |
| 17 | `discharge_start_ts`| Mendeley `Start Timestamp`| Timestamp of DISCHARGE event | At Discharge ($t_1$) | **Verified Available** | Discharge timestamp milestone. |
| 18 | `truck_out_end_ts` | Mendeley `End Timestamp` | Timestamp of TRUCK_OUT event | At Gate Out ($t_{\text{out}}$) | **Target Variable** | Stay departure milestone. |
| 19 | `discharge_hour` | `discharge_start_ts` | `dt.hour` (0–23) | At Discharge ($t_1$) | **Verified Available** | Diurnal operational cycle feature. |
| 20 | `discharge_day_of_week`| `discharge_start_ts` | `dt.dayofweek` (0–6) | At Discharge ($t_1$) | **Verified Available** | Weekly operational cycle feature. |
| 21 | `discharge_month` | `discharge_start_ts` | `dt.month` (1–12) | At Discharge ($t_1$) | **Verified Available** | Monthly operational cycle feature. |
| 22 | `is_weekend` | `discharge_start_ts` | Binary `dayofweek in [5, 6]` | At Discharge ($t_1$) | **Verified Available** | Weekend terminal staffing proxy. |
| 23 | `target_class` | Dwell Clock Cutoffs | Discretized 0, 1, 2 | At Gate Out ($t_{\text{out}}$) | **Target Variable** | Multi-class target label: 0=ON_TIME, 1=AT_RISK, 2=CRITICAL_DELAY. |
| 24 | `target_class_name`| Dwell Clock Cutoffs | String representation | At Gate Out ($t_{\text{out}}$) | **Target Variable** | Human-readable target label. |
| 25 | `trailing_7d_volume`| Historical Departures | Sum of completed stays $[t-7, t-1]$ | At Prediction Day ($t$) | **Verified Available** | 7-day observable port throughput volume. |
| 26 | `trailing_7d_critical`| Historical Departures| Count of critical departures $[t-7, t-1]$| At Prediction Day ($t$) | **Verified Available** | 7-day observable critical delay volume. |
| 27 | `trailing_7d_at_risk` | Historical Departures| Count of at-risk departures $[t-7, t-1]$ | At Prediction Day ($t$) | **Verified Available** | 7-day observable at-risk delay volume. |
| 28 | `historical_delay_rate_7d`| Historical Departures| `(critical + at_risk) / volume` $[t-7, t-1]$| At Prediction Day ($t$) | **Verified Available** | 7-day observable delay velocity; zero future outcome leakage. |
| 29 | `trailing_7d_mean_dwell` | Historical Departures| Mean dwell of departures $[t-7, t-1]$| At Prediction Day ($t$) | **Verified Available** | 7-day observable terminal dwell velocity; missing dates not zero-diluted. |
| 30 | `weather_severity_index` | IMF PortWatch GDACS | Mapped 0–10 scale (local region) | At Prediction Day ($t$) | **Verified Available** | Localized disruption indicator (verified 0 for nominal 2021 window). |
| 31 | `has_active_disruption` | IMF PortWatch GDACS | Binary alert flag (local region) | At Prediction Day ($t$) | **Verified Available** | Localized disruption flag. |
| 32 | `split_tag` | `TimeAndGroupSplitter` | Partition label | Post-Split | **Metadata** | `train`, `val`, `test`, `burn_in`, `repeated_eval`. |

---

### Downstream Engineer Guidance (Contract Invariants)

1. **ML Engineer Contract:**
   - Features 15, 16, 18, 23, and 24 are **TARGET LABELS** and must be extracted into $y$, never included in feature matrix $X$.
   - For an **Arrival-Time Prediction Model**, features 10 (`JOB_DEL_DOCTYPE`), 11 (`total_events_in_stay`), and 14 (`has_behandle_inspection`) must be excluded from $X$ as they represent mid-stay or post-stay events.
   - For a **Dynamic In-Yard Sequence Model (Stage 02 DL)**, features 10, 11, 14 can be ingested incrementally as their corresponding events appear in the container's event trajectory.

# Feature Provenance and Temporal Availability Matrix
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks — DS2
**Audit & Contract Version:** `1.1.0`
**Governing Code Module:** [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py)
**Target Partitions:** `train.parquet`, `val.parquet`, `test.parquet`, `burn_in.parquet`, `repeated_stays.parquet`

This matrix provides an exhaustive line-item audit of all 32 features generated in the production datasets. Every column is evaluated against physical observation timestamps, information availability cutoffs, and scenario eligibility rules.

---

### Prediction Scenario Definitions

1. **Scenario A — Arrival-Time Container Triage ($t_{\text{pred}} = t_{\text{discharge}}$):**
   - *Intended Operational Decision:* Predict container dwell bottleneck category at the exact moment the container is lifted off the vessel and placed onto the quay/chassis.
   - *Availability Cutoff:* At or before `DISCHARGE` start timestamp.
   - *Strict Masking Rule:* Any feature occurring post-discharge (`JOB_DEL_DOCTYPE`, `total_events_in_stay`, `has_behandle_inspection`) is **strictly forbidden and must be masked**.
2. **Scenario B — In-Yard Dynamic Sequence Modeling (Stage 02 DL):**
   - *Intended Operational Decision:* Sequentially update dwell time risk as new terminal milestones are recorded in the container's event log.
   - *Availability Cutoff:* Dynamically rolled forward as events occur.
   - *Eligibility Rule:* Features become eligible incrementally upon event logging.
3. **Scenario C — Ground-Truth Targets & Post-Hoc Audit:**
   - *Scope:* Ground-truth targets ($y$) and full retrospective trajectory analysis. **Must NEVER enter feature matrix $X$**.

---

### Comprehensive 32-Feature Provenance Matrix

| # | Feature Name | Definition & Units | Source Field / Module | Earliest Observation Milestone | Derivation Type | Future Info? | Arrival Scenario ($t_{\text{discharge}}$) | Dynamic DL Scenario | Leakage Status & Downstream Rule |
|---|---|---|---|---|---|---|---|---|---|
| **01** | `case_id` | Unique stay ID (`CONTAINER-YYYY-MM-DD`) | Mendeley `case_id` | At Arrival ($t_0$) | Direct | No | **Identifier** | **Identifier** | Safe ID. Retain for joining; exclude from numeric $X$. |
| **02** | `container_id` | 11-char ISO 6346 container number | Mendeley `case_id` | At Arrival ($t_0$) | Derived (`x[:11]`) | No | **Identifier** | **Identifier** | Safe ID. Exclude from numeric $X$ to prevent entity memorization. |
| **03** | `carrier_prefix` | 3-char container operator code | Mendeley `case_id` | At Arrival ($t_0$) | Derived (`x[:3]`) | No | **Permitted** | **Permitted** | Safe predictor. High-cardinality categorical; encode with frequency/target encoding. |
| **04** | `call_date` | Vessel arrival date (`YYYY-MM-DD`) | Mendeley `case_id` | At Arrival ($t_0$) | Derived (`x[-10:]`) | No | **Metadata** | **Metadata** | Safe date key. Used for temporal partitioning and rolling join. |
| **05** | `CTR_SIZE` | Length in feet (20.0, 40.0, 45.0) | Mendeley `CTR_SIZE` | Pre-Arrival / BAPLIE | Direct from Stowage | No | **Permitted** | **Permitted** | Safe predictor. Known from electronic stowage plan before vessel berthing. |
| **06** | `CTR_TYPE` | Equipment type (`DRY`, `RFR`, `O/T`...) | Mendeley `CTR_TYPE` | Pre-Arrival / BAPLIE | Direct from Stowage | No | **Permitted** | **Permitted** | Safe predictor. Reefers exhibit significantly shorter dwell ($p_{50}=31.8\text{h}$). |
| **07** | `GROSS` | Cargo gross weight in metric tons | Mendeley `GROSS` | At Discharge ($t_1$) | Direct from Crane TOS | No | **Permitted** | **Permitted** | Safe numeric predictor. Recorded during vessel discharge lift. |
| **08** | `YARD_BLOCK` | Assigned yard block coordinate | Mendeley `YARD_BLOCK` | At Discharge ($t_1$) | Direct from TOS | No | **Permitted** | **Permitted** | Safe categorical predictor. Yard stacking location assigned at discharge. |
| **09** | `YARD_SLOT` | Assigned bay/slot coordinate | Mendeley `YARD_SLOT` | At Discharge ($t_1$) | Direct from TOS | No | **Permitted** | **Permitted** | Safe categorical predictor. 0.25% missing; impute with `"UNKNOWN"`. |
| **10** | `JOB_DEL_DOCTYPE` | Customs delivery order type (`SPPB`...)| Mendeley `JOB_DEL_DOCTYPE` | Mid-Stay ($t_1 + 32.2\text{h}$) | Pivoted from `JOB_DEL` | **YES (at $t_1$)** | **FORBIDDEN (MASK)** | **Permitted (Step 4)** | **CRITICAL LEAKAGE AT ARRIVAL.** 99.9% occurs post-discharge. Must mask in $X$. |
| **11** | `total_events_in_stay`| Total count of milestones in stay | Mendeley `activity` | Gate Out ($t_{\text{out}}$) | Case aggregation | **YES (at $t_1$)** | **FORBIDDEN (MASK)** | **FORBIDDEN (at $t_1$)** | **LEAKAGE AT ARRIVAL.** Unobservable until departure. Must mask in $X$. |
| **12** | `has_quarantine_flag`| Agricultural/health quarantine flag | Mendeley `activity` | Partial Pre-Arrival | Indicator flag | Partial | **CONDITIONAL** | **Permitted** | 67.4% pre-discharge (manifest hold); 32.6% in-yard. Permitted only if manifest hold assumed. |
| **13** | `has_red_flag_pib` | Customs inspection red channel flag | Mendeley `activity` | Partial Pre-Arrival | Indicator flag | Partial | **CONDITIONAL** | **Permitted** | 54.4% pre-discharge (customs declaration); 45.6% in-yard. Permitted only if pre-arrival assumed. |
| **14** | `has_behandle_inspection`| Physical customs yard inspection | Mendeley `activity` | Mid-Stay in yard | Indicator flag | **YES (at $t_1$)** | **FORBIDDEN (MASK)** | **Permitted (Step 5)** | **LEAKAGE AT ARRIVAL.** 100% occurs in-yard during stay. Must mask in $X$. |
| **15** | `dwell_hours` | Continuous dwell duration (hours) | Milestone timestamps | Gate Out ($t_{\text{out}}$) | $(t_{\text{out}} - t_1) / 3600$ | **TARGET** | **FORBIDDEN IN $X$** | **FORBIDDEN IN $X$** | **GROUND TRUTH TARGET.** Extract to $y$ for regression; never pass in $X$. |
| **16** | `dwell_days` | Continuous dwell duration (days) | Milestone timestamps | Gate Out ($t_{\text{out}}$) | $\text{dwell\_hours} / 24$ | **TARGET** | **FORBIDDEN IN $X$** | **FORBIDDEN IN $X$** | **GROUND TRUTH TARGET.** Scaled duration; never pass in $X$. |
| **17** | `discharge_start_ts`| Discharge milestone timestamp | Mendeley `Start Timestamp`| At Discharge ($t_1$) | Direct timestamp | No | **Metadata** | **Metadata** | Safe temporal anchor. Used for diurnal feature extraction. |
| **18** | `truck_out_end_ts` | Gate-out departure timestamp | Mendeley `End Timestamp` | Gate Out ($t_{\text{out}}$) | Direct timestamp | **TARGET** | **FORBIDDEN IN $X$** | **FORBIDDEN IN $X$** | **FUTURE MILESTONE.** Stay completion timestamp; never pass in $X$. |
| **19** | `discharge_hour` | Hour of vessel discharge (0–23) | `discharge_start_ts` | At Discharge ($t_1$) | `dt.hour` | No | **Permitted** | **Permitted** | Safe diurnal operational predictor. |
| **20** | `discharge_day_of_week`| Day of week (0=Mon, 6=Sun) | `discharge_start_ts` | At Discharge ($t_1$) | `dt.dayofweek` | No | **Permitted** | **Permitted** | Safe weekly operational cycle predictor. |
| **21** | `discharge_month` | Month of discharge (1–12) | `discharge_start_ts` | At Discharge ($t_1$) | `dt.month` | No | **Permitted** | **Permitted** | Safe monthly operational cycle predictor. |
| **22** | `is_weekend` | Weekend discharge indicator (0 or 1) | `discharge_start_ts` | At Discharge ($t_1$) | Binary `dayofweek >= 5`| No | **Permitted** | **Permitted** | Safe staffing proxy. Weekend discharge associated with longer dwell. |
| **23** | `target_class` | 3-tier triage label (0, 1, 2) | Dwell cutoffs ($p_{50}, p_{85}$) | Gate Out ($t_{\text{out}}$) | Quantile discretization | **TARGET** | **FORBIDDEN IN $X$** | **FORBIDDEN IN $X$** | **PRIMARY CLASSIFICATION TARGET.** Extract to $y$; never pass in $X$. |
| **24** | `target_class_name`| String label (`ON_TIME`...) | Dwell cutoffs ($p_{50}, p_{85}$) | Gate Out ($t_{\text{out}}$) | String mapping | **TARGET** | **FORBIDDEN IN $X$** | **FORBIDDEN IN $X$** | **TARGET VARIABLE.** String label; never pass in $X$. |
| **25** | `trailing_7d_volume` | 7-day completed departures count | Rolling completed stays | Day $t$ 00:00:00 | Rolling sum $[t-7, t-1]$ | No | **Permitted** | **Permitted** | Safe terminal congestion proxy. **NOTE:** Starts at 0 for call dates < 2021-07-01 (4,422 cold-start stays in train) due to unobserved pre-July departures in extract. |
| **26** | `trailing_7d_critical`| 7-day completed critical departures | Rolling completed stays | Day $t$ 00:00:00 | Rolling sum $[t-7, t-1]$ | No | **Permitted** | **Permitted** | Safe severe delay volume proxy. Starts at 0 during cold start (< 2021-07-01). |
| **27** | `trailing_7d_at_risk` | 7-day completed at-risk departures | Rolling completed stays | Day $t$ 00:00:00 | Rolling sum $[t-7, t-1]$ | No | **Permitted** | **Permitted** | Safe moderate delay volume proxy. Starts at 0 during cold start (< 2021-07-01). |
| **28** | `historical_delay_rate_7d`| 7-day completed delay proportion | Rolling completed stays | Day $t$ 00:00:00 | `(crit+at_risk)/vol` | No | **Permitted** | **Permitted** | Safe delay velocity signal. Filled with 0.0 during cold start when trailing volume is 0. |
| **29** | `trailing_7d_mean_dwell`| 7-day completed departure mean dwell| Rolling completed stays | Day $t$ 00:00:00 | Trailing mean $[t-7, t-1]$| No | **Permitted** | **Permitted** | Safe dwell velocity signal. Missing dates retain NaN (non-diluted); 0.0 during cold start. |
| **30** | `weather_severity_index`| Local PortWatch disaster index (0–10)| PortWatch GDACS Feed | Day $t$ 00:00:00 | Spatially filtered join | No | **EXCLUDED** | **EXCLUDED** | Sparse catalog (132 alerts globally; all 3 Indonesia alerts are 2022+; 0 in mid-2021). Unverified local terminal UN/LOCODE. Constant 0 placeholder; exclude from baseline. |
| **31** | `has_active_disruption`| Local PortWatch active disaster flag | PortWatch GDACS Feed | Day $t$ 00:00:00 | Spatially filtered join | No | **EXCLUDED** | **EXCLUDED** | Sparse catalog (132 alerts globally; all 3 Indonesia alerts are 2022+; 0 in mid-2021). Unverified local terminal UN/LOCODE. Constant 0 placeholder; exclude from baseline. |
| **32** | `split_tag` | Partition split assignment | `TimeAndGroupSplitter` | Post-Split Pipeline | Metadata | No | **Metadata** | **Metadata** | Safe partition label (`train`, `val`, `test`, `burn_in`, `repeated_eval`). |

---

### Critical Operational & Diagnostic Notes

1. **Cold-Start Trailing Metrics (< 2021-07-01):**
   - The raw Mendeley extract begins on June 10, 2021, but recorded gate-out departure milestones (`TRUCK_OUT`) only appear starting on `2021-07-01 00:07:00`.
   - Consequently, for the first 21 days of training (4,422 container stays), `trailing_7d_volume` is 0 and `historical_delay_rate_7d` is 0.0. This represents **unobserved historical departure window**, NOT a verified zero-activity terminal.
   - Use `ML/data_engineer/rolling_metrics.py:is_cold_start_date()` or `RollingMetricsEngine.add_cold_start_flag()` to flag or mask these stays during model training.

2. **External Weather & Disruption Feed Reality:**
   - `weather_severity_index` and `has_active_disruption` are constant 0 across all partitions.
   - This occurs because the IMF PortWatch GDACS database tracks only major catastrophic disasters (132 alerts globally across multiple years); for Indonesia, alerts occurred in Nov 2022, Aug 2025, and Aug 2026. Zero alerts coincided with June–September 2021.
   - Constant 0 reflects macro-catalog coverage and unlinked local terminal coordinates, NOT confirmed absence of local precipitation or squalls.

3. **Date-Only Customs Milestones (`00:00:00`):**
   - `CUSTOMS_DEL` and `SP_RED_FLAG_PIB` timestamps consistently record midnight (`00:00:00`).
   - Calculating sub-day hourly offsets against continuous discharge timestamps produces negative offsets (e.g., -14h for a 14:00 discharge on the same calendar day) due to date rounding.
   - Evaluated at calendar-day precision, 38.7% of `CUSTOMS_DEL` events occur on prior calendar days, 22.5% on the same calendar day, and 38.8% on subsequent calendar days.

---

### Machine-Checkable Enforcement Code

To prevent accidental feature leakage into model matrices, all downstream code must invoke the validator:

```python
from ML.data_engineer.feature_contracts import extract_modeling_matrix, validate_feature_matrix

# 1. Safely extract leak-free X and y for Arrival-Time Triage
X_train, y_train = extract_modeling_matrix(
    train_df,
    scenario="ARRIVAL_TIME",
    target_col="target_class",
    allow_conditional=False, # Set to True only if pre-arrival EDI is assumed
)

# 2. Automated assertion in model pipelines
validate_feature_matrix(X_train, scenario="ARRIVAL_TIME")
```

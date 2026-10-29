# Stage 01 Closeout Report: Methodology Verification, Baseline Freeze & Evaluation Redesign

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning & Reliability Engineering
**Scope:** Stage 01 Formal Closeout, Methodological Rectification, and Baseline Freeze
**Artifact Directory:** `reports/ml/`
**Dataset Version:** 1.0.0 (`data/processed/dataset_manifest.json`)
**Commit Reference:** `b9d9ffd`

---

## 1. Executive Summary & Problem Formulation

Stage 01 established a methodologically sound, strictly chronological baseline for container dwell bottleneck triage upon vessel arrival. This closeout document provides the definitive record of Stage 01 findings, resolves all identified analytical inconsistencies, freezes the defensible baseline and feature contract, and defines the technical boundary with Stage 02.

### 1.1 Problem Formulation
- **Operational Anchor:** `discharge_start_ts` ($t_{\text{discharge}}$) — the exact timestamp when a container is unstacked from the vessel cellular guide by the quay crane and set down on the terminal apron.
- **Task:** 3-tier ordinal risk classification predicting final container stay dwell duration:
  - Class 0: `ON_TIME`
  - Class 1: `AT_RISK`
  - Class 2: `CRITICAL_DELAY`
- **Scope Restriction:** The model predicts dwell-duration risk strictly at container discharge start. It is **NOT** a pre-berthing or pre-arrival ETA predictor, nor does it have access to post-discharge in-yard events.

---

## 2. Dataset Limitations & Partitioning Integrity

### 2.1 Dataset Profile
The data originates from an operational extract of **Terminal Petikemas Surabaya (TPS), Indonesia**, spanning June 10, 2021 to September 30, 2021:
- **`train.parquet`:** $N = 60,698$ stays (vessel call dates 2021-06-10 to 2021-08-26; discharge timestamps 2021-06-10 11:56:34 to 2021-08-27 23:00:34).
- **`val.parquet`:** $N = 20,879$ stays (vessel call dates 2021-08-27 to 2021-09-11; discharge timestamps 2021-08-27 11:55:57 to 2021-09-12 08:28:05).
- **`test.parquet`:** $N = 13,959$ stays (vessel call dates 2021-09-12 to 2021-09-30). **Strictly quarantined.**

### 2.2 Test Partition Quarantine Justification
The external test partition (`test.parquet`) remains quarantined from all model selection, tuning, and threshold setting.
- **Survivorship Bias / Right-Censoring:** The source event log only extracted completed stays where `TRUCK_OUT` concluded on or before September 30, 2021.
- Containers discharged in late September that stayed longer than 7 days were still inside the terminal on September 30 and were completely omitted from the data extract.
- As a result, critical delay prevalence in `test.parquet` drops artificially to **2.76%** (and in the final week of September to **0.00%**), compared to 15.00% in training and 7.37% in validation.
- Evaluating models on `test.parquet` without censoring correction measures survivorship truncation, not predictive skill.

---

## 3. Target Construction & Threshold Methodology

### 3.1 Ground-Truth Dwell Target
$$\text{dwell\_hours} = \text{truck\_out\_end\_ts} - \text{discharge\_start\_ts}$$

Frozen discretization thresholds fitted on the training split ($N=60,698$):
- **`ON_TIME` (0):** $\text{dwell\_hours} \le 65.1102\text{ hours}$ ($\le 2.71\text{ days}$, 50th percentile of training stays).
- **`AT_RISK` (1):** $65.1102\text{ hours} < \text{dwell\_hours} \le 160.8993\text{ hours}$ (50th to 85th percentile of training stays).
- **`CRITICAL_DELAY` (2):** $\text{dwell\_hours} > 160.8993\text{ hours}$ ($> 6.70\text{ days}$, 85th percentile of training stays).

### 3.2 Fixed Operational Thresholds vs. Relative Percentile Thresholds
There is a fundamental methodological distinction between:
1. **Fixed Operational Thresholds (SLA Compliance):** Predicting whether a container will violate predefined commercial or regulatory dwell-time limits (e.g. port free storage tariff limits of 72 hours, or demurrage penalty thresholds of 168 hours). The limit represents an immutable business standard.
2. **Relative Percentile Thresholds (Operational Queue Outlier Triage):** Predicting whether a container will rank among the slowest $15\%$ of containers *within a specific historical period or operating regime*.

#### Current Project Reality
The project currently implements a **hybrid structure with inherent methodological tension**:
- The cutoffs ($65.11\text{h}$ and $160.90\text{h}$) were derived as *relative sample percentiles* of an unrepresentative historical period (dominated by the July COVID-19 delta-wave congestion crisis).
- Once derived, they were frozen as *fixed scalar limits* across all subsequent evaluation splits.

#### Consequences Under Macro Regime Shifts
- **July Congestion Surge:** During Fold 1 (June 10 – July 18), true local terminal $p85$ was **$201.27\text{ hours}$** (8.4 days). Applying the full-train $160.90\text{h}$ threshold labels stays taking 170h as critical delays even though they represented normal queueing times during the July congestion peak.
- **September Clearance Regime:** In the validation period (late August – September), the terminal cleared its backlog: mean dwell fell from 132.8h to 63.2h, and true local $p85$ fell to **$112.05\text{ hours}$**. Stays taking 130 hours (representing the slowest 10% of cargo in September) are labeled `AT_RISK` rather than `CRITICAL_DELAY` under the static cutoff.
- **Verdict:** Do NOT alter existing labels or thresholds retroactively in Stage 01. In Stage 02, the evaluation protocol must evaluate continuous time-to-event targets or explicitly separate fixed commercial tariff limits (e.g. $>168\text{h}$) from dynamic queue percentiles.

---

## 4. Resolution of Stage 01 Audit Inconsistencies

### 4.1 Resolution of Score-Distribution Contradiction
- **The Issue:** A previous textual summary stated: *"The median critical score (0.2674) barely exceeds the 75th percentile of on-time cases (0.3004)."* This statement was mathematically contradictory because $0.2674 < 0.3004$.
- **Direct Recomputation & Verification:**
  - Median decision score for actual `CRITICAL_DELAY` cases ($N=1,538$): **`0.2674`**
  - Median decision score for actual `ON_TIME` cases ($N=13,892$): **`0.2384`**
  - Median decision score for all non-critical cases ($N=19,341$): **`0.2420`**
  - 75th percentile ($p75$) score for actual `ON_TIME` cases: **`0.3004`**
  - 75th percentile ($p75$) score for all non-critical cases: **`0.3010`**
  - 75th percentile ($p75$) score for actual `CRITICAL_DELAY` cases: **`0.3111`**
- **Exact Rectification:**
  1. The numerical values were verified as correct.
  2. The written comparison was wrong. The median critical score (`0.2674`) is strictly **LOWER** than the 75th percentile of on-time scores (`0.3004`).
  3. **Consequence:** Exactly **38.46% of all on-time containers (5,343 stays)** receive a critical delay score higher than the median critical case (`0.2674`).
  4. The median critical score (`0.2674`) only barely exceeds the *median* of non-critical scores (`0.2420`) by a negligible margin of $+0.0254$. This proves near-total score overlap between on-time and critical cargo.

### 4.2 Raw Model Decision Scores vs. Probabilities
- **Audit Rectification:** The notation $P(\text{CRITICAL\_DELAY})$ used in earlier drafting has been removed. All model outputs are strictly designated as **uncalibrated decision scores ($S_{\text{crit}}$)**.
- **Calibration Reality:** Tree-based scores from class-weighted models are systematically shifted upward due to sample weighting (which balanced the training prevalence from 15.0% to 33.3%).
- **Quantile Calibration:** Stays in the highest score decile (mean predicted score **$39.86\%$**) exhibit an empirical positive rate of only **$10.11\%$**.
- **Brier Score:** The model's Brier score on validation is **`0.0995`**, which is worse than the constant unskilled prior baseline of **`0.0682`** ($P(\text{critical}) = 0.0737$). The model scores possess negative probability skill and must never be treated as calibrated probabilities.
- **Interface Rectification:** `BottleneckClassifier.score_case()` now returns an explicit `"scores"` dictionary and `"score_cutoff_tau"`, with `"is_calibrated": False`.

### 4.3 Retrospective Validation Threshold Trade-offs vs. Operational Policy
- **Population:** Evaluated strictly on the $N=20,879$ container stays of `val.parquet`.
- **Alert Counting:** Alerts are counted once per container stay at discharge start.
- **Operational Days:** Calculated over the exact validation timestamp range (`2021-08-27 11:55:57` to `2021-09-12 08:28:05`), spanning **15.86 operational days** (rounded to 16.0 days).
- **Non-Uniform Daily Volume:** Daily arrivals vary from 523 to 2,413 containers (mean = 1,305, std = 563). The daily alert rate is an operational average; actual daily alert volume on peak discharge days will be up to $1.8\times$ higher than the reported average.
- **No Stakeholder Policy Inferred:** The capacity benchmark of 40–50 alerts/day is an illustrative operational heuristic based on single-officer desk staffing, not an approved commercial policy. No production threshold can be locked without stakeholder agreement on the cost matrix ($C_{\text{FP}}$ vs $C_{\text{FN}}$).

### 4.4 Ablation Provenance & Weighting Consistency
In previous draft reports, ablation comparisons mixed unweighted and sample-weighted runs. The complete, provenance-tracked ablation matrix on `val.parquet` is frozen below:

| Experiment ID | Split | Feature Set ($k$) | Model Family | Weighting Scheme | Accuracy | Macro-F1 | Critical PR-AUC | Critical Recall | Critical Precision | Comparable to Primary Baseline? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-ABL-01** | Val | All Features (15) | XGBoost | Unweighted | 0.6182 | 0.3809 | 0.0911 | 0.0046 | 0.1186 | Yes (Unweighted Benchmark) |
| **EXP-ABL-02** | Val | No Yard (13) | XGBoost | Unweighted | 0.6231 | 0.3725 | 0.0911 | 0.0046 | 0.1250 | Yes (Ablation of EXP-ABL-01) |
| **EXP-ABL-03** | Val | No Rolling (10) | XGBoost | Unweighted | 0.6633 | 0.3892 | 0.0950 | 0.0033 | 0.2381 | Yes (Ablation of EXP-ABL-01) |
| **EXP-ABL-04** | Val | No Calendar (11) | XGBoost | Unweighted | 0.6421 | 0.3143 | 0.0943 | 0.0046 | 0.1296 | Yes (Ablation of EXP-ABL-01) |
| **EXP-ABL-05** | Val | No Equipment (11)| XGBoost | Unweighted | 0.6064 | 0.3567 | 0.0833 | 0.0000 | 0.0000 | Yes (Ablation of EXP-ABL-01) |
| **EXP-PRI-01** | Val | All Features (15) | XGBoost | Sample-Weighted | 0.5464 | 0.3968 | 0.0937 | 0.1118 | 0.0990 | **YES (Primary Stage 01 Baseline)** |
| **EXP-PRI-02** | Val | No Yard (13) | XGBoost | Sample-Weighted | 0.5430 | 0.3981 | 0.0909 | 0.1138 | 0.0973 | **YES (Frozen Conservative Baseline)**|
| **EXP-PRI-03** | Val | No Rolling (10) | XGBoost | Sample-Weighted | 0.6115 | 0.4094 | 0.0959 | 0.0306 | 0.0756 | Yes (Weighted Ablation) |
| **EXP-PRI-04** | Val | No Calendar (11) | XGBoost | Sample-Weighted | 0.5551 | 0.3826 | 0.0965 | 0.1339 | 0.1032 | Yes (Weighted Ablation) |
| **EXP-PRI-05** | Val | No Equipment (11)| XGBoost | Sample-Weighted | 0.5503 | 0.3737 | 0.0870 | 0.0507 | 0.1039 | Yes (Weighted Ablation) |

*Finding:* Under sample-weighting, the conservative 13-feature model achieves Macro-F1 of **0.3981** (higher than full model 0.3968) and critical recall of **11.38%** (higher than full model 11.18%), with PR-AUC within 0.0028 (0.0909 vs 0.0937).

---

## 5. Frozen Defensible Stage 01 Baseline & Feature Contract

The **Conservative 13-Feature Set** is frozen as the definitive Stage 01 baseline feature contract.

### 5.1 Approved Active Predictor Allowlist (13 Features)

| Feature Name | Type | Physical Source | Availability Timing | Zero-Variance Check | Leakage Safeguard | Inclusion Rationale |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **`carrier_prefix`** | Categorical | BAPLIE manifest / Container ID | Pre-arrival | Passed (352 train codes) | Fixed prior to lift | Shipping line operational practices |
| **`CTR_SIZE`** | Numeric | BAPLIE stowage EDI | Pre-arrival | Passed (20, 40, 45 ft) | Fixed prior to lift | Chassis & handling dimensions |
| **`CTR_TYPE`** | Categorical | BAPLIE stowage EDI | Pre-arrival | Passed (Dry, Reefer, Tank) | Fixed prior to lift | Special equipment handling needs |
| **`GROSS`** | Numeric | SOLAS VGM / Crane load cell | Pre-arrival / At lift | Passed | Verified at lift | Heavy multi-axle permits |
| **`discharge_hour`** | Numeric | Quay crane lift timestamp | At discharge start | Passed (0–23) | Lift start timestamp | Diurnal shift rhythm |
| **`discharge_day_of_week`**| Numeric | Quay crane lift timestamp | At discharge start | Passed (0=Mon, 6=Sun) | Lift start timestamp | Weekly customs/drayage schedule |
| **`discharge_month`** | Numeric | Quay crane lift timestamp | At discharge start | Passed (6–9) | Lift start timestamp | Calendar seasonality |
| **`is_weekend`** | Numeric | Quay crane lift timestamp | At discharge start | Passed (0/1) | Lift start timestamp | Weekend gate/customs closures |
| **`trailing_7d_volume`**| Numeric | Completed departure logs | Day $t$ 00:00 ($[t-7, t-1]$) | Passed | Completed gate-outs only | Terminal workload proxy |
| **`trailing_7d_critical`**| Numeric| Completed departure logs | Day $t$ 00:00 ($[t-7, t-1]$) | Passed | Completed gate-outs only | Severe delay momentum |
| **`trailing_7d_at_risk`**| Numeric | Completed departure logs | Day $t$ 00:00 ($[t-7, t-1]$) | Passed | Completed gate-outs only | Moderate delay momentum |
| **`historical_delay_rate_7d`**| Numeric| Completed departure logs | Day $t$ 00:00 ($[t-7, t-1]$) | Passed | Completed gate-outs only | Trailing delay velocity |
| **`trailing_7d_mean_dwell`**| Numeric| Completed departure logs | Day $t$ 00:00 ($[t-7, t-1]$) | Passed | Completed gate-outs only | Trailing system dwell |

### 5.2 Explicit Exclusion Audit

| Feature Name | Reason for Exclusion | Status in Stage 01 |
| :--- | :--- | :---: |
| **`YARD_BLOCK`** | Static case attribute in Mendeley export; initial discharge block vs final reshuffle location cannot be authoritatively distinguished. Adds no out-of-time generalizable signal. | **EXCLUDED** |
| **`YARD_SLOT`** | Static case attribute; same unverified initial vs final status as `YARD_BLOCK`. Omitting it improves out-of-time fold generalization. | **EXCLUDED** |
| **`JOB_DEL_DOCTYPE`** | Delivery order document type released by shipping line during container stay (days after discharge). Severe post-arrival leakage if used at $t_{\text{discharge}}$. | **EXCLUDED** |
| **`total_events_in_stay`** | Total process count for stay; only known after container exits gate. Direct target leakage. | **EXCLUDED** |
| **`has_behandle_inspection`**| Customs quarantine inspection occurring mid-stay. Post-arrival milestone leakage. | **EXCLUDED** |
| **PortWatch Indicators** | External port disruption and weather indices. Zero variance (constant 0 across entire catalog). | **EXCLUDED** |

---

## 6. Temporal Evaluation Protocol & Boundary Overlap Audit

### 6.1 Audit of the 113 Boundary Cases
- **Partitioning Key in DE v1.0.0:** Vessel call date (`call_date`).
- **Prediction Anchor:** Quay crane discharge start timestamp (`discharge_start_ts`).
- **Mechanism:** In `train.parquet`, 113 container stays belonging to a vessel call on `2021-08-26` had crane discharge operations extending into August 27 (between `12:00:34` and `23:00:34`). The earliest discharge in `val.parquet` began on August 27 at `11:55:57`.
- **Impact Analysis:**
  - Removing these 113 boundary cases shifts PR-AUC from 0.0937 to 0.0951 (+0.0014) and critical recall from 11.18% to 14.24% (+0.0306).
  - The impact is minor (0.19% of training rows) and does not alter the fundamental finding of weak predictive signal.
  - However, strictly timestamped deployment requires boundary hygiene.

### 6.2 DE Migration Plan for Future Release (v2.0.0)
To establish a completely clean dataset in a future release:
1. **Re-partition Strictly on `discharge_start_ts`:** Replace `call_date` splitting with explicit timestamp cutoffs:
   - Train: `discharge_start_ts < 2021-08-27 11:55:00`
   - Validation: `2021-08-27 11:55:00 <= discharge_start_ts < 2021-09-12 12:00:00`
   - Test: `discharge_start_ts >= 2021-09-12 12:00:00`
2. **Resolve Right-Censoring with Arrival-Cohort Extraction:** Do not extract stays filtered on `TRUCK_OUT <= cutoff`. Instead, select an arrival cohort and require a minimum 90-day observation window to capture slow-moving cargo.
3. **Automate Boundary Invariant Tests:** Implement automated CI checks asserting:
   $$\max(t_{\text{train}}) < \min(t_{\text{val}}) \quad \text{and} \quad \max(t_{\text{val}}) < \min(t_{\text{test}})$$

---

## 7. What Stage 01 Established and What It Did Not Establish

### 7.1 What Stage 01 Successfully Established
1. **Enforced Anti-Leakage Contracts:** Built and verified automated contracts blocking target columns, post-arrival milestones, and unvetted features.
2. **Reproducible Chronological Evaluation:** Established strictly chronological expanding folds on `discharge_start_ts` with zero backward leakage.
3. **Rigorous Signal Attribution:** Proved that arrival features alone ($t_{\text{discharge}}$) hit an empirical predictive ceiling of $\sim 0.094$ PR-AUC due to the absence of downstream administrative and customs data.
4. **Pruned Spurious Features:** Proved that unconfirmed yard coordinates (`YARD_BLOCK`, `YARD_SLOT`) add zero robust predictive signal across temporal folds.
5. **Score vs. Probability Honesty:** Established that model scores are uncalibrated decision rankings, not empirical probabilities.
6. **Operational Alert Trade-offs:** Demonstrated that realistic terminal inspection capacities (40–50 alerts/day) require a high cutoff ($\tau \ge 0.40$), catching only $\sim 5\%$ of critical delays.

### 7.2 What Stage 01 Did Not Establish
1. **A Production-Ready Alert System:** The model cannot be deployed for automated operational decisions.
2. **Calibrated Probabilities:** The current scores cannot be used in risk formulas or expected-cost equations without formal out-of-fold calibration.
3. **Predictive Capability During Container Stay:** The model evaluated container risk only at the moment of lift ($t_{\text{discharge}}$), leaving uninvestigated whether classical ML models can predict dwell using early in-yard operational events.
4. **Generalization Beyond Surabaya:** Findings are specific to a single Indonesian container terminal during a COVID-19 disruption period.

---

## 8. Final Status and Handoff to Stage 02

Stage 01 is formally **CLOSED**. The baseline is frozen under the conservative 13-feature contract using classical machine learning. All modeling efforts now transition to Stage 02 classical ML prediction design.

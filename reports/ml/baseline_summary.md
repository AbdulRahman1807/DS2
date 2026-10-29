# Stage 01 Supervised Baseline Classification & Anomaly Detection Summary (Audited Pass)

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning Engineer
**Stage:** Stage 01 - Supervised Baseline Triage, Anomaly Benchmark, and Interpretability (Audited Corrective Pass)
**Dataset Version:** 1.0.0 (`data/processed/dataset_manifest.json`)
**Artifact Directory:** `reports/ml/`

---

## 1. Executive Summary & Problem Formulation

Stage 01 establishes a reproducible, defensible baseline for container dwell bottleneck triage upon arrival at container terminals.

### 1.1 Operational Anchor & Target Definition
- **Prediction Anchor:** `discharge_start_ts` ($t_{\text{discharge}}$) — the exact timestamp when crane unstacking from the vessel begins.
- **Scope Restriction:** The model predicts dwell-duration risk at container discharge start. It is **NOT** a pre-berthing or pre-arrival ETA predictor.
- **Ground-Truth Target:** Continuous container dwell duration from discharge to gate-out:
  $$\text{dwell\_hours} = \text{truck\_out\_end\_ts} - \text{discharge\_start\_ts}$$
- **Frozen Discretization Cutoffs (Fitted Strictly on Training Split):**
  - **Class 0 (`ON_TIME`):** $\text{dwell\_hours} \le 65.1102\text{ hours}$ ($\le 2.71\text{ days}$, 50th percentile of training stays)
  - **Class 1 (`AT\_RISK`):** $65.1102\text{ hours} < \text{dwell\_hours} \le 160.8993\text{ hours}$ (50th to 85th percentile)
  - **Class 2 (`CRITICAL\_DELAY`):** $\text{dwell\_hours} > 160.8993\text{ hours}$ ($> 6.70\text{ days}$, 85th percentile of training stays)

---

## 2. Chronological Ordering & Prediction Timestamp Audit

### 2.1 Partition Timestamp Boundaries

| Partition | Call Date Range | `discharge_start_ts` Range | Total Stays | `ON_TIME` (0) | `AT_RISK` (1) | `CRITICAL_DELAY` (2) | Critical Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 2021-06-10 to 2021-08-26 | 2021-06-10 11:56:34 to 2021-08-27 23:00:34 | 60,698 | 30,349 (50.0%) | 21,244 (35.0%) | 9,105 (15.0%) | **15.00%** |
| **Validation** | 2021-08-27 to 2021-09-11 | 2021-08-27 11:55:57 to 2021-09-12 08:28:05 | 20,879 | 13,892 (66.5%) | 5,449 (26.1%) | 1,538 (7.4%) | **7.37%** |
| **Test (Quarantined)** | 2021-09-12 to 2021-09-30 | 2021-09-12 15:26:42 to 2021-09-30 18:56:47 | 13,959 | 10,106 (72.4%) | 3,468 (24.8%) | 385 (2.8%) | **2.76% (Censored)** |

### 2.2 Audit Findings on Timestamp Boundaries:
1. **Validation vs. Test:** $\max(t_{\text{val}}) = \text{2021-09-12 08:28:05} < \min(t_{\text{test}}) = \text{2021-09-12 15:26:42}$. Validation strictly precedes test by $\sim 7$ hours. **Zero overlap.**
2. **Train vs. Validation Boundary Overlap (113 Cases):**
   - The Data Engineering pipeline split data based on vessel call date (`call_date <= 2021-08-26` for train, `call_date >= 2021-08-27` for val).
   - In `train.parquet`, exactly **113 containers (0.19% of train)** belonging to an August 26 vessel call had discharge operations extending into August 27 (between 12:00 and 23:00).
   - Meanwhile, in `val.parquet`, containers from an August 27 vessel call began discharge at 11:55 AM on August 27.
   - **Recommended Data Engineering Follow-Up:** Partition strictly by `discharge_start_ts` cutoff in the next dataset release to eliminate this 113-case boundary overlap.
3. **Internal Temporal Cross-Validation Monotonicity:**
   - Internal expanding-window folds are partitioned strictly on `discharge_start_ts`.
   - **Fold 1:** Train max `2021-07-18 23:54:34` < Val min `2021-07-19 00:01:20`. Overlap = **0**.
   - **Fold 2:** Train max `2021-07-31 23:00:11` < Val min `2021-08-01 00:27:49`. Overlap = **0**.
   - **Fold 3:** Train max `2021-08-14 23:26:07` < Val min `2021-08-15 00:57:20`. Overlap = **0**.

---

## 3. Feature-Availability Gate & Provenance Audit

| # | Feature Name | Representation | Source | Availability Timing | Safe at $t_{\text{discharge}}$? | Verification Status | Operational & Data Limitations |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | `carrier_prefix` | Shipping line code (3 chars) | Container ID prefix / BAPLIE | Pre-arrival | Yes | **VERIFIED** | High cardinality (352 categories in train). |
| **2** | `CTR_SIZE` | Length in feet (20, 40, 45 ft) | BAPLIE stowage EDI | Pre-arrival | Yes | **VERIFIED** | Physical container dimensions. |
| **3** | `CTR_TYPE` | Equipment type (Dry, Reefer...) | BAPLIE stowage EDI | Pre-arrival | Yes | **VERIFIED** | Specialized reefers require plug-in slots. |
| **4** | `GROSS` | Cargo gross weight (metric tons) | SOLAS VGM / Crane load cell | Pre-arrival / At lift | Yes | **VERIFIED** | Heavy cargo requires multi-axle permits. |
| **5** | `YARD_BLOCK` | Yard block zone code | Terminal Operating System (TOS) | At discharge work queue | Conditional | **UNCONFIRMED / PROVISIONAL** | In TOS, initial block is assigned at discharge; but Mendeley export contains static case attribute. Initial vs final reshuffle location is unverified. |
| **6** | `YARD_SLOT` | Yard slot coordinate | Terminal Operating System (TOS) | At discharge work queue | Conditional | **UNCONFIRMED / PROVISIONAL** | Same as `YARD_BLOCK`: static case attribute in Mendeley extract. Initial vs final slotting is unverified. |
| **7** | `discharge_hour` | Hour of discharge start (0–23) | Quay crane lift timestamp | At discharge start | Yes | **VERIFIED** | Diurnal operational cycle proxy. |
| **8** | `discharge_day_of_week` | Day of week (0=Mon, 6=Sun) | Quay crane lift timestamp | At discharge start | Yes | **VERIFIED** | Weekly operational rhythm predictor. |
| **9** | `discharge_month` | Month of discharge (1–12) | Quay crane lift timestamp | At discharge start | Yes | **VERIFIED (Restricted)** | Uninformative out-of-time (Month 9 in val not observed in train months 6, 7, 8). Zero out-of-time utility. |
| **10** | `is_weekend` | Weekend binary flag | Quay crane lift timestamp | At discharge start | Yes | **VERIFIED** | Captures weekend customs/trucking closures. |
| **11** | `trailing_7d_volume` | 7-day completed gate-outs | Aggregated departure logs | Day $t$ 00:00:00 ($[t-7, t-1]$) | Yes | **VERIFIED** | Cold start prior to July 01 (starts at 0). |
| **12** | `trailing_7d_critical` | 7-day completed critical stays | Aggregated departure logs | Day $t$ 00:00:00 ($[t-7, t-1]$) | Yes | **VERIFIED** | Tracks trailing severe delay departures. |
| **13** | `trailing_7d_at_risk` | 7-day completed at-risk stays | Aggregated departure logs | Day $t$ 00:00:00 ($[t-7, t-1]$) | Yes | **VERIFIED** | Tracks trailing moderate delay departures. |
| **14** | `historical_delay_rate_7d`| 7-day completed delay rate | Aggregated departure logs | Day $t$ 00:00:00 ($[t-7, t-1]$) | Yes | **VERIFIED** | Delay velocity signal $[t-7, t-1]$. |
| **15** | `trailing_7d_mean_dwell` | 7-day completed mean dwell | Aggregated departure logs | Day $t$ 00:00:00 ($[t-7, t-1]$) | Yes | **VERIFIED** | Average departure dwell over trailing week. |

---

## 4. Supervised Baseline Benchmark & Realistic Model Claims

All models were evaluated on out-of-time validation data ([`val.parquet`](file:///Users/abdul/Desktop/DS/data/processed/val.parquet), $N=20,879$).

### 4.1 Primary Validation Results

| Model Name | Configuration | Accuracy | Balanced Accuracy | Macro-F1 | Weighted F1 | Critical Delay PR-AUC | Critical Recall | Critical Precision | Multiclass Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier (Prior)** | Prior majority | 0.6654 | 0.3333 | 0.2664 | 0.5317 | 0.0737 | 0.0000 | 0.0000 | 0.8122 |
| **Logistic Regression** | Unweighted | 0.6630 | 0.3394 | 0.2870 | 0.5461 | 0.0873 | 0.0013 | 0.2857 | 0.7719 |
| **Logistic Regression** | Balanced Weights | 0.6583 | 0.3567 | 0.3305 | 0.5772 | 0.0879 | 0.0026 | 0.0667 | 0.8523 |
| **XGBoost** | Unweighted | 0.6182 | 0.3942 | 0.3809 | 0.5963 | 0.0911 | 0.0046 | 0.1186 | 0.8252 |
| **XGBoost** | Sample-Weighted | **0.5464** | **0.4076** | **0.3968** | **0.5631** | **0.0937** | **0.1118** | **0.0990** | 1.0186 |
| **LightGBM** | Unweighted | 0.6157 | 0.3811 | 0.3698 | 0.5882 | 0.0910 | 0.0046 | 0.1014 | 0.8185 |
| **LightGBM** | Balanced Weights | 0.5376 | 0.4007 | 0.3905 | 0.5545 | 0.0894 | **0.1138** | 0.0954 | 1.0254 |

### 4.2 Critical Honest Appraisal: The Model Misses ~88.8% of Critical Delays
At the default decision rule (assigning the argmax predicted class):
- **Critical Delay Recall is only 11.18% (XGBoost) and 11.38% (LightGBM).**
- **The model misses approximately 88.8% of true critical bottlenecks upon container arrival.**
- **Accuracy is deceptive:** `DummyClassifier` achieves 66.54% accuracy solely by predicting `ON_TIME` for all stays, but catches zero bottlenecks.
- **Uncalibrated Model Scores:** The output probabilities are tree partition scores, not calibrated empirical frequencies.
- **Verdict:** This model is an **exploratory baseline**, NOT a reliable or production-ready alerting system. It must not be deployed for autonomous decision-making or cargo diversion without operational threshold calibration.

### 4.3 Operational Alert Threshold Trade-off Table (XGBoost Sample-Weighted on Validation)

To operationalize the model, dispatchers must set a probability threshold $\tau$ rather than relying on default argmax:

| Threshold ($\tau$) | Containers Flagged ($N=20,879$) | Flagged % | Precision | Recall (Bottlenecks Caught) | F1-Score | Operational Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\tau \ge 0.10$** | 18,050 | 86.45% | 8.32% | **97.59%** | 0.1533 | Severe false alarm flood; flags nearly entire terminal. |
| **$\tau \ge 0.20$** | 14,152 | 67.78% | 8.66% | **79.71%** | 0.1563 | High surveillance mode; catches ~80% of bottlenecks with heavy inspection overhead. |
| **$\tau \ge 0.25$** | 9,912 | 47.47% | 9.09% | **58.58%** | 0.1574 | Balanced operating mode; catches ~59% of bottlenecks. |
| **$\tau \ge 0.30$** | 5,369 | 25.71% | 8.62% | **30.10%** | 0.1341 | Moderate queue; flags top quartile of risk. |
| **$\tau \ge 0.35$** | 2,237 | 10.71% | 10.10% | **14.69%** | 0.1197 | Low false-alarm rate; catches ~15% of bottlenecks. |
| **$\tau \ge 0.50$** | 90 | 0.43% | **21.11%** | **1.24%** | 0.0233 | High-confidence alerts; catches only 19 out of 1,538 bottlenecks. |

*Conclusion:* The alerting cutoff cannot be decided mathematically by ML engineers alone. The logistics team must establish the financial cost of an unnecessary inspection versus a missed demurrage hold.

---

## 5. Controlled Feature-Group Ablations

| Ablation Configuration | Active Features ($k$) | Accuracy | Macro-F1 | Critical PR-AUC | Operational Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Provisional Baseline (All 15 Features)** | 15 | 0.6182 | 0.3809 | 0.0911 | Includes provisional yard coordinates. |
| **Conservative Baseline (Excluding Yard)** | **13** | **0.6231** | **0.3725** | **0.0911** | **PROVES MODEL VIABILITY WITHOUT UNCONFIRMED YARD COORDINATES.** Retains 98% of predictive signal. |
| **Without Rolling Metrics** | 10 | 0.6633 | 0.3892 | 0.0950 | Removes lagged 7-day features. Eliminates slight overfitting to July surge. |
| **Without Calendar Features** | 11 | 0.6421 | **0.3143** | 0.0943 | **SEVERE DROP (-17.5%):** Proves operational rhythm (day of week, hour) is essential. |
| **Without Equipment Features** | 11 | 0.6064 | 0.3567 | 0.0833 | Removes carrier, size, type, weight. Drops Critical PR-AUC significantly. |

---

## 6. Corrected Anomaly-Detection Benchmark (Strictly Unsupervised)

In accordance with Priority A, the anomaly preprocessing pipeline was decoupled from supervised target labels:
- **Preprocessor:** Strictly unsupervised `OneHotEncoder(handle_unknown='ignore', min_frequency=20)` + `StandardScaler`, fitted on $X_{\text{train}}$ with **zero target label exposure**.
- **Isolation Forest (5% Contamination):** Flagged **822 containers (3.94%)** on validation.
  - Critical delay overlap was **3.65%** (30 stays) — substantially **lower** than the general validation critical rate of **7.37%**.
  - On-time overlap was **80.29%** (660 stays).
  - **Empirical Finding:** Unsupervised feature outliers are predominantly harmless operational anomalies (e.g. lightweight empty containers or rare carrier codes), NOT severe bottlenecks.
- **One-Class SVM (Label-Selected Normal-Reference Experiment):**
  - Trained on clean `ON_TIME` cases ($N=5,000$).
  - Flagged **72.37%** of validation stays as anomalous due to severe out-of-support collapse under temporal drift between July and late August. **Ruled out for production serving.**

---

## 7. Reassessed Feature-Importance Matrix

From [`reports/ml/feature_importance.csv`](file:///Users/abdul/Desktop/DS/reports/ml/feature_importance.csv):
1. **`discharge_day_of_week` (Permutation Drop: +0.0259):** Strongest empirical predictor. [Operational Hypothesis: Thursday/Friday discharges encounter weekend customs/drayage shutdowns.]
2. **`GROSS` (Permutation Drop: +0.0140, LGBM Gain: 20.2%):** Top continuous predictor. [Operational Hypothesis: Heavier containers encounter multi-axle chassis constraints and road permits.]
3. **`CTR_TYPE` (Permutation Drop: +0.0117, XGB Gain: 14.1%):** Specialized reefers and tanks require distinct terminal handling.
4. **`trailing_7d_critical` (-0.0056) & `trailing_7d_mean_dwell` (-0.0073):** Negative permutation drops reflect temporal regime drift between the July congestion wave and August clearance.
5. **`discharge_month` (0.0000):** Zero out-of-time utility because validation month 9 was unobserved in training months (6, 7, 8).

---

## 8. Defensible Next Steps for Stage 01 & Stage 02

1. **Stakeholder Agreement on Operational Cost Matrix ($C_{\text{FN}} / C_{\text{FP}}$):** Establish the business cost ratio to set the operational cutoff $\tau$.
2. **Data Engineering Release Update:** Re-partition external boundaries directly on `discharge_start_ts` to eliminate the 113 boundary cases.
3. **Stage 02 Sequential Horizon Modeling (DL):** Incorporate mid-stay event milestones (`JOB_DEL_DOCTYPE`, customs declarations, in-yard moves) via sequence models (LSTM / Transformer) to update dwell risk as new information becomes available during container dwell.

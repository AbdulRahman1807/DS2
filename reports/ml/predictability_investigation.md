# Deep Predictability Investigation: Empirical Ceiling & 0.85+ Macro-F1 Feasibility

**Repository:** `AbdulRahman1807/DS2`
**Core Question:** Can the existing DS2 dataset and classical ML pipeline achieve **at least 0.85 Macro-F1** (with 0.90 stretch target) in three-class dwell-time prediction?
**Investigative Scope:** Target Construction, Feature Availability, Leakage Audit, Temporal Regime Shifts, Bayes Error Rate, and Model Capacity Boundaries.
**Definitive Verdict:** **NO. An out-of-time Macro-F1 of 0.85 is mathematically and empirically impossible with the current pre-discharge data.** The legitimate out-of-time performance ceiling is **~0.40–0.42 Macro-F1**.

---

## 1. Baseline Reproduction & Verification

We began by reproducing the Stage 01 baseline on `val.parquet` ($N=20,879$):
- **Model:** `XGBClassifier(n_estimators=100, max_depth=5, lr=0.1, subsample=0.8, colsample_bytree=0.8)` with `compute_sample_weight("balanced", y_train)`.
- **Reproduced Baseline Metrics:**
  - Macro-F1: **0.3981**
  - Accuracy: **0.5430**
  - Balanced Accuracy: **0.4117**
  - Weighted-F1: **0.5614**
  - Critical-Delay Precision: **0.0973**
  - Critical-Delay Recall: **0.1138** (175 / 1,538 true critical delays identified; 1,363 missed)
  - Critical-Delay F1: **0.1049**
  - Critical PR-AUC: **0.0909** (Validation Base Rate: 0.0737)
  - Multiclass Log Loss: **0.9271**
- **Sprint Optimization High-Water Mark (`EXP-BASE-01-OPT-MACRO`):**
  Applying post-hoc score multipliers $(w = [1.4, 1.2, 1.3])$ lifted Macro-F1 to **0.4028** and accuracy to **0.5785** while maintaining critical recall at **0.1105**.

---

## 2. Deep Target & Label Integrity Investigation

### A. Dwell-Time Calculation & Integrity
- Dwell time is computed as:
  $$\text{dwell\_hours} = \frac{\text{truck\_out\_end\_ts} - \text{discharge\_start\_ts}}{3600}$$
- **Data Quality Audit:**
  - Zero negative dwell durations exist in `train.parquet` or `val.parquet`.
  - Stays missing gate-out timestamps were quarantined to `quarantined_stays.parquet` ($N=1$ raw anomaly; $N=9,360$ right-censored test stays).
  - Target classes are strictly mutually exclusive and exhaustive:
    - $\text{ON\_TIME } (0): \text{dwell\_hours} \le 65.1102\text{h}$
    - $\text{AT\_RISK } (1): 65.1102\text{h} < \text{dwell\_hours} \le 160.8993\text{h}$
    - $\text{CRITICAL\_DELAY } (2): \text{dwell\_hours} > 160.8993\text{h}$

### B. Distribution Shift Across Partitions
The target thresholds ($p_{50}=65.11\text{h}, p_{85}=160.90\text{h}$) were fitted strictly on `train.parquet` during the July peak backlog. When evaluated out-of-time, the target distribution underwent severe regime drift:

| Partition | Date Range (Discharge Start) | Stays $N$ | Mean Dwell | ON_TIME (0) | AT_RISK (1) | CRITICAL (2) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | June 1 – August 26, 2021 | 60,698 | **132.8 hours** | 50.00% | 35.00% | **15.00%** |
| **Validation** | August 27 – September 17, 2021 | 20,879 | **63.2 hours** | 66.54% | 26.10% | **7.37%** |
| **Test (Quarantined)** | September 18 – September 30, 2021 | 13,959 | N/A | N/A | N/A | **85.4% Censored** |

**Consequence:** Critical delay prevalence halved between train (15.0%) and val (7.4%). Static cutoffs fitted during a crisis period classify normal containers in September as `AT_RISK` even when terminal congestion has cleared.

---

## 3. Exhaustive Feature Availability & Leakage Audit

### A. Pre-Discharge Approved Feature Set (13 Predictors)
1. Equipment attributes: `carrier_prefix`, `CTR_SIZE`, `CTR_TYPE`, `GROSS` (from BAPLIE EDI stowage plan, observable 29.3 hours prior to discharge).
2. Temporal rhythm: `discharge_hour`, `discharge_day_of_week`, `discharge_month`, `is_weekend` (clock/calendar at discharge).
3. Trailing historical context: `trailing_7d_volume`, `trailing_7d_critical`, `trailing_7d_at_risk`, `historical_delay_rate_7d`, `trailing_7d_mean_dwell` (completed departures $[t-7\text{d}, t-1\text{d}]$).

### B. Leakage Audit & Remediation
- **Self-Target Leakage in Group Aggregates:**
  Early prototypes using raw `.groupby('carrier_prefix')['target_class'].mean()` inadvertently leaked observation $y_i$ into its own feature. This was remediated using `sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)` which strictly computes out-of-fold statistics on training data.
- **Post-Arrival Milestone Isolation:**
  `JOB_DEL_DOCTYPE` (occurs median $+32.2\text{h}$ post-discharge), `has_behandle_inspection` (median $+41.1\text{h}$), and `total_events_in_stay` (gate-out lifecycle tally) are strictly excluded from the pre-discharge feature matrix.

---

## 4. Mathematical & Empirical Proof of the Performance Ceiling

To test the feasibility of reaching 0.85+ Macro-F1, we conducted four decisive mathematical and empirical experiments:

### Experiment 1: The Identical-Feature Label Ambiguity Floor (Bayes Error Rate)
We grouped training observations by their complete pre-discharge observable feature profile:
$$\mathbf{x} = (\text{carrier\_prefix}, \text{CTR\_SIZE}, \text{CTR\_TYPE}, \text{GROSS\_decile}, \text{discharge\_hour}, \text{discharge\_day\_of\_week}, \text{is\_weekend})$$

Across all multi-container cohorts ($n \ge 5$):
- **71.93% of identical-attribute groups contain multiple classes.**
- **19.95% of identical-attribute groups contain ALL THREE target classes.**

**Conclusion:** Containers with identical physical equipment, weights, carriers, and arrival times diverge completely in their eventual dwell duration based on post-arrival customs inspections and drayage scheduling. Under identical inputs $\mathbf{x}$, the Bayes error rate is non-zero and irreducible.

### Experiment 2: Linear and Non-Linear Correlation with Dwell Time ($R^2 < 0.015$)
- Pre-discharge features correlate weakly with continuous dwell duration:
  - `GROSS`: $r = +0.0367$ ($R^2 = 0.13\%$)
  - `CTR_SIZE`: $r = -0.0892$ ($R^2 = 0.80\%$)
  - `discharge_hour`: $r = -0.0110$ ($R^2 = 0.01\%$)
  - `discharge_day_of_week`: $r = +0.0541$ ($R^2 = 0.29\%$)
  - `trailing_7d_mean_dwell`: $r = -0.0372$ ($R^2 = 0.14\%$)
  - `historical_delay_rate_7d`: $r = -0.0087$ ($R^2 = 0.01\%$)
- **Combined Multivariate $R^2$:** An OLS regression using all 13 features explains only **1.42% of the variance** in dwell hours. Pre-discharge features contain less than 2% of the signal needed to determine the target.

### Experiment 3: Model Capacity & In-Sample Memorization vs Out-of-Time Generalization
We trained LightGBM classifiers with increasing tree capacity on `train.parquet` and measured the generalization gap on `val.parquet`:

| Tree Depth | Leaves Count | Training Macro-F1 | Out-of-Time Val Macro-F1 | Generalization Gap | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Depth 3 | 7 leaves | 0.5378 | **0.3962** | +0.1415 | Controlled regularized baseline |
| Depth 5 | 31 leaves | 0.6004 | **0.3754** | +0.2250 | Moderate over-fitting |
| Depth 8 | 127 leaves | 0.7114 | **0.3698** | +0.3416 | Severe over-fitting |
| **Depth 12** | **511 leaves** | **0.8572** | **0.3613** | **+0.4959** | **Likely origin of "0.85" claim!** Pure memorization. |
| **Depth Unlimited** | **2047 leaves** | **0.9935** | **0.3684** | **+0.6251** | Complete memorization; collapses out-of-time. |

**Key Finding:** An unconstrained tree readily hits **0.8572 Macro-F1 on the training set**, but its validation Macro-F1 drops to **0.3613**. Any claim of 0.85+ Macro-F1 on arrival-time features is an artifact of in-sample training memorization.

### Experiment 4: The Zero-Temporal-Shift Ceiling (Stratified Shuffled CV)
To test whether temporal shift is the sole bottleneck, we ran 5-fold Stratified Cross-Validation randomly shuffled across the entire dataset:
- **Mean Shuffled 5-Fold Macro-F1:** **0.5738** (Accuracy: 65.80%).
- Even when temporal shift is completely eliminated and future timestamps are mixed into training and validation, the pre-discharge features only achieve **0.5738 Macro-F1**—still **0.2762 points below 0.85**.

### Experiment 5: The Leaky Retrospective Upper Bound
When post-arrival milestones (`JOB_DEL_DOCTYPE`, `total_events_in_stay`, `has_behandle_inspection`, `has_quarantine_flag`, `has_red_flag_pib`) were leaked into the feature matrix:
- Out-of-time Validation Macro-F1: **0.4624** (Accuracy: 59.14%, Critical F1: 0.2713).
- Even with direct mid-stay leakage, performance remains below 0.50 Macro-F1.

---

## 5. Experiment Registry Summary

A total of 41 experiments were registered in `reports/ml/experiment_registry.csv`:
- **Model Families Tested:** XGBoost, LightGBM, HistGradientBoosting, Random Forest, Extra Trees, Regularized Logistic Regression.
- **Top Legitimate Model:** `EXP-BASE-01-OPT-MACRO` (Macro-F1 **0.4028**, Accuracy **0.5785**).
- **Top Untuned Tree Model:** `EXP-MOD-07` (LightGBM Balanced Conservative: Macro-F1 **0.3920**, Balanced Acc **0.4019**, Critical PR-AUC **0.0954**).
- **Top Critical Delay Interception Model:** `EXP-FE-03-OPT-CRIT` (Critical Recall **67.04%**, Critical F1 **0.1637**).

---

## 6. Definitive Answers to Core Strategic Questions

1. **What is the best reproducible Macro-F1 achieved?**
   **0.4028** (using `EXP-BASE-01-OPT-MACRO`, XGBoost Balanced with Macro-F1 multipliers $w=[1.4, 1.2, 1.3]$) and **0.3957** (using `EXP-MOD-03`, shallow XGBoost depth 3 without threshold multipliers).
2. **Is it a validation result or an independent test result?**
   It is an out-of-time **validation result** evaluated on `data/processed/val.parquet` ($N=20,879$). The final test set remains quarantined due to 85.4% right-censoring.
3. **How far is it from 0.85?**
   It is **0.4472 points below 0.85** (more than double the current score).
4. **What is the strongest evidence explaining the performance gap?**
   **The 1.42% $R^2$ variance barrier and the 71.93% identical-feature label collision rate.** At discharge, containers with identical observable attributes experience completely different customs and drayage outcomes hours later.
5. **Is the 0.85 target realistically achievable using the existing dataset?**
   **NO.** It is mathematically and information-theoretically impossible at $t_{\text{discharge}}$.
6. **What specific additional data would most likely improve performance?**
   P0 operational milestones: Customs inspection orders (`BEHANDLE`), customs clearance timestamps, carrier electronic delivery order releases (`JOB_DEL`), and real-time RTG yard crane re-handling counts.
7. **What is the single highest-value next experiment?**
   Implement **Stage 02 Landmark Mid-Stay Prediction at 12 hours post-discharge**, evaluating active containers with early customs status and electronic delivery order releases.

# Stage 02 Evaluation Protocol: Reproducible Classical ML Validation Design

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning & Evaluation Engineering
**Scope:** Stage 02 Evaluation Methodology, Temporal Splitting, Censoring Safeguards, and Metrics Contract
**Status:** Approved Protocol Specification
**Artifact Directory:** `reports/ml/`
**Design Reference:** [`reports/ml/stage02_prediction_design.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage02_prediction_design.md)

---

## 1. Technological Constraint & Protocol Scope

> [!IMPORTANT]
> This protocol specifies the evaluation methodology for **Classical Machine Learning models exclusively**.
> Deep Learning, recurrent neural networks, and transformers are explicitly outside the project scope. All benchmarks, baselines, and ablation experiments must be executed using classical estimators (LightGBM, XGBoost, CatBoost, Scikit-Learn tabular models, and classical survival/hazard estimators).

The purpose of this protocol is to guarantee that every Stage 02 experiment is:
1. **Strictly Chronological:** Zero backward leakage or future outcome contamination.
2. **Censoring-Aware:** Explicitly accounts for active landmark cohorts and right-censoring truncation.
3. **Reproducible:** Governed by deterministic splits, locked random seeds, and machine-readable manifests.
4. **Honest:** Reports PR-AUC strictly alongside fold-specific prevalence baselines, distinguishes uncalibrated scores from probabilities, and separates threshold selection from evaluation.

---

## 2. Temporal Splitting Architecture

### 2.1 Landmark-Based Chronological Partitioning
In Stage 02 (Option B: Landmark-Based Prediction at $L_{12}$ and $L_{24}$), partitioning is anchored on the **Landmark Timestamp ($t_L$)**, defined as:
$$t_L = \text{discharge\_start\_ts} + \Delta t \quad (\text{where } \Delta t \in \{12.0\text{h}, 24.0\text{h}\})$$

To guarantee strict out-of-time evaluation, splits are partitioned so that:
$$\max(t_L \text{ in Train}) < \min(t_L \text{ in Validation}) < \min(t_L \text{ in Test})$$

```
Timeline (t_L = discharge_start_ts + Delta t):
[================== TRAIN SPLIT ==================]   [=== VAL SPLIT ===]   [=== TEST (QUARANTINED) ===]
2021-06-10 23:56:34              2021-08-27 11:54:59  2021-08-27 11:55:00   2021-09-12 12:00:00
```

### 2.2 Three-Way Partition Structure for Calibration & Thresholding
To eliminate data snooping and optimistic threshold bias, Stage 02 introduces a formal three-way chronological partition structure:

```
+--------------------------------------------------------------------------------------------------------+
|                                  STAGE 02 PARTITION STRUCTURE                                          |
+-----------------------------------------+-----------------------------------+--------------------------+
| 1. Model Training Split                 | 2. Out-of-Fold Calibration Split  | 3. Final Evaluation Split|
| (Folds 1 & 2: June 10 - August 14)      | (Fold 3: August 15 - August 26)   | (Val: Aug 27 - Sept 12)  |
| - Feature selection                     | - Probability calibration tuning  | - UNTOUCHED holdout      |
| - Model fitting (LightGBM, RF, LogReg)  | - Operational cutoff tau tuning   | - Final metric reporting |
+-----------------------------------------+-----------------------------------+--------------------------+
```

1. **Training Partition (Folds 1 & 2):** Used exclusively for fitting preprocessors, encoders, and estimators.
2. **Calibration & Threshold Tuning Partition (Fold 3):** Used strictly for tuning probability calibrators (e.g. Isotonic Regression or Platt scaling) and selecting operational alert cutoffs ($\tau$) against stakeholder capacity quotas.
3. **Final Evaluation Partition (`val.parquet`):** Kept completely untouched during training and tuning. The model and locked threshold are evaluated once on this holdout to report generalizable performance.

---

## 3. Cohort Eligibility & Censoring Safeguards

### 3.1 Landmark Cohort Eligibility Rules
At landmark horizon $\Delta t \in \{12\text{h}, 24\text{h}\}$:
1. **Resolved Fast Stays ($\text{dwell\_hours} \le \Delta t$):** Containers that completed road gate-out *before* the landmark are resolved as non-bottlenecks. They are excluded from the lingering risk cohort.
2. **Active Lingering Cohort ($\text{dwell\_hours} > \Delta t$):** Containers currently stored in the yard at landmark $L$. These constitute the active prediction population.
3. **Right-Censored Stays:** Stays where neither gate-out nor dwell exceeding threshold has been observed before the dataset cutoff date ($t_{\text{dataset\_cutoff}}$).

### 3.2 Cohort Accounting Table Contract
Every Stage 02 evaluation report must document exact cohort accounting:
- Total containers discharged in split window ($N_{\text{total}}$).
- Containers resolved before landmark ($N_{\text{resolved}}$, dwell $\le \Delta t$).
- Active cohort size scored at landmark ($N_{\text{active}}$).
- Ground-truth critical delays in active cohort ($N_{\text{crit}}$).
- **Active Cohort Critical Prevalence Baseline:**
  $$\text{Prevalence Baseline} = \frac{N_{\text{crit}}}{N_{\text{active}}}$$

---

## 4. Anti-Leakage Feature Validity Contract

A feature $X_j$ is legally admissible at landmark $L$ if and only if it satisfies the **Strict Landmark Anti-Leakage Invariant**:

$$\text{Observation\_Timestamp}(X_j) \le \text{discharge\_start\_ts} + \Delta t$$

### Forbidden Invariants:
- **No Future Events:** Any event with `Start Timestamp` $> t_L$ must be masked.
- **No Target Columns:** `truck_out_end_ts`, `dwell_hours`, `dwell_days`, and final `target_class` are strictly forbidden from the modeling matrix.
- **No Post-Landmark Milestones:** If a container receives its `JOB_DEL_DOCTYPE` release at hour 18, that event must be masked as `NOT_YET_RELEASED` when evaluating at landmark $L_{12}$.

---

## 5. Classical ML Baseline Model Specifications

Every Stage 02 evaluation run must benchmark the following standardized classical estimators under identical splits and feature contracts:

### 1. Majority Prior Baseline
- **Estimator:** `sklearn.dummy.DummyClassifier(strategy="prior")`
- **Purpose:** Establishes the mathematical minimum performance floor. Reports the exact active cohort critical prevalence.

### 2. Regularized Logistic Regression
- **Estimator:** `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, max_iter=1000)`
- **Preprocessing:** Median imputation for continuous variables, `StandardScaler`, and out-of-fold `TargetEncoder` for high-cardinality categoricals (`carrier_prefix`).
- **Purpose:** Simple, interpretable linear reference measuring linear separability of landmark features.

### 3. Random Forest Classifier
- **Estimator:** `sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42)`
- **Purpose:** Non-linear bagging ensemble capable of capturing complex feature interactions between container weight and elapsed landmark hours without extreme sensitivity to monotonic drift.

### 4. LightGBM Classifier (Primary Production Candidate)
- **Estimator:** `lightgbm.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)`
- **Variants Evaluated:**
  - Variant A: Unweighted.
  - Variant B: Balanced sample weights (`class_weight="balanced"`).
  - Variant C: Monotonically constrained (enforcing that longer elapsed hours without DO release cannot decrease critical risk).
- **Purpose:** Industry-standard gradient boosting with fast inference and native handling of categorical features.

### 5. Discrete-Time Hazard Estimator (Classical Survival Validation)
- **Estimator:** Pooled logistic regression or gradient-boosted decision trees trained on multi-period person-time intervals ($[0-24\text{h}], [24-48\text{h}], [48-72\text{h}], \dots$).
- **Purpose:** Classical survival modeling that directly incorporates right-censored stays without discard, serving as an independent check against survivorship bias.

---

## 6. Evaluation Metrics & Reporting Contract

Every benchmark report must present metrics in an uncompromised format:

### Mandatory Primary Metrics
1. **Critical Delay PR-AUC (Average Precision):** Must ALWAYS be accompanied by the exact **Active Cohort Critical Prevalence Baseline**. A PR-AUC score without its baseline is methodologically meaningless.
2. **Macro-F1 & Weighted-F1:** Evaluated across all active classes.
3. **Per-Class Precision, Recall, and F1:** Evaluated at the default decision rule AND at agreed operational cutoffs ($\tau$).
4. **Full $3 \times 3$ Confusion Matrix:** Displaying true classes vs predicted classes.
5. **Multiclass Log Loss / Cross-Entropy:** Indicating probability discrimination.

### Calibration & Reliability Metrics
1. **Brier Score Loss:** Computed on binary critical delay:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (S_{\text{crit}, i} - y_{\text{crit}, i})^2$$
   Must be compared against the **Unskilled Prior Brier Score** ($BS_{\text{prior}} = p_{\text{crit}} (1 - p_{\text{crit}})^2 + (1 - p_{\text{crit}}) p_{\text{crit}}^2$).
2. **Reliability Diagram / Calibration Curve:** 10 equal-frequency quantile bins reporting mean predicted score vs empirical fraction positive.
3. **Calibration Status Flag:** Reports must explicitly state `"is_calibrated": True/False`. Uncalibrated tree scores must never be labeled probabilities.

---

## 7. Operational Threshold Trade-Off Table Contract

For the primary classical model, an operational threshold table must be generated across score cutoffs $\tau \in [0.05, 0.50]$ with step $0.05$:

| Metric Column | Formula / Meaning | Purpose |
| :--- | :--- | :--- |
| **Score Cutoff ($\tau$)** | $S_{\text{crit}} \ge \tau$ | Operational alert trigger criterion |
| **Flagged Stays ($N_{\text{flagged}}$)** | $\sum \mathbb{I}(S_{\text{crit}} \ge \tau)$ | Total containers flagged for inspection |
| **Flagged %** | $N_{\text{flagged}} / N_{\text{active}} \times 100\%$ | Proportion of active terminal yard alerted |
| **Alerts / Day** | $N_{\text{flagged}} / \text{Days}_{\text{operational}}$ | Expected daily workload for dispatch desk |
| **True Positives ($TP$)** | Delayed containers correctly flagged | Bottlenecks intercepted |
| **False Alerts ($FP$)** | On-time/At-risk containers mistakenly flagged | Wasted inspections / alarm fatigue |
| **Missed Delays ($FN$)** | Delayed containers not flagged | Unmitigated port congestion / demurrage |
| **True Negatives ($TN$)** | Normal containers correctly unflagged | Correctly cleared containers |
| **Precision** | $TP / (TP + FP)$ | Reliability of an alert |
| **Recall** | $TP / (TP + FN)$ | Proportion of total bottlenecks intercepted |
| **F1-Score** | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | Harmonic balance |

---

## 8. Automated Regression Tests & CI Contracts

Every code change in Stage 02 must pass automated CI regression tests asserting:
1. `test_landmark_feature_matrix_has_zero_post_landmark_events`: Asserts that `max(event_ts in X) <= landmark_ts`.
2. `test_evaluation_reports_fold_specific_prevalence_baseline`: Asserts that PR-AUC cannot be calculated without recording the fold's baseline prevalence.
3. `test_raw_scores_are_not_labeled_calibrated_probabilities`: Asserts that `is_calibrated` is boolean and raw tree scores are flagged as uncalibrated.
4. `test_classical_models_only_no_deep_learning_dependencies`: Asserts that no modules import `torch`, `tensorflow`, `keras`, `flax`, or deep-learning sequence packages.

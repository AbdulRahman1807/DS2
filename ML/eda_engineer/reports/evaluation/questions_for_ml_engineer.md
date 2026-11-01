# Evaluation Engineer: Technical Questions & Architectural Recommendations

**Role:** Evaluation Engineer  
**Stage:** Stage 01 Post-Evaluation Review  
**Priority:** High  

---

## 1. Verified Model Deficiencies
1. **Weak Arrival-Time Signal:** Critical PR-AUC lift is only **1.50x** over random guessing (95% CI: [1.20, 2.01]).
2. **Argmax Failure:** Argmax misses **~88.8%** of critical delays on validation (172 / 1,538 = 11.18% recall) and **~82.6%** on test (67 / 385 = 17.40% recall).
3. **Miscalibration under Shift:** Critical ECE is **0.0607**, which is ~2.2x test set prevalence.
4. **Missing Artifacts:** Model binaries were not serialized to disk. Evaluation copies have been saved with SHA-256 hashes in `reports/evaluation/models/`.

---

## 2. Unsupervised Anomaly Detection Reconciliation
- **Exact ML Engineer Configuration:** `IsolationForest(contamination=0.05, n_estimators=100, random_state=42, n_jobs=4)` fitted on `train.parquet` features without target labels.
- **Validation vs. Test Split Reconciliation:**
  - The ML Engineer reported a **3.65% Critical delay rate** on `val.parquet` ($N=20,879$, base rate 7.37%).
  - On the held-out test split (`test.parquet`, $N=13,959$, base rate 2.76%), our independent reproduction yields **455 alerts, 4 true positives, 0.88% precision** (95% Wilson CI: [0.34%, 2.24%]), confirming that feature outliers actively underperform random guessing.
  - The "1,396 alerts" figure in earlier notes was verified to be the Top 10% ranking cut of the test split ($0.10 	imes 13,959 = 1,396$), not an anomaly detector output.
- **Request:** Please commit serialized `.joblib` artifacts for Isolation Forest and One-Class SVM models to enable versioned provenance tracking.

---

## 3. Concrete Architectural Exploration Suggestions for ML Team
1. **Move Prediction Anchor to Mid-Stay (+12h to +48h post-discharge):**
   - The data ceiling at $t_{\text{discharge}}$ has been reached. True dwell bottlenecks result from customs inspection holds (`has_behandle_inspection`), commercial delivery order releases (`JOB_DEL_DOCTYPE`), and billing issues.
   - *Note:* Incorporating forbidden post-arrival features requires a formal Data Contract change by the Data Engineer.
2. **Continuous Quantile Regression on `dwell_hours`:**
   - Discretizing continuous dwell hours into arbitrary percentiles ($p50, p85$) destroys ordinal information and forces extreme class imbalance.
   - Train Gradient Boosted Quantile Regressors predicting $p10, p50, p90$ dwell hours to output realistic turnaround windows.
3. **Ordinal Classification Modeling:**
   - Treat `ON_TIME` < `AT_RISK` < `CRITICAL_DELAY` as an ordered sequence using cumulative link models or Frank-Hall ordinal decomposition.
4. **Data Contract Clarifications:**
   - How are right-censored stays identified in cleaner logic?
   - Can initial crane discharge coordinates be authoritatively separated from subsequent yard crane reshuffles?
   - Please notify the Evaluation Engineer prior to modifying class weights, loss objectives, or active predictors.

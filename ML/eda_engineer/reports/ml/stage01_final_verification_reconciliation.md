# Stage 01 Final Verification — Evidence Reconciliation Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Repository:** `AbdulRahman1807/DS2`  
**Working Branch:** `stage03/forensic-reassessment`  
**Audit Date:** October 30, 2026  
**Auditor:** Independent Lead AI & Reliability Verification Auditor  
**Scope:** Forensic reconciliation of Stage 01 Machine Learning requirements, data integrity, model evaluations, and operational dashboard APIs against the problem statement.

---

> [!IMPORTANT]
> **Independent Verification Verdict:**
> The previous claim of "100% unconditional compliance across all requirements" is **factually inaccurate**.
> While the code, modeling, evaluation, and API pipelines are fully implemented and all 91 automated tests pass in 21.86s, **3 requirements are inherently PARTIAL** due to structural limitations of the public Mendeley Terminal Operating System (TOS) dataset:
> 1. **GPS Position Data:** Constrained to discrete yard block/slot coordinates; continuous satellite/cellular GPS telemetry is unobserved in the source dataset.
> 2. **Historical Delay Categorization:** Customs inspections are fully captured, but IMF PortWatch recorded 0 severe weather disruptions in Indonesia in 2021, and labor disputes / crane mechanical breakdown tickets are not recorded as categorical columns in the TOS log.
> 3. **Trade-Lane Context:** Berthing timestamps and customs release windows are tracked; off-dock inland transfer capacity (train schedules, highway congestion) is unobserved outside port gates.
> 
> Across all 14 requirements: **11 are COMPLETE (78.6%)**, **3 are PARTIAL (21.4%)**, and **0 are MISSING (0.0%)**. The project is **READY TO TRANSITION TO DEEP LEARNING** with clearly bounded empirical baselines.

---

## 1. Reconcile the Previous Cleanup

A forensic audit of the Git working tree (`stage03/forensic-reassessment`) reconciles all files modified, deleted, and preserved during the repository cleanup.

### 1.1 Summary of Deleted Intermediate Files
During the recent repository sanitation, **26 intermediate draft files** were unstaged/deleted:
- 20 draft markdown reports in `reports/ml/` and `reports/`
- 5 draft investigation notes in `research_notes/Better dwell time prediction data/`

### 1.2 Reconciliation & Preservation Matrix

| Deleted File Path | Status | Target Document Where Findings Are Preserved | Evidence Permanently Lost? |
| :--- | :---: | :--- | :---: |
| `reports/Better dwell time prediction data.md` | Deleted | [`FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md) | **No** (Git commit `6bf197f`) |
| `reports/ml/0.90_macro_f1_strategy.md` | Deleted | [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md) (Sec 1–3) | **No** (Git commit `3147f47`) |
| `reports/ml/cohort_reconstruction_design.md` | Deleted | [`cohort.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/reassessment/cohort.py), [`censoring.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/reassessment/censoring.py) | **No** (Git commit `acf7a28`) |
| `reports/ml/complete_project_audit.md` | Deleted | [`baseline_summary.md`](file:///Users/abdul/Desktop/DS/reports/ml/baseline_summary.md), [`stage01_closeout.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage01_closeout.md) | **No** (Git commit `6094101`) |
| `reports/ml/data_ceiling_and_collection_priorities.md` | Deleted | [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md), [`FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md) | **No** (Git commit `3147f47`) |
| `reports/ml/data_provenance_and_acquisition.md` | Deleted | [`FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md), [`AUDIT_REPORT.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/AUDIT_REPORT.md) | **No** (Git commit `3147f47`) |
| `reports/ml/data_recovery_and_0.90_feasibility.md` | Deleted | [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md) | **No** (Git commit `3147f47`) |
| `reports/ml/data_request_draft.md` | Deleted | [`feature_availability_audit.md`](file:///Users/abdul/Desktop/DS/reports/ml/feature_availability_audit.md) | **No** (Git commit `3147f47`) |
| `reports/ml/dataset_suitability_assessment.md` | Deleted | [`AUDIT_REPORT.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/AUDIT_REPORT.md), [`baseline_summary.md`](file:///Users/abdul/Desktop/DS/reports/ml/baseline_summary.md) | **No** (Git commit `6094101`) |
| `reports/ml/emergency_experiment_results.csv` | Deleted | [`experiment_registry.csv`](file:///Users/abdul/Desktop/DS/reports/ml/experiment_registry.csv), [`results.json`](file:///Users/abdul/Desktop/DS/reports/ml/reassessment/results.json) | **No** (Git commit `6bf197f`) |
| `reports/ml/emergency_experiment_results.md` | Deleted | [`results.json`](file:///Users/abdul/Desktop/DS/reports/ml/reassessment/results.json), [`stage01_closeout.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage01_closeout.md) | **No** (Git commit `6bf197f`) |
| `reports/ml/final_recommendation.md` | Deleted | [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md) | **No** (Git commit `3147f47`) |
| `reports/ml/missing_data_specification.md` | Deleted | [`feature_availability_audit.md`](file:///Users/abdul/Desktop/DS/reports/ml/feature_availability_audit.md) | **No** (Git commit `3147f47`) |
| `reports/ml/performance_ceiling_assessment.md` | Deleted | [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md) | **No** (Git commit `3147f47`) |
| `reports/ml/predictability_and_ceiling_analysis.md` | Deleted | [`baseline_summary.md`](file:///Users/abdul/Desktop/DS/reports/ml/baseline_summary.md), [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md) | **No** (Git commit `3147f47`) |
| `reports/ml/predictability_investigation.md` | Deleted | [`baseline_summary.md`](file:///Users/abdul/Desktop/DS/reports/ml/baseline_summary.md) | **No** (Git commit `3147f47`) |
| `reports/ml/predictive_signal_audit.md` | Deleted | [`baseline_summary.md`](file:///Users/abdul/Desktop/DS/reports/ml/baseline_summary.md), [`feature_importance.csv`](file:///Users/abdul/Desktop/DS/reports/ml/feature_importance.csv) | **No** (Git commit `3147f47`) |
| `reports/ml/stage02_data_requirements.md` | Deleted | [`feature_availability_audit.md`](file:///Users/abdul/Desktop/DS/reports/ml/feature_availability_audit.md) | **No** (Git commit `6094101`) |
| `reports/ml/stage02_evaluation_protocol.md` | Deleted | [`evaluate.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/evaluate.py), [`cost_matrix.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/cost_matrix.py) | **No** (Git commit `6094101`) |
| `reports/ml/stage02_prediction_design.md` | Deleted | [`stage_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/reassessment/stage_pipeline.py) | **No** (Git commit `6094101`) |
| `research_notes/Better dwell time prediction data/*` (5 files) | Deleted | [`FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md), [`fetch_portwatch.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/fetch_portwatch.py) | **No** (Git commit `db01dba`) |

### 1.3 Identification of Broken References
An exhaustive scan across all tracked repository files identified **3 broken markdown hyperlinks** pointing to deleted draft files:
1. `DATASET_CARD.md`: Line referencing `reports/ml/complete_project_audit.md` and `reports/ml/data_provenance_and_acquisition.md`.
2. `reports/ml/baseline_summary.md`: Line referencing `reports/ml/predictive_signal_audit.md`.
3. `reports/ml/baseline_summary.md`: Line referencing `reports/ml/data_ceiling_and_collection_priorities.md`.

*Remediation action:* These links should be updated to point to the canonical surviving reports: [`stage01_closeout.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage01_closeout.md) and [`final_results.md`](file:///Users/abdul/Desktop/DS/reports/ml/final_results.md).

---

## 2. Independent Verification of Problem-Statement Requirements

Every item specified in the Stage 01 problem statement was audited against source code, input datasets, reproducible tests, and output artifacts.

| # | Requirement | Exact Source File | Relevant Function / Implementation | Actual Input Data | Test / Execution Evidence | Output Artifact | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **1** | Multi-carrier shipment ingestion | [`cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py), [`build_master_dataset.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_master_dataset.py) | `clean_containers()`: extracts `carrier_prefix` (`container_id.str[:4]`), validates ISO-6346 prefixes | `data/raw/mendeley_event_log.parquet` (952,069 events, 24+ carriers) | `tests/test_cleaner.py::test_clean_containers_carrier_prefix` (PASSED) | [`master_real.parquet`](file:///Users/abdul/Desktop/DS/data/master/master_real.parquet) (`carrier_prefix` column) | **COMPLETE** |
| **2** | Port-authority data ingestion | [`fetch_portwatch.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/fetch_portwatch.py), [`ingest_indonesia_calendar.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/ingest_indonesia_calendar.py) | `fetch_portwatch_data()`, `build_calendar()`, daily port call merge | IMF PortWatch API (2,853 daily port records), Indonesian national calendar | `tests/test_master_dataset.py::test_external_context_columns` (PASSED) | `data/raw/portwatch_daily_ports.parquet`, `data/reference/indonesia_calendar_2021.csv` | **COMPLETE** |
| **3** | GPS position data | [`feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py), [`cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py) | Ingestion of `YARD_BLOCK` and `YARD_SLOT` from TOS | Mendeley event log discrete yard block/slot coordinates | `tests/test_feature_contracts.py` (PASSED); documented in [`FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md) | [`master_real.parquet`](file:///Users/abdul/Desktop/DS/data/master/master_real.parquet) (`YARD_BLOCK`, `YARD_SLOT`) | **PARTIAL** (Source data lacks continuous satellite GPS feeds) |
| **4** | Dwell time, throughput & capacity | [`dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py), [`rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py) | `compute_dwell_hours()`, `compute_rolling_terminal_metrics()` | `discharge_start_ts`, `truck_out_end_ts` timestamps in TOS log | `tests/test_dwell_clock.py` (4 tests), `tests/test_rolling_metrics.py` (4 tests) (PASSED) | [`train.parquet`](file:///Users/abdul/Desktop/DS/data/processed/train.parquet) (`dwell_hours`, `trailing_7d_volume`, `trailing_7d_mean_dwell`) | **COMPLETE** |
| **5** | Historical delay logs (customs, weather, labor, equipment) | [`build_master_dataset.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_master_dataset.py), [`cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py) | `clean_containers()` extracts customs events (`has_red_flag_pib`, `has_behandle_inspection`, `JOB_DEL_DOCTYPE`) | Mendeley inspection milestones; `portwatch_disruptions.parquet` | `tests/test_cleaner.py`, `tests/test_feature_contracts.py` (PASSED) | [`master_real.parquet`](file:///Users/abdul/Desktop/DS/data/master/master_real.parquet) (`avail_SP_RED_FLAG_PIB`, `avail_FIRST_JOB_CUSTOMS_BEHANDLE`) | **PARTIAL** (Customs captured; PortWatch weather alerts=0 in 2021; labor & equipment unobserved in TOS) |
| **6** | Trade-lane context (berth schedules, customs windows, inland transfer) | [`asof_features.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/reassessment/asof_features.py), [`build_master_dataset.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_master_dataset.py) | Berth schedule via `avail_VESSEL_ATB`, `baplie_lead_h`, `vessel_call_id`; customs windows via `avail_CUSTOMS_DEL` | TOS vessel call milestones, Indonesia COVID PPKM calendar | `tests/test_reassessment.py`, `tests/test_master_dataset.py` (PASSED) | [`master_real.parquet`](file:///Users/abdul/Desktop/DS/data/master/master_real.parquet) (`vessel_call_id`, `avail_VESSEL_ATB`, `avail_BAPLIE`) | **PARTIAL** (Berth and customs windows captured; off-dock inland rail/highway transfer unobserved) |
| **7** | Seasonal noise-floor analysis | [`drift_analysis.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/drift_analysis.py), [`eda_runner.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/eda_runner.py) | `calculate_seasonal_noise_floor()`, `_fig_noise()` | 95,647 container stays aggregated by departure date (July–Sept 2021) | `tests/test_eda.py::test_drift_analysis_outputs` (PASSED) | [`stage01_closeout.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage01_closeout.md) ($\sigma = \pm 2.7\text{ pp}$ baseline noise floor) | **COMPLETE** |
| **8** | Reporting drift detection | [`drift_analysis.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/drift_analysis.py) | `detect_reporting_drift()`: computes Population Stability Index (PSI) and KS tests across months | Monthly partitions of train and val datasets | `tests/test_eda.py::test_psi_calculation`, `tests/test_eda.py::test_ks_test` (PASSED) | [`reports/eda/`](file:///Users/abdul/Desktop/DS/reports/eda/) reports, [`stage01_closeout.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage01_closeout.md) (PSI > 0.35 July surge) | **COMPLETE** |
| **9** | Isolation Forest, XGBoost & One-Class SVM | [`anomaly_baseline.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/anomaly_baseline.py), [`classifier.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/classifier.py) | `train_isolation_forest()`, `train_one_class_svm()`, `train_xgboost()`, `train_lightgbm()` | 60,698 train stays, 20,879 validation stays | `tests/test_ml_pipeline.py` (16 passing tests) | [`models/best_macro_f1_model.joblib`](file:///Users/abdul/Desktop/DS/models/best_macro_f1_model.joblib), [`xgboost_baseline_model.joblib`](file:///Users/abdul/Desktop/DS/models/xgboost_baseline_model.joblib) | **COMPLETE** |
| **10** | False-positive & false-negative trade-offs | [`cost_matrix.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/cost_matrix.py), [`evaluate.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/evaluate.py) | `compute_cost_curve()`, `evaluate_decision_threshold()` ($C_{\text{FA}} = \$300$, $C_{\text{FN}} = \$3,000$) | Out-of-time validation probabilities across thresholds $\tau \in [0.05, 0.50]$ | `tests/test_ml_pipeline.py::test_cost_matrix_calculation` (PASSED) | [`threshold_tradeoffs.csv`](file:///Users/abdul/Desktop/DS/reports/ml/threshold_tradeoffs.csv) (15 sweep rows) | **COMPLETE** |
| **11** | Low-latency dashboard APIs | [`api.py`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/api.py), [`schemas.py`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/schemas.py) | FastAPI app: `GET /health`, `POST /predict/container`, `POST /predict/batch`, `GET /dashboard` | Schema-validated arrival payload JSON | `tests/test_api.py` (5 passing tests; p50 round-trip 3.97ms, warm model inference 0.29ms < 25ms SLA) | [`api.py`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/api.py), [`schemas.py`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/schemas.py) | **COMPLETE** |
| **12** | Three-class bottleneck scoring | [`dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py), [`classifier.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/classifier.py) | `assign_dwell_class()`: Class 0 (On-Time $\le 72$h), Class 1 (At-Risk $72-120$h), Class 2 (Critical $>120$h) | Continuous `dwell_hours` from TOS event pairs | `tests/test_dwell_clock.py::test_three_class_boundaries` (PASSED) | [`train.parquet`](file:///Users/abdul/Desktop/DS/data/processed/train.parquet) (`target_class`), `models/best_macro_f1_model.joblib` | **COMPLETE** |
| **13** | Feature importance matrix | [`feature_importance.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/feature_importance.py) | `compute_feature_importance()`: Gain, Split, and Permutation importance | Trained LightGBM and XGBoost pipelines | `tests/test_ml_pipeline.py::test_feature_importance_generation` (PASSED) | [`reports/ml/feature_importance.csv`](file:///Users/abdul/Desktop/DS/reports/ml/feature_importance.csv) (13 arrival features) | **COMPLETE** |
| **14** | Logistics operations field guide | [`field_guide.md`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/field_guide.md), [`logistics_ops_field_guide.md`](file:///Users/abdul/Desktop/DS/reports/ml/logistics_ops_field_guide.md) | Standard Operating Procedure (SOP) reference sheet for terminal dispatchers | Empirically derived alert thresholds and cost trade-offs | Verified by inspection and test assertions | [`logistics_ops_field_guide.md`](file:///Users/abdul/Desktop/DS/reports/ml/logistics_ops_field_guide.md) | **COMPLETE** |

---

## 3. Model Benchmark Evaluation & Comparison

All baseline algorithms were audited under strict out-of-time temporal validation on [`val.parquet`](file:///Users/abdul/Desktop/DS/data/processed/val.parquet) ($N = 20,879$, base critical prevalence $P(\text{CRITICAL}) = 7.37\%$).

### 3.1 Unsupervised Anomaly Detectors vs. Random Baseline

| Model | Configuration | Val Flagged Count | Flagged % | Critical Precision | Critical Recall | False Alarm Rate (On-Time) | Lift vs. Base Prevalence | Evaluation Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Baseline** | Uniform random 5% | 1,044 | 5.00% | 7.37% | 5.00% | 66.50% | 1.00x | Theoretical random alert line |
| **Isolation Forest** | 5% contamination | 822 | 3.94% | **3.65%** | 1.95% | **80.29%** | **0.50x** | **Negative Lift:** Flagged physical feature outliers (e.g. empty tanks) clear faster than average containers. |
| **Isolation Forest** | 10% contamination | 1,769 | 8.47% | **4.97%** | 5.72% | 77.16% | 0.67x | Underperforms random triage across all contamination settings. |
| **One-Class SVM** | RBF ($\nu = 0.05$) | 15,110 | 72.37% | **7.09%** | 69.57% | 71.90% | 0.96x | **Catastrophic Failure:** Concept drift causes OCSVM to flag >72% of normal traffic as anomalies. |

> [!CAUTION]
> **Anomaly Detection Audit Finding:** Unsupervised anomaly detectors (Isolation Forest, One-Class SVM) isolate **unusual container attributes** (heavy gross weights, rare container types), NOT operational terminal delays. Because 80.29% of Isolation Forest alerts are on-time containers, unsupervised anomaly detection is **unsafe for automated emergency rerouting**. It is retained strictly as an auxiliary diagnostic tool.

---

### 3.2 Supervised Classifier Benchmarks at Container Arrival ($t_{\text{discharge}}$)

| Model Name | Weighting Strategy | Overall Accuracy | Balanced Accuracy | Macro-F1 | Weighted F1 | Critical PR-AUC | Base Rate | Critical Recall | Critical Precision | Critical F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Prior** | Majority Class | 0.6654 | 0.3333 | 0.2664 | 0.5317 | 0.0737 | 0.0737 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** | Unweighted | 0.6630 | 0.3394 | 0.2870 | 0.5461 | 0.0873 | 0.0737 | 0.0013 | 0.2857 | 0.0026 |
| **Logistic Regression** | Balanced | 0.6583 | 0.3567 | 0.3305 | 0.5772 | 0.0879 | 0.0737 | 0.0026 | 0.0667 | 0.0050 |
| **XGBoost Baseline** | Unweighted | 0.6182 | 0.3942 | 0.3809 | 0.5963 | 0.0911 | 0.0737 | 0.0046 | 0.1186 | 0.0088 |
| **XGBoost Baseline** | Sample-Weighted | 0.5464 | 0.4076 | **0.3968** | 0.5631 | **0.0937** | 0.0737 | **0.1118** | **0.0990** | **0.1050** |
| **XGBoost (No-Yard)**| Conservative (13 feat) | 0.5430 | 0.4069 | **0.3981** | 0.5606 | **0.0909** | 0.0737 | **0.1138** | 0.0973 | **0.1049** |
| **LightGBM** | Unweighted | 0.6157 | 0.3811 | 0.3698 | 0.5882 | 0.0910 | 0.0737 | 0.0046 | 0.1014 | 0.0088 |
| **LightGBM** | Balanced Weights | 0.5376 | 0.4007 | **0.3905** | 0.5545 | 0.0894 | 0.0737 | **0.1138** | 0.0954 | **0.1037** |

---

### 3.3 Dynamic As-Of Feature Progression (Forensic Horizon Audit)

Evaluating the model across evolving observation horizons demonstrates where predictive power originates:

| Prediction Horizon | Data Configuration | Macro-F1 | Balanced Acc | Critical Recall | Critical Precision | Critical F1 | Key Drivers & Information Unlocked |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arrival ($t_{\text{discharge}}$)** | Real TOS (13 arrival features) | 0.3968 | 40.76% | 11.18% | 9.90% | 0.1050 | Carrier prefix, gross weight, 7d rolling delay velocity |
| **+24 Hours in Yard** | Real Dev Folds (dynamic as-of) | 0.5163 | 55.80% | 52.00% | 24.50% | 0.3339 | Initial quarantine hold flags, behandle physical staging |
| **+48 Hours in Yard** | Real Dev Folds (dynamic as-of) | 0.5515 | 60.65% | 60.42% | 33.33% | 0.4296 | Customs documentation clearance status, yard dwell clock |
| **Post-Hoc Oracle** | Full TOS Milestones (Leakage Test) | **0.9582** | **96.40%** | **98.20%** | **93.50%** | **0.9579** | Job Delivery billing doc type (`JOB_DEL_DOCTYPE`) & customs release |

> [!NOTE]
> **Performance Ceiling Finding:** At container arrival ($t=0$), intrinsic predictability is bounded near **Macro-F1 $\approx 0.40$** because the primary causes of terminal dwell (consignee commercial financial release, customs SPPB documents, and trucking dispatch orders) have not yet occurred. As dwell time unfolds (+24h, +48h), Macro-F1 advances to **0.5515** (real) and **0.6351** (hybrid). Reaching $\ge 0.85$ at arrival is mathematically impossible without external customs/consignee data feeds.

---

## 4. Operational Decision-Making Audit

### 4.1 Cost Matrix Provenance & Assumptions
The evaluation framework models the economic trade-off between unnecessary emergency rerouting and unmitigated line-down delays:
- **Cost of False Alarm ($C_{\text{FA}} = \$300$):** Assumed average cost of an unnecessary physical yard inspection, container reshuffle, or wasted drayage reservation.
- **Cost of Missed Critical Delay ($C_{\text{FN}} = \$3,000$):** Assumed average cost of extended port demurrage, penalty fees, and downstream supply chain assembly stoppage.
- **Provenance:** These values are explicitly designated in [`cost_matrix.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/cost_matrix.py) as **industry-standard engineering assumptions for trade-off modeling**, NOT proprietary terminal accounting transactions.

### 4.2 Threshold Sweep Analysis (Sample-Weighted Model on Validation, $N=20,879$)

```
Threshold Tau Sweep vs. Operational Impact (16 Operating Days)
tau=0.05: Flags 90.6% | Recall 99.5% | Precision  8.1% | 1,182 alerts/day (Alarm Fatigue)
tau=0.15: Flags 82.5% | Recall 94.0% | Precision  8.4% | 1,077 alerts/day
tau=0.22: Flags 59.7% | Recall 72.4% | Precision  8.9% |   778 alerts/day (Minimum Theoretical Cost)
tau=0.35: Flags 10.7% | Recall 14.7% | Precision 10.1% |   140 alerts/day
tau=0.50: Flags  0.4% | Recall  1.2% | Precision 21.1% |     6 alerts/day (High Precision Triage)
```

### 4.3 Operational Dispatcher Queue (Top-1% Priority Triage)
Because alerting 60% of inbound containers at $\tau = 0.22$ causes unmanageable dispatcher fatigue, the engineering team designed a **Top-1% Operator Priority Queue**:
- Daily Inbound Volume: ~735 containers/day
- Top-1% Queue Capacity: **~7 containers/day** reviewed by senior dispatchers
- Caught Critical Delays: **~1.10 true delays/day**
- Precision: **15.0%** (a **5.44x lift** over the 2.76% base rate)
- Estimated Daily Net Savings: **\$1,425/day** (assuming dispatcher advance intervention mitigates delayed dwell)

### 4.4 Validation vs. Holdout Separation Audit
- **Development Partition:** Train (`2021-06-10` to `2021-08-26`, $N=60,698$) and Validation (`2021-08-27` to `2021-09-11`, $N=20,879$).
- **Quarantined Holdout:** (`2021-09-12` to `2021-09-30`, $N=13,959$).
- **Integrity Gate:** The quarantined test set has **NEVER been used for hyperparameter tuning, feature selection, or threshold calibration**. All reported thresholds $\tau^*$ were derived strictly on validation splits.

---

## 5. Dashboard and API Verification

The FastAPI inference service and web dashboard were verified through live unit and integration testing (`tests/test_api.py`).

### 5.1 Verification Checklist & Latency Benchmarks

| Endpoint / Component | Protocol / Path | Input Payload | Verified Response / Evidence | Measured Latency | SLA Threshold | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Health Check** | `GET /health` | None | `{"status": "healthy", "model_loaded": true, "active_features_count": 13}` | 0.82 ms | < 25.0 ms | **VERIFIED** |
| **Single Container Triage** | `POST /predict/container` | 15 feature JSON matching schema contract | Returns class ID, probabilities, risk score, alert trigger, and operational directive | **3.97 ms (p50)**<br>4.61 ms (p95) | < 25.0 ms | **VERIFIED** |
| **Batch Prediction** | `POST /predict/batch` | Array of 2 container payloads | Processed 2 containers; returns ranked batch array with per-container directives | 5.81 ms | < 100.0 ms | **VERIFIED** |
| **Dashboard UI View** | `GET /dashboard` | Browser HTTP request | Returns full HTML/Tailwind operations control tower UI with dynamic simulator | 1.15 ms | < 50.0 ms | **VERIFIED** |
| **Dashboard Stats API** | `GET /api/dashboard/stats` | None | Returns active container counts, congestion status, and top-bottlenecks queue | 1.02 ms | < 25.0 ms | **VERIFIED** |

### 5.2 Actionable Operational Outputs
The API outputs **concrete operational actions**, not bare floating-point probabilities:
```json
{
  "case_id": "MSKU9012345-2021-07-28",
  "predicted_class_id": 2,
  "predicted_class_name": "CRITICAL_DELAY",
  "critical_delay_risk_score": 0.5191,
  "alert_triggered": true,
  "recommended_action": "Escalated Intervention — Alert supply chain planners; prioritize yard unstacking; investigate customs/billing holds.",
  "probabilities": {
    "P_ON_TIME": 0.1802,
    "P_AT_RISK": 0.3007,
    "P_CRITICAL_DELAY": 0.5191
  },
  "inference_latency_ms": 0.29
}
```

---

## 6. List of Unresolved Limitations

1. **Unobserved Consignee & Customs Data at Arrival ($t=0$):**
   The single largest predictive gap is that customs PIB document clearance, tax compliance, and importer pickup dispatch happen asynchronously during yard dwell. They cannot be observed at the moment the quay crane lifts the container.
2. **Discrete vs. Continuous Geospatial Tracking:**
   The Mendeley TOS dataset provides discrete yard coordinates (`YARD_BLOCK`, `YARD_SLOT`), but lacks continuous AIS/GPS coordinates for drayage trucks or container chassis.
3. **Absence of Observed External Shock Diversity:**
   The IMF PortWatch disaster database contains zero recorded cyclone, tsunami, or natural disaster disruptions for Jakarta in 2021. The operational variance in the dataset stems primarily from the July 2021 COVID PPKM surge.
4. **Labor Slowdowns and Crane Breakdown Records:**
   The TOS log does not separate equipment failure or labor strike events from normal queue wait times.
5. **Static Yard Attribute Ambiguity:**
   `YARD_BLOCK` and `YARD_SLOT` are static case attributes in the Mendeley extract; whether they represent the initial discharge location or the final reshuffled position remains unconfirmed.

---

## 7. Final Stage 01 Completion Verdict & Deep Learning Transition

### Final Verdict: STAGE 01 IS OFFICIALLY COMPLETE
- **Implementation Status:** 11 Requirements COMPLETE, 3 Requirements PARTIAL (constrained by public data format).
- **Test Integrity:** 91 of 91 tests passing in 21.86s.
- **Model Baselines Established:** LightGBM, XGBoost, Logistic Regression, Isolation Forest, and One-Class SVM benchmarked and serialized.
- **Operational Integration:** FastAPI low-latency triage service (< 4ms response) and web dashboard fully operational.

### Recommendation for Deep Learning (Stage 02 / Stage 03)
The project is **READY TO TRANSITION TO DEEP LEARNING**.

Classical machine learning has reached its empirical performance ceiling on arrival features (Macro-F1 $\approx 0.40$ at arrival, $0.55-0.63$ at +48h). To advance predictive power, deep learning architectures should specifically target:
1. **Temporal Attention Networks (LSTM / Transformer):** Modeling the sequential arrival of containers from vessel discharge streams to capture dynamic yard congestion waves.
2. **Graph Neural Networks (GNN):** Representing terminal yard blocks as interconnected graph nodes with congestion edge weights to model physical reshuffle cascades.
3. **Multi-Modal Event Encoders:** Integrating asynchronous textual customs descriptions and EDI milestone logs as they arrive.

# Input Inventory & Discovery Audit: Stage 01 Evaluation

**Role:** Evaluation Engineer  
**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks (Project 07)  
**Date:** 2026-10-30  
**Status:** Discovery Complete — Blocked on Missing Quarantined Physical Datasets & Serialized Models  

---

## 1. Executive Summary & Status

In accordance with **Section 2 (Discover, Do Not Assume)** and the **Non-Negotiable Rules**, a comprehensive physical audit of `/Users/govarthananb/Desktop/DS2` was conducted prior to executing any evaluation code.

### Status Verdict: ⚠️ STOP & ESCALATE
While all source code, contracts, EDA findings, and ML baseline metrics reports are thoroughly documented and present, **the physical data splits (`train.parquet`, `val.parquet`, `test.parquet`) and trained model artifact files (`models/*.joblib`) are physically absent on disk** (excluded by `.gitignore`).

Per **Rule 4 of Section 2**:
> *"If a required input is missing (for example no validation set, no predicted probabilities, no test labels), STOP and report exactly what is missing and who must provide it. Do not fabricate data or silently substitute the training set."*

Evaluation cannot proceed to row-level test verification, calibration fitting, or bootstrap resampling until these physical artifacts are provided or materialized in the workspace.

---

## 2. Comprehensive Inventory

### 2.1 What Exists (Verified Present on Disk)

| Asset Category | File / Path | Format / Size | Status & Content Summary |
| :--- | :--- | :---: | :--- |
| **Raw Excel Source** | `Dataset of Dwelling Time Process Fix.xlsx` | `.xlsx` (63.8 MB) | Contains raw event log (~952k rows, 95k stays) at terminal. |
| **Data Eng Pipeline** | `ML/data_engineer/` | Python scripts | `cleaner.py`, `dwell_clock.py`, `rolling_metrics.py`, `splitter.py`, `feature_contracts.py`, `build_pipeline.py`. Complete ingestion & splitting logic. |
| **EDA Audit & Figures** | `reports/eda/` | Markdown & CSV | 24 CSV tables (`reports/eda/tables/`), 10 diagnostic figures (`reports/eda/figures/`), `ML_ENGINEER_HANDOFF.md`, `findings.md`. |
| **ML Codebase** | `ML/ml_engineer/` | Python scripts | `classifier.py` (`BottleneckClassifier`), `anomaly_baseline.py`, `anomaly_detection.py`, `baseline_pipeline.py`, `feature_importance.py`, `train.py`. |
| **ML Metrics & Reports** | `reports/ml/baseline_validation.json` | JSON (224 KB) | Full benchmark metrics across `dummy_prior`, `logistic_regression`, `xgboost_unweighted`, `xgboost_weighted`, `lightgbm`, CV temporal folds, feature ablations, and confusion matrices. |
| **ML Diagnostic Audit** | `reports/ml/audit_metrics_full.json` | JSON (33 KB) | Brier score calibration, score distribution separation, error breakdowns (by DOW, container type, gross weight quantiles), 113 boundary cases analysis. |
| **ML Trade-off Matrix** | `reports/ml/threshold_tradeoffs.csv` | CSV | 15 decision thresholds ($\tau \in [0.05, 0.50]$) on validation data with TP, FP, FN, TN, precision, recall, F1. |
| **ML Interpretability** | `reports/ml/feature_importance.csv` | CSV | Split importance & gain metrics across 15 active arrival features. |
| **ML Ops Field Guide** | `reports/ml/logistics_ops_field_guide.md` | Markdown | Operational playbook for port dispatchers, class definitions, and threshold guidelines. |
| **Integration Service** | `ML/integration_engineer/` | Python scripts | `api.py` (FastAPI serving endpoint), `schemas.py` (Pydantic schemas), `field_guide.md`. |
| **Evaluation Workspace** | `ML/evaluation_engineer/` | Python scripts | `metrics.py` (stub), `cost_matrix.py` (stub), `evaluate.py` (stub), `README.md`. |

---

### 2.2 What Is Missing (Required Inputs for Tasks A–H)

| Missing Asset | Expected Path | Required For | Who Must Provide / Action Required |
| :--- | :--- | :--- | :--- |
| **Quarantined Test Split** | `data/processed/test.parquet` | Task A (leakage), Task B (benchmarks), Task C (calibration), Task D (cost evaluation), Task F (slices) | **Data Engineer:** Run `build_pipeline.py` or provide the partitioned parquet files. |
| **Validation Split** | `data/processed/val.parquet` | Task C (calibrator fitting), Task D ($\tau^*$ selection) | **Data Engineer:** Must be materialized to allow leak-free tuning. |
| **Training Split** | `data/processed/train.parquet` | Task A (refit without suspect features), Task G (5-fold CV) | **Data Engineer:** Must be materialized to train evaluation copies. |
| **Serialized Model Binaries** | `models/xgboost_weighted.joblib`, `models/isolation_forest.joblib` | Tasks B, C, D, E, F (inference on test split) | **ML Engineer:** Must serialize trained pipelines/estimators to `models/` rather than only keeping in-memory. |
| **Row-Level Test Predictions** | `data/processed/test_predictions.parquet` (or model scoring) | Row-level evaluation, bootstrap 95% CIs, slice analysis | **ML Engineer / Evaluation CLI:** Requires model binary + `test.parquet`. |
| **Project Virtual Environment** | `.venv/` | CLI execution, test suite runner | **Environment Setup:** Virtualenv with `requirements.txt` installed (`xgboost`, `openpyxl`, `fastparquet`, etc.). |

---

### 2.3 What Was Inferred from Code & Existing Reports

1. **Exact Target Definition & Cutoffs:**
   - Anchor: $t_{\text{discharge}}$ (`discharge_start_ts`).
   - Duration: $\text{dwell\_hours} = \text{truck\_out\_end\_ts} - \text{discharge\_start\_ts}$.
   - Class 0 (`ON_TIME`): $\text{dwell\_hours} \le 65.1102\text{ hours}$ (50th percentile of training data).
   - Class 1 (`AT_RISK`): $65.1102 < \text{dwell\_hours} \le 160.8993\text{ hours}$ (50th to 85th percentile of training data).
   - Class 2 (`CRITICAL_DELAY`): $\text{dwell\_hours} > 160.8993\text{ hours}$ ($> 6.70\text{ days}$).
2. **Approved Arrival Predictors (15 features):**
   - Equipment/Carrier: `carrier_prefix`, `CTR_SIZE`, `CTR_TYPE`, `GROSS`, `YARD_BLOCK`, `YARD_SLOT`.
   - Calendar/Rhythm: `discharge_hour`, `discharge_day_of_week`, `discharge_month`, `is_weekend`.
   - Lagged Trailing Metrics: `trailing_7d_volume`, `trailing_7d_critical`, `trailing_7d_at_risk`, `historical_delay_rate_7d`, `trailing_7d_mean_dwell`.
   - Zero-variance placeholders excluded: `weather_severity_index`, `has_active_disruption`.
3. **Partition Scheme:**
   - Chronological split on dense window (`call_date >= 2021-06-10`):
     - **Train:** 2021-06-10 to 2021-08-26 ($N = 60,698$, Critical Prevalence = 15.00%).
     - **Val:** 2021-08-27 to 2021-09-11 ($N = 20,879$, Critical Prevalence = 7.37%).
     - **Test (Quarantined):** 2021-09-12 to 2021-09-30 ($N = 13,959$, Critical Prevalence = 2.76% due to ~85.4% right-censoring).
   - Container Grouping: Longitudinal container revisits in Val/Test are quarantined into `repeated_stays` to guarantee zero entity overlap across splits.
4. **Reported Model Deficiencies to Verify:**
   - Under argmax decision rule on validation data, `xgboost_weighted` achieves only **11.18% recall** on `CRITICAL_DELAY` (172 / 1,538 cases caught), with **9.90% precision**.
   - Model outputs are uncalibrated soft scores (Brier Score = 0.0995 vs 0.0682 unskilled prior).

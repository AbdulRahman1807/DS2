# Container Dwell Bottleneck Prediction Prototype — Runbook
## AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks (TPS 2021)
**Audit & Architecture Version:** `1.2.0` (Production Hardened Prototype)  
**Governing Branch:** `stage03/forensic-reassessment`  
**Repository:** `AbdulRahman1807/DS2`

---

## 1. System Overview & Pre-requisites

This runbook provides unambiguous, reproducible instructions for operating the hardened classical ML prototype for container dwell bottleneck triage. All instructions assume execution from the root directory of the repository (`/Users/abdul/Desktop/DS`).

### 1.1 Environment Setup

- **Python Version:** Python 3.10 to 3.13 (Verified on macOS Python 3.13.5).
- **Virtual Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 1.2 Verification of Core Dependencies
```bash
python3 -c "import pandas, numpy, scipy, sklearn, xgboost, lightgbm, fastapi, uvicorn, streamlit, pydantic; print('All core dependencies imported successfully.')"
```

---

## 2. Dataset Pipeline & Integrity Enforcement

The system ingests 952,069 raw container event records from Terminal Petikemas Surabaya (TPS), Indonesia (June–September 2021), accounting for 95,648 reconstructed container stays.

### 2.1 Reconciled Event Accounting & Cohort Integrity
- **Total Raw Events:** 952,069 records.
- **Total Reconstructed Stays:** 95,648 stays.
- **Completed Stays:** 95,647 completed stays with verified `TRUCK_OUT` departure timestamps.
- **Quarantined Open Stay:** 1 stay (`CSNU6526902-2021-04-19`) remaining in yard past extract boundary without gate-out; excluded from closed-dwell cohort.
- **Holdout Set (Quarantined):** Exactly 12,918 in-cohort stays discharged in $[2021-08-27, 2021-09-06)$ with $\ge 600$h follow-up. Kept strictly unobserved during model development.

### 2.2 Data Ingestion & Master Dataset Construction
To regenerate the unified master dataset with external context (PPKM social restriction calendar and PortWatch GDACS disruption feeds):
```bash
python3 ML/data_engineer/build_master_dataset.py
```
*Outputs generated in `data/master/`:*
- `master_real.parquet`: Real container stays with normalized gross weight in tonnes and observation timestamps.
- `master_synthetic.parquet`: Training-only synthetic stays generated via process-segment recombination (never evaluated in validation/test).
- `master_manifest.json`: Cryptographic SHA-256 hashes, row counts, and generation parameters.

### 2.3 Gross Weight Normalization
Cargo weight is recorded inconsistently in the raw Mendeley dataset (81.1% in metric tonnes, 18.9% in kilograms). Automatic normalization is enforced across all entry points:
- Any `GROSS` $> 100.0$ is converted via `GROSS / 1000.0`.
- Enforced at schema level in [`ML/integration_engineer/schemas.py`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/schemas.py).
- Enforced at API level in [`ML/integration_engineer/api.py`](file:///Users/abdul/Desktop/DS/ML/integration_engineer/api.py).

---

## 3. Feature Availability Contracts & Leakage Prevention

Machine-readable contracts strictly prevent post-discharge feature leakage into arrival-time triage matrices.

- **Machine-Readable Contract:** [`ML/data_engineer/feature_availability_contract.json`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_availability_contract.json)
- **Feature Provenance Matrix:** [`ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md)
- **Programmatic Enforcement:**
```python
from ML.data_engineer.feature_contracts import extract_modeling_matrix, validate_feature_matrix

# Extracts 15 leak-free predictors for Arrival-Time Triage (Horizon A)
X_train, y_train = extract_modeling_matrix(train_df, scenario="ARRIVAL_TIME")
validate_feature_matrix(X_train, scenario="ARRIVAL_TIME")
```

---

## 4. Model Training & Rolling-Origin Cross-Validation

The benchmark models use temporal rolling-origin cross-validation (folds F1, F2, F3) with a mandatory 7-day embargo and 160.9-hour label maturity gap to ensure statistical separation without lookahead bias.

### 4.1 Running Diagnostic Classical Benchmarks
Execute the complete multi-horizon diagnostic experiment suite (Horizon A, B12, B24, B48, inner-tuned, cost-sensitive, ordinal, and survival formulations):
```bash
python3 ML/ml_engineer/reassessment/run_reassessment.py
```
*Outputs:*
- Serialized out-of-fold prediction matrices in `reports/ml/reassessment/predictions/`.
- Experiment summary metrics in `reports/ml/reassessment/results.json`.
- Feature importance rankings in `reports/ml/reassessment/importance_*.csv`.

### 4.2 Benchmark Results (Development Cohort)
- **Horizon A (Discharge / Arrival):**
  - LightGBM Asof: Macro-F1 = 0.4072, Accuracy = 53.6%
  - XGBoost Asof: Macro-F1 = 0.4069
  - Dummy Prior Baseline: Macro-F1 = 0.2831
- **Horizon B24 (+24h In-Yard):**
  - LightGBM B24: Macro-F1 = 0.4789, Accuracy = 55.4%
  - Cost-Sensitive B24: Macro-F1 = 0.4639 (Prioritizes Critical Delay recall)
- **Horizon B48 (+48h In-Yard):**
  - LightGBM B48: Macro-F1 = 0.5283, Accuracy = 58.7%

---

## 5. Statistical Rigor: Paired Vessel-Clustered Bootstrap

To prevent inflated t-statistics from intra-vessel container correlation, all model comparisons use paired vessel-clustered bootstrap difference testing across distinct vessel voyage calls ($B=1000$ iterations).

### 5.1 Programmatic Execution
```python
import numpy as np
from ML.evaluation_engineer.metrics import compute_paired_vessel_clustered_macro_f1_difference

res = compute_paired_vessel_clustered_macro_f1_difference(
    y_true=y_test,
    y_pred_candidate=lgbm_preds,
    y_pred_reference=dummy_preds,
    vessel_clusters=vessel_call_ids,
    n_bootstraps=1000,
    random_state=42,
)

print(f"Delta Macro-F1: {res['macro_f1_delta']:.4f}")
print(f"95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
print(f"Statistically Significant (p < 0.05): {res['statistically_significant']}")
```

---

## 6. Running Automated Tests

The repository maintains an automated test suite verifying data contracts, feature leakage masking, API SLAs, schema coercion, dwell clock logic, and statistical bootstrap validity.

```bash
# Run complete test suite (137 tests)
pytest tests/ -v

# Run only the hardening test suite
pytest tests/test_ml_prototype_hardening.py -v
```
All 137 tests must pass before deploying or creating a release.

---

## 7. Starting the FastAPI Microservice

The FastAPI integration service exposes single-container triage (< 25ms SLA), vectorized manifest batch scoring, operational alert threshold configuration ($\tau$), and reference benchmarks.

### 7.1 Launching the API
```bash
# Option A: Local runner
python3 ML/integration_engineer/run_server.py

# Option B: Direct Uvicorn invocation
uvicorn ML.integration_engineer.api:app --host 127.0.0.1 --port 8000 --log-level info
```

### 7.2 Service Endpoints
- **Health Check:** `GET http://127.0.0.1:8000/health`
- **Interactive Swagger Docs:** `GET http://127.0.0.1:8000/docs`
- **Single Container Scoring:** `POST http://127.0.0.1:8000/predict/container`
- **Batch Manifest Scoring:** `POST http://127.0.0.1:8000/predict/batch`
- **Operational Threshold Tradeoffs:** `GET http://127.0.0.1:8000/api/v1/metrics/operational`
- **Feature Importance Matrix:** `GET http://127.0.0.1:8000/api/v1/features/importance`

### 7.3 Testing Single Container Scoring via cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict/container" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "TEST-CONTAINER-001",
    "carrier_prefix": "MSCU",
    "CTR_SIZE": 40.0,
    "CTR_TYPE": "DRY",
    "GROSS": 26500.0,
    "YARD_BLOCK": "B",
    "YARD_SLOT": "05",
    "discharge_hour": 14,
    "discharge_day_of_week": 4,
    "discharge_month": 7,
    "is_weekend": 0,
    "trailing_7d_volume": 420.0,
    "trailing_7d_critical": 35.0,
    "trailing_7d_at_risk": 95.0,
    "historical_delay_rate_7d": 0.22,
    "trailing_7d_mean_dwell": 78.5,
    "alert_threshold_tau": 0.25
  }'
```

---

## 8. Starting the Streamlit Control Tower

The Streamlit Control Tower delivers real-time operations queue triage, multi-horizon risk tracking, and terminal overview analytics.

### 8.1 Launching Streamlit
```bash
streamlit run streamlit_app.py --server.port 8501
```
Navigate to `http://localhost:8501` in your browser. The default initial view is set to **Container Triage**.

### 8.2 Application Pages
1. **Container Triage (Default):** Interactive container simulator, risk score breakdown, SOP playbooks, and key empirical risk factor attribution.
2. **Overview:** Executive KPIs, terminal throughput, dwell distribution ($p_{50}=65.11$h, $p_{85}=160.90$h), and carrier dwell distributions.
3. **Operations Queue:** Top-1% bottleneck queue sorting containers by predicted critical risk score.
4. **Model Intelligence:** ROC/PR curves, multi-horizon progression (Horizon A $\to$ B12 $\to$ B24 $\to$ B48), and feature importance.
5. **Port Intelligence:** Ministry of Ports benchmarks and comparative turnaround indicators.

---

## 9. Rollback & Checkpoint Recovery

If any unexpected regression occurs or if a clean restore point is needed:

```bash
# Return to the pre-hardening checkpoint commit
git checkout backup/pre-hardening-checkpoint-20261031

# Or reset current branch to the checkpoint commit (Caution: working directory changes will be lost)
git reset --hard 1294081
```

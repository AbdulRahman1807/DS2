# Final Classical ML Model Recommendation & Operational Specification

**Repository:** `AbdulRahman1807/DS2`
**Model Identifier:** `EXP-BASE-01-OPT-MACRO` (Primary Recommendation) & `EXP-MOD-07` (Untuned Reference)
**Evaluation Standard:** Out-of-time Validation (`data/processed/val.parquet`, $N=20,879$)
**Target Variable:** 3-Class Dwell Risk (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`)
**Deployment Status:** Model-Ready for Stage 01 Operational Piloting (Subject to Pre-Discharge Information Boundaries)

---

## 1. Recommended Model Architecture & Configuration

Following a 41-experiment empirical investigation across linear, tree-based, and ensemble classical ML families, the recommended production candidate is **`EXP-BASE-01-OPT-MACRO`**:

### Primary Production Configuration
- **Estimator Family:** Extreme Gradient Boosting Classifier (`XGBClassifier`)
- **Hyperparameters:**
  - `max_depth`: `5`
  - `learning_rate`: `0.10`
  - `n_estimators`: `100`
  - `subsample`: `0.80`
  - `colsample_bytree`: `0.80`
  - `objective`: `"multi:softprob"`
  - `eval_metric`: `"mlogloss"`
  - `random_state`: `42`
- **Class Imbalance Strategy:** Training-sample weighting computed via `sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)`.
- **Decision Rule Policy:** Post-hoc score multiplier vector applied to raw model probability outputs:
  $$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} \left( w_c \cdot S_c \right) \quad \text{where } \mathbf{w} = [w_0 = 1.4, w_1 = 1.2, w_2 = 1.3]$$
- **Untuned Production Alternative (Zero-Thresholding Baseline):**
  For deployments requiring standard unweighted $\arg\max$, the recommended model is **`EXP-MOD-07`** (`LGBMClassifier(num_leaves=15, max_depth=4, learning_rate=0.05, class_weight="balanced")`), which delivers the highest untuned critical PR-AUC (**0.0954**) and Macro-F1 (**0.3920**).

---

## 2. Exact Feature Contract

The recommended model strictly ingests **13 pre-discharge features** verified to be available at or before $t_{\text{discharge}}$:

### Feature Specification Table
| Feature Name | Type | Source System | Preprocessing Transformation | Lookahead Verification |
| :--- | :--- | :--- | :--- | :--- |
| `carrier_prefix` | Categorical | BAPLIE EDI Manifest | Out-of-fold `TargetEncoder(smooth="auto", cv=5)` | Observable 29.3h prior to discharge. |
| `CTR_TYPE` | Categorical | BAPLIE EDI Manifest | Out-of-fold `TargetEncoder(smooth="auto", cv=5)` | Observable 29.3h prior to discharge. |
| `CTR_SIZE` | Numerical | BAPLIE EDI Manifest | Median imputation (20.0), standard scaled | Observable 29.3h prior to discharge. |
| `GROSS` | Numerical | VGM EDI Declaration | Median imputation (21.4t), standard scaled | Observable prior to discharge. |
| `discharge_hour` | Numerical | TOS Discharge Event | Standard scaled | Exactly at discharge milestone. |
| `discharge_day_of_week` | Numerical | TOS Discharge Event | Standard scaled | Exactly at discharge milestone. |
| `discharge_month` | Numerical | TOS Discharge Event | Standard scaled | Exactly at discharge milestone. |
| `is_weekend` | Numerical | TOS Discharge Event | Pass-through binary integer $\{0, 1\}$ | Exactly at discharge milestone. |
| `trailing_7d_volume` | Numerical | Historical Gate-Out Log | Standard scaled | Departures completed $[t-7\text{d}, t-1\text{d}]$. |
| `trailing_7d_critical` | Numerical | Historical Gate-Out Log | Median imputation, standard scaled | Departures completed $[t-7\text{d}, t-1\text{d}]$. |
| `trailing_7d_at_risk` | Numerical | Historical Gate-Out Log | Median imputation, standard scaled | Departures completed $[t-7\text{d}, t-1\text{d}]$. |
| `historical_delay_rate_7d` | Numerical | Historical Gate-Out Log | Standard scaled | Departures completed $[t-7\text{d}, t-1\text{d}]$. |
| `trailing_7d_mean_dwell` | Numerical | Historical Gate-Out Log | Standard scaled | Departures completed $[t-7\text{d}, t-1\text{d}]$. |

---

## 3. Validation Methodology & Verification

1. **Chronological Out-of-Time Validation:**
   Trained strictly on `train.parquet` ($N=60,698$, June 1 – August 26, 2021) and evaluated on `val.parquet` ($N=20,879$, August 27 – September 17, 2021).
   - Strict condition verified: $\max(t_{\text{train}}) < \min(t_{\text{val}})$. Zero backward lookahead leakage.
2. **Out-of-Fold Preprocessing Isolation:**
   All scalers, imputers, and target encoders are fitted strictly on `train.parquet` and transformed on `val.parquet`.
3. **Quarantined Test Set:**
   The test set (`test.parquet`, September 18–30, 2021) remains strictly quarantined due to its verified 85.4% right-censoring survivorship bias.

---

## 4. Reproducible Validation Performance Metrics

Evaluated on `val.parquet` ($N=20,879$, critical delay prevalence = 7.37%):

### Global Summary Metrics
- **Macro-F1 (Primary Optimization Objective):** **0.4028** (Stage 01 Baseline: 0.3981, $\Delta = +0.0047$)
- **Accuracy:** **0.5785** (Stage 01 Baseline: 0.5430, $\Delta = +0.0355$)
- **Balanced Accuracy:** **0.4047** (Stage 01 Baseline: 0.4117)
- **Weighted-F1:** **0.5555**
- **Critical-Delay PR-AUC:** **0.0911** (Naive Base Rate: 0.0737)
- **Runtime:** 0.84 seconds training; $<1.5\text{ms}$ single-case inference latency.

### Complete Per-Class Performance
| Class Name | Class ID | Precision | Recall | F1-Score | Support | Prevalence in Val |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ON_TIME`** ($\le 65.11\text{h}$) | 0 | **0.7512** | **0.6543** | **0.6994** | 13,892 | 66.54% |
| **`AT_RISK`** ($65.11\text{h} - 160.90\text{h}$) | 1 | **0.3541** | **0.4493** | **0.3961** | 5,449 | 26.10% |
| **`CRITICAL_DELAY`** ($> 160.90\text{h}$) | 2 | **0.0984** | **0.1105** | **0.1041** | 1,538 | 7.37% |

### Confusion Matrix
```
Predicted Class:      ON_TIME (0)    AT_RISK (1)    CRITICAL (2)    Total True
True ON_TIME (0)          9,090          3,692          1,110         13,892
True AT_RISK (1)          2,367          2,448            634          5,449
True CRITICAL (2)           643            725            170          1,538
Total Predicted          12,100          6,865          1,914         20,879
```

---

## 5. Alternative Operational Profile: High-Recall Interception

For terminal dispatch scenarios where missing a severe bottleneck carries extreme demurrage penalties, the recommended alternative configuration is **`EXP-FE-03-OPT-CRIT`**:
- **Model:** `LGBMClassifier(n_estimators=200, max_depth=6, num_leaves=45, lr=0.05, class_weight="balanced")` on 29 engineered features.
- **Decision Multipliers:** $\mathbf{w} = [w_0 = 1.0, w_1 = 0.6, w_2 = 1.6]$.
- **Operational Performance:**
  - **Critical-Delay Recall:** **67.04%** (1,031 out of 1,538 critical delays successfully intercepted).
  - **Critical-Delay F1:** **0.1637** (highest critical F1 of all 41 experiments).
  - **Trade-off:** Overall accuracy drops to 38.55% due to 9,519 false alarms on non-critical containers.

---

## 6. Serialized Model Artifacts

Artifacts are persisted using `joblib` and available at:
- Primary Macro-F1 Model: `ML/ml_engineer/artifacts/best_macro_f1_model.joblib` (and mirrored in `models/best_macro_f1_model.joblib`).
- High-Recall Feature-Engineered Model: `ML/ml_engineer/artifacts/lightgbm_fe_model.joblib`.
- Baseline Model: `ML/ml_engineer/artifacts/xgboost_baseline_model.joblib`.

Each artifact contains:
```python
{
    "estimator": <Fitted Classical Estimator>,
    "preprocessor": <Fitted ColumnTransformer>,
    "features": [<Exact list of active feature names>]
}
```

---

## 7. Remaining Limitations & Handoff to Stage 02

1. **Pre-Discharge Information Boundary:**
   This model represents the mathematical ceiling (~0.40 Macro-F1) of what can be predicted at container discharge using static manifest and rolling terminal attributes.
2. **Path to 0.60+ Performance:**
   Transcendence of the 0.40 ceiling requires Stage 02: **12h/24h Landmark Mid-Stay Prediction**, ingesting dynamic customs holds (`has_quarantine_flag`), delivery order submissions (`JOB_DEL_DOCTYPE`), and physical examination milestones (`has_behandle_inspection`).

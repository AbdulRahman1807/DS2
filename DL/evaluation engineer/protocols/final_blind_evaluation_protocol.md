# Final 2025 Blind-Test Evaluation Protocol

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Deep Learning Engineering  
**Workspace:** `DL/ml engineer/`  
**Date:** 2026-11-02  
**Status:** **FROZEN PRIOR TO 2025 BLIND TEST EVALUATION**  

---

## 1. Protocol Objective & Governance

This protocol specifies the exact, immutable evaluation methodology for the out-of-sample 2025 blind-test evaluation of frozen deep learning models. 

To ensure complete scientific integrity, this protocol is committed and frozen **before** materializing, inspecting, or calculating any statistics on 2025 target values. Under non-negotiable governance restrictions:
- No model architecture or checkpoint weights may be modified.
- No retraining, fine-tuning, or hyperparameter optimization may be performed.
- No 2025 blind-test results may be used for model selection, threshold adjustments, or feature engineering.
- All evaluation scripts operate in read-only inference mode (`torch.no_grad()`).

### 1.1 Pre-Evaluation Smoke-Test Disclosure
Prior to full batch inference (Step 3960), an ad-hoc python verification command (`toolSummary="Test PortSequenceDataset on test split"`) was run to confirm that `PortSequenceDataset` would instantiate over `sequence_metadata.parquet` without memory error. This command printed sample 0's target value (`target: 0.0`). Zero test files or test suites were created or modified. All checkpoint weights, hyperparameters, and evaluation scripts remained strictly frozen prior to full evaluation.

---

## 2. Frozen Checkpoints & Cryptographic Verification

Only models whose weights are verified against the established cryptographic registry will be evaluated:

| Model Role | Model Implementation Class | Checkpoint Path | Expected SHA-256 Checksum | Parameters |
|---|---|---|---|---:|
| **Global Representation Backbone** | `ResidualTransformerForecaster` (Mean-Pooling) | `artifacts/transformer_mean_pooling/best_model.pt` | `ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e` | 68,481 |
| **Multi-Target Throughput Forecaster** | `DirectThroughputResidualForecaster` (Dual Heads) | `artifacts/throughput_forecasting/best_model.pt` | `0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d` | 4,226 |
| **Weekly Anchorage Queue Forecaster** | `ResidualDwellGRUForecaster` | `artifacts/dwell_forecasting/best_model.pt` | `0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371` | 13,114 |

---

## 3. Data Contract, Features & Input Availability

### 3.1 Universal 20-Feature Operational Core
The model consumes exactly 20 features per timestep in the following fixed index order:
0. `portcalls` — Total port calls (vessel arrivals)
1. `portcalls_container` — Container vessel port calls
2. `portcalls_dry_bulk` — Dry bulk vessel port calls
3. `portcalls_general_cargo` — General cargo vessel port calls
4. `portcalls_roro` — Ro-Ro vessel port calls
5. `portcalls_tanker` — Tanker vessel port calls
6. `portcalls_cargo` — Total cargo vessel port calls
7. `import_container` — Container import metric volume
8. `import_cargo` — Cargo import metric volume
9. `export_container` — Container export metric volume
10. `export_cargo` — Cargo export metric volume
11. `total_cargo_tonnage` — Total cargo tonnage loaded/discharged
12. `pw_calls_rolling_7d_mean` — 7-day rolling mean of port calls
13. `pw_calls_rolling_14d_mean` — 14-day rolling mean of port calls
14. `pw_calls_rolling_30d_mean` — 30-day rolling mean of port calls
15. `pw_container_rolling_7d_mean` — 7-day rolling mean of container port calls
16. `pw_container_rolling_14d_mean` — 14-day rolling mean of container port calls
17. `day_of_week` — Day of week integer (0=Monday, 6=Sunday)
18. `is_weekend` — Binary weekend indicator (1=Saturday/Sunday)
19. `day_of_year` — Day of year integer (1–366)

### 3.2 Lookback Window & Operational Availability Policy ($T-2$)
- **Sequence Length:** Exactly $L=30$ consecutive daily timesteps.
- **Input Observation Availability:** Strict $T-2$ operational latency policy. For a forecast issued at reference date $t$, the input sequence covers dates $[t-31, t-2]$. Timesteps $t-1$ and $t$ are masked/excluded due to maritime AIS transmission and verification latency.
- **Normalization:** Ingests the frozen `TrainingStandardScaler` parameters from `artifacts/preprocessing/scaler_params.json` fitted exclusively on training data ($\le 2023-12-22$).

---

## 4. Target Definitions & Forecast Horizons

### 4.1 Global Direct Throughput Targets
1. **Target A — Container Vessel Calls:**
   - **Column:** `target_container_calls_7d_future_sum`
   - **Definition:** Cumulative container-carrying vessel arrivals over the 7-day forecast window $[t+1, t+7]$.
   - **Unit:** Vessel Arrivals (integer ship counts).
   - **Domain Clarification:** Measures vessel traffic flow, not container volume (TEU).
2. **Target B — Cargo Tonnage:**
   - **Column:** `target_cargo_tonnage_7d_future_sum`
   - **Definition:** Cumulative deadweight metric tons loaded or discharged over the 7-day forecast window $[t+1, t+7]$.
   - **Unit:** Metric Tons ($t$).
   - **Domain Clarification:** Measures physical deadweight cargo flow, not container-yard inventory.

### 4.2 Anchorage Waiting-Time Target (Dwell Model)
- **Target Variable:** `avgwaiting_time`
- **Definition:** Mean duration (in hours) spent by vessels waiting at outer anchorages before berth access.
- **Horizons:** $H=1$ (week $t+1$) and $H=4$ (mean over weeks $[t+1, t+4]$).
- **Unit:** Hours ($h$).
- **Data Boundary Note:** Public ADB AIS data ends at calendar year 2022. No 2023–2025 observations exist in public records. The 2022 held-out evaluation stands as the final out-of-sample dwell evaluation.

---

## 5. Baseline Benchmark Definitions

To ensure methodological fairness, the blind evaluation uses the exact same analytical persistence baselines established during the 2024 validation milestone:

### 5.1 Container Vessel Calls Persistence Baseline
The persistence forecast for cumulative 7-day future container calls is defined as 7 times the unscaled 7-day rolling mean observed at timestep $T-2$ (Feature Index 15):
$$\hat{y}_{\text{naive}}^{\text{container}} = 7.0 \times \left(x_{T-2, 15} \cdot \sigma_{\text{pw\_container\_roll7}} + \mu_{\text{pw\_container\_roll7}}\right)$$

### 5.2 Cargo Tonnage Persistence Baseline
The persistence forecast for cumulative 7-day future cargo tonnage is defined as 7 times the unscaled daily cargo tonnage observed at timestep $T-2$ (Feature Index 11):
$$\hat{y}_{\text{naive}}^{\text{tonnage}} = 7.0 \times \left(x_{T-2, 11} \cdot \sigma_{\text{tonnage}} + \mu_{\text{tonnage}}\right)$$

---

## 6. Evaluation Metrics & Parameters

For each target and baseline, the following regression metrics will be calculated:
1. **Huber Loss:**
   $$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
   - For Container Calls: $\delta = 1.0$ calls.
   - For Cargo Tonnage: $\delta = 50,000.0$ metric tons.
2. **Mean Absolute Error (MAE):** $\frac{1}{N}\sum |y - \hat{y}|$
3. **Median Absolute Error (Median AE):** $\text{median}(|y - \hat{y}|)$
4. **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$
5. **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$
6. **Mean Prediction Bias:** $\frac{1}{N}\sum (\hat{y} - y)$

---

## 7. Expected Blind-Test Population & Partition Bounds

- **Source File:** `DL/data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet`
- **Filter:** `split_tag == "test"`
- **Expected Sequence Count:** Exactly **741,335 sequences**.
- **Expected Port Count:** Exactly **2,065 commercial ports**.
- **Expected Sequences per Port:** Exactly **359 sequences per port** ($359 \times 2,065 = 741,335$).
- **Reference Date Range:** `2024-12-31` to `2025-12-24`.
- **Target Forecast Range:** `2025-01-01` to `2025-12-31`.

---

## 8. Aggregation & Diagnostic Stratification

All evaluations will produce:
1. **Global Aggregate Metrics:** Panel-wide evaluation across all 741,335 sequences.
2. **Temporal Monthly Stratification:** Monthly evaluation from 2024-12 through 2025-12 to assess stability and seasonal drift.
3. **Activity-Level Diagnostic Buckets:**
   - *Container Calls:*
     - Zero Traffic: $[0.0]$ calls
     - Low Activity: $(0.0, 7.0]$ calls (1–7 calls/week)
     - Medium Activity: $(7.0, 28.0]$ calls (8–28 calls/week)
     - High Activity: $>28.0$ calls (>4 calls/day)
   - *Cargo Tonnage:*
     - Zero Freight: $[0.0]$ metric tons
     - Low Tonnage: $(0.0, 20,000.0]$ metric tons
     - Medium Tonnage: $(20,000.0, 150,000.0]$ metric tons
     - High Tonnage: $>150,000.0$ metric tons
4. **Port-Level Spatial Performance:**
   - Improved ports ($\text{MAE}_{\text{model}} < \text{MAE}_{\text{naive}}$).
   - Tied ports ($|\Delta_{\text{MAE}}| \le 10^{-4}$ for calls; $|\Delta_{\text{MAE}}| \le 1.0$ for tonnage).
   - Regressed ports ($\text{MAE}_{\text{model}} > \text{MAE}_{\text{naive}}$).
   - Identification of top 10 improved and top 10 regressed ports.

---

## 9. Output Artifact Specifications

The evaluation will generate the following immutable artifacts:
- **Prediction Parquet:** `artifacts/throughput_forecasting/blind_test_predictions_2025.parquet` (containing `sequence_id`, `portwatch_id`, `reference_date`, actuals, naive predictions, model predictions, and errors).
- **JSON Metrics Summary:** `artifacts/throughput_forecasting/blind_test_metrics_2025.json`.
- **Evaluation Report:** `reports/final_blind_test_evaluation_report.md`.
- **Generalization Analysis:** `reports/final_generalization_analysis.md`.

*This protocol is formally frozen.*


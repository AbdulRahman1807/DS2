# Milestone 3.6.1 — Final Deep Learning Model Registry (Updated with 2025 Blind-Test Results)

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** 2 — Deep Learning Engineering  
**Workspace:** `DL/ml engineer/`  
**Date:** 2026-11-02  
**Auditor / Engineer:** Senior Deep Learning Engineer, Independent Evaluator & Technical Handoff Owner  
**Status:** **AUTHORITATIVE MASTER MODEL REGISTRY — FROZEN & BLIND-TEST VERIFIED**  

---

## 1. Executive Overview

This registry establishes the authoritative, immutable inventory of all deep learning model architectures developed, evaluated, and dispositioned across Milestones 3.1 through Final Closure. Every entry records exact parameter counts, input feature spaces, target definitions, temporal horizons, empirical baseline comparisons, checkpoint file paths, cryptographic SHA-256 checksums, and out-of-sample blind-test validation results.

---

## 2. Master Model Inventory & Final Disposition Matrix

| # | Model Identifier | Task / Target | Framework / Backbone | Trainable Params | Checkpoint Path & SHA-256 Checksum | 2024 Val / 2025 Test Performance | Final Disposition |
|---|---|---|---|---:|---|---|---|
| **M1** | `ResidualTransformerForecaster` (Mean-Pooling) | Global 7-day cumulative port calls (`target_portcalls_7d_future_sum`) | 2-layer Pre-LN Transformer, $d_{\text{model}}=64$, 4 heads, sinusoidal PE, ReLU residual | 68,481 (71,043 tensor elements) | `artifacts/transformer_mean_pooling/best_model.pt`<br>`ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e` | Val Huber: 2.0089, MAE: 2.3924 calls, $R^2$: 0.9858 (vs. Persistence MAE 3.0271) | **PRIMARY GLOBAL FLOW BACKBONE — FROZEN & BLIND-TEST VERIFIED** |
| **M2** | `DirectThroughputResidualForecaster` (Dual Heads) | 7-day container vessel calls & 7-day cargo tonnage | Frozen M1 Transformer Encoder + Dual 2-layer residual MLPs ($64 \to 32 \to 1$) | 4,226 (Container: 2,113; Tonnage: 2,113) | `artifacts/throughput_forecasting/best_model.pt`<br>`0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d` | **2025 Blind Test:**<br>• Calls: MAE 0.8323 (-2.13%), $R^2$ 0.9824<br>• Tonnage: MAE 41,578t (-1.54%), $R^2$ 0.9609 | **PRIMARY THROUGHPUT ARCHITECTURE — FROZEN & BLIND-TEST VERIFIED** |
| **M3** | `ResidualDwellGRUForecaster` | Weekly mean vessel anchorage waiting hours (`avgwaiting_time`) | 2-layer GRU ($d_h=32$) + 19-port entity embedding ($d=8$) + dual heads ($H=1, H=4$) | 13,114 | `artifacts/dwell_forecasting/best_model.pt`<br>`0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371` | **Held-Out Test 2022 ($H=1$):**<br>MAE: 4.0824h (-7.03%), $R^2$: 0.6103<br>**Horizon $H=4$:** MAE: 5.8675h (-11.05%) | **PRIMARY DWELL ARCHITECTURE — FROZEN WITH LIMITATIONS (HELD-OUT TEST VERIFIED)** |
| **M4** | `TemporalAttentionLSTMForecaster` | Global 7-day cumulative port calls | 2-layer LSTM ($d_h=64$) + additive Bahdanau attention ($d_a=32$) | 57,473 | `artifacts/attention_lstm/best_model.pt`<br>`738baa0177f383a6a92734ff2253ef229eead5b383e2649db56497fee71d5e6b` | Val Huber: 2.4432, MAE: 2.8929 calls, $R^2$: 0.9815 | **BENCHMARK RECURRENT ARCHITECTURE — RETAIN AS SECONDARY REFERENCE** |
| **M5** | `ResidualLSTMForecaster` (Exp03) | Global 7-day cumulative port calls | 2-layer LSTM ($d_h=64$) + linear residual connection | 55,361 | `artifacts/lstm/experiments/exp03_residual_persistence/best_model.pt`<br>`93a4be3ccb39067487223499038f20dba8bca8272435513c682c3c3d3e4f525e` | Val Huber: 2.3974, MAE: 2.7878 calls, $R^2$: 0.9809 | **BASELINE DEEP LEARNING MODEL — HISTORICAL BENCHMARK REFERENCE** |
| **M6** | `ResidualTransformerForecaster` (Last Token) | Global 7-day cumulative port calls | 2-layer Transformer, $d_{\text{model}}=64$, last-timestep token pooling | 68,481 | `artifacts/transformer/best_model.pt`<br>`3a6edb10961481e37e08eeb27b3ecb360d1495cb8e702a4f20f0de2bff3bfafb` | Val Huber: 2.0734, MAE: 2.4560 calls, $R^2$: 0.9851 | **SUPERSEDED** (Outperformed globally by Mean-Pooling variant M1) |
| **M7** | `CaribbeanResidualSTGNN` (GATv2 / GCN) | Regional 7-day Caribbean port calls | GATv2 / GCN + residual temporal connection over regional maritime graph | 59,556 | `artifacts/gnn/caribbean/gatv2_actual/best_model.pt`<br>`1f028fdab40c569297bc1c0e06b7ee0a32cdc8813062cfb1ff2fe6d17fae230e` | Failed to beat persistence ($t=-2.44, p=0.0181$) | **REJECTED** (Spatial GNN message passing underperformed persistence) |
| **M8** | `PortEmbeddedLSTMForecaster` (Exp02) | Global 7-day cumulative port calls | 2-layer LSTM ($d_h=64$) + port entity embeddings (direct target regression) | 67,825 | `artifacts/lstm/experiments/exp02_port_embedding/best_model.pt`<br>`4f413e377baf84bc1cb9c54f5bce4e6c1c4876a3936631e415d5da35ef2172b2` | Severe target under-prediction on raw regression | **SUPERSEDED** (Replaced by residual formulations) |

---

## 3. Authoritative Specifications for Primary Production Models

### 3.1. Primary Global Backbone: `ResidualTransformerForecaster` (Mean-Pooling)
- **Model ID:** M1
- **Architecture Class:** `ResidualTransformerForecaster` in [`models/transformer_forecaster.py`](file:///Users/abdul/Desktop/DS/DL/ml%20engineer/models/transformer_forecaster.py)
- **Mathematical Formulation:**
  $$\hat{y}_{t+1:t+7} = \operatorname{ReLU}\left( 7.0 \cdot \bar{y}_{\text{roll7}}(t-2) + \mathbf{w}^T \left( \frac{1}{30} \sum_{\tau=1}^{30} h_\tau \right) + b \right)$$
- **Input Dimensions:** $[B, 30, 20]$ conforming strictly to the Milestone 3.1 universal operational core.
- **Availability Policy:** Strict $T-2$ convention (quarantining $t-1$ and $t$).
- **Latent Dimensions:** $d_{\text{model}} = 64$, 4 attention heads, $d_{\text{ff}} = 128$, Pre-LayerNorm.
- **Parameter Count:** Exactly 68,481 trainable parameters (71,043 tensor elements including position buffers).
- **Training Population:** 3,696,350 sequences (2019–2023).
- **Validation Population:** 743,400 sequences (2024).
- **Validation Performance:** Huber: **2.0089**, MAE: **2.3924 calls**, RMSE: **5.6420 calls**, $R^2$: **0.9858** (beats Naive Persistence Huber 2.6558 by 24.36%).
- **Checkpoint Location:** `artifacts/transformer_mean_pooling/best_model.pt`
- **SHA256 Hash:** `ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e`
- **Status:** **FROZEN & IMMUTABLE**.

---

### 3.2. Primary Throughput Model: `DirectThroughputResidualForecaster`
- **Model ID:** M2
- **Architecture Class:** `DirectThroughputResidualForecaster` in [`models/throughput_forecaster.py`](file:///Users/abdul/Desktop/DS/DL/ml%20engineer/models/throughput_forecaster.py)
- **Backbone Integration:** Operates on the frozen representation $h_{\text{mean}} \in \mathbb{R}^{64}$ of Model M1 (zero gradient updates to backbone).
- **Trainable Heads:**
  1. *Container Vessel Calls Head:* 2-layer MLP ($64 \to 32 \to 1$) producing additive residual over 7-day trailing container calls:
     $$\hat{y}_{\text{container}} = \operatorname{ReLU}\left( y_{\text{naive, container}} + \Delta_c(h_{\text{mean}}) \right)$$
  2. *Cargo Tonnage Head:* 2-layer MLP ($64 \to 32 \to 1$) producing additive residual over 7-day trailing total cargo tonnage:
     $$\hat{y}_{\text{tonnage}} = \operatorname{ReLU}\left( y_{\text{naive, tonnage}} + \Delta_t(h_{\text{mean}}) \cdot \sigma_{\text{tonnage}} \right)$$
- **Parameter Breakdown:** 4,226 trainable parameters; 68,481 frozen backbone parameters.
- **2024 Full Validation Benchmark ($N=743,400$ Sequences across 2,065 Ports):**
  - *Container Calls:* Huber: **0.6397** (-6.23%), MAE: **0.8396 calls** (-2.06%), RMSE: **2.3215** (-6.34%), $R^2$: **0.9819** (vs. Persistence MAE 0.8573, $R^2$ 0.9794).
  - *Cargo Tonnage:* Huber: **1.606B** (-2.63%), MAE: **41,840.3 metric tons** (-1.34%, saving 566.5 t/seq), RMSE: **144,398.5** (-2.41%), $R^2$: **0.9583** (vs. Persistence MAE 42,406.8, $R^2$ 0.9562).
- **2025 Full Blind-Test Benchmark ($N=741,335$ Sequences across 2,065 Ports):**
  - *Container Calls:* Huber: **0.6356** (-6.31%), MAE: **0.8323 calls** (-2.13%), RMSE: **2.3185** (-6.59%), $R^2$: **0.9824** (vs. Persistence MAE 0.8504, $R^2$ 0.9799).
  - *Cargo Tonnage:* Huber: **1.595B** (-2.79%), MAE: **41,577.7 metric tons** (-1.54%, saving 650.0 t/seq), RMSE: **139,051.6** (-2.51%), $R^2$: **0.9609** (vs. Persistence MAE 42,227.7, $R^2$ 0.9589).
  - *Monthly Stability:* Outperforms persistence across **100% of calendar months** in both 2024 and 2025.
- **Checkpoint Location:** `artifacts/throughput_forecasting/best_model.pt`
- **SHA256 Hash:** `0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d`
- **Prediction Artifacts:**
  - `artifacts/throughput_forecasting/validation_predictions_full_2024.parquet` (18.45 MB)
  - `artifacts/throughput_forecasting/blind_test_predictions_2025.parquet` (18.94 MB)
- **Status:** **FROZEN & IMMUTABLE — FULLY VALIDATED AND BLIND-TEST VERIFIED**.

---

### 3.3. Primary Dwell Architecture: `ResidualDwellGRUForecaster`
- **Model ID:** M3
- **Architecture Class:** `ResidualDwellGRUForecaster` in [`models/dwell_forecaster.py`](file:///Users/abdul/Desktop/DS/DL/ml%20engineer/models/dwell_forecaster.py)
- **Target Semantics:** Weekly mean vessel anchorage waiting duration in hours (`avgwaiting_time`).
- **Input Dimensions:** 8-week lookback window ($L=8$) of 7 normalized continuous features (waiting time, service time, call volume, entrance count, Neo-Panamax percentage, harmonic week calendar signals).
- **Entity Conditioning:** 19-port learnable embedding matrix ($d_e = 8$).
- **Recurrent Encoder:** 2-layer GRU, $d_h = 32$, dropout $p=0.1$.
- **Residual Formulation:** $\hat{y}_{t+H} = \operatorname{ReLU}(y_t + \text{MLP}_{40 \to 32 \to 1}([h_T; e_p]))$.
- **Parameter Count:** Exactly 13,114 parameters.
- **Chronological Partitions:**
  - *Training:* 2019-W01 to 2020-W53 (1,843 sequences across 19 ports).
  - *Validation (Crisis Surge):* 2021-W01 to 2021-W53 (1,007 sequences).
  - *Held-Out Test (Normalization):* 2022-W01 to 2022-W52 (931 sequences).
- **Empirical Performance:**
  - *Validation 2021 ($H=1$):* Huber: **13.5554** (-3.40%), MAE: **5.7353h** (-2.24%), RMSE: **10.8556h** (-1.03%), $R^2$: **0.7185** (vs. Persistence MAE 5.8669h, $R^2$ 0.7126).
  - *Held-Out Test 2022 ($H=1$):* Huber: **8.7176** (-9.91%), MAE: **4.0824h** (-7.03%), RMSE: **6.4028h** (-5.84%), $R^2$: **0.6103** (vs. Persistence MAE 4.3910h, $R^2$ 0.5604).
  - *4-Week Horizon ($H=4$):* MAE: **5.8675h** (-11.05%), RMSE: **10.8131h** (-6.94%), $R^2$: **0.6677** (vs. Persistence MAE 6.5963h, $R^2$ 0.6163).
  - *Spatial Generalization:* **15 out of 19 ports (78.9%)** achieve lower MAE than persistence.
- **Checkpoint Location:** `artifacts/dwell_forecasting/best_model.pt`
- **SHA256 Hash:** `0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371`
- **Known Limitations:** Predicts outer anchorage vessel waiting queue time; does not predict container yard storage dwell. No 2023–2025 public weekly AIS observations exist; the 2022 held-out evaluation stands as the final out-of-sample evaluation.
- **Status:** **FROZEN & IMMUTABLE WITH DOCUMENTED LIMITATIONS**.

---

## 4. Integrity and Compliance Checklist

- [x] All model weights and biases verified through cryptographic SHA-256 checksums.
- [x] Frozen Transformer backbone confirmed 100% immutable and unmodified (`ff67c307183a...`).
- [x] 2025 blind test set (741,335 sequences) evaluated strictly in out-of-sample read-only inference mode.
- [x] Preprocessing scaler parameters fitted strictly on observations $\le 2023-12-22$ with zero test target leakage.
- [x] Full blind-test predictions exported to `artifacts/throughput_forecasting/blind_test_predictions_2025.parquet` (18.94 MB).
- [x] Authoritative metrics exported to `artifacts/throughput_forecasting/blind_test_metrics_2025.json`.
- [x] Generalization assessment completed confirming strong out-of-sample generalization.

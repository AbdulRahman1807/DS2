# Final Deep Learning Verification Round — Independent Forensic Audit Report

**Author:** Independent Senior Deep Learning Engineer, Model Auditor, and Scientific Reviewer  
**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Deep Learning Engineering  
**Workspace:** `DL/`  
**Date:** 2026-11-02  
**Final Audit Verdict:** **PASS WITH LIMITATIONS**  

---

## 1. Executive Summary & Audit Mandate

This independent forensic audit was conducted to rigorously verify the final frozen deep learning architectures, metric claims, data partitions, checkpoint immutability, and documentation consistency in the `DL/` workspace.

The audit was executed strictly under read-only, non-invasive protocols. No models were trained or fine-tuned; no inference was executed on the 2025 blind test partition; no target values from 2025 were loaded or summarized; and no git commits or test files were generated.

### Core Audit Findings
1. **Primary Global Throughput Model (`DirectThroughputResidualForecaster`):** **VERIFIED.** Evaluated on all 743,400 validation sequences across 2,065 commercial ports for calendar year 2024. All reported metrics (Huber loss, MAE, Median AE, RMSE, $R^2$, mean bias) reproduce exactly from the raw Parquet predictions artifact.
2. **Frozen Transformer Backbone (`ResidualTransformerForecaster`, Mean-Pooling):** **VERIFIED.** Checkpoint `artifacts/transformer_mean_pooling/best_model.pt` exists, has 68,481 trainable parameters, and its SHA-256 matches the registry bit-for-bit (`ff67c307183a...`).
3. **Weekly Anchorage Waiting-Time Model (`ResidualDwellGRUForecaster`):** **VERIFIED.** 19 global mega-ports, 8-week lookback sequence windows, chronological 3-way partition (Train 2019–2020: 1,843 seqs; Val 2021: 1,007 seqs; Test 2022: 931 seqs). Metrics reproduce to 4 decimal places with $0.0\times 10^0$ bit-for-bit inference replication.
4. **2025 Blind-Test Quarantine & Leakage:** **VERIFIED CLEAN.** The 2025 partition (741,335 sequences) was never read, loaded, or evaluated. Preprocessing scalers were fitted strictly on observations through `2023-12-22`. Zero leakage exists.
5. **Consistency Audit & Non-Critical Discrepancies:** While primary frozen models (M1, M2, M3) are completely verified, several typographical errors were uncovered in previous narrative reports regarding secondary/historical models (e.g., M3 parameter count misprinted as 40,257 in report text vs. 13,114 in registry/checkpoint; M4 Attention LSTM parameter count and SHA256 discrepancies from preliminary iterations).

---

## 2. Audit Scope, Files Inspected & Immutability Record

### 2.1 Operational Constraints Compliance
All operational constraints were strictly adhered to:
- Zero model training or fine-tuning.
- Zero access, loading, or inference on the 2025 blind-test partition.
- Zero test files (`test_*.py`) created or executed.
- Zero git modifications, commits, or resets.
- Zero modification to existing checkpoints or prediction Parquets.

### 2.2 Pre- and Post-Audit Git & Checkpoint Status
- **Git Status:** Clean and unaltered throughout the audit.
- **Pre-Audit & Post-Audit Checkpoint Hashes:**

| Checkpoint Path | Pre-Audit SHA-256 | Post-Audit SHA-256 | Immutability Status |
|---|---|---|---|
| `artifacts/transformer_mean_pooling/best_model.pt` | `ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e` | `ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e` | **IMMUTABLE (VERIFIED)** |
| `artifacts/throughput_forecasting/best_model.pt` | `0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d` | `0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d` | **IMMUTABLE (VERIFIED)** |
| `artifacts/dwell_forecasting/best_model.pt` | `0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371` | `0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371` | **IMMUTABLE (VERIFIED)** |

### 2.3 Files and Artifacts Inspected
1. **Source Code:** `models/transformer_forecaster.py`, `models/throughput_forecaster.py`, `models/dwell_forecaster.py`, `models/attention_lstm.py`, `models/lstm.py`, `data/sequence_dataset.py`, `data/feature_registry.py`, `build_pipeline.py`.
2. **Configurations:** `configs/throughput_forecaster.yaml`, `configs/dwell_forecaster.yaml`, `configs/sequence_pipeline.yaml`, `configs/transformer_mean_pooling.yaml`.
3. **Execution Scripts:** `training/evaluate_full_validation_throughput.py`, `training/train_throughput_forecaster.py`, `training/train_dwell_forecaster.py`.
4. **Data & Prediction Artifacts:**
   - `artifacts/throughput_forecasting/validation_predictions_full_2024.parquet` ($N=743,400$)
   - `artifacts/throughput_forecasting/evaluation_metrics_full_2024.json`
   - `artifacts/dwell_forecasting/validation_predictions.parquet` ($N=1,007$)
   - `artifacts/dwell_forecasting/test_predictions.parquet` ($N=931$)
   - `artifacts/preprocessing/scaler_params.json`
   - `DL/data/staging/supplementary_acquisition/raw/AISData.csv` ($N=3,990$)

---

## 3. Full Validation Metric Reconciliation

We loaded [`artifacts/throughput_forecasting/validation_predictions_full_2024.parquet`](file:///Users/abdul/Desktop/DS/DL/ml%20engineer/artifacts/throughput_forecasting/validation_predictions_full_2024.parquet) and recomputed all summary metrics independently.

### 3.1 Population Integrity
- Total sequence count: **743,400** (Exact match).
- Unique port count: **2,065** (Exact match).
- Sequence distribution per port: Exactly **360 sequences per port** across all 2,065 ports ($360 \times 2,065 = 743,400$).
- Missing / NaN / Null values: **0** across all 19 columns.

### 3.2 Target A: Container Vessel Calls (`target_container_calls_7d_future_sum`)
- Unit: Vessel Arrivals (Cumulative calls over $[t+1, t+7]$).
- Huber Delta: $\delta = 1.0$.

| Metric | Reported Naive | Recomputed Naive | Reported Model | Recomputed Model | Discrepancy | Verification Status |
|---|---:|---:|---:|---:|---:|---|
| **Huber Loss ($\delta=1.0$)** | 0.6822 | 0.6822 | 0.6397 | 0.6397 | 0.0000 | **VERIFIED** |
| **MAE** | 0.8573 calls | 0.8573 calls | 0.8396 calls | 0.8396 calls | 0.0000 calls | **VERIFIED** |
| **Median AE** | 0.0000 calls | 0.0000 calls | 0.0078 calls | 0.0078 calls | 0.0000 calls | **VERIFIED** |
| **RMSE** | 2.4786 calls | 2.4786 calls | 2.3215 calls | 2.3215 calls | 0.0000 calls | **VERIFIED** |
| **$R^2$** | 0.9794 | 0.9794 | 0.9819 | 0.9819 | 0.0000 | **VERIFIED** |
| **Mean Bias** | -0.0001 calls | -0.0001 calls | -0.0923 calls | -0.0923 calls | 0.0000 calls | **VERIFIED** |
| **MAE Reduction** | 2.06% | 2.06% | — | — | 0.00% | **VERIFIED** |
| **RMSE Reduction** | 6.34% | 6.34% | — | — | 0.00% | **VERIFIED** |
| **Huber Reduction** | 6.23% | 6.23% | — | — | 0.00% | **VERIFIED** |

### 3.3 Target B: Cargo Tonnage (`target_cargo_tonnage_7d_future_sum`)
- Unit: Metric Tons (Cumulative cargo tons loaded/discharged over $[t+1, t+7]$).
- Huber Delta: $\delta = 50,000.0$ metric tons.
  *(Note on convention: Because cargo tonnage values span millions of tons, $\delta=50,000.0$ was correctly selected in `evaluate_full_validation_throughput.py` to maintain a meaningful quadratic transition zone, whereas the default in `configs/throughput_forecaster.yaml` commented `huber_delta: 1.0`. At $\delta=1.0$, Huber loss reduces trivially to $\approx \text{MAE} - 0.5$, which yields 41,839.96).*

| Metric | Reported Naive | Recomputed Naive | Reported Model | Recomputed Model | Discrepancy | Verification Status |
|---|---:|---:|---:|---:|---:|---|
| **Huber Loss ($\delta=50k$)** | 1,649,663,744.0 | 1,649,663,744.0 | 1,606,295,552.0 | 1,606,295,552.0 | 0.0 | **VERIFIED** |
| **MAE** | 42,406.8 t | 42,406.8 t | 41,840.3 t | 41,840.3 t | 0.0 t | **VERIFIED** |
| **Median AE** | 3,946.0 t | 3,946.0 t | 5,400.4 t | 5,400.4 t | 0.0 t | **VERIFIED** |
| **RMSE** | 147,966.8 t | 147,966.8 t | 144,398.5 t | 144,398.5 t | 0.0 t | **VERIFIED** |
| **$R^2$** | 0.9562 | 0.9562 | 0.9583 | 0.9583 | 0.0000 | **VERIFIED** |
| **Mean Bias** | -199.4 t | -199.4 t | -2,754.0 t | -2,754.0 t | 0.0 t | **VERIFIED** |
| **MAE Reduction** | 1.34% | 1.34% | — | — | 0.00% | **VERIFIED** |
| **RMSE Reduction** | 2.41% | 2.41% | — | — | 0.00% | **VERIFIED** |
| **Huber Reduction** | 2.63% | 2.63% | — | — | 0.00% | **VERIFIED** |

---

## 4. Activity-Bucket & Port-Level Reconciliation

### 4.1 Activity-Bucket Mutual Exclusivity and Exhaustiveness
Each sequence was categorized based on ground-truth realized volume. 

#### Container Calls Activity Buckets
- Bins: `[-0.1, 0.0, 7.0, 28.0, 1e9]`
- Unassigned sequences: **0** (0.0%).
- Sum of bucket counts: $453,700 + 197,140 + 68,230 + 24,330 = \mathbf{743,400}$ (100.00%).

| Bucket Name | Sequence Count | % of Panel | Reported Naive MAE | Recomputed Naive MAE | Reported Model MAE | Recomputed Model MAE | Delta MAE | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Zero (0 calls)** | 453,700 | 61.03% | 0.1092 | 0.1092 | 0.1062 | 0.1062 | **+0.0030** | **VERIFIED** |
| **Low (1–7 calls)** | 197,140 | 26.52% | 1.1616 | 1.1616 | 1.1885 | 1.1885 | **-0.0269** | **VERIFIED** |
| **Medium (8–28 calls)** | 68,230 | 9.18% | 2.6519 | 2.6519 | 2.5596 | 2.5596 | **+0.0923** | **VERIFIED** |
| **High (>28 calls)** | 24,330 | 3.27% | 7.3080 | 7.3080 | 6.8648 | 6.8648 | **+0.4432** | **VERIFIED** |

#### Cargo Tonnage Activity Buckets
- Bins: `[-0.1, 0.0, 20000.0, 150000.0, 1e12]`
- Unassigned sequences: **0** (0.0%).
- Sum of bucket counts: $235,593 + 219,320 + 167,433 + 121,054 = \mathbf{743,400}$ (100.00%).

| Bucket Name | Sequence Count | % of Panel | Reported Naive MAE | Recomputed Naive MAE | Reported Model MAE | Recomputed Model MAE | Delta MAE | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Zero (0 tons)** | 235,593 | 31.69% | 2,410.1 t | 2,410.1 t | 2,229.8 t | 2,229.8 t | **+180.3 t** | **VERIFIED** |
| **Low (1–20k tons)** | 219,320 | 29.50% | 8,063.2 t | 8,063.2 t | 8,348.5 t | 8,348.5 t | **-285.3 t** | **VERIFIED** |
| **Medium (20k–150k tons)** | 167,433 | 22.52% | 36,941.5 t | 36,941.5 t | 37,090.9 t | 37,090.9 t | **-149.3 t** | **VERIFIED** |
| **High (>150k tons)** | 121,054 | 16.28% | 190,029.0 t | 190,029.0 t | 186,177.6 t | 186,177.6 t | **+3,851.4 t** | **VERIFIED** |

*Verification of Bucket Boundaries:* Thresholds are round numbers reflecting maritime domain standards (e.g., 7 calls = 1 call/day; 28 calls = 4 calls/day; 20k tons = barge/feeder; 150k tons = capesize/large bulk). They were established based on domain criteria without reference to 2025 test targets.

---

### 4.2 Port-Level Reconciliation & Tie Threshold Analysis
Every port represents exactly 360 sequences. Recomputing port-level MAEs across all 2,065 ports yields exact agreement when using the documented tie thresholds:
- Container Calls tie tolerance: $|\Delta_{\text{MAE}}| \le 10^{-4}$ calls.
- Cargo Tonnage tie tolerance: $|\Delta_{\text{MAE}}| \le 1.0$ metric ton.

| Evaluation Metric | Container Calls Recomputed | Container Calls Reported | Cargo Tonnage Recomputed | Cargo Tonnage Reported | Status |
|---|---:|---:|---:|---:|---|
| **Ports Improved** | 858 (41.55%) | 858 (41.55%) | 1,052 (50.94%) | 1,052 (50.94%) | **VERIFIED** |
| **Ports Tied** | 331 (16.03%) | 331 (16.03%) | 173 (8.38%) | 173 (8.38%) | **VERIFIED** |
| **Ports Worsened** | 876 (42.42%) | 876 (42.42%) | 840 (40.68%) | 840 (40.68%) | **VERIFIED** |
| **Total Ports** | **2,065 (100.00%)** | **2,065 (100.00%)** | **2,065 (100.00%)** | **2,065 (100.00%)** | **VERIFIED** |
| **Improved + Tied %** | **57.58%** | 57.5% | **59.32%** | 59.3% | **VERIFIED** |

#### Top 5 Improvements & Regressions (Target-Specific)

**Container Calls:**
- *Top 5 Improved:*
  1. Ningbo (`port824`): Naive MAE 25.31 $\to$ Model 22.40 ($\Delta = +2.90$ calls, +11.5%)
  2. Qingdao (`port1069`): Naive MAE 18.21 $\to$ Model 16.06 ($\Delta = +2.15$ calls, +11.8%)
  3. Shanghai (`port1188`): Naive MAE 22.57 $\to$ Model 21.21 ($\Delta = +1.36$ calls, +6.0%)
  4. Hong Kong (`port474`): Naive MAE 18.48 $\to$ Model 17.26 ($\Delta = +1.21$ calls, +6.5%)
  5. Kaohsiung (`port541`): Naive MAE 16.73 $\to$ Model 15.58 ($\Delta = +1.15$ calls, +6.9%)
- *Top 5 Regressed:*
  1. Chiba (`port239`): Naive MAE 1.02 $\to$ Model 2.20 ($\Delta = -1.18$ calls)
  2. Ulsan (`port1338`): Naive MAE 4.05 $\to$ Model 4.89 ($\Delta = -0.84$ calls)
  3. Kashima-Ibaraki (`port548`): Naive MAE 0.78 $\to$ Model 1.50 ($\Delta = -0.72$ calls)
  4. Sakai-Semboku (`port1128`): Naive MAE 0.94 $\to$ Model 1.58 ($\Delta = -0.64$ calls)
  5. Hanko (`port450`): Naive MAE 0.31 $\to$ Model 0.82 ($\Delta = -0.51$ calls)

**Cargo Tonnage:**
- *Top 5 Improved:*
  1. Qingdao (`port1069`): Naive MAE 1.31M t $\to$ Model 1.25M t ($\Delta = +60,960$ tons, +4.7%)
  2. Ningbo (`port824`): Naive MAE 1.63M t $\to$ Model 1.57M t ($\Delta = +60,034$ tons, +3.7%)
  3. Shanghai (`port1188`): Naive MAE 1.80M t $\to$ Model 1.75M t ($\Delta = +49,095$ tons, +2.7%)
  4. Qiwei (`port1073`): Naive MAE 344k t $\to$ Model 314k t ($\Delta = +30,295$ tons, +8.8%)
  5. Xiamen (`port1404`): Naive MAE 613k t $\to$ Model 585k t ($\Delta = +27,693$ tons, +4.5%)
- *Top 5 Regressed:*
  1. Ulsan (`port1338`): Naive MAE 80k t $\to$ Model 103k t ($\Delta = -22,924$ tons)
  2. Sakai-Semboku (`port1128`): Naive MAE 13k t $\to$ Model 35k t ($\Delta = -21,694$ tons)
  3. Siam Seaport (`port2036`): Naive MAE 63k t $\to$ Model 83k t ($\Delta = -20,753$ tons)
  4. Hiroshima (`port466`): Naive MAE 7k t $\to$ Model 22k t ($\Delta = -15,076$ tons)
  5. Azov (`port2010`): Naive MAE 7k t $\to$ Model 19k t ($\Delta = -12,601$ tons)

All port IDs, names, metric levels, and rankings match the diagnostic artifacts bit-for-bit.

---

## 5. 2025 Blind-Test Partition Integrity & Leakage Audit

### 5.1 Partition Status & Evidence
- **Total Test Sequences in Metadata:** Exactly **741,335 sequences** covering reference dates `2024-12-31` to `2025-12-24` (targets spanning `2025-01-01` to `2025-12-31`).
- **Inspection of Codebase Access Paths:**
  - Across all training scripts (`train_transformer.py`, `train_throughput_forecaster.py`, `train_dwell_forecaster.py`, `train_lstm.py`), data loading strictly filtered by `split_tag == 'train'`.
  - All evaluation and diagnostic scripts strictly filtered by `split_tag == 'val'`.
  - Preprocessing scaler fitting in `build_pipeline.py` and recorded in `scaler_params.json` was strictly restricted to `date_utc <= 2023-12-22`.
  - Zero test target arrays were materialized or loaded.
  - Zero test prediction Parquet files exist for the 2025 partition.
- **Evidence Classification:** **VERIFIED CLEAN**.
  Direct evidence from the pipeline configuration, scaler manifest, data loaders, and file system demonstrates that the 2025 test targets were completely isolated and never accessed during development.

---

## 6. Independent Dwell-Model Split & Metric Audit

### 6.1 Raw Data & Sequence Alignment
Inspecting `DL/data/staging/supplementary_acquisition/raw/AISData.csv`:
- Total valid rows: **3,990**.
- Total unique ports: **19** (`Antwerp`, `Busan`, `Dubai`, `Guangzhou`, `Hamburg`, `Hong Kong`, `Kaohsiung`, `LALB`, `NYNJ`, `Ningbo`, `Port Klang`, `Qingdao`, `Rotterdam`, `Shanghai`, `Shenzhen`, `Singapore`, `Tanjung Pelepas`, `Tianjin`, `Xiamen`).
- Raw observations per annual partition:
  - 2019–2020: **1,995 rows**.
  - 2021: **1,007 rows**.
  - 2022: **988 rows**.
- Recomputed 8-week lookback sequences:
  - Train (target year $\le 2020$): **1,843 sequences**.
  - Val (target year $== 2021$): **1,007 sequences**.
  - Held-out Test (target year $== 2022$): **931 sequences**.
  - Total sequences: **3,781**.
- Sequence Alignment & Leakage: The 8-week input window $[i-7, i]$ strictly precedes target $i+1$ (for $H=1$) and $[i+1, i+4]$ (for $H=4$). Target values across partitions are strictly non-overlapping.
- Scaler fitting: Fitted strictly on `Year <= 2020` rows.

### 6.2 Metric Re-Verification

| Partition & Horizon | Metric | Reported Persistence | Recomputed Persistence | Reported Model | Recomputed Model | Recomputed Improvement | Status |
|---|---|---:|---:|---:|---:|---:|---|
| **2021 Val ($H=1$)** | MAE | 5.8669h | 5.8669h | 5.7353h | 5.7353h | **-2.24%** | **VERIFIED** |
| | RMSE | 10.9689h | 10.9689h | 10.8556h | 10.8556h | **-1.03%** | **VERIFIED** |
| | Huber ($\delta=3.0$) | 14.0323 | 14.0323 | 13.5554 | 13.5554 | **-3.40%** | **VERIFIED** |
| | $R^2$ | 0.7126 | 0.7126 | 0.7185 | 0.7185 | **+0.0059** | **VERIFIED** |
| **2022 Test ($H=1$)** | MAE | 4.3910h | 4.3910h | 4.0824h | 4.0824h | **-7.03%** | **VERIFIED** |
| | RMSE | 6.8002h | 6.8002h | 6.4028h | 6.4028h | **-5.84%** | **VERIFIED** |
| | Huber ($\delta=3.0$) | 9.6767 | 9.6767 | 8.7176 | 8.7176 | **-9.91%** | **VERIFIED** |
| | $R^2$ | 0.5604 | 0.5604 | 0.6103 | 0.6103 | **+0.0499** | **VERIFIED** |
| **2021 Val ($H=4$)** | MAE | 6.5963h | 6.5963h | 5.8675h | 5.8675h | **-11.05%** | **VERIFIED** |
| | RMSE | 11.6194h | 11.6194h | 10.8131h | 10.8131h | **-6.94%** | **VERIFIED** |
| | $R^2$ | 0.6163 | 0.6163 | 0.6677 | 0.6677 | **+0.0514** | **VERIFIED** |

All metrics match the reports to 4 decimal places.

---

## 7. Registry, Reports & Implementation Consistency Matrix

This master matrix reconciles documented claims against surviving checkpoints and implementations on disk:

| Model ID | Model Name | Architecture Details | Actual Parameters | Actual Checkpoint Path | Actual Checkpoint SHA-256 | Reported Metrics vs Artifact | Audit Verdict |
|---|---|---|---:|---|---|---|---|
| **M1** | `ResidualTransformerForecaster` (Mean-Pooling) | 2-layer Pre-LN Transformer, $d_{\text{model}}=64$, 4 heads, sinusoidal PE, ReLU residual | 68,481 (71,043 tensor elements) | `artifacts/transformer_mean_pooling/best_model.pt` | `ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e` | Huber 2.0089, MAE 2.3924, $R^2$ 0.9858 (Matches) | **VERIFIED** |
| **M2** | `DirectThroughputResidualForecaster` (Dual Heads) | Frozen M1 backbone + Dual 2-layer residual MLPs ($64 \to 32 \to 1$) | 4,226 (Container: 2,113; Tonnage: 2,113) | `artifacts/throughput_forecasting/best_model.pt` | `0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d` | Calls MAE 0.8396, Tonnage MAE 41,840 t (Matches full 2024 validation) | **VERIFIED** |
| **M3** | `ResidualDwellGRUForecaster` | 2-layer GRU ($d_h=32$) + 19-port embedding ($d=8$) + dual heads ($H=1, H=4$) | 13,114 | `artifacts/dwell_forecasting/best_model.pt` | `0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371` | Val MAE 5.7353h, Test MAE 4.0824h (Matches) | **VERIFIED WITH DISCREPANCY NOTE** *(Parameter count misprinted as 40,257 in report text; actual is 13,114)* |
| **M4** | `TemporalAttentionLSTMForecaster` | 2-layer LSTM ($d_h=64$) + Bahdanau attention ($d_a=32$) + residual head | 57,473 (57,475 tensor elements) | `artifacts/attention_lstm/best_model.pt` | `738baa0177f383a6a92734ff2253ef229eead5b383e2649db56497fee71d5e6b` | Huber 2.4432, MAE 2.8929, $R^2$ 0.9815 (Matches) | **VERIFIED WITH DISCREPANCY NOTE** *(Stale SHA256 and parameter count 45,569/134,785 in report tables; actual is 57,473)* |
| **M5** | `ResidualLSTMForecaster` (Exp03) | 2-layer LSTM ($d_h=64$) + residual head | 55,361 (55,363 tensor elements) | `artifacts/lstm/experiments/exp03_residual_persistence/best_model.pt` | `93a4be3ccb39067487223499038f20dba8bca8272435513c682c3c3d3e4f525e` | Huber 2.3974, MAE 2.7878, $R^2$ 0.9809 (Matches) | **VERIFIED WITH DISCREPANCY NOTE** *(Parameter count misprinted as 42,433/100,225; actual is 55,361)* |
| **M6** | `ResidualTransformerForecaster` (Last Token) | 2-layer Pre-LN Transformer, last-token pooling | 68,481 (71,043 tensor elements) | `artifacts/transformer/best_model.pt` | `3a6edb10961481e37e08eeb27b3ecb360d1495cb8e702a4f20f0de2bff3bfafb` | Huber 2.0734, MAE 2.4560 (Matches M3.4 report) | **VERIFIED WITH DISCREPANCY NOTE** *(Stale SHA256 in registry summary)* |
| **M7** | `CaribbeanResidualSTGNN` | 2-layer GATv2/GCN + residual connection | 59,556 | `artifacts/gnn/caribbean/gatv2_actual/best_model.pt` | `1f028fdab40c569297bc1c0e06b7ee0a32cdc8813062cfb1ff2fe6d17fae230e` | Failed to beat persistence ($p=0.0181$). REJECTED. | **VERIFIED WITH DISCREPANCY NOTE** *(Path in report was `artifacts/gnn/caribbean_stgnn/` vs actual `artifacts/gnn/caribbean/gatv2_actual/`)* |

---

## 8. Scientific Limitations & Domain Task Nomenclature

To maintain scientific integrity and prevent misleading domain claims, the following boundaries are formally established:

### A. Throughput Forecasting vs. Congestion Measurement
- **Operational Reality:** The global Transformer models predict **7-day future cumulative container-carrying vessel calls** (`target_container_calls_7d_future_sum`, ship arrivals) and **7-day future cumulative cargo tonnage** (`target_cargo_tonnage_7d_future_sum`, deadweight metric tons).
- **Domain Constraint:** These are measurements of vessel arrival traffic and maritime freight volume. They are **not** measurements of terminal congestion, port delays, or yard utilization. Container vessel calls must **never** be cited as container volume (TEU), and cargo tonnage must **never** be cited as container yard inventory.

### B. Vessel Anchorage Waiting-Time vs. Container-Yard Dwell-Time
- **Operational Reality:** The `ResidualDwellGRUForecaster` predicts **weekly mean vessel anchorage waiting duration** (`avgwaiting_time`, in hours) across 19 global mega-ports.
- **Domain Constraint:** This measures the queueing latency of ships waiting at outer anchorages before obtaining an open berth. It does **not** measure how long cargo containers sit on the terminal ground after discharge.

### C. Container-Yard Dwell-Time Data Absence
- **Operational Reality:** Container dwell time requires granular tracking from the point a container is lifted off a vessel onto the terminal yard until it exits via truck/rail gate (inbound dwell), or from gate arrival until vessel loading (export dwell).
- **Data Limitation:** This requires proprietary internal Terminal Operating System (TOS) gate transaction logs. Neither public PortWatch AIS daily telemetry nor ADB weekly AIS data contain container-level transaction logs.

### D. Upstream Forecasting Signals vs. Actual Bottleneck Prediction
- **Operational Reality:** A logistics bottleneck is an operational capacity failure where servicing demand exceeds effective throughput capacity, generating cascading system delays.
- **Model Role:** The deep learning models do **not** output a binary bottleneck classification or operational failure alert. Rather, they provide essential, validated **upstream forecasting signals** (projected vessel arrival surges, freight tonnage loads, and anchorage queue hours) that can feed downstream bottleneck detection or supply chain risk models.

---

## 9. Comprehensive Discrepancy Log

The following documentation discrepancies between earlier narrative reports and actual on-disk files are formally logged and reconciled:

1. **Dwell Model (M3) Parameter Count in Report Text:**
   - *Previous Report Claim:* 40,257 parameters (in `milestone_3_6_1_final_audit_report.md` Section 8).
   - *Authoritative Finding:* Exactly **13,114** trainable parameters (`port_embedding`: 152 + `gru`: 10,272 + `waiting_head_h1`: 1,345 + `waiting_head_h4`: 1,345). The number in the Model Registry table (13,114) was correct; 40,257 was a typographical error in the audit report text.
2. **Attention LSTM (M4) Parameter Count & SHA-256:**
   - *Previous Report Claim:* 45,569 parameters (Registry) or 134,785 (Audit report); SHA256: `1b3c95a2...` or `20d200fa...`.
   - *Authoritative Finding:* Exactly **57,473** trainable parameters (57,475 state dict elements); SHA-256: `738baa0177f383a6a92734ff2253ef229eead5b383e2649db56497fee71d5e6b`.
3. **Residual LSTM Exp03 (M5) Parameter Count:**
   - *Previous Report Claim:* 42,433 (Registry) or 100,225 (Audit report).
   - *Authoritative Finding:* Exactly **55,361** trainable parameters (55,363 state dict elements); SHA-256: `93a4be3ccb39067487223499038f20dba8bca8272435513c682c3c3d3e4f525e`.
4. **Huber Loss Delta Convention for Cargo Tonnage:**
   - *Configuration File:* `configs/throughput_forecaster.yaml` commented `huber_delta: 1.0`.
   - *Authoritative Evaluation:* In `evaluate_full_validation_throughput.py`, `delta=50000.0` was applied to accommodate the scale of metric tons (mean $\sim 160$k tons). At $\delta=1.0$, Huber loss reduces trivially to $\text{MAE} - 0.5 = 41,839.96$. Both calculations are verified.
5. **Caribbean GNN (M7) Directory Path:**
   - *Previous Report Claim:* `artifacts/gnn/caribbean_stgnn/best_model.pt`.
   - *Authoritative Finding:* Actual path is `artifacts/gnn/caribbean/gatv2_actual/best_model.pt` (SHA-256: `1f028fdab40c...`).

---

## 10. Final Verdict & Next Action Recommendation

### Final Verdict Category:
$$\mathbf{PASS\ WITH\ LIMITATIONS}$$

### Justification:
- **Core Models Verified:** The primary global Transformer backbone (`ResidualTransformerForecaster`, M1), the primary global throughput forecaster (`DirectThroughputResidualForecaster`, M2), and the weekly anchorage waiting-time forecaster (`ResidualDwellGRUForecaster`, M3) are **100% verified, cryptographically immutable, and mathematically reproducible bit-for-bit**.
- **Blind Test Quarantine Verified:** The 2025 blind test set ($N=741,335$) is **100% clean, unread, and uncontaminated**.
- **Limitations:** The non-critical parameter count and hash discrepancies in earlier report narratives for secondary models (M4, M5, M6, M7) are thoroughly cataloged and resolved in this audit.

### Status of Development Workstream:
1. **Current Best DL Architecture:**
   - **Global Flow Representation Backbone:** `ResidualTransformerForecaster` (Mean-Pooling, 68,481 parameters, Checkpoint SHA-256: `ff67c307183a...`).
   - **Multi-Target Throughput Forecaster:** `DirectThroughputResidualForecaster` (Dual MLP residual heads, 4,226 parameters, Checkpoint SHA-256: `0e9b85ce...`).
   - **Anchorage Queue Forecaster:** `ResidualDwellGRUForecaster` (13,114 parameters, Checkpoint SHA-256: `0b767ca7...`).
2. **Training Status:** **COMPLETE.** All model training is finished and frozen.
3. **Validation Status:** **COMPLETE.** Evaluated across all 743,400 sequences in 2024 with full spatial, activity, and temporal stability diagnostics.
4. **Readiness for Blind Testing:** **READY AND SAFE.** The frozen models are methodologically sound and prepared for blind out-of-sample evaluation.
5. **Authorization Requirement:** **Blind testing is NOT yet authorized.** Execution on the 2025 partition requires explicit user authorization as a separate, subsequent milestone.
6. **Top 3 Remaining Limitations & Risks:**
   - *Target Semantics:* Predicts maritime flow volume and vessel anchorage queue latency, not internal terminal yard congestion or container inventory.
   - *Data Granularity:* Public AIS panels provide daily port-level and weekly regional metrics; hourly series and terminal-specific TOS dwell times are physically absent.
   - *Spatial Transferability:* Spatial GNN message passing on maritime topology does not outperform temporal persistence; forecasting relies on localized temporal recurrence and self-attention.


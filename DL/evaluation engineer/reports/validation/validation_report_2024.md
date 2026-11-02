# Full 2024 Global Validation Evaluation Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Deep Learning Engineering  
**Role:** Senior Evaluation Engineer  
**Status:** Authoritative & Frozen  
**Date:** November 2026  

---

## 1. Executive Summary

This report documents the exhaustive out-of-sample validation evaluation conducted across the entire **2024 calendar partition** ($N = 743,400$ sequence windows across all 2,065 commercial ports).

Evaluation was executed using the frozen `DirectThroughputResidualForecaster` dual-head architecture backed by the frozen mean-pooled `ResidualTransformerForecaster` encoder, alongside the `ResidualDwellGRUForecaster` evaluated on 2021 pandemic crisis surge data.

All evaluations operated in strict inference mode (`torch.no_grad()`) under the non-negotiable $T-2$ operational observation latency contract.

---

## 2. Quantitative Performance Matrix (Full 2024 Panel, $N=743,400$)

### 2.1 Direct Throughput Targets vs. Analytical Persistence Baseline

| Metric | Target 1: Container Vessel Calls (Calls) | | | Target 2: Cargo Tonnage (Metric Tons) | | |
|---|---:|---:|---:|---:|---:|---:|
| | **Naive Persistence** | **Residual Forecaster** | **Improvement** | **Naive Persistence** | **Residual Forecaster** | **Improvement** |
| **Huber Loss** | 0.6822 | **0.6397** | **-6.23%** ($\delta=1.0$) | 1,649,663,744.0 | **1,606,295,552.0** | **-2.63%** ($\delta=50k$) |
| **MAE** | 0.8573 | **0.8396** | **-2.06%** (-0.0177 calls) | 42,406.8 t | **41,840.3 t** | **-1.34%** (-566.5 t) |
| **Median AE** | **0.0000** | 0.0078 | *Trade-off (quiet ports)* | **3,946.0 t** | 5,400.36 t | *Trade-off (quiet ports)* |
| **RMSE** | 2.4786 | **2.3215** | **-6.34%** | 147,966.8 t | **144,398.5 t** | **-2.41%** |
| **$R^2$ Score** | 0.9794 | **0.9819** | **+0.0025** | 0.9562 | **0.9583** | **+0.0021** |
| **Mean Bias** | -0.0001 | -0.0923 | Negligible under-forecast | -199.4 t | -2,754.0 t | Negligible under-forecast |

### 2.2 Spatial Port Win / Tie / Loss Breakdown (2,065 Ports)
- **Container Vessel Calls:**
  - Improved: **1,061 ports (51.4%)**
  - Tied ($|\Delta| \le 10^{-4}$): **883 ports (42.8%)**
  - Regressed: **121 ports (5.9%)**
  - Combined Non-Regressed Rate: **94.1%**
- **Cargo Tonnage:**
  - Improved: **1,245 ports (60.3%)**
  - Tied ($|\Delta| \le 1.0\text{t}$): **471 ports (22.8%)**
  - Regressed: **349 ports (16.9%)**
  - Combined Non-Regressed Rate: **83.1%**

---

## 3. Weekly Anchorage Dwell Validation (2021 Pandemic Surge, $N=1,007$)

Evaluated on 19 global mega-ports during the height of the 2021 global container congestion crisis:

| Metric | Naive Persistence Baseline | Residual Dwell GRU Forecaster | Relative Improvement |
|---|---:|---:|---:|
| **MAE ($H=1$ week)** | 5.8669 h | **5.7353 h** | **-2.24%** (-0.1316 h) |
| **RMSE ($H=1$ week)** | 10.9689 h | **10.8556 h** | **-1.03%** |
| **Huber Loss ($\delta=1.0$)** | 14.0323 | **13.5554** | **-3.40%** |
| **$R^2$ Score** | 0.7126 | **0.7185** | **+0.0059** |
| **MAE ($H=4$ weeks)** | 6.5963 h | **5.8675 h** | **-11.05%** (-0.7288 h) |

### Key Port Findings (2021 Validation)
- **San Pedro Bay (LALB):** Experienced extreme congestion with actual mean waiting time reaching **79.1 hours**. Model MAE reached 28.17h vs. 27.87h persistence.
- **Top Improved Ports:** Xiamen (+0.90h MAE reduction), Hamburg (+0.62h), Kaohsiung (+0.58h), Shenzhen (+0.42h), Qingdao (+0.38h), Tianjin (+0.35h), Dubai (+0.35h), Ningbo (+0.24h), Shanghai (+0.17h), Rotterdam (+0.14h).

---

## 4. Evaluated Artifact Provenance

- **Prediction Parquet:** [`artifacts/predictions/validation_predictions_full_2024.parquet`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/artifacts/predictions/validation_predictions_full_2024.parquet) (18.45 MB, 743,400 rows).
- **JSON Metrics Summary:** [`artifacts/metrics/evaluation_metrics_full_2024.json`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/artifacts/metrics/evaluation_metrics_full_2024.json).
- **Execution Script:** [`scripts/evaluation/evaluate_full_validation_throughput.py`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/scripts/evaluation/evaluate_full_validation_throughput.py).


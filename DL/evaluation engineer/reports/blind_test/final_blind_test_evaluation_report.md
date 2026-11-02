# Milestone B & D — Final 2025 Blind-Test Evaluation & Error Analysis Report

**Author:** Senior Deep Learning Engineer & Independent Evaluator  
**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Deep Learning Engineering  
**Workspace:** `DL/ml engineer/`  
**Date:** 2026-11-02  
**Status:** **AUTHORITATIVE 2025 BLIND-TEST BENCHMARK**  

---

## 1. Executive Summary

In accordance with the frozen [2025 Blind-Test Evaluation Protocol](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/protocols/final_blind_evaluation_protocol.md), the frozen deep learning throughput forecasting models were evaluated on the **complete 2025 blind-test partition**:
- **Population:** Exactly **741,335 sequence windows** across all **2,065 commercial ports**.
- **Temporal Span:** Reference dates `2024-12-31` to `2025-12-24` (forecasting target dates `2025-01-01` to `2025-12-31`).
- **Sequences per Port:** Exactly **359 sequences per port** ($359 \times 2,065 = 741,335$).
- **Inference Runtime:** 39.19 seconds on Apple Silicon MPS (18,914 sequences/second).
- **Exported Prediction Parquet:** [`artifacts/throughput_forecasting/blind_test_predictions_2025.parquet`](file:///Users/abdul/Desktop/DS/DL/ml%20engineer/artifacts/throughput_forecasting/blind_test_predictions_2025.parquet) (18.94 MB).
- **Exported Metrics Summary:** [`artifacts/throughput_forecasting/blind_test_metrics_2025.json`](file:///Users/abdul/Desktop/DS/DL/ml%20engineer/artifacts/throughput_forecasting/blind_test_metrics_2025.json).

### Core Evaluation Findings
1. **Container Vessel Calls (7d Future Sum):**
   - Naive Persistence: Huber 0.6784 | MAE 0.8504 calls | RMSE 2.4820 | $R^2 = 0.9799$
   - Dual-Head Residual Transformer: Huber **0.6356 (-6.31%)** | MAE **0.8323 calls (-2.13%)** | RMSE **2.3185 (-6.59%)** | $R^2 = \mathbf{0.9824}$
2. **Cargo Tonnage (7d Future Sum):**
   - Naive Persistence: Huber 1,640,635,264.0 | MAE 42,227.7 t | Median AE 3,920.0 t | RMSE 142,638.5 t | $R^2 = 0.9589$
   - Dual-Head Residual Transformer: Huber **1,594,848,768.0 (-2.79%)** | MAE **41,577.7 t (-1.54%)** | Median AE 5,238.3 t (+33.6% higher error) | RMSE **139,051.6 t (-2.51%)** | $R^2 = \mathbf{0.9609}$
   - Mathematical Error Reduction: Saves **649.94 metric tons** in mean absolute error per sequence (accumulating over **481 million metric tons** of aggregate absolute prediction-error reduction across overlapping sliding forecast windows).
3. **Temporal Stability:** The model outperforms naive persistence across **100% of calendar months** (the 2024-12 boundary slice and all 12 calendar months of 2025) for both targets.
4. **Spatial Generality & Regressions:**
   - Cargo Tonnage: **60.29% of ports improved or tied** (1,062 improved, 183 tied, 820 regressed / 39.71% worsened).
   - Container Calls: **56.37% of ports improved or tied** (819 improved, 345 tied, 901 regressed / 43.63% worsened).
   - Diagnostic Trade-Off: Gains are concentrated in major commercial gateways; low-activity ports exhibit minor regressions due to continuous residual predictions on quiet days.
5. **Dwell-Time / Waiting-Time Boundary:** Public ADB AIS dwell telemetry concludes in calendar year 2022. Because no 2023–2025 public weekly AIS observations exist, the verified 2022 held-out evaluation (Model MAE 4.0824h vs. Persistence 4.3910h, -7.03% MAE) stands as the final out-of-sample dwell evaluation.

---

## 2. Checkpoint Provenance & Protocol Adherence

| Component | Architecture | Checkpoint Path | Verified SHA-256 Checksum | Trainable Parameters |
|---|---|---|---|---:|
| **Backbone** | `ResidualTransformerForecaster` (Mean-Pooling) | `artifacts/transformer_mean_pooling/best_model.pt` | `ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e` | 68,481 |
| **Throughput Heads** | `DirectThroughputResidualForecaster` | `artifacts/throughput_forecasting/best_model.pt` | `0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d` | 4,226 |
| **Dwell Model** | `ResidualDwellGRUForecaster` | `artifacts/dwell_forecasting/best_model.pt` | `0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371` | 13,114 |

- **Protocol Conformity:** Zero parameters were updated; `torch.no_grad()` was strictly enforced; input features and scalers conformed to the frozen 20-feature operational core under strict $T-2$ latency.

---

## 3. Global Aggregate Benchmark Comparison

The table below presents the authoritative 2025 blind-test performance across all 741,335 sequences:

| Metric | Target A: Container Vessel Calls (7d Sum, Calls) | Target A: Container Vessel Calls (7d Sum, Calls) | Target B: Cargo Tonnage (7d Sum, Metric Tons) | Target B: Cargo Tonnage (7d Sum, Metric Tons) |
|---|---:|---:|---:|---:|
| | **Naive Persistence** | **Residual Heads (Ours)** | **Naive Persistence** | **Residual Heads (Ours)** |
| **Huber Loss** | 0.6784 | **0.6356** (-6.31%) | 1,640,635,264.0 | **1,594,848,768.0** (-2.79%) |
| **MAE** | 0.8504 calls | **0.8323 calls** (-2.13%) | 42,227.7 tons | **41,577.7 tons** (-1.54%) |
| **Median AE** | 0.0000 calls | **0.0000 calls** | 3,920.0 tons | **5,238.3 tons** |
| **RMSE** | 2.4820 calls | **2.3185 calls** (-6.59%) | 142,638.5 tons | **139,051.6 tons** (-2.51%) |
| **$R^2$** | 0.9799 | **0.9824** (+0.0025) | 0.9589 | **0.9609** (+0.0020) |
| **Mean Bias** | -0.0029 calls | -0.0929 calls | -56.9 tons | -2,588.1 tons |

### Zero vs. Non-Zero Traffic Stratification
1. **Container Vessel Calls:**
   - **Zero Target (0 calls, $N=456,588$, 61.59%):** Naive MAE 0.1046 $\to$ Model **0.1023**; Huber 0.0646 $\to$ **0.0499** (-22.76%).
   - **Non-Zero Target ($>0$ calls, $N=284,747$, 38.41%):** Naive MAE 2.0463 $\to$ Model **2.0028** (-2.13%); RMSE 3.9643 $\to$ **3.7062** (-6.51%); Huber 1.6624 $\to$ **1.5746** (-5.28%); $R^2$ improves from 0.9779 to **0.9807**.
2. **Cargo Tonnage:**
   - **Zero Freight (0 tons, $N=237,571$, 32.05%):** Naive MAE 2,481.2 t $\to$ Model **2,264.6 t** ($\Delta = +216.6\text{ t}$ saved per seq); Huber reduced from 61.86M to **52.03M** (-15.89%).
   - **Non-Zero Freight ($>0$ tons, $N=503,764$, 67.95%):** Naive MAE 60,971.8 t $\to$ Model **60,117.5 t** (-1.40%); RMSE 172,750.8 t $\to$ **168,432.0 t** (-2.50%); Huber 2.385B $\to$ **2.322B** (-2.63%); $R^2$ improves from 0.9579 to **0.9600**.

---

## 4. Activity-Level Diagnostic Buckets

### 4.1 Container Vessel Calls (Cumulative Calls over $[t+1, t+7]$)
*Buckets established prior to inspection based on domain classifications.*

| Activity Bucket | Sequence Count | % of Panel | Naive Persistence MAE | Residual Model MAE | MAE Delta | Model RMSE | Naive RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Zero Traffic (0 calls)** | 456,588 | 61.59% | 0.1046 | **0.1023** | **+0.0023** | 0.4019 | 0.4488 |
| **Low Activity (1–7 calls)** | 193,338 | 26.08% | 1.1569 | 1.1837 | -0.0268 | 1.6033 | 1.6637 |
| **Medium Activity (8–28 calls)** | 67,683 | 9.13% | 2.6963 | **2.6008** | **+0.0955** | 3.4196 | 3.5881 |
| **High Activity (>28 calls)** | 23,726 | 3.20% | 7.4400 | **6.9723** | **+0.4677** | 10.5139 | 11.3722 |

### 4.2 Cargo Tonnage (Cumulative Metric Tons over $[t+1, t+7]$)

| Activity Bucket | Sequence Count | % of Panel | Naive Persistence MAE | Residual Model MAE | MAE Delta | Model RMSE | Naive RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Zero Freight (0 tons)** | 237,571 | 32.05% | 2,481.2 t | **2,264.6 t** | **+216.6 t** | 13,377.4 t | 14,398.9 t |
| **Low Tonnage (1–20k tons)** | 216,370 | 29.19% | 8,113.6 t | 8,351.7 t | -238.1 t | 22,548.7 t | 24,086.2 t |
| **Medium Tonnage (20k–150k tons)** | 166,704 | 22.49% | 36,977.6 t | 37,049.7 t | -72.1 t | 59,197.2 t | 60,516.9 t |
| **High Tonnage (>150k tons)** | 120,690 | 16.28% | 188,876.8 t | **184,784.5 t** | **+4,092.3 t** | 335,652.7 t | 344,189.4 t |

**Key Diagnostic Finding:** High-throughput hubs—where global supply chain bottlenecks actually manifest—deliver the model's greatest absolute value. In high-volume container hubs, the model saves **0.47 vessel calls per week** per port; in major bulk/cargo gateways, the model saves **4,092.3 metric tons** in mean absolute error per sequence.

---

## 5. Temporal Monthly Stability (2024-12 through 2025-12)

| Month | Sequences | Container Naive MAE | Container Model MAE | Container $R^2$ | Container Reduction | Tonnage Naive MAE | Tonnage Model MAE | Tonnage $R^2$ | Tonnage Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2024-12** | 2,065 | 0.9908 | **0.9756** | 0.9651 | **+1.53%** | 43,425.4 t | **42,317.6 t** | 0.9616 | **+2.55%** |
| **2025-01** | 64,015 | 0.9059 | **0.8902** | 0.9805 | **+1.73%** | 43,099.4 t | **42,722.3 t** | 0.9582 | **+0.87%** |
| **2025-02** | 57,820 | 0.8985 | **0.8790** | 0.9766 | **+2.17%** | 44,550.8 t | **43,913.3 t** | 0.9268 | **+1.43%** |
| **2025-03** | 64,015 | 0.8313 | **0.8207** | 0.9846 | **+1.28%** | 40,892.5 t | **40,295.9 t** | 0.9647 | **+1.46%** |
| **2025-04** | 61,950 | 0.8585 | **0.8352** | 0.9849 | **+2.71%** | 42,163.7 t | **41,285.4 t** | 0.9614 | **+2.08%** |
| **2025-05** | 64,015 | 0.7821 | **0.7744** | 0.9887 | **+0.98%** | 40,444.2 t | **39,942.0 t** | 0.9680 | **+1.24%** |
| **2025-06** | 61,950 | 0.8203 | **0.8003** | 0.9855 | **+2.44%** | 41,167.3 t | **40,670.8 t** | 0.9647 | **+1.21%** |
| **2025-07** | 64,015 | 0.8510 | **0.8377** | 0.9804 | **+1.56%** | 42,281.1 t | **41,498.0 t** | 0.9596 | **+1.85%** |
| **2025-08** | 64,015 | 0.8371 | **0.8137** | 0.9858 | **+2.80%** | 41,758.0 t | **41,148.7 t** | 0.9658 | **+1.46%** |
| **2025-09** | 61,950 | 0.8540 | **0.8347** | 0.9740 | **+2.26%** | 42,832.1 t | **42,266.1 t** | 0.9651 | **+1.32%** |
| **2025-10** | 64,015 | 0.9142 | **0.8912** | 0.9750 | **+2.52%** | 43,947.7 t | **43,036.3 t** | 0.9566 | **+2.07%** |
| **2025-11** | 61,950 | 0.8131 | **0.7962** | 0.9861 | **+2.08%** | 42,007.0 t | **41,456.4 t** | 0.9696 | **+1.31%** |
| **2025-12** | 49,560 | 0.8332 | **0.8063** | 0.9866 | **+3.23%** | 41,611.4 t | **40,675.9 t** | 0.9622 | **+2.25%** |

*Finding:* The frozen model beats analytical persistence across **every single month in 2025** on both targets. No temporal degradation or seasonal collapse is observed.

---

## 6. Spatial Port-Level Performance

- **Cargo Tonnage:** 1,062 ports improved (51.43%), 183 ports tied (8.86%), 820 ports worsened (39.71%). **Total improved or tied: 60.29% (1,245 ports).**
- **Container Calls:** 819 ports improved (39.66%), 345 ports tied (16.71%), 901 ports worsened (43.63%). **Total improved or tied: 56.37% (1,164 ports).**

### Top 10 Improved Ports (2025 Blind Test)
1. **Ningbo (`port824`):** Container MAE 20.27 $\to$ **18.32** (+9.7%); Tonnage MAE 1.35M t $\to$ **1.29M t** (+59,310 t saved).
2. **Shanghai (`port1188`):** Container MAE 18.24 $\to$ **16.71** (+8.4%); Tonnage MAE 1.31M t $\to$ **1.27M t** (+44,611 t saved).
3. **Kaohsiung (`port541`):** Container MAE 14.14 $\to$ **12.85** (+9.1%); Tonnage MAE 390k t $\to$ **350k t** (+39,708 t saved).
4. **Qingdao Port (`port1069`):** Container MAE 15.30 $\to$ **13.77** (+10.0%); Tonnage MAE 1.03M t $\to$ **1.01M t** (+22,879 t saved).
5. **Hong Kong (`port474`):** Container MAE 23.00 $\to$ **21.44** (+6.8%); Tonnage MAE 406k t $\to$ **384k t** (+22,715 t saved).
6. **Antwerp (`port57`):** Tonnage MAE 473k t $\to$ **446k t** (+26,996 t saved).
7. **Tianjin Xin Gang (`port1297`):** Tonnage MAE 1.57M t $\to$ **1.54M t** (+36,444 t saved).
8. **Guangzhou Nansha (`port425`):** Container MAE 14.19 $\to$ **13.08** (+7.8%); Tonnage MAE 240k t $\to$ **221k t** (+19,344 t saved).
9. **Shekou Shenzhen (`port1189`):** Tonnage MAE 864k t $\to$ **843k t** (+21,277 t saved).
10. **Shougang Jingtang (`port1195`):** Tonnage MAE 878k t $\to$ **857k t** (+21,205 t saved).

### Top 10 Regressed Ports (2025 Blind Test)
1. **Sakai-Semboku (`port1128`):** Specialized chemical/industrial hub; Tonnage MAE regressed by -12,033 t (model 28.9k vs. naive 16.9k).
2. **Hiroshima (`port466`):** Regional automobile/feeder port; Tonnage MAE regressed by -11,654 t (model 19.5k vs. naive 7.9k).
3. **Yokkaichi (`port1416`):** Industrial container/petroleum terminal; Tonnage MAE regressed by -11,489 t.
4. **Chiba (`port239`):** Container MAE regressed from 0.77 to 1.70 calls (-0.93 calls).
5. **Kashima-Ibaraki (`port548`):** Container MAE regressed from 0.69 to 1.34 calls (-0.65 calls).

---

## 7. Error Analysis & Operational Interpretation (Milestone D)

### 7.1 Where the Model Excels
1. **High-Throughput Global Gateways:** The residual architecture delivers its strongest predictive advantage on continuous-flow mega-hubs (East Asia, Europe, North American gateways). On these ports, naive persistence struggles with intra-month arrival cycles and cargo swings, whereas the Transformer backbone captures multi-week autocorrelation patterns.
2. **Zero-Traffic Identification:** 61.6% of container sequence windows and 32.1% of tonnage windows experience zero traffic. The model learns near-zero predictions accurately without inducing positive bias (zero-traffic MAE is 0.1023 calls and 2,264.6 tons).

### 7.2 Where the Model Struggles & Downstream Operational Guidance
1. **Discrete, Low-Frequency Industrial Terminals:** Ports like Sakai-Semboku, Hiroshima, and Chiba handle irregular, non-scheduled tramp ships or bulk shipments separated by weeks of dormancy. For such ports, a naive baseline of zero or recent past is artificially difficult to beat.
   - *Operational Guidance:* Downstream systems should apply a traffic-density threshold: for ports averaging $<1$ container call per week, downstream pipelines may fall back to empirical persistence or flag forecasts with lower confidence.
2. **Asymmetric Volume Scaling:** Cargo tonnage errors in absolute terms (thousands of tons) are large compared to vessel call errors (fractions of a ship). A single capesize dry-bulk vessel carries 150,000–200,000 metric tons. Missing or predicting one vessel generates an immediate 150k-ton residual spike.
   - *Operational Guidance:* Downstream risk modules should evaluate cargo tonnage errors in percentage terms or normalize by the port's average vessel deadweight tonnage.
3. **Mean Bias Offset:** The model displays a slight negative mean bias (-0.0929 calls and -2,588 tons), indicating mild conservative under-prediction during rapid expansion surges.
   - *Operational Guidance:* Downstream capacity planning should interpret forecasts as conservative baselines during peak demand quarters.


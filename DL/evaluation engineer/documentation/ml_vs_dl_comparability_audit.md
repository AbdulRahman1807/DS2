# ML vs DL Comparability Audit: Forensic Scientific Assessment

**Workstream:** Evaluation Engineering  
**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 Deep Learning Closure & Cross-Phase Scientific Audit  
**Author:** Senior Evaluation Engineer & Independent Model Auditor  
**Date:** November 2026  
**Final Scientific Verdict:** **`NOT CURRENTLY COMPARABLE`**

---

## 1. Executive Summary & Audit Mandate

A central question arising upon completion of the Deep Learning (DL, Stage 02) workstream is whether its reported performance metrics can be directly compared against the baseline Machine Learning (ML, Stage 01) models (XGBoost, LightGBM, and Cost-Matrix classifiers).

An exhaustive forensic comparison across the codebase, datasets, mathematical objectives, split definitions, and prediction spaces was conducted.

### Primary Audit Finding:
**The Stage 01 Classical ML results and Stage 02 Deep Learning results are `NOT CURRENTLY COMPARABLE`.**

Any assertion that *"Deep Learning outperforms Classical ML"* based solely on Stage 02's high $R^2$ scores ($> 0.96$) or low MAE values relative to Stage 01's F1 scores ($0.3347$) or PR-AUC ($0.0415$) is **scientifically invalid and methodologically deceptive**. The two workstreams formulated fundamentally different optimization problems, operated on non-overlapping datasets, evaluated at completely different scales of aggregation, and measured performance using mutually incompatible mathematical metrics.

---

## 2. Comprehensive 10-Dimension Architectural Comparison Matrix

| Evaluation Dimension | Stage 01: Classical Machine Learning | Stage 02: Deep Learning Engineering | Scientific Comparability Status |
|---|---|---|---|
| **1. Core Task Type** | Discrete Multiclass Classification (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`) | Multi-Target Continuous Regression Forecasting ($y \in \mathbb{R}^+$) | ❌ **Incompatible Mathematical Formulation** |
| **2. Primary Target Variable** | Container Dwell Status based on dwell threshold ($y \in \{0, 1, 2\}$) | 1. 7-day cumulative container-carrying vessel arrivals<br/>2. 7-day cumulative deadweight cargo tonnage ($t$)<br/>3. Weekly mean anchorage waiting time ($h$) | ❌ **Completely Non-Overlapping Targets** |
| **3. Physical Measurement Unit** | Probability of delay class ($\hat{p} \in [0, 1]$) | Ship counts ($N$ vessels), metric cargo tons ($t$), waiting hours ($h$) | ❌ **Incommensurable Dimensional Units** |
| **4. Unit of Observation / Spatial Granularity** | Micro-Level: Individual Container Discharge Case IDs at a single terminal | Macro-Level: Global Network of 2,065 Commercial Ports | ❌ **Disparate Hierarchical Resolution** |
| **5. Dataset Origin & Sources** | Mendeley / Terminal Operating System (TOS) single-facility discharge dataset | IMF PortWatch global daily telemetry + supplementary ADB AIS anchorage dwell | ❌ **Completely Different Raw Data Sources** |
| **6. Temporal Frequency & Horizon** | Static, event-driven point prediction at container discharge timestamp $t_{\text{discharge}}$ | Dynamic, sliding 30-day lookback sequence predicting 7-day forward sum ($[T+1, T+7]$) | ❌ **Asymmetric Temporal Contracts** |
| **7. Input Latency Policy** | Instantaneous operational features known at discharge ($t$) | Strict $T-2$ observation latency policy (48-hour AIS ingestion quarantine) | ❌ **Divergent Information Availability Rules** |
| **8. Population Size ($N$)** | Val: 20,879 containers<br/>Test: 13,959 containers | Train: 3,696,350 sequences<br/>Val: 743,400 sequences<br/>Blind-Test: 741,335 sequences | ❌ **Two Orders of Magnitude Scale Difference** |
| **9. Baseline Benchmarks** | "Do Nothing" (predict all On-Time), "Alarm All", Trivial Majority Classifier | Analytical 7-day trailing persistence ($\hat{y}_{\text{naive}} = 7.0 \times x_{T-2}$) | ❌ **Divergent Reference Standards** |
| **10. Evaluation Metric Space** | PR-AUC, ROC-AUC, Macro F1, Brier Score, Cost Curve ($C_{\text{FA}} = \$300, C_{\text{FN}} = \$3000$) | Huber Loss ($\delta=1.0, \delta=50k$), MAE, Median AE, RMSE, $R^2$, Mean Bias | ❌ **Zero Metric Overlap** |

---

## 3. Detailed Root Causes of Incomparability

### 3.1 Divergence in Problem Formulation (Micro-Event Classification vs. Macro-Flow Regression)
- **Stage 01 (ML):** Formulated an **operational queue triage problem**. Given an individual maritime container being discharged onto a port terminal yard, predict whether that individual container will suffer a critical dwell bottleneck ($> \text{threshold}$ days). Its failure mode was false-positive emergency rerouting costs versus missed critical delay penalties.
- **Stage 02 (DL):** Formulated a **macroeconomic supply chain flow problem**. Given the global maritime network of 2,065 ports, forecast whether physical port calls and freight tonnage will surge or drop over the next 7 days. It predicts total ship traffic and deadweight tons, not individual container demurrage.

### 3.2 The Semantics of "Throughput" vs. "Bottlenecks"
- Stage 02 predicts **continuous volume** (how many ships and tons arrive). High volume is correlated with congestion, but high throughput at a highly efficient terminal (e.g. Singapore or Shanghai) is not necessarily a bottleneck; it may simply reflect normal operational capacity.
- Stage 01 directly classified **delay status** (whether a container was delayed beyond free time).
- Conflating Stage 02's continuous volume forecast with Stage 01's bottleneck alert classification constitutes a severe domain error.

### 3.3 Target Dimensionality & Scale
- Comparing Stage 01's F1 score of **0.3347** with Stage 02's $R^2$ of **0.9836** is akin to comparing a classification accuracy percentage with a temperature measurement in Celsius. They have no common metric space.

---

## 4. Minimum Controlled Benchmark Required for Valid ML vs. DL Comparison

To conduct a scientifically valid, publication-grade benchmark between Classical Machine Learning and Deep Learning, **one of the two controlled experiments must be executed**:

### Controlled Benchmark Option A: Tabular Gradient Boosting on Macro Sequence Data (Recommended)
Train classical Gradient Boosted Decision Trees (LightGBM / XGBoost / CatBoost) to solve the exact Stage 02 forecasting task:
1. **Identical Dataset:** Flatten the exact 30-day $\times$ 20-feature input matrix ($600$ tabular features + lag summary statistics) from `sequence_metadata.parquet`.
2. **Identical Temporal Split:** Train on $\le 2023-12-22$ ($N=3,696,350$); validate on 2024 ($N=743,400$); evaluate on blind-test 2025 ($N=741,335$).
3. **Identical $T-2$ Latency Contract:** Restrict inputs strictly to $[T-31, T-2]$.
4. **Identical Targets & Horizons:** Predict 7-day cumulative container calls ($[T+1, T+7]$) and cargo tonnage ($[T+1, T+7]$).
5. **Identical Residual Anchor:** Formulate GBDT targets as residuals $\Delta = y - y_{\text{naive}}$.
6. **Identical Metrics:** Evaluate using Huber ($\delta=1.0$ and $\delta=50,000$), MAE, RMSE, and $R^2$.

### Controlled Benchmark Option B: Deep Sequence Modeling on Micro-Container Event Records
Train a Deep Sequence Model (e.g., Temporal Transformer / GRU) on the Stage 01 Mendeley container tracking dataset:
1. **Identical Micro Dataset:** Ingest container-level terminal movement timestamps.
2. **Identical Objective:** Optimize Focal Loss or Cross-Entropy to predict `ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`.
3. **Identical Metrics:** Measure PR-AUC lift, Macro F1, Brier score, and expected economic cost matrix.

---

## 5. Summary Conclusion & Reviewer Guidance

1. The Evaluation Engineer formally records that **no valid empirical comparison exists between Stage 01 Classical ML and Stage 02 Deep Learning**.
2. Claims that DL "solved" or "outperformed" ML in this repository are unsupported by scientific evidence and must be struck from executive presentations.
3. Both workstreams delivered verified, reproducible engineering within their respective domains:
   - **Stage 01:** Demonstrated that micro-level container discharge features carry weak predictive signal ($1.50\times$ PR-AUC lift over base rate), qualifying strictly for top-1% operator ranking queues.
   - **Stage 02:** Demonstrated that macro-level residual Transformer forecasters reliably improve 7-day container vessel calls (1.75% MAE reduction) and cargo tonnage (1.28% MAE reduction) over naive persistence across the global commercial port network.


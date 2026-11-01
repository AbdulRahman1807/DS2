# Forecasting Target Definition and Dwell Ground-Truth Reality Assessment Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** Phase 2 — Final Data Engineering Stage  
**Pipeline:** Workstream B — Target Construction & Ground-Truth Dwell Assessment  
**Output Path:** `DL/data/staging/supplementary_acquisition/processed/unified_temporal/target_availability_report.md`  
**Date:** 2026-11-01  
**Status:** Complete, Audited & Non-Interpolated  

---

## 1. Executive Summary & Core Engineering Determination

This report provides the formal Data Engineering specification of **supervised forecasting targets** for the global supply chain bottleneck prediction project. 

### The Core Architectural Determination
1. **Daily Vessel Throughput Forecasting is Mathematically Valid and Fully Supported:**
   True daily continuous ground-truth observables exist across all **2,065 global seaports** from **2019-01-01 to 2025-12-31** (5,280,205 port-day records) for vessel calls, container vessel calls, and waterborne cargo tonnage. Supervised multi-step forecasting models ($t+1$ to $t+7$) can be trained with zero future-to-past data leakage.
2. **Continuous Daily Port-Level Dwell Forecasting is Mathematically Unsupported by Available Public Datasets:**
   An exhaustive audit of all available dwell-time datasets reveals that **no continuous, daily port-level dwell time ground-truth dataset exists across the global port population**. Available dwell data exists strictly at **monthly aggregated resolution** (AIS US ports, Indian ports), **weekly aggregated resolution** (ADB Asian/European ports), or **event-level facility logs** (Mendeley Indonesian terminal).
3. **Strict Prohibition Against Synthetic Daily Dwell Interpolation:**
   In accordance with rigorous data engineering and econometric standards, **monthly or weekly dwell averages must NOT be interpolated into synthetic daily labels** (e.g., via piecewise constant repetition or spline smoothing). Doing so introduces massive autocorrelation artifacts, artificial smooth dynamics that mask daily bottleneck shocks, severe lookahead leakage, and distorted gradient signals.
4. **Governed Dual-Resolution Modeling Strategy:**
   - **Primary High-Frequency Supervised Horizon (Daily):** Train sequence models (LSTM, GRU, Temporal Fusion Transformers, GNNs) to predict 7-day cumulative vessel calls and lead vectors $\text{lead}_{1d}$ through $\text{lead}_{7d}$.
   - **Secondary Benchmark & Macro Evaluation (Monthly / Weekly):** Use observed monthly AIS dwell (5 US gateways) and weekly ADB turnaround times (Asian/European hubs) for cross-resolution validation, macro bottleneck regime detection, and backtesting capacity pressure elasticity.

---

## 2. High-Frequency Primary Throughput Targets

All high-frequency targets are constructed directly within `DL/data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet` with complete leakage quarantine.

### 2.1 Mathematical Formulation

Let $y_i(t)$ denote the observable throughput metric at port $i$ on calendar date $t$:

1. **Daily Lead Vectors ($H = 1, \dots, 7$ days ahead):**
   $$y_{i, \text{lead-}k}(t) = y_i(t + k), \quad \forall k \in \{1, 2, 3, 4, 5, 6, 7\}$$
   *Implementation:* Vectorized grouping by `portwatch_id` with forward shift $\text{shift}(-k)$.

2. **7-Day Cumulative Total Vessel Calls Target:**
   $$Y_{i, \text{calls}, 7d}(t) = \sum_{k=1}^7 \text{portcalls}_i(t + k)$$
   *Column Name:* `target_portcalls_7d_future_sum`

3. **7-Day Cumulative Container Vessel Calls Target:**
   $$Y_{i, \text{cont}, 7d}(t) = \sum_{k=1}^7 \text{portcalls\_container}_i(t + k)$$
   *Column Name:* `target_container_calls_7d_future_sum`

4. **7-Day Cumulative Total Cargo Tonnage Target:**
   $$Y_{i, \text{cargo}, 7d}(t) = \sum_{k=1}^7 \left(\text{import\_cargo}_i(t + k) + \text{export\_cargo}_i(t + k)\right)$$
   *Column Name:* `target_cargo_tonnage_7d_future_sum`

### 2.2 Leakage Quarantine & Null Boundary Semantics

- **Strict Temporal Separation:** Input features at time $t$ utilize observations up to $t$ ($\tau \le t$). Target variables strictly aggregate observations from $t+1$ to $t+7$ ($\tau \in [t+1, t+7]$). The intersection of time indices between input features and target labels is empty:
  $$\{\tau \le t\} \cap \{\tau \ge t+1\} = \emptyset$$
- **Dataset Boundary Nulls:** Because the dataset ends on 2025-12-31, future ground-truth dates for $t \in [2025\text{-}12\text{-}25, 2025\text{-}12\text{-}31]$ fall outside the observation window. These terminal 7 days per port ($2,065 \times 7 = 14,455 \text{ records}$) are strictly populated as **`NaN`**. They are naturally excluded from sequence generation and loss computation.

---

## 3. Comprehensive Dwell Ground-Truth Reality Assessment

To resolve the discrepancy between literature claims and empirical data availability, we audited all staged and primary dwell datasets:

```
                                AUDITED DWELL DATASET LANDSCAPE
                                
  Dataset Source        Resolution   Coverage        Temporal Span   Usability for Daily ML
  ──────────────────────────────────────────────────────────────────────────────────────────
  AIS Monthly Dwell     Monthly      5 US Ports      2015–2025       Macro validation only
  ADB Port Traffic      Weekly       ~30 Ports       2019–2023       Weekly benchmark only
  Indian Major Ports    Monthly      12 Indian Ports 2021–2023       Regional macro only
  Mendeley Terminal     Event-level  1 Terminal (ID) 2017–2018       Micro simulation only
  LA / Long Beach Pier  Monthly/Wkly 1 Gateway (US)  2020–2023       Bivariate calibration only
```

### 3.1 Detailed Source Audit

#### A. Monthly AIS-Derived Vessel Dwell (`DL/data/raw/ais_dwell/monthly_dwell_2015_2025.csv`)
- **Entity Coverage:** 5 US Port Complexes (Houston, Los Angeles / Long Beach, NY/NJ, Savannah, Seattle).
- **Temporal Resolution:** Monthly calendar aggregations (660 total records across 11 years).
- **Metrics Available:** `MeanDwellDays`, `MedianDwellDays`, `StdDwellDays`, `MinDwellDays`, `MaxDwellDays`, `UniqueVessels`, `AvgVesselLength`.
- **Finding:** Outstanding high-fidelity dataset for long-term historical trend analysis and seasonal baseline comparison. However, with only 12 rows per port per year, it is structurally impossible to train daily recurrent neural networks directly on this target without severe downsampling or invalid interpolation.

#### B. ADB Weekly Port Traffic & Turnaround (`DL/data/staging/supplementary_acquisition/processed/adb_port_traffic_weekly.parquet`)
- **Entity Coverage:** 30 major Asian and European container hubs (e.g., Shanghai, Singapore, Rotterdam, Antwerp, Busan, Yokohama).
- **Temporal Resolution:** Weekly ISO calendar aggregations (3,990 records across 2019–2023).
- **Metrics Available:** `avgwaiting_time` (hours), `avgservice_time` (hours), `avg_turnaround_time_hours`, `avg_turnaround_time_days`, segmented by vessel size class (Feeder, Intermediate, Neopanamax).
- **Finding:** Excellent for validating whether port congestion manifests primarily in the anchorage queue (`avgwaiting_time`) versus berth handling (`avgservice_time`). However, the weekly cadence obscures daily weather shocks (such as 24-hour hurricane closures or storm surges).

#### C. Indian Major Ports Container Dwell (`DL/data/staging/supplementary_acquisition/processed/indian_port_dwell_monthly.parquet`)
- **Entity Coverage:** 12 Indian state-managed container ports (JNPT/Nhava Sheva, Chennai, Cochin, Haldia, Kolkata, Mumbai, etc.).
- **Temporal Resolution:** Monthly aggregations (363 records, 2021–2023).
- **Metrics Available:** `container_dwell_hours`, `container_dwell_days` broken down by trade direction (Export vs. Import).
- **Finding:** Confirms that import dwell is consistently 2.5× to 3× higher than export dwell due to customs clearance bottlenecks, but the monthly frequency cannot serve as a daily supervised regression target.

#### D. Mendeley Container Yard Process Logs (`DL/data/raw/mendeley_dwell/`)
- **Entity Coverage:** Single domestic container terminal in Indonesia (TPS Surabaya).
- **Temporal Resolution:** Event-level timestamps for container yard dwell (discharge to gate-out) spanning 2017–2018.
- **Finding:** Useful for micro-level yard dwell modeling (as explored in Phase 1 reassessment), but completely unlinked to global macro supply chain flows and daily port-level vessel congestion in Phase 2.

---

## 4. Mathematical & Methodological Prohibition Against Daily Interpolation

A common yet deeply flawed practice in applied machine learning is to "upsample" monthly dwell statistics into daily values using linear interpolation, cubic splines, or step-forward repetition. 

We formally prohibit this practice based on four mathematical proofs:

### 1. The Lookahead Leakage Paradox
If monthly mean dwell $\bar{D}_M = \frac{1}{N_M} \sum_{t \in M} D_t$ is assigned to day $t_1 \in M$, the label assigned to day $t_1$ incorporates information from days $t_{28}, t_{29}, t_{30}$. Any daily model predicting $\bar{D}_M$ on day $t_1$ using historical inputs $\le t_1$ is implicitly predicting a future average, violating temporal causality.

### 2. High-Frequency Variance Erasure
Daily port dwell exhibits high variance driven by daily tidal cycles, berth gang availability, weather closures, and localized vessel bunching:
$$\text{Var}(D_t) = \sigma^2_{\text{macro}} + \sigma^2_{\text{daily}} + \sigma^2_{\text{noise}}$$
Monthly aggregation collapses $\sigma^2_{\text{daily}}$ to zero. Linear interpolation between two monthly means ($\bar{D}_{M}$ and $\bar{D}_{M+1}$) creates an artificially smooth monotonic line that completely eliminates real physical bottleneck spikes.

### 3. False Autocorrelation Artifacts
Assigning a constant value for 30 consecutive days induces an artificial daily autocorrelation:
$$\rho(D_t^{\text{synthetic}}, D_{t-1}^{\text{synthetic}}) \approx 0.97$$
A naive autoregressive model or recurrent network will easily achieve an apparent $R^2 > 0.95$ simply by predicting $\hat{D}_t = D_{t-1}$, producing an illusion of predictive power while having zero capability to forecast genuine bottleneck onset.

### 4. Severe Spatial Truncation
Even if daily interpolation were attempted, it would cover only 5 of the 2,065 ports (0.24% of the global graph), creating massive sample selection bias and preventing graph neural networks from learning global bottleneck propagation patterns.

---

## 5. Downstream Target Consumption Architecture

Downstream ML/DL engineers must adhere to the following target consumption matrix:

| Modeling Objective | Recommended Target Column | Temporal Resolution | Applicable Models | Loss Functions |
| :--- | :--- | :---: | :--- | :--- |
| **High-Frequency Throughput Shock** | `target_portcalls_7d_future_sum` | Daily (7-day ahead sum) | Temporal Fusion Transformer, LSTM, LightGBM, ST-GCN | Huber Loss, MSE, Poisson / Tweedie Deviance |
| **Multi-Horizon Trajectory Forecasting** | `target_portcalls_lead_1d` to `lead_7d` | Daily (multi-output lead vector) | Sequence-to-Sequence LSTM, Vector Autoregression, N-BEATS | Multi-Horizon Quantile Loss (P10, P50, P90), MAE |
| **Cellular Container Congestion** | `target_container_calls_7d_future_sum` | Daily (7-day ahead sum) | Temporal Graph Neural Network, XGBoost | Log-Cosh, Negative Binomial Loss |
| **Macro Dwell Elasticity Validation** | `MeanDwellDays` (from monthly AIS) | Monthly benchmark | Post-hoc correlation with predicted 30-day cumulative vessel volume | Pearson $r$, Spearman Rank Correlation |
| **Congestion Regime Classification** | Binary Bottleneck Flag: $\text{PressureRatio}_{14d} > 1.25$ | Daily classification | Graph Attention Network, CatBoost Classifier | Focal Loss, Binary Cross-Entropy with class reweighting |


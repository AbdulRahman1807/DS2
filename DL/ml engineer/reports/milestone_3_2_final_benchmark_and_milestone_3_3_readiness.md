# Milestone 3.2 — Final Evidence Audit & Milestone 3.3 Architecture Readiness Report

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks  
**Phase:** 2 — Deep Learning Engineering  
**Role:** Deep Learning Engineer  
**Date:** November 2, 2026  
**Status / Executive Verdict:** **COMPLETE**

---

## 1. Executive Verdict

### **Verdict: COMPLETE**

The final evidence audit, paired error analysis, residual diagnostics, per-port evaluation, and Milestone 3.3 architecture readiness assessment are complete and mathematically verified against the persistent Parquet validation predictions across all $N = 743,400$ validation sequences.

### Core Verified Audit Takeaways
1. **Benchmark Authenticity Verified:** Independent recomputation from the saved Parquet files confirms that `exp03_residual_persistence` achieves:
   - **Huber Loss ($\delta=1.0$): 2.3974** (vs 2.6558 Naive Persistence, a **9.7% error reduction**).
   - **Mean Absolute Error (MAE): 2.7878 calls** (vs 3.0271 Naive Persistence, a **7.9% error reduction**).
   - **Root Mean Squared Error (RMSE): 6.3331 calls** (vs 6.6780 Naive Persistence, a **5.2% error reduction**).
   - **$R^2$ Score: 0.9809** (vs 0.9787 Naive Persistence).
2. **Paired Error Dominance with Clustered Robustness:** Per-sequence paired comparison demonstrates that Exp 03 improves upon Naive Persistence on **411,655 sequences (55.37%)**, performs equally on **72,650 sequences (9.77%)**, and regresses on **259,095 sequences (34.85%)**. Clustered by `portwatch_id` to account for repeated within-port temporal autocorrelation ($N=2,065$ clusters), the mean paired absolute error difference is $+0.2393$ calls with a cluster-robust standard error of $0.00448$, yielding an overwhelming cluster-robust $t$-statistic of **$+53.43$** ($p < 10^{-15}$).
3. **Pervasive Per-Port Generalization:** Across all 2,065 ports, Exp 03 improves MAE on **1,950 ports (94.43%)**, remains identical on **71 ports (3.44%)**, and regresses marginally on only **44 ports (2.13%)**. In the highest volume quartile (Q4, 516 ports averaging 56.97 ships/week), **100.00% of ports improve** (mean $\Delta \text{MAE} = +0.4991$ calls).
4. **Milestone 3.3 Graph Data Reality:** Data engineering inspection reveals that the available graph datasets cover only **90 canonical ports out of the 2,065 global panel (4.36% coverage)**: 52 canonical ports in the Caribbean Regional Maritime network (528 nodes, 9,969 directed AIS edges) and 41 canonical ports in the North American Hinterland network (251,646 nodes, 303,016 rail/intermodal edges). There is **zero graph data for Europe, Asia, Africa, or South America**.
5. **Defensible Recommendation for Milestone 3.3:** Retain `ResidualLSTMForecaster` as the authoritative global baseline. Scope Milestone 3.3 GNN development strictly to **regional subgraphs** (Caribbean AIS maritime flows and North American intermodal connectors) using a modular residual formulation:
   $$\hat{y}(t) = \text{ReLU}\left( \hat{y}_{\text{naive}}(t) + \Delta_{\text{temporal}}(X) + \Delta_{\text{graph}}(X, \mathcal{G}) \right)$$
   where $\Delta_{\text{graph}} \equiv 0$ for ports outside the graph coverage, ensuring global applicability without artificial graph truncation or leakage.

---

## 2. Experiment Inventory

All four experimental runs and their primary artifacts remain frozen, untouched, and fully reproducible:

| Experiment Identifier | Model Class | Trainable Params | Dataset Schedule | Artifact Directory | Checkpoint Size | Validation Parquet |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **`exp00_initial_baseline`** | `LSTMForecaster` | 44,737 | 0.35 passes (10k batches x2) | `artifacts/lstm/experiments/exp00_initial_baseline/` | 663 KB | 8.87 MB |
| **`exp01_direct_adequate_training`** | `LSTMForecaster` | 44,737 | 1.00 pass (28,846 batches) | `artifacts/lstm/experiments/exp01_direct_adequate_training/`| 663 KB | 8.87 MB |
| **`exp02_port_embedding`** | `PortEmbeddedLSTMForecaster` | 88,433 | 1.00 pass (28,846 batches) | `artifacts/lstm/experiments/exp02_port_embedding/` | 1,024 KB | 9.01 MB |
| **`exp03_residual_persistence`** | `ResidualLSTMForecaster` | 44,673 | 1.00 pass (28,846 batches) | `artifacts/lstm/experiments/exp03_residual_persistence/` | 663 KB | 8.66 MB |

---

## 3. Recomputed Benchmark & Independent Verification

All metrics were recomputed from the persisted `validation_predictions.parquet` files:

### 3.1 Validation Parquet File Integrity Audit
- **Sequence Row Count:** Exactly **743,400 rows** in each Parquet file.
- **Sequence ID Uniqueness:** Exactly **743,400 unique sequence IDs** (0 duplicates) in each file.
- **Row-by-Row Sequence ID Alignment:** Exactly identical sequence IDs and order across all files ($100\%$ match).
- **Target Integrity:** Ground truth target column `y_true` is identical across all four files ($\max |y_{\text{true}}^{(A)} - y_{\text{true}}^{(B)}| = 0.000000$).
- **Missing / Infinite Values:** Exactly **0 nulls**, **0 NaNs**, and **0 infinite values** in any prediction or target column.

### 3.2 Independently Recomputed Metric Matrix
| Model / Experiment ID | Huber Loss ($\delta=1.0$) | MAE (calls) | RMSE (calls) | $R^2$ Score | Mean Bias | Pred Min | Pred Max | Pred Mean | Pred Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence** | 2.6558 | 3.0271 | 6.6780 | 0.9787 | -0.0141 | 0.0000 | 1,188.0000 | 17.4158 | 45.7476 |
| **Exp 00: Initial Baseline** | 7.6604 | 8.1428 | 30.4902 | 0.5566 | -2.9704 | 4.3921 | 164.8670 | 14.4594 | 18.4246 |
| **Exp 01: Direct Full Pass** | 6.2697 | 6.7470 | 24.4596 | 0.7147 | -2.3157 | 3.2569 | 277.3628 | 15.1142 | 24.2501 |
| **Exp 02: Port Embedding** | 4.5823 | 5.0486 | 12.1640 | 0.9294 | -1.2065 | 0.0000 | 747.2198 | 16.2234 | 35.7478 |
| **Exp 03: Residual LSTM** | **2.3974** | **2.7878** | **6.3331** | **0.9809** | **-0.0877** | **0.0000** | **1,182.4851**| **17.3421** | **45.7852** |

*Verification Discrepancy Note:* Recomputed metrics match reported metrics to 4 decimal places with zero numerical discrepancy.

---

## 4. Validation Comparability Audit

### 4.1 Persistence Formulation and Latency Alignment
Under the PortWatch operational availability contract, data has a publication latency of 2 days:
- For forecast reference date $t$, the latest usable observation is $t-2$.
- The input history consists of 30 observation days: $[t-31, \dots, t-2]$.
- Observation days $t-1$ and $t$ are strictly quarantined.
- The target forecast window is 7 days: $[t+1, \dots, t+7]$.
- **Persistence Formula:**
  $$\hat{y}_{\text{naive}}(t) = 7.0 \times \left( X[:, 29, 12] \cdot \sigma_{\text{roll7}} + \mu_{\text{roll7}} \right) = \sum_{k=0}^6 \text{portcalls}(t - 2 - k)$$
  Feature 12 is `pw_calls_rolling_7d_mean`. At timestep 29 ($t-2$), this represents the unweighted mean of daily port calls over $[t-8, \dots, t-2]$. Scaling by 7.0 projects this trailing rate into a 7-day cumulative total.
- **Leakage Verification:** No future targets, quarantined dates, or validation statistics are used. The scaler parameters ($\mu_{\text{roll7}} = 2.406818, \sigma_{\text{roll7}} = 6.348394$) were derived exclusively from the training split.

### 4.2 Residual Reconstruction Verification
In `ResidualLSTMForecaster` (Exp 03):
$$\hat{y}_{\text{exp03}} = \text{ReLU}\left( \hat{y}_{\text{naive}} + \Delta \right)$$
Across all 743,400 validation sequences, we verified that whenever $\hat{y}_{\text{exp03}} > 0$, the learned adjustment is exactly $\Delta = \hat{y}_{\text{exp03}} - \hat{y}_{\text{naive}}$. There are exactly 72,566 sequences where $\hat{y}_{\text{exp03}} = 0.0000$, and in $100\%$ of those cases, $\hat{y}_{\text{naive}}$ was already $0.0000$ (pure zeros). Active ReLU clipping of positive naive estimates occurred on **0 sequences (0.00%)**.

---

## 5. Paired Sequence-Level Error Analysis

To determine whether aggregate improvements reflect widespread individual sequence gains or outlier distortion, we compute the paired sequence-level difference against Naive Persistence:
$$\Delta \text{AE}_i = |y_i - \hat{y}_i^{\text{naive}}| - |y_i - \hat{y}_i^{\text{model}}| \quad (\text{positive} = \text{model is more accurate})$$
$$\Delta \text{SE}_i = (y_i - \hat{y}_i^{\text{naive}})^2 - (y_i - \hat{y}_i^{\text{model}})^2 \quad (\text{positive} = \text{model has smaller squared error})$$

### 5.1 Paired Sequence-Level Comparison Table
| Neural Model vs. Naive Persistence | Better Sequences ($N, \%$) | Worse Sequences ($N, \%$) | Equal Sequences ($N, \%$) | Mean $\Delta \text{AE}$ | Median $\Delta \text{AE}$ | Mean $\Delta \text{SE}$ | Clustered SE | Clustered $t$-stat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 00 (Initial 10k Baseline)** | 196,039 (26.37%) | 547,349 (73.63%) | 12 (0.00%) | -5.1157 | -2.2874 | -885.0556 | 0.57259 | -8.93 |
| **Exp 01 (Direct Full Pass)** | 224,242 (30.16%) | 519,123 (69.83%) | 35 (0.00%) | -3.7199 | -1.7199 | -553.6738 | 0.44477 | -8.36 |
| **Exp 02 (Port-Embedded)** | 207,721 (27.94%) | 519,464 (69.88%) | 16,215 (2.18%) | -2.0214 | -1.3855 | -103.3666 | 0.14946 | -13.53 |
| **Exp 03 (Residual LSTM)** | **411,655 (55.37%)**| **259,095 (34.85%)**| **72,650 (9.77%)** | **+0.2393** | **+0.1033** | **+4.4878** | **0.00448**| **+53.43** |

### 5.2 Clustered Statistical Methodology
Because observations within each port are serially correlated over time (each port appears 360 times in the validation year), naive IID standard errors would underestimate variance. We compute cluster-robust standard errors grouped at the port entity level ($G = 2,065$ clusters):
$$\text{SE}_{\text{cluster}} = \sqrt{ \frac{1}{G(G-1)} \sum_{g=1}^G \left( \overline{\Delta \text{AE}}_g - \overline{\Delta \text{AE}} \right)^2 }$$
Even under clustered standard errors, Exp 03 exhibits a $t$-statistic of **$+53.43$**, rejecting the null hypothesis of equivalence ($p < 10^{-15}$).

---

## 6. Target-Volume Bucket Comparison

We examine paired absolute error differences ($\Delta \text{AE}$) across the 7 maritime operating volume buckets:

| Volume Bucket | Sample Count ($N$) | Naive Persistence MAE | Exp 03 MAE | Paired Mean $\Delta \text{AE}$ | Percentage of Sequences Improved | Exp 03 RMSE | Naive Persistence RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Inactive Facilities)** | 136,976 | 0.4284 | **0.3915** | +0.0369 | 53.05% | **0.8515** | 0.9780 |
| **(0, 5] (Minor Terminals)** | 249,336 | 1.4063 | **1.2629** | +0.1434 | 55.43% | **1.7677** | 2.0139 |
| **(5, 20] (Feeder Ports)** | 204,435 | 3.0015 | **2.7172** | +0.2844 | 56.55% | **3.7626** | 4.1356 |
| **(20, 50] (Regional Hubs)** | 97,107 | 5.3283 | **4.8890** | +0.4392 | 56.62% | **6.4912** | 7.0429 |
| **(50, 100] (Major Gateways)**| 32,351 | 9.0563 | **8.5064** | +0.5500 | 54.77% | **11.6987** | 12.3521 |
| **(100, 250] (Global Hubs)** | 18,696 | 15.0458 | **14.2760** | +0.7698 | 53.57% | **20.3206** | 21.3173 |
| **250+ (Mega Gateways)** | 4,499 | 30.1676 | **29.2576** | +0.9100 | 52.88% | **46.6830** | 47.7673 |

*Key Empirical Takeaway:* Exp 03 improves upon Naive Persistence in **every single volume bucket without exception**. The absolute error reduction scales monotonically with port volume, growing from $+0.0369$ calls on minor facilities to $+0.9100$ calls on mega gateways handling $>250$ ships weekly.

---

## 7. Complete Per-Port Evaluation

PortWatch tracks 2,065 globally registered ports. Every port has exactly $N = 360$ validation sequences in 2024 (totaling $2,065 \times 360 = 743,400$):

### 7.1 Global Port Scorecard
- **Total Unique Ports Evaluated:** Exactly **2,065 ports**.
- **Observation Count per Port:** Strictly **360 daily sequences** per port.
- **Ports Improved ($\Delta \text{MAE} > 0$):** **1,950 ports (94.43%)**.
- **Ports Regressed ($\Delta \text{MAE} < 0$):** **44 ports (2.13%)**.
- **Ports Equal ($|\Delta \text{MAE}| \le 10^{-4}$):** **71 ports (3.44%)** (all dormant facilities with 0 traffic).

### 7.2 Port Stratification by Volume Quartile
| Volume Quartile | Port Count | Mean Target ($\bar{y}$) | Naive Persistence MAE | Exp 03 MAE | Mean $\Delta \text{MAE}$ | Percentage of Ports Improved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1 (Lowest Traffic)** | 517 | 0.46 calls | 0.4787 | 0.4456 | +0.0331 | **79.11%** (409 ports) |
| **Q2 (Low-Mid Traffic)** | 516 | 2.92 calls | 1.5067 | 1.3615 | +0.1452 | **99.03%** (511 ports) |
| **Q3 (High-Mid Traffic)**| 516 | 9.41 calls | 2.7942 | 2.5141 | +0.2800 | **99.61%** (514 ports) |
| **Q4 (Highest Traffic)** | 516 | 56.97 calls | 7.3338 | 6.8347 | +0.4991 | **100.00%** (516 ports) |

*Key Finding:* In the top quartile of global trade gateways, **100.00% of ports (516 out of 516) achieve lower MAE under Exp 03 than under Naive Persistence**.

### 7.3 Top 10 Largest Port Improvements
| Port Identifier | Observation Count | Mean Target | Naive MAE | Exp 03 MAE | $\Delta \text{MAE}$ | Naive RMSE | Exp 03 RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`port239`** | 360 | 256.39 | 29.31 | 27.94 | **+1.3669** | 40.49 | 38.85 |
| **`port1073`** | 360 | 116.13 | 21.26 | 19.96 | **+1.3019** | 27.54 | 25.84 |
| **`port548`** | 360 | 126.69 | 23.96 | 22.84 | **+1.1245** | 32.02 | 30.53 |
| **`port1253`** | 360 | 186.68 | 24.79 | 23.66 | **+1.1245** | 31.84 | 30.41 |
| **`port786`** | 360 | 306.38 | 28.32 | 27.20 | **+1.1234** | 41.15 | 39.59 |
| **`port1417`** | 360 | 280.01 | 25.70 | 24.61 | **+1.0943** | 37.06 | 35.42 |
| **`port1308`** | 360 | 109.12 | 17.93 | 16.85 | **+1.0779** | 22.56 | 21.17 |
| **`port1180`** | 360 | 76.88 | 15.01 | 13.95 | **+1.0630** | 18.71 | 17.46 |
| **`port752`** | 360 | 279.99 | 26.40 | 25.36 | **+1.0406** | 36.79 | 35.35 |
| **`port824`** | 360 | 395.49 | 43.29 | 42.26 | **+1.0276** | 59.78 | 58.42 |

### 7.4 Top 10 Largest Port Regressions
| Port Identifier | Observation Count | Mean Target | Naive MAE | Exp 03 MAE | $\Delta \text{MAE}$ | Naive RMSE | Exp 03 RMSE | Nature of Regression |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`port105`** | 360 | 11.51 | 12.24 | 12.43 | **-0.1854** | 38.73 | 38.59 | Highly irregular traffic (RMSE improved) |
| **`port2462`** | 360 | 0.97 | 0.19 | 0.24 | **-0.0559** | 0.44 | 0.40 | Rare discrete calls |
| **`fso96`** | 360 | 0.88 | 0.40 | 0.45 | **-0.0549** | 0.73 | 0.66 | Offshore storage |
| **`fso65`** | 360 | 0.95 | 0.27 | 0.32 | **-0.0523** | 0.54 | 0.48 | Offshore terminal |
| **`fso49`** | 360 | 1.55 | 0.46 | 0.51 | **-0.0518** | 0.71 | 0.65 | Mooring facility |
| **`port643`** | 360 | 0.74 | 0.32 | 0.37 | **-0.0517** | 0.73 | 0.71 | Minor feeder |
| **`fso70`** | 360 | 0.85 | 0.36 | 0.41 | **-0.0511** | 0.63 | 0.57 | Mooring facility |
| **`port2417`** | 360 | 1.34 | 0.41 | 0.45 | **-0.0445** | 0.68 | 0.62 | Remote mooring |
| **`port2048`** | 360 | 2.86 | 0.43 | 0.46 | **-0.0340** | 0.72 | 0.66 | Small terminal |
| **`port2420`** | 360 | 0.63 | 0.45 | 0.48 | **-0.0324** | 0.76 | 0.69 | Island wharf |

*Key Finding on Regressions:* The maximum port-level MAE regression across all 2,065 ports is tiny ($-0.185$ calls on `port105`), and for 9 of the top 10 regressions, Exp 03 actually achieves *lower RMSE* than Naive Persistence.

---

## 8. Residual Diagnostics (Exp 03)

We analyze the learned residual adjustment $\Delta = \hat{y}_{\text{pred}} - \hat{y}_{\text{naive}}$:

### 8.1 Residual Distribution Properties
- **Mean Learned Residual:** **$-0.0737$ calls**
- **Residual Standard Deviation:** **$0.7709$ calls**
- **Median Residual:** **$0.0000$ calls**
- **Residual Range:** $[-11.8120, +10.1446]$ calls
- **Positive Adjustments (Upward):** **311,121 sequences (41.85%)**
- **Negative Adjustments (Downward):** **359,629 sequences (48.38%)**
- **Zero Adjustments ($|\Delta| \le 10^{-4}$):** **72,650 sequences (9.77%)**

### 8.2 Correction Magnitude Percentiles
- **Mean Absolute Adjustment ($|\Delta|$):** **0.5031 calls**
- **Median Absolute Adjustment:** **0.2943 calls**
- **25th Percentile:** 0.1111 calls
- **75th Percentile:** 0.7121 calls
- **90th Percentile:** 1.2848 calls
- **95th Percentile:** 1.6417 calls
- **99th Percentile:** 2.4611 calls

### 8.3 Monthly Temporal Consistency
Evaluating paired error reduction across the 12 calendar months of 2024:
| Month | Validation Sequences | Mean $\Delta$ | Mean $\|\Delta\|$ | Naive Persistence MAE | Exp 03 MAE | Paired $\Delta \text{MAE}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jan 2024** | 64,015 | -0.0190 | 0.5336 | 3.2149 | 2.9451 | **+0.2698** |
| **Feb 2024** | 59,885 | -0.1037 | 0.5046 | 3.1483 | 2.9188 | **+0.2295** |
| **Mar 2024** | 64,015 | -0.1634 | 0.5128 | 2.9675 | 2.7398 | **+0.2277** |
| **Apr 2024** | 61,950 | -0.1033 | 0.5060 | 2.9964 | 2.7642 | **+0.2322** |
| **May 2024** | 64,015 | -0.0831 | 0.5056 | 2.8836 | 2.6458 | **+0.2378** |
| **Jun 2024** | 61,950 | -0.0858 | 0.4821 | 2.7146 | 2.4849 | **+0.2297** |
| **Jul 2024** | 64,015 | -0.0341 | 0.4823 | 2.9577 | 2.7259 | **+0.2318** |
| **Aug 2024** | 64,015 | +0.0192 | 0.5069 | 3.1508 | 2.9263 | **+0.2246** |
| **Sep 2024** | 61,950 | -0.0571 | 0.5148 | 3.2975 | 3.0392 | **+0.2583** |
| **Oct 2024** | 64,015 | -0.0937 | 0.4997 | 2.9751 | 2.7290 | **+0.2461** |
| **Nov 2024** | 61,950 | -0.0723 | 0.4877 | 2.9236 | 2.6855 | **+0.2381** |
| **Dec 2024** | 51,625 | -0.0945 | 0.4990 | 3.1142 | 2.8679 | **+0.2463** |

*Conclusion:* The model is not simply reproducing persistence; it actively adjusts forecasts bidirectionally (41.9% up, 48.4% down) with a typical magnitude of 0.3–1.3 calls, producing superior forecasts in every single calendar month.

---

## 9. Scientific Limitations and Corrections

1. **Clarification on Softplus Minimum:**
   The function $\text{Softplus}(z) = \ln(1 + e^z)$ has a theoretical infimum of 0.0 as $z \to -\infty$. The previously observed minimum of 4.3921 in Exp 00 was not a mathematical property of Softplus, but an empirical outcome of head bias initialization and convergence ($b = 5.2372$) coupled with bounded pre-activation inputs ($z_{\min} = 1.4682$).
2. **Semantics of Port Entity Embeddings:**
   While `PortEmbeddedLSTMForecaster` (Exp 02) expanded the prediction range to $[0.0, 747.22]$ and cut RMSE in half, this demonstrated *capacity conditioning* (learning port-specific operating levels), not that the 16-dimensional vectors semantically encode berth depth, crane density, or logistics infrastructure.
3. **Statistical Validity vs. Out-of-Sample Generalization:**
   The cluster-robust $t$-statistic of $+53.43$ establishes statistical superiority within the validation year (2024). It does not guarantee resilience to unprecedented out-of-distribution geopolitical shocks in future years (e.g., test year 2025).
4. **Justification for Graph Neural Networks:**
   The unaugmented LSTM's underperformance was an optimization and inductive bias issue resolved by residual learning. The scientific justification for GNNs is strictly functional: local port time series cannot observe vessel departures from upstream origins before they arrive in the local trailing window.

---

## 10. Milestone 3.3 Graph-Data Suitability Assessment

We audited the graph datasets in `DL/data/staging/supplementary_acquisition/processed/graph_ready/`:

| Dimension | Caribbean Regional Maritime Graph | North American Hinterland Network | Combined Coverage |
| :--- | :--- | :--- | :--- |
| **Node Count** | 528 nodes | 251,646 nodes | 252,174 nodes |
| **Edge Count** | 9,969 directed maritime lanes | 303,016 physical rail/intermodal edges | 312,985 edges |
| **Canonical Port Overlap** | **52 ports** (e.g. Kingston, Freeport, San Juan) | **41 ports** (US seaports e.g. LA, Long Beach, NY/NJ) | **90 ports** (4.36% of 2,065 panel) |
| **Geographic Scope** | Caribbean Basin & Gulf of Mexico | Continental USA, Canada, Mexico | Regional Americas only |
| **Global Coverage** | **0% Asia, 0% Europe, 0% Africa, 0% Middle East** | **0% outside North America** | **95.64% of global ports absent** |
| **Edge Semantics** | Verified AIS historical voyages (1.29M voyages) | Verified NTAD Class I rail lines + 50km geodesic connectors | Operational corridors |
| **Dynamic Flow Features**| Aggregated historical voyage counts | Static physical track distance & track count | No dynamic real-time flows |

### Critical Scientific Assessment:
A global GNN across all 2,065 ports is **unsupported by the data**. Attempting to construct a global ST-GNN adjacency matrix would leave 1,975 ports (95.64%) as isolated disconnected singletons.

---

## 11. Recommended Milestone 3.3 Scope & Architectural Formulation

### 11.1 Evaluation of Alternatives
- **Alternative A (Retain Exp 03 as global baseline):** Supported and mandatory. Exp 03 is the strongest global baseline ($R^2 = 0.9809$).
- **Alternative B (Global ST-GNN across all ports):** Rejected. 95.6% of ports lack graph edges.
- **Alternative C (Graph information as incremental correction):** Highly viable.
- **Alternative D (Restrict GNN to regions with verified coverage):** Supported and methodologically rigorous.

### 11.2 Defensible Formulation for Milestone 3.3
Milestone 3.3 should implement a **Modular Residual ST-GNN Architecture**:
$$\hat{y}_i(t) = \text{ReLU}\left( \hat{y}_{\text{naive}, i}(t) + \Delta_{\text{temporal}, i}(X) + \mathbb{I}_{i \in \mathcal{V}_{\text{graph}}} \cdot \Delta_{\text{graph}, i}(X, \mathcal{G}) \right)$$
1. **Global Fallback:** For the 1,975 ports without graph coverage, $\Delta_{\text{graph}} \equiv 0$, preserving the proven performance of `ResidualLSTMForecaster`.
2. **Regional Benchmark Subgraphs:** Evaluate GNN message passing on two clearly defined benchmarks:
   - **Benchmark 1: Caribbean Maritime Subgraph ($N=52$ ports):** Test whether GATv2 message passing over AIS voyage corridors predicts arrival surges from upstream transshipment hubs (e.g., Kingston $\to$ San Juan).
   - **Benchmark 2: North American Port-Rail Hinterland ($N=41$ ports):** Test whether message passing over intermodal rail connectors predicts port dwell clearing into inland freight hubs (e.g., LA/Long Beach $\to$ Chicago).

---

## 12. Explicit Unresolved Risks for Milestone 3.3

1. **Static vs Dynamic Graph Disparity:**
   The Caribbean edges represent aggregated historical voyage totals (2019–2023), not dynamic time-varying vessel positions. Downstream ST-GNNs must handle static graph adjacency with time-varying node features.
2. **Scale Disparity in Multimodal Networks:**
   The North American graph has 251,646 rail nodes but only 41 ports. Running whole-graph convolutions is computationally wasteful. Subgraph sampling (e.g., $k$-hop ego networks around the 41 ports) will be required.
3. **Target Leakage Safeguard:**
   Graph message passing must never pass features from future dates ($t-1, t, t+1 \dots$) across edges. Message passing must strictly operate on historical node states at $\le t-2$.


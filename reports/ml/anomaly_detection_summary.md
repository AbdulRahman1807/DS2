# Stage 01 Statistical Anomaly Detection Benchmark Summary (Audited Pass)

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning Engineer
**Stage:** Stage 01 Baseline Benchmark (Corrective Audit Pass)
**Methodology Note:** All anomaly preprocessing is strictly unsupervised (`OneHotEncoder` + `StandardScaler`), fitted on $X_{\text{train}}$ without exposure to target labels $y$.

---

## 1. Executive Summary & Core Principle

The Stage 01 specification requires baseline unsupervised anomaly detectors (**Isolation Forest** and **One-Class SVM**) to evaluate anomalous logistics behavior across dwell-time, throughput, and capacity dimensions.

### Governing Principle: Statistical Outlier $\neq$ Operational Bottleneck
An unsupervised anomaly detector isolates containers with **rare multivariate attribute combinations** (e.g., highly unusual gross weights, rare refrigerated/tank cargo, or uncommon yard slotting coordinates). **A statistical feature outlier is NOT automatically a critical bottleneck.**

In empirical testing on held-out out-of-time validation data (`val.parquet`, $N=20,879$):
- **Isolation Forest (5% contamination)** flagged **822 containers (3.94%)** as anomalous.
- Among these flagged anomalies, the retrospective `CRITICAL_DELAY` rate was **3.65%** — substantially **LOWER** than the general validation critical-delay prevalence of **7.37%**.
- Over **80%** of containers flagged as feature outliers were actually **`ON_TIME`** (80.29%).
- This occurs because containers with unusual physical characteristics (e.g., lightweight empty dry containers or non-standard carriers) frequently clear customs rapidly, whereas true systemic bottlenecks result from broad yard saturation affecting ordinary standard containers.

Therefore, **unsupervised anomaly detection must be treated as an auxiliary diagnostic for operational surveillance, NOT as a substitute for the 3-class supervised triage classifier.**

---

## 2. Preprocessing Architecture: Strictly Unsupervised

In response to the methodology audit, the anomaly preprocessing pipeline was completely decoupled from supervised target encoding:

1. **Numeric Features (11 columns):** Imputed with training medians; standardized using `StandardScaler`.
2. **Categorical Features (4 columns):** Missing values imputed with `'UNKNOWN'`; encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=20)`.
3. **No Target Exposure:** Preprocessor `fit()` takes only feature matrix $X_{\text{train}}$. Target vector $y_{\text{train}}$ is neither accepted nor required.
4. **Dimension:** Expands active arrival features into a 340-dimensional sparse feature representation suitable for random tree partition scoring.

---

## 3. Supported vs. Unsupported Signal Analysis

In strict accordance with repo safety and empirical integrity gates, we explicitly distinguish which signals are supported by current dataset artifacts:

| Signal Dimension | Supported by Current Data? | Methodology & Observability Horizon | Operational Meaning & Limitations |
| :--- | :--- | :--- | :--- |
| **Container-Level Feature Outliers** | **YES (Fully Supported)** | Evaluated via Isolation Forest on 15 approved arrival predictors at $t_{\text{discharge}}$. | Identifies extreme gross weights, rare container types (Tank/Reefer), unusual carrier free-time contracts, and outlier yard positions. |
| **Lagged Terminal Throughput Dynamics** | **YES (Fully Supported)** | Trailing 7-day completed gate-out volume (`trailing_7d_volume`) and delay rates. | Detects systemic throughput drops and clearance stalls across the terminal on a daily rolling basis. |
| **Temporal Congestion Regimes** | **YES (Fully Supported)** | Weekly aggregated delay rates (`historical_delay_rate_7d`) and trailing critical stay counts. | Accurately isolates the major July 2021 surge episode where daily terminal delay rates exceeded 35%. |
| **Continuous TEU Yard Capacity Utilization** | **NO (Unsupported)** | Requires dynamic terminal operating system (TOS) yard survey snapshots and TEU stacking capacities. | **UNSUPPORTED:** Cannot fabricate high-frequency yard occupancy percentages from single-milestone completed truck-out logs. |
| **Berth Occupancy & Crane Telemetry** | **NO (Unsupported)** | Requires vessel berthing schedules, crane moves/hour, and gang allocation logs. | **UNSUPPORTED:** Mendeley dataset contains case-level container milestones, not berth telemetry. |
| **Hourly Gate-In/Out Transaction Spikes** | **NO (Unsupported)** | Requires continuous OCR gate optical sensor feeds. | **UNSUPPORTED:** Gate events are observed only upon completed truck-out transaction. |

---

## 4. Model Benchmark Results

### 4.1 Primary Candidate: Isolation Forest (Purely Unsupervised)

- **Algorithm:** Isolation Forest (`n_estimators=100`, purely unsupervised `OneHotEncoder` + `StandardScaler`).
- **Feature Matrix:** 15 active arrival predictors (all post-arrival inspections and target columns strictly excluded).
- **Contamination Sensitivity Analysis:**

| Contamination Setting | Train Flagged ($N=60,698$) | Val Flagged ($N=20,879$) | Val Critical Delay Overlap | Val AT_RISK Overlap | Val ON_TIME Overlap | Val Baseline Crit Prev |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1% Contamination** | 606 (1.0%) | 117 (0.56%) | **2.56%** | 7.69% | 89.74% | 7.37% |
| **5% Contamination (Baseline)** | 3034 (5.0%) | 822 (3.94%) | **3.65%** | 16.06% | 80.29% | 7.37% |
| **10% Contamination** | 6070 (10.0%) | 1769 (8.47%) | **4.97%** | 17.86% | 77.16% | 7.37% |

#### Score Distribution (Decision Function on Validation)
The continuous anomaly score represents the average path length in isolation trees (offset-adjusted; negative scores indicate anomalies):
- **Minimum Score:** `-0.022` (most extreme outlier)
- **25th Percentile:** `0.0114`
- **Median (50th):** `0.0175`
- **75th Percentile:** `0.0229`
- **Maximum Score:** `0.037` (most normal container)

---

### 4.2 Comparative Reference: One-Class SVM (Label-Selected Normal-Reference Experiment)

- **Methodological Design:** Label-selected normal-reference experiment (semi-supervised, NOT pure unsupervised).
- **Cohort:** Clean `ON_TIME` training cases ($N=5,000$ sample to avoid cubic computational complexity).
- **Hyperparameters:** RBF kernel, `gamma='scale'`, `nu=0.05`.
- **Validation Flagged Count:** **15110 out of 20,879 stays (72.37%)**.
- **Critical Delay Overlap:** 7.09% (comparable to general population prevalence of 7.37%).

> [!CAUTION]
> **One-Class SVM Finding:** One-Class SVM with an RBF kernel completely breaks down under out-of-time temporal drift. Because the training period (June/July) exhibited severe COVID congestion while validation (late August/September) reflects a clearance regime with shifting carrier mixes, One-Class SVM classifies over 70% of normal incoming containers as anomalous. **One-Class SVM is ruled out for production serving.** Isolation Forest remains the only defensible statistical anomaly baseline.

---

## 5. Temporal Congestion Regimes (Surge Detection)

Aggregating the historical 7-day rolling operational features reveals a marked temporal concentration of systemic congestion:

- **Mean Historical Weekly Delay Rate:** `34.47%`
- **Surge Anomaly Threshold ($+1.5\sigma$):** `70.37%`

### Chronological Congestion Regimes:
1. **June 10 – July 04 (Cold-Start & Pre-Surge Phase):**
   - Zero completed departure history prior to July 01.
   - Low observed departure volumes; historical delay rate initialized at 0.0.
2. **July 05 – July 25 (Peak Congestion Surge):**
   - Severe operational blockage at the terminal.
   - Historical delay rates peaked above **35%**, and mean dwell times climbed over **180 hours**.
   - Accounts for the vast majority of all `CRITICAL_DELAY` cases in the training corpus.
3. **July 26 – August 26 (Recovery & Clearance Phase):**
   - Delay rates gradually decreased from 35% down to 18%.
   - Terminal throughput normalized as container backlogs cleared.
4. **August 27 – September 11 (Validation Partition):**
   - Stable operational rhythm with on-time rate reaching 66.5%.
   - Demonstrates why static thresholding without temporal re-calibration experiences drift.

---

## 6. False-Alarm Limitations & Operational Warnings

1. **No Established Operational Cost Matrix:**
   - In commercial logistics, raising an anomaly alert triggers physical yard inspections, customs re-checks, or off-dock drayage rerouting.
   - Without an agreed corporate cost ratio ($C_{\text{False Alarm}} / C_{\text{Missed Delay}}$), setting an arbitrary anomaly alert threshold will flood yard dispatchers with false alarms (over 80% of Isolation Forest flagged cases were actually on-time).
2. **Lack of Verified Disruption Diversity:**
   - The PortWatch external disaster feed is constant 0 throughout mid-2021 (no typhoons or recorded tsunamis).
   - The single July congestion spike reflects local Indonesian pandemic disruptions.
   - The anomaly detector cannot be claimed as validated for overseas port closures, typhoons, or labor strikes.

# Stage 01 Statistical Anomaly Detection Benchmark Summary

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning Engineer
**Stage:** Stage 01 Baseline Benchmark
**Evaluation Scope:** Unsupervised statistical outlier detection on 15 approved arrival features

---

## 1. Executive Summary & Core Principle

The Stage 01 specification explicitly requires baseline unsupervised anomaly detectors (**Isolation Forest** and **One-Class SVM**) to identify anomalous logistics behavior across dwell-time, throughput, and capacity dimensions.

### Governing Principle: Statistical Outlier $\neq$ Operational Bottleneck
An unsupervised anomaly detector isolates containers with **rare multivariate attribute combinations** (e.g., highly unusual gross weights, rare refrigerated/tank cargo, or uncommon yard slotting coordinates). **A statistical feature outlier is NOT automatically a critical bottleneck.**

In empirical testing on held-out out-of-time validation data (`val.parquet`, $N=20,879$):
- **Isolation Forest (5% contamination)** flagged **940 containers (4.50%)** as anomalous.
- Among these flagged anomalies, the retrospective `CRITICAL_DELAY` rate was **0.96%** — substantially **LOWER** than the general validation critical-delay prevalence of **7.37%**.
- This occurs because containers with unusual physical characteristics (e.g., lightweight empty dry containers or non-standard carriers) frequently clear customs rapidly, whereas true systemic bottlenecks result from broad yard saturation affecting ordinary standard containers.

Therefore, **unsupervised anomaly detection must be treated as an auxiliary diagnostic for operational surveillance, NOT as a primary substitute for the 3-class supervised triage classifier.**

---

## 2. Supported vs. Unsupported Signal Analysis

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

## 3. Model Benchmark Results

### 3.1 Primary Candidate: Isolation Forest

- **Algorithm:** Isolation Forest (`n_estimators=100`, out-of-fold `TargetEncoder` + `StandardScaler`).
- **Feature Matrix:** 15 active arrival predictors (all post-arrival inspections and target columns strictly excluded).
- **Contamination Sensitivity Analysis:**

| Contamination Setting | Train Flagged ($N=60,698$) | Val Flagged ($N=20,879$) | Val Critical Delay Overlap | Val AT_RISK Overlap | Val ON_TIME Overlap | Val Baseline Crit Prev |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1% Contamination** | 607 (1.00%) | 185 (0.89%) | **0.54%** (1 stay) | 18.92% (35 stays) | 80.54% (149 stays) | 7.37% |
| **5% Contamination (Baseline)** | 3,035 (5.00%) | 940 (4.50%) | **0.96%** (9 stays) | 19.36% (182 stays) | 79.68% (749 stays) | 7.37% |
| **10% Contamination** | 6,070 (10.00%) | 1,883 (9.02%) | **1.70%** (32 stays) | 21.08% (397 stays) | 77.22% (1,454 stays) | 7.37% |

#### Score Distribution (Decision Function on Validation)
The continuous anomaly score represents the average path length in isolation trees (offset-adjusted; negative scores indicate anomalies):
- **Minimum Score:** `-0.0735` (most extreme outlier)
- **25th Percentile:** `0.1146`
- **Median (50th):** `0.141`
- **75th Percentile:** `0.1563`
- **Maximum Score:** `0.1844` (most normal container)

---

### 3.2 Comparative Reference: One-Class SVM

- **Cohort:** Clean `ON_TIME` training cases ($N=5,000$ sample to avoid cubic computational complexity).
- **Hyperparameters:** RBF kernel, `gamma='scale'`, `nu=0.05`.
- **Validation Flagged Count:** **15,068 out of 20,879 stays (72.17%)**.
- **Critical Delay Overlap:** 7.13% (essentially identical to general population prevalence of 7.37%).

> [!CAUTION]
> **One-Class SVM Finding:** One-Class SVM with an RBF kernel completely breaks down under out-of-time temporal drift. Because the training period (June/July) exhibited severe COVID congestion while validation (late August/September) reflects a clearance regime with shifting carrier mixes, One-Class SVM classifies nearly three-quarters of normal incoming containers as anomalous. **One-Class SVM is ruled out for production serving.** Isolation Forest remains the only defensible statistical anomaly baseline.

---

## 4. Temporal Congestion Regimes (Surge Detection)

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

## 5. False-Alarm Limitations & Operational Warnings

1. **No Established Operational Cost Matrix:**
   - In commercial logistics, raising an anomaly alert triggers physical yard inspections, customs re-checks, or off-dock drayage rerouting.
   - Without an agreed corporate cost ratio ($C_{\text{False Alarm}} / C_{\text{Missed Delay}}$), setting an arbitrary anomaly alert threshold will flood yard dispatchers with false alarms (80% of Isolation Forest flagged cases were actually on-time).
2. **Lack of Verified Disruption Diversity:**
   - The PortWatch external disaster feed is constant 0 throughout mid-2021 (no typhoons or recorded tsunamis).
   - The single July congestion spike reflects local Indonesian pandemic disruptions.
   - The anomaly detector cannot be claimed as validated for overseas port closures, typhoons, or labor strikes.

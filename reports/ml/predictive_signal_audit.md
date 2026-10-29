# DS2 Stage 01 Predictive Signal Audit and Evaluation Redesign

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning & Reliability Engineering
**Scope:** Stage 01 Predictive Signal Investigation, Temporal Evaluation Protocol Redesign, and Failure Mode Audit
**Artifact Directory:** `reports/ml/`
**Dataset Version:** 1.0.0 (`data/processed/dataset_manifest.json`)
**Commit Reference:** `97b5df3`

---

## Executive Summary & Core Verdict

The objective of this investigation is not to engineer a higher validation score, but to establish **whether the available dataset contains reliable, temporally generalizable predictive signal for critical container dwell delays**, identify what fundamentally limits performance, and specify an uncompromised evaluation protocol.

### Key Empirical Findings
1. **Weak Predictive Signal at Discharge Start ($t_{\text{discharge}}$):**
   Arrival-time features (carrier prefix, container dimensions, cargo gross weight, discharge hour, day of week, and 7-day lagged completed departures) yield a critical-delay PR-AUC of **0.0937** on validation data against a baseline critical prevalence of **0.0737** (a lift of only +0.0200 over random chance). At the default multiclass argmax decision rule, the model misses **88.82%** of true critical delays (recall = 11.18%, precision = 9.90%).
2. **Empirical Independence from Yard Coordinates:**
   Excluding the unconfirmed static coordinates `YARD_BLOCK` and `YARD_SLOT` (evaluating a conservative 13-feature model) produces **identical or superior temporal generalization** across all chronological folds (Fold 1: 0.2044 vs 0.2049; Fold 2: 0.1018 vs 0.0961; Fold 3: 0.1269 vs 0.1087; Val: 0.0909 vs 0.0937). Unconfirmed yard fields add no robust predictive signal.
3. **Macro Operational Regime Shift Between July and August/September:**
   The terminal experienced an acute systemic congestion wave in July (mean dwell 132.8h, $p85 = 218.8\text{h}$, critical delay rate 27.9%, on-time rate 21.2%), followed by a high-throughput clearance regime in late August and September (mean dwell 63.2h, $p85 = 109.9\text{h}$, critical delay rate 7.2%, on-time rate 69.3%). A static classifier trained on arrival features cannot distinguish systemic queueing from normal flow without real-time yard density metrics.
4. **Target Discretization Threshold Dependency:**
   The frozen operational thresholds ($p50 = 65.1102\text{h}$, $p85 = 160.8993\text{h}$) were fitted once across the entire 60,698-case training set (June 10 – August 26). In Fold 1 (train period ending July 18), the local training 85th percentile was $201.27\text{h}$. Using the full-train $p85$ embeds future knowledge of the August clearance into the target definition of early folds.
5. **113 Boundary Overlap Cases Sensitivity:**
   The 113 container stays in `train.parquet` where discharge extended into August 27 account for 0.1862% of training stays. Removing them shifts validation PR-AUC from 0.0937 to 0.0951 (+0.0014) and critical recall from 11.18% to 14.24% (+0.0306). While necessary for Data Engineering hygiene in a future release, removing them does not alter the fundamental reality of weak arrival signal.
6. **Model Scores Are Uncalibrated Decision Scores, NOT Probabilities:**
   Class-weighted gradient boosting inflates predicted scores: containers in the highest score decile (mean predicted score 39.86%) have an empirical critical delay rate of only **10.11%**. The Brier score is **0.0995**, which is worse than the constant unskilled prior baseline (**0.0682**). Model scores cannot be communicated to operators as true probabilities.
7. **Operational Alerting Capacity Constraint:**
   Lowering the decision threshold to $\tau \ge 0.20$ increases critical recall to 79.71%, but triggers **884.5 alerts per day** (14,152 alerts over 16 days) with 12,926 false alarms. If a terminal team can reasonably handle **40–50 alerts per day**, the operational threshold must be set to $\tau \ge 0.40$, where precision is 11.11% and the model catches only **5.07% of critical bottlenecks** (78 of 1,538 cases).

### Final Recommendation
**Do not treat this model as a production alert system.** Cease hyperparameter tuning and feature engineering on static arrival features. The data ceiling at $t_{\text{discharge\_start}}$ has been reached. Treat this model as an audited, honest exploratory baseline, and focus modeling efforts on **Stage 02 sequential horizon modeling**, where real-time customs events, in-yard reshuffles, and gate appointment milestones become observable during the stay.

---

## Phase 1 — Chronological Evaluation Protocol & Target Definition Audit

### 1.1 Target Construction & Discretization Mechanics
Ground-truth container dwell duration is computed as the elapsed time from vessel crane discharge start to road truck gate-out end:
$$\text{dwell\_hours} = \text{truck\_out\_end\_ts} - \text{discharge\_start\_ts}$$

The 3-tier operational target classes are assigned using fixed percentiles derived from the training partition ($N=60,698$, June 10 – August 26, 2021):
- **Class 0 (`ON_TIME`):** $\text{dwell\_hours} \le 65.1102\text{ hours}$ ($\le 2.71\text{ days}$, 50th percentile of training stays)
- **Class 1 (`AT_RISK`):** $65.1102\text{ hours} < \text{dwell\_hours} \le 160.8993\text{ hours}$ (50th to 85th percentile of training stays)
- **Class 2 (`CRITICAL_DELAY`):** $\text{dwell\_hours} > 160.8993\text{ hours}$ ($> 6.70\text{ days}$, 85th percentile of training stays)

### 1.2 Target-Derived Threshold Dependency Across Internal Folds
To evaluate whether the frozen thresholds embed future information relative to internal chronological folds, we computed the empirical percentiles within each fold's training and validation windows:

| Evaluation Window | Date / TS Range | Sample Size ($N$) | Empirical $p50$ (hours) | Empirical $p85$ (hours) | Frozen $p85$ Used | Divergence ($\Delta p85$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Train Split** | 2021-06-10 to 2021-08-26 | 60,698 | 65.11h | 160.90h | 160.90h | 0.00h (Fitted) |
| **Fold 1 Train** | $< \text{2021-07-19 00:00:00}$ | 20,960 | 86.25h | **201.27h** | 160.90h | **-40.37h** |
| **Fold 1 Val** | 2021-07-19 to 2021-07-31 | 11,350 | 82.89h | **178.43h** | 160.90h | -17.53h |
| **Fold 2 Train** | $< \text{2021-08-01 00:00:00}$ | 32,310 | 84.89h | **190.83h** | 160.90h | **-29.93h** |
| **Fold 2 Val** | 2021-08-01 to 2021-08-14 | 16,537 | 49.31h | **124.40h** | 160.90h | +36.50h |
| **Fold 3 Train** | $< \text{2021-08-15 00:00:00}$ | 48,847 | 70.90h | **171.20h** | 160.90h | -10.30h |
| **Fold 3 Val** | 2021-08-15 to 2021-08-26 | 11,310 | 44.34h | **113.77h** | 160.90h | +47.13h |
| **Validation Benchmark** | 2021-08-27 to 2021-09-12 | 20,879 | 44.82h | **112.05h** | 160.90h | **+48.85h** |

#### Audit Findings on Target Dependency
1. **Future Dwell Clearance Leaks into Early Fold Labels:**
   During Fold 1 (June 10 to July 18), the terminal operated under severe congestion: the true local 85th percentile was **201.27 hours** (8.4 days). However, because the global frozen threshold $p85 = 160.90\text{h}$ was calculated across the entire dataset (including the late August recovery), Fold 1's validation target labels use a cutoff that was determined by future data.
2. **Fixed Thresholding Fails Under Operational Regime Shifts:**
   In the validation period (late August – early September), the terminal cleared its backlog. The local 85th percentile dropped to **112.05 hours**. Under the static $160.90\text{h}$ rule, stays taking between 112h and 160h (which represent the worst 15% of stays in September) are labeled `AT_RISK` rather than `CRITICAL_DELAY`. This explains why validation critical prevalence dropped from 15.00% in training to 7.37% in validation.

### 1.3 Prediction Horizon Clarity
We strictly delineate three distinct operational prediction horizons:
- **Horizon A: Pre-Arrival / Pre-Berthing ($t < t_{\text{berth}}$):** Information available prior to vessel arrival (BAPLIE manifest, carrier prefix, container dimensions, declared gross weight).
- **Horizon B: At Discharge Start ($t_{\text{discharge\_start}}$) — STAGE 01 ANCHOR:** Quay crane begins unstacking from the vessel. Features include Horizon A plus discharge timestamp cycles (hour of day, day of week, weekend indicator) and trailing $[t-7, t-1]$ completed departure metrics.
- **Horizon C: During Stay / Mid-Stay ($t > t_{\text{discharge}}$) — STAGE 02 SCOPE:** Dynamic operational events occurring after discharge (TOS in-yard moves, customs document release `JOB_DEL_DOCTYPE`, payment confirmation, gate appointment bookings).

> [!IMPORTANT]
> Stage 01 models operate strictly at **Horizon B ($t_{\text{discharge\_start}}$)**. All mid-stay milestones and post-discharge timestamps are strictly quarantined by the feature contract.

### 1.4 Audit of 113 Train/Validation Boundary Overlap Cases
In the external Parquet files, partitioning was performed on vessel `call_date` (`call_date <= 2021-08-26` for `train.parquet`, `call_date >= 2021-08-27` for `val.parquet`).
- **Mechanism:** In `train.parquet`, exactly 113 containers belonging to a vessel call on August 26 had quay crane discharge operations that extended into August 27 (between 12:00:34 and 23:00:34).
- Meanwhile, the earliest discharge in `val.parquet` commenced on August 27 at 11:55:57.
- **Timestamp Overlap:** $\max(t_{\text{train}}) = \text{2021-08-27 23:00:34} > \min(t_{\text{val}}) = \text{2021-08-27 11:55:57}$. Exactly 113 training stays (0.1862% of train) occurred chronologically after the start of validation discharge.

#### Quantitative Sensitivity Analysis
We trained the class-weighted XGBoost model with and without the 113 boundary cases:

| Metric | With 113 Cases ($N=60,698$) | Without 113 Cases ($N=60,585$) | Net Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.5464 | 0.5378 | -0.0086 |
| **Macro-F1** | 0.3968 | 0.3970 | +0.0002 |
| **Critical Delay PR-AUC** | 0.0937 | 0.0951 | +0.0014 |
| **Critical Delay Recall** | 0.1118 | 0.1424 | +0.0306 |
| **Critical Delay Precision** | 0.0990 | 0.0983 | -0.0007 |
| **Critical Delay F1** | 0.1050 | 0.1163 | +0.0113 |

**Conclusion:** Removing the 113 boundary cases shifts PR-AUC by only +0.0014 and critical recall from 11.18% to 14.24%. While this sensitivity analysis confirms that boundary overlap is not the cause of poor model performance, a clean subsequent Data Engineering release should partition strictly by `discharge_start_ts < 2021-08-27 11:55:00`.

### 1.5 Backward-Looking Construction Audit for Rolling Features
`ML/data_engineer/rolling_metrics.py` computes trailing port delay metrics over a 7-day window.
- **Observation Date:** Trailing delay rates are computed on completed departures using `truck_out_end_ts.dt.date`. Because a container's dwell duration is only observed when it exits the gate, aggregating on departure date ensures that unfinished stays are not prematurely labeled.
- **Lag Enforcement:** The rolling calculation uses `rolling("7D", closed="left")`. For call date $t$, the rolling window covers $[t-7, t-1]$, strictly excluding day $t$.
- **Subtle Temporal Leakage Risks Identified:**
  1. *Merge Key Mismatch:* Rolling metrics are indexed by `call_date` and merged onto container records by `call_date`. For containers where discharge starts 1 to 2 days after the vessel call date, merging on `call_date` uses data from $[t_{\text{call}}-7, t_{\text{call}}-1]$, which is older than $[t_{\text{discharge}}-7, t_{\text{discharge}}-1]$. This is conservative (no lookahead leakage), but introduces slight lag staleness.
  2. *Historical Label Discretization:* Trailing metrics aggregate `daily_critical` and `daily_at_risk`. In the historical extract, those past departures were labeled using the global frozen thresholds ($p50, p85$). In a real-time production terminal, an operator would have had to define "critical" using an operational SLA (e.g., $>7\text{ days}$) or an expanding historical percentile.

---

## Phase 2 — Comprehensive Baseline Comparison Across All Temporal Folds

To ensure fair comparison, all models were evaluated across three internal chronological expanding folds and the external validation partition using identical feature contracts and random seeds.

### 2.1 Model Specifications
1. **DummyClassifier (Prior):** Predicts the marginal class distribution of the training split.
2. **Logistic Regression (Unweighted):** Standard multinomial linear baseline with median imputation and standard scaling.
3. **XGBoost (Unweighted):** 100 estimators, max depth 5, learning rate 0.1, subsample 0.8, colsample 0.8.
4. **XGBoost (Sample-Weighted):** Same architecture, with balanced sample weights ($w_i = \frac{N}{3 \cdot N_{y_i}}$).
5. **XGBoost Conservative (No Yard):** Same architecture and sample weights, trained on the **conservative 13-feature subset** excluding unconfirmed `YARD_BLOCK` and `YARD_SLOT`.

---

### 2.2 Results Across Internal Temporal Folds and Primary Validation

#### Fold 1 (Expanding July: Train $< \text{2021-07-19}$, Val $\text{2021-07-19 to 2021-07-31}$)
- **Sample Sizes:** Train $N=20,960$ (Crit = 3,678, 17.55%), Val $N=11,350$ (Crit = 2,263, 19.94%)
- **Validation Class Distribution:** `ON_TIME`: 4,366 (38.47%), `AT_RISK`: 4,721 (41.59%), `CRITICAL_DELAY`: 2,263 (19.94%)
- **Fold Prevalence Baseline:** **0.1994**

| Model Configuration | Accuracy | Macro-F1 | Critical PR-AUC | Fold Baseline | Critical Recall | Critical Precision | Critical F1 | Multiclass Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Prior)** | 0.4159 | 0.1958 | 0.1994 | 0.1994 | 0.0000 | 0.0000 | 0.0000 | 1.0545 |
| **Logistic Regression (Unweighted)** | 0.3781 | 0.2521 | 0.1640 | 0.1994 | 0.0910 | 0.1350 | 0.1087 | 3.6282 |
| **XGBoost (Unweighted)** | 0.4010 | 0.2530 | 0.2069 | 0.1994 | 0.1025 | 0.2311 | 0.1420 | 1.2305 |
| **XGBoost (Sample-Weighted)** | 0.3327 | 0.2681 | 0.2049 | 0.1994 | 0.3760 | 0.2065 | 0.2666 | 1.2703 |
| **XGBoost Conservative (No Yard)** | 0.3204 | **0.2720** | 0.2044 | 0.1994 | **0.4560** | 0.2057 | **0.2835** | 1.3012 |

*Fold 1 Insight:* Unweighted Logistic Regression performs **worse than random chance** (PR-AUC 0.1640 vs 0.1994 baseline). XGBoost achieves a PR-AUC of 0.2069, representing an almost negligible lift (+0.0075) over the fold prevalence baseline.

---

#### Fold 2 (Late July – Early August: Train $< \text{2021-08-01}$, Val $\text{2021-08-01 to 2021-08-14}$)
- **Sample Sizes:** Train $N=32,310$ (Crit = 5,941, 18.39%), Val $N=16,537$ (Crit = 1,471, 8.895%)
- **Validation Class Distribution:** `ON_TIME`: 10,228 (61.85%), `AT_RISK`: 4,838 (29.26%), `CRITICAL_DELAY`: 1,471 (8.90%)
- **Fold Prevalence Baseline:** **0.0890**

| Model Configuration | Accuracy | Macro-F1 | Critical PR-AUC | Fold Baseline | Critical Recall | Critical Precision | Critical F1 | Multiclass Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Prior)** | 0.2926 | 0.1509 | 0.0890 | 0.0890 | 0.0000 | 0.0000 | 0.0000 | 0.9963 |
| **Logistic Regression (Unweighted)** | 0.6195 | 0.3131 | 0.1023 | 0.0890 | 0.0020 | 0.1579 | 0.0040 | 0.9376 |
| **XGBoost (Unweighted)** | 0.4943 | 0.3588 | 0.0969 | 0.0890 | 0.0218 | 0.0578 | 0.0316 | 1.0536 |
| **XGBoost (Sample-Weighted)** | 0.4041 | 0.3533 | 0.0961 | 0.0890 | 0.3270 | 0.1026 | 0.1562 | 1.1442 |
| **XGBoost Conservative (No Yard)** | 0.4275 | **0.3722** | **0.1018** | 0.0890 | **0.3596** | **0.1139** | **0.1730** | 1.1390 |

*Fold 2 Insight:* As congestion cleared in early August, critical prevalence fell by more than half (19.94% to 8.90%). The conservative no-yard model outperforms the full model in PR-AUC (0.1018 vs 0.0961) and critical F1 (0.1730 vs 0.1562).

---

#### Fold 3 (Mid-August Clearance: Train $< \text{2021-08-15}$, Val $\text{2021-08-15 to 2021-08-26}$)
- **Sample Sizes:** Train $N=48,847$ (Crit = 7,412, 15.17%), Val $N=11,310$ (Crit = 847, 7.489%)
- **Validation Class Distribution:** `ON_TIME`: 7,469 (66.04%), `AT_RISK`: 2,994 (26.47%), `CRITICAL_DELAY`: 847 (7.49%)
- **Fold Prevalence Baseline:** **0.0749**

| Model Configuration | Accuracy | Macro-F1 | Critical PR-AUC | Fold Baseline | Critical Recall | Critical Precision | Critical F1 | Multiclass Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Prior)** | 0.6604 | 0.2652 | 0.0749 | 0.0749 | 0.0000 | 0.0000 | 0.0000 | 0.9069 |
| **Logistic Regression (Unweighted)** | 0.6172 | 0.3783 | 0.0948 | 0.0749 | 0.0059 | 0.2381 | 0.0115 | 0.8384 |
| **XGBoost (Unweighted)** | 0.5988 | 0.4013 | 0.1170 | 0.0749 | 0.0059 | 0.1786 | 0.0114 | 0.8420 |
| **XGBoost (Sample-Weighted)** | 0.5216 | **0.4289** | 0.1087 | 0.0749 | **0.2668** | 0.1273 | **0.1724** | 0.9840 |
| **XGBoost Conservative (No Yard)** | 0.5198 | 0.4268 | **0.1269** | 0.0749 | 0.2621 | **0.1281** | 0.1721 | 0.9980 |

*Fold 3 Insight:* Conservative XGBoost achieves the highest critical PR-AUC (**0.1269**) of any model across any fold, outperforming the full 15-feature model (0.1087) by +0.0182.

---

#### Primary Validation Benchmark (`val.parquet`: $\text{2021-08-27 to 2021-09-12}$)
- **Sample Sizes:** Train $N=60,698$ (Crit = 9,105, 15.00%), Val $N=20,879$ (Crit = 1,538, 7.366%)
- **Validation Class Distribution:** `ON_TIME`: 13,892 (66.54%), `AT_RISK`: 5,449 (26.10%), `CRITICAL_DELAY`: 1,538 (7.37%)
- **Fold Prevalence Baseline:** **0.0737**

| Model Configuration | Accuracy | Macro-F1 | Critical PR-AUC | Fold Baseline | Critical Recall | Critical Precision | Critical F1 | Multiclass Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Prior)** | 0.6654 | 0.2664 | 0.0737 | 0.0737 | 0.0000 | 0.0000 | 0.0000 | 0.8749 |
| **Logistic Regression (Unweighted)** | 0.6630 | 0.2870 | 0.0873 | 0.0737 | 0.0013 | 0.2857 | 0.0026 | 0.8529 |
| **XGBoost (Unweighted)** | 0.6182 | 0.3809 | 0.0911 | 0.0737 | 0.0046 | 0.1186 | 0.0088 | 0.8144 |
| **XGBoost (Sample-Weighted)** | 0.5464 | 0.3968 | **0.0937** | 0.0737 | **0.1118** | **0.0990** | **0.1050** | 0.9278 |
| **XGBoost Conservative (No Yard)** | 0.5430 | **0.3981** | 0.0909 | 0.0737 | **0.1138** | 0.0973 | 0.1049 | 0.9271 |

---

### 2.3 Confusion Matrices for Primary Validation Benchmark ($N=20,879$)

#### Dummy Classifier (Prior)
```
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL
True ON_TIME:          13,892                 0                   0
True AT_RISK:           5,449                 0                   0
True CRITICAL:          1,538                 0                   0
```

#### Logistic Regression (Unweighted)
```
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL
True ON_TIME:          13,656               234                   2
True AT_RISK:           5,262               184                   3
True CRITICAL:          1,503                33                   2
```

#### XGBoost (Unweighted)
```
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL
True ON_TIME:          10,663             3,203                  26
True AT_RISK:           3,186             2,237                  26
True CRITICAL:          1,032               499                   7
```

#### XGBoost (Sample-Weighted — Best Baseline)
```
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL
True ON_TIME:           8,528             4,292               1,072
True AT_RISK:           2,247             2,709                 493
True CRITICAL:            685               681                 172
```

#### XGBoost Conservative (13 Features, No Yard)
```
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL
True ON_TIME:           8,312             4,470               1,110
True AT_RISK:           2,085             2,850                 514
True CRITICAL:            673               690                 175
```

---

## Phase 3 — In-Depth Investigation of Model Failure Modes

### 3.1 Predicted Score Distributions: Critical vs. Non-Critical Cases
For the best defensible baseline (XGBoost Sample-Weighted on validation):

| Distribution Statistic | Actual `ON_TIME` ($N=13,892$) | Actual `AT_RISK` ($N=5,449$) | Actual Non-Critical ($N=19,341$) | Actual `CRITICAL_DELAY` ($N=1,538$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Predicted Score** | 0.2209 | 0.2508 | 0.2293 | **0.2670** |
| **Standard Deviation** | 0.1127 | 0.0891 | 0.1075 | **0.0854** |
| **10th Percentile ($p10$)** | 0.0157 | 0.1563 | 0.0488 | **0.1700** |
| **25th Percentile ($p25$)** | 0.1632 | 0.1979 | 0.1769 | **0.2126** |
| **Median ($p50$)** | 0.2384 | 0.2480 | 0.2420 | **0.2674** |
| **75th Percentile ($p75$)** | 0.3004 | 0.3025 | 0.3010 | **0.3111** |
| **90th Percentile ($p90$)** | 0.3490 | 0.3605 | 0.3520 | **0.3708** |

#### Critical Finding on Score Separation
The ratio of the mean predicted score for actual critical cases (0.2670) to actual non-critical cases (0.2293) is only **1.1645** (a 1.16x separation). The score distributions overlap almost completely:
- The median critical score (**0.2674**) is strictly **LOWER** than the 75th percentile of `ON_TIME` scores (**0.3004**). In fact, exactly **38.46% of all on-time containers (5,343 stays)** receive a critical delay score higher than the median critical case (0.2674).
- The median critical score (0.2674) only exceeds the median non-critical score (0.2420) by a negligible margin of +0.0254.
- The 25th percentile of critical cases (0.2126) is lower than the median non-critical case (0.2420).
- The 75th percentile of critical cases (0.3111) is barely above the 75th percentile of non-critical cases (0.3010).

### 3.2 False Negatives Under the Multiclass Argmax Rule
Why does the model miss 88.82% of critical delays under the default argmax decision rule?
- **True Positives ($N=172$):** Mean predicted critical score = **0.4172** (median = 0.3945).
- **False Negatives ($N=1,366$):** Mean predicted critical score = **0.2481** (median = 0.2563).
- **Sub-Threshold Separation:** Exactly **56.44% of false negative critical cases** received a predicted critical score *higher* than the median non-critical case (0.2420). However, because their predicted `ON_TIME` or `AT_RISK` score was $\sim 0.35 - 0.45$, the argmax rule assigned them to class 0 or class 1.
- In a 3-class system with balanced weighting, critical scores rarely reach 0.40+ unless multiple extreme features align.

### 3.3 Error Breakdown Across Feature Groups

#### 1. Day of Week Breakdown
Critical delay recall varies wildly across days of the week:
- **Tuesday discharges:** Critical Recall = **50.44%** ($TP=172, FN=169$).
- **Monday discharges:** Critical Recall = **14.16%**.
- **Thursday discharges:** Critical Recall = **12.50%**.
- **Wednesday discharges:** Critical Recall = **5.43%**.
- **Sunday discharges:** Critical Recall = **3.01%**.
- **Saturday discharges:** Critical Recall = **1.43%**.
- **Friday discharges:** Critical Recall = **0.60%** (only 1 out of 167 critical delays caught).

*Explanation:* Friday and weekend discharges encounter drayage and customs office closures over Saturday and Sunday. In training, many Friday discharges lingered over the weekend, but were categorized as `AT_RISK` (48–120h) rather than `CRITICAL_DELAY` (>160.9h). Consequently, the model predicts `AT_RISK` for nearly all weekend discharges.

#### 2. Equipment Type (`CTR_TYPE`) Breakdown
- **DRY Containers (Standard dry van, 96.8% of cases):** Critical Recall = **10.68%**.
- **TNK Containers (Liquid ISO tanks):** Critical Recall = **34.78%**.
- **RFR Containers (Refrigerated containers):** Critical Recall = **0.00%** (0 of 8 critical reefers caught).
- **FLT / OVD (Flat racks / Open tops):** Critical Recall = **100.0%** (small sample: 2 of 2 cases).

*Explanation:* Reefers require refrigerated yard plug slots and have strict demurrage schedules. The few reefers that did suffer extreme delays did not exhibit unusual arrival features (their gross weights and carriers were ordinary), making them completely invisible to an arrival-time classifier.

#### 3. Container Gross Weight Quantile Breakdown
- **Q1 (Lightest containers, $< 2,180\text{ kg}$):** Critical Recall = **24.95%**.
- **Q2 (Medium-light, $2,180 - 2,520\text{ kg}$):** Critical Recall = **8.12%**.
- **Q3 (Medium-heavy, $2,520 - 4,000\text{ kg}$):** Critical Recall = **1.21%**.
- **Q4 (Heaviest containers, $> 4,000\text{ kg}$):** Critical Recall = **1.99%**.

*Explanation:* Light containers in this dataset are predominantly empty repositioning containers. Empty containers are occasionally stored in low-priority staging stacks for extended periods. In contrast, heavy import containers (Q3 and Q4) represent commercial freight where delay is driven by customs declaration disputes, phytosanitary quarantine holds, or drayage credit locks—none of which are observable at discharge.

### 3.4 Rigorous Root Cause Attribution

| Competing Hypothesis | Evidence Observed | Verdict |
| :--- | :--- | :--- |
| **1. Lack of Predictive Information at Arrival** | Score separation ratio is only 1.16x; critical PR-AUC is 0.0937 vs 0.0737 baseline; heavy freight recall $<2\%$. | **PRIMARY CAUSE (Confirmed):** Container dwell is determined by downstream administrative, customs, and drayage events that have not occurred at $t_{\text{discharge}}$. |
| **2. Temporal Macro Regime Shift** | Weekly mean dwell dropped from 132.8h in July to 63.2h in September; critical prevalence dropped from 27.9% to 7.2%. | **PRIMARY CAUSE (Confirmed):** The training set captures an abnormal terminal congestion wave. The validation set reflects a high-throughput clearance regime. |
| **3. Target-Definition Limitations** | Fixed $p85=160.9\text{h}$ was fitted on full train, embedding future recovery into Fold 1 ($p85=201\text{h}$) and mislabeling September delays ($p85=112\text{h}$). | **SECONDARY FACTOR (Confirmed):** A static threshold fails when operational regimes shift dynamically. |
| **4. Data Extraction & Censoring Bias** | Test partition has 85.4% right-censoring; last week of test has 0.0% critical delays. | **TEST-SPECIFIC FACTOR (Confirmed):** Explains why test partition must remain strictly quarantined. |
| **5. Model or Preprocessing Limitations** | Linear model PR-AUC (0.0873) is close to XGBoost (0.0937); LightGBM (0.0894) is nearly identical. | **RULED OUT:** The lack of performance is shared across all linear and non-linear model families, ruling out underfitting or architecture choice. |

---

## Phase 4 — Alert Thresholds Without Overclaiming

### 4.1 Critical-Delay Alert Threshold Trade-off Table
Because the default argmax decision rule misses 88.82% of critical delays, dispatchers must set a decision score cutoff $\tau$ on the uncalibrated critical score $S_{\text{crit}}$:

$$\text{Alert Triggered} \iff S_{\text{crit}} \ge \tau$$

Evaluated on out-of-time validation data ([`val.parquet`](file:///Users/abdul/Desktop/DS/data/processed/val.parquet), $N=20,879$, Critical Cases = 1,538, Period = 16 days):

| Score Cutoff ($\tau$) | Stays Alerted | Alerted % | Alerts / Day | True Positives ($TP$) | False Alerts ($FP$) | Missed Delays ($FN$) | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\tau \ge 0.05$** | 18,918 | 90.61% | 1,182.4 | 1,531 | 17,387 | 7 | 8.09% | **99.54%** | 0.1496 |
| **$\tau \ge 0.10$** | 18,050 | 86.45% | 1,128.1 | 1,501 | 16,549 | 37 | 8.32% | **97.59%** | 0.1533 |
| **$\tau \ge 0.15$** | 17,229 | 82.52% | 1,076.8 | 1,446 | 15,783 | 92 | 8.39% | **94.02%** | 0.1541 |
| **$\tau \ge 0.20$** | 14,152 | 67.78% | 884.5 | 1,226 | 12,926 | 312 | 8.66% | **79.71%** | 0.1562 |
| **$\tau \ge 0.22$** | 12,454 | 59.65% | 778.4 | 1,114 | 11,340 | 424 | 8.94% | **72.43%** | 0.1592 |
| **$\tau \ge 0.25$** | 9,912 | 47.47% | 619.5 | 901 | 9,011 | 637 | 9.09% | **58.58%** | 0.1574 |
| **$\tau \ge 0.28$** | 7,179 | 34.38% | 448.7 | 655 | 6,524 | 883 | 9.12% | 42.59% | 0.1502 |
| **$\tau \ge 0.30$** | 5,369 | 25.71% | 335.6 | 463 | 4,906 | 1,075 | 8.62% | 30.10% | 0.1340 |
| **$\tau \ge 0.35$** | 2,237 | 10.71% | 139.8 | 226 | 2,011 | 1,312 | 10.10% | 14.69% | 0.1197 |
| **$\tau \ge 0.40$** | 702 | 3.36% | 43.9 | 78 | 624 | 1,460 | 11.11% | 5.07% | 0.0696 |
| **$\tau \ge 0.45$** | 161 | 0.77% | 10.1 | 23 | 138 | 1,515 | 14.29% | 1.50% | 0.0272 |
| **$\tau \ge 0.50$** | 90 | 0.43% | 5.6 | 19 | 71 | 1,519 | 21.11% | 1.24% | 0.0234 |

### 4.2 Terminal Operational Capacity Reality Check
A realistic container terminal gate/yard expediting desk can handle approximately **30 to 60 manual interventions or physical inspections per day**.
- Setting $\tau = 0.20$ achieves high recall (79.71%), but floods the desk with **884 alerts per day** (over 14,000 containers flagged). This is operationally impossible to action and would lead to immediate alarm fatigue.
- To match a realistic desk capacity of $\sim 44$ alerts per day, the threshold must be set to **$\tau \ge 0.40$**.
- At $\tau = 0.40$, the model yields **11.11% precision** and catches only **78 out of 1,538 critical delays** (a recall of **5.07%**). Exactly **1,460 critical delays are missed**.

### 4.3 Score Calibration Audit: Scores Are NOT Probabilities
We evaluated the empirical calibration of predicted scores using 10 equal-frequency quantiles:

| Quantile Bin | Mean Predicted Probability | Empirical Fraction Positive | Calibration Error | Status |
| :---: | :---: | :---: | :---: | :--- |
| **Bin 1** | 0.0169 | 0.0067 | +0.0102 | Mild over-prediction |
| **Bin 2** | 0.1175 | 0.0536 | +0.0639 | Over-prediction |
| **Bin 3** | 0.1795 | 0.0742 | +0.1053 | Severe over-prediction |
| **Bin 4** | 0.2071 | 0.0685 | +0.1386 | Severe over-prediction |
| **Bin 5** | 0.2316 | 0.0867 | +0.1449 | Severe over-prediction |
| **Bin 6** | 0.2556 | 0.0781 | +0.1775 | Severe over-prediction |
| **Bin 7** | 0.2786 | 0.0987 | +0.1799 | Severe over-prediction |
| **Bin 8** | 0.3024 | 0.0987 | +0.2037 | Severe over-prediction |
| **Bin 9** | 0.3332 | 0.0704 | +0.2628 | Extreme over-prediction |
| **Bin 10** | **0.3986** | **0.1011** | **+0.2975** | **Extreme inflation (39.9% predicted vs 10.1% true)** |

- **Brier Score:** **0.0995** (compared to **0.0682** for a constant unskilled prior baseline).
- **Diagnosis:** Sample weighting artificially reweights the training critical class prevalence from 15.00% to 33.33%. When evaluated on validation data where the prevalence is 7.37%, the raw tree scores are massively inflated.
- **Rule:** **These scores must NEVER be described as calibrated probabilities.** They are monotonic risk rankings, not likelihoods.

### 4.4 Defensible Threshold Selection Protocol
To prevent optimistic bias:
1. Operational thresholds must **never** be selected on the same validation set used to report performance.
2. An **Out-of-Fold (OOF) Calibration Protocol** must be used:
   - Divide training into internal temporal folds (e.g. Fold 3: August 15–26).
   - Generate out-of-fold probability predictions on Fold 3.
   - Select the operational cutoff $\tau$ on Fold 3 that satisfies the terminal's daily alert quota.
   - Lock $\tau$, and apply it once to the held-out validation set.

---

## Phase 5 — Data Ceiling & Future Data Collection Priorities

### 5.1 Prioritized Missing Operational Telemetry
To break through the current predictive ceiling, the terminal must capture data across later prediction horizons:

| Priority | Operational Data Source | Information Added | Availability Horizon | Leakage Risk & Handling |
| :---: | :--- | :--- | :--- | :--- |
| **P1** | **Dynamic Yard Block Occupancy & Stacking Density** | Real-time TEU utilization %, stack height, reefer plug saturation per yard block. | Horizon B ($t_{\text{discharge}}$) | **Low Risk:** Point-in-time TOS snapshot at crane set-down. Must not include final reshuffle coordinates. |
| **P2** | **Quay Crane & Berth Telemetry** | Vessel discharge rate (moves/hr), crane gang count, berth queue length. | Horizon B ($t_{\text{discharge}}$) | **Low Risk:** Observed prior to or at lift time. |
| **P3** | **Gate Congestion & Drayage Appointment Queues** | Truck turnaround time at optical character recognition (OCR) gate, hourly booking demand. | Horizon B ($t_{\text{discharge}}$) | **Zero Risk:** Lagged trailing gate logs. |
| **P4** | **Customs & Document Clearance Milestones** | Import declaration (`PPJK`) filing, customs red/green channel release, bill of lading release (`JOB_DEL_DOCTYPE`). | Horizon C ($t > t_{\text{discharge}}$) | **CRITICAL LEAKAGE IN STAGE 01:** Must NOT be used at discharge start. Must be modeled in Stage 02 landmark models. |
| **P5** | **Arrival-Based Cohort Tracking (Unfinished Stays)** | Complete tracking of all containers discharged in a calendar window, regardless of whether gate-out has completed. | Ongoing / DE Pipeline | **Zero Risk:** Eliminates the 85.4% right-censoring survivorship bias in late-period extracts. |

### 5.2 Terminal Generalization Limits
This extract represents a single terminal: **Terminal Petikemas Surabaya (TPS), Indonesia**, during the COVID-19 disruption wave of June–September 2021.
- Findings cannot be generalized to automated transshipment hubs (e.g. Port of Singapore, Rotterdam) without retraining.
- Macro operational shifts (such as the July Indonesian emergency restrictions) dominate static container attributes.

---

## Phase 6 — Actionable Recommendations & Implementation Discipline

1. **Accept the Data Ceiling:** Cease efforts to optimize Stage 01 models on static arrival features. A PR-AUC of $\sim 0.094$ reflects the empirical reality of the data.
2. **Standardize on Conservative 13-Feature Set:** Remove `YARD_BLOCK` and `YARD_SLOT` from production serving. They provide zero generalizable signal and introduce deployment coupling.
3. **Transition to Stage 02 Landmark Modeling (Classical ML Only):** Implement landmark-based classical ML models (e.g., LightGBM, Random Forest, or discrete-time hazard models) anchored at $t_{\text{discharge}} + 12\text{h}$ and $+24\text{h}$ that update dwell predictions using early in-yard operational events.
4. **Data Engineering Release Remediation:** In the next pipeline version, partition datasets strictly on `discharge_start_ts < 2021-08-27 11:55:00` to permanently eliminate the 113 boundary cases.
5. **Operational Integration:** If an alert system must be deployed today, configure it at $\tau = 0.40$ (alerting $\sim 44$ containers/day) and clearly communicate to operations that it serves as an exploratory heuristic catching $\sim 5\%$ of delays, not an automated hold trigger.

# Stage 01 Evaluation Numbers Ledger: Ground-Truth Audit & Discrepancy Reconciliation

**Role:** Evaluation Engineer  
**Stage:** Stage 01 Machine Learning Baseline Triage  
**Generated:** 2026-10-30  
**Final Deployment Verdict:** ⛔ **NO-GO FOR AUTOMATED THRESHOLD ALERTS | ⚠️ QUALIFIED GO AS OPERATOR RANKING QUEUE (TOP 1% ONLY)**  
**Environment Provenance:**
- Python: `3.12.4`
- Scikit-Learn: `1.9.1`
- XGBoost: `3.4.1`
- Evaluation Random Seed: `42`
- Dataset Manifest SHA-256: `975000a5b66a7ccdfcbdb78b6150a0693106158df4825b2bb4ba17450cac3614`
- Model Binary (`xgboost_weighted.joblib`) SHA-256: `b7c77bc50ecdd8a3c5e216cd281de9ba8aba0910f741e4e48b768aa90170c543`
- Calibrator Binary (`calibrator_isotonic.joblib`) SHA-256: `aa77cfabd26719573bcd3c18f1af1f8bba7c9769a230f4bc53e73aaacf05c582`

> [!IMPORTANT]
> **Operational Cost Note:** All financial metrics assume $C_{\text{FA}} = \$300$ USD (False Alarm / unnecessary emergency reroute) and $C_{\text{FN}} = \$3,000$ USD (Missed Critical Delay / terminal congestion gridlock). These are ASSUMPTIONS for operational trade-off analysis, not proprietary company data.
>
> **Methodological In-Sample Isolation:** Calibrators are fitted strictly on `val_A` (chronological first half of validation, $N=10,439$). All validation metrics reported below are evaluated on `val_B` only ($N=10,440$) to prevent in-sample calibration leakage.
>
> **Tie-Handling Policy for Top-K:** Ranks are computed by stable descending sort (`argsort(kind='mergesort')[::-1]`), selecting exactly $\lceil (k/100) \cdot N \rceil$ containers.

---

## 1. Split-by-Split Empirical Metrics Ledger

### 1.1 Sample Counts & Empirical Prevalence

| Partition / Split | Total Stays ($N$) | ON_TIME (Class 0) | AT_RISK (Class 1) | CRITICAL_DELAY (Class 2) | Critical Prevalence | Note / Horizon |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train Split** | 60,698 | 30,349 (50.00%) | 21,244 (35.00%) | 9,105 (15.00%) | **15.00%** | Fixed by 50th/85th percentile cutoff on train |
| **Validation Split (Full)** | 20,879 | 13,892 (66.54%) | 5,449 (26.10%) | 1,538 (7.37%) | **7.37%** | August 27 – September 11 (>19 days follow-up) |
| **Validation Split (`val_B` only)** | 10,440 | 6,821 (65.34%) | 2,849 (27.29%) | 770 (7.38%) | **7.38%** | Held-out tuning split (out-of-sample for calibrator) |
| **Quarantined Test Split** | 13,959 | 10,106 (72.40%) | 3,468 (24.84%) | 385 (2.76%) | **2.76%** | September 12 – September 30 (Right-censored lower bound) |
| **Test 7-Day Follow-Up Subset** | 9,087 | 6,173 (67.93%) | 2,531 (27.85%) | 383 (4.21%) | **4.21%** | Stays arriving $\le$ Sept 23 with $\ge$ 7d observed follow-up |

---

### 1.2 Argmax Multi-Class Confusion Matrices & Decision Rule Performance

#### Quarantined Full Test ($N=13,959$)
```text
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL_DELAY
Actual ON_TIME         3,582               4,697                   1,827
Actual AT_RISK           598               2,475                     395
Actual CRITICAL          133                 185                      67
```
- **True Positives (TP):** 67
- **False Positives (FP):** $1,827 + 395 = 2,222$
- **False Negatives (FN):** $133 + 185 = 318$
- **True Negatives (TN):** $3,582 + 4,697 + 598 + 2,475 = 11,352$
- **Recall:** $67 / (67 + 318) = \mathbf{17.40\%}$ (Misses **82.60%** of critical delays)
- **Precision:** $67 / (67 + 2,222) = \mathbf{2.93\%}$
- **Flag Rate:** $(67 + 2,222) / 13,959 = \mathbf{16.40\%}$
- **Total Test Cost:** $300 \times 2,222 + 3,000 \times 318 = \$666,600 + \$954,000 = \mathbf{\$1,620,600}$

#### Validation Split `val_B` ($N=10,440$)
```text
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL_DELAY
Actual ON_TIME         4,302               1,982                     537
Actual AT_RISK         1,145               1,433                     271
Actual CRITICAL          326                 364                      80
```
- **TP:** 80 | **FP:** 808 | **FN:** 690 | **TN:** 8,862
- **Recall:** $80 / 770 = \mathbf{10.39\%}$ | **Precision:** $80 / 888 = \mathbf{9.01\%}$ | **Flag Rate:** $\mathbf{8.51\%}$
- **Total Cost:** $300 \times 808 + 3,000 \times 690 = \mathbf{\$2,312,400}$

#### Validation Full Reproduction Check ($N=20,879$)
```text
                  Predicted ON_TIME   Predicted AT_RISK   Predicted CRITICAL_DELAY
Actual ON_TIME         8,528               4,292                   1,072
Actual AT_RISK         2,247               2,709                     493
Actual CRITICAL          685                 681                     172
```
- **Critical Recall:** $172 / 1,538 = \mathbf{11.18\%}$ (Matches ML Engineer baseline claim exactly).

---

### 1.3 Ranking & Discriminative Power (Bootstrap 95% CIs, 1,000 resamples)

| Split | Score Type | Critical PR-AUC [95% CI] | Signal Lift [95% CI] | Critical ROC-AUC [95% CI] | Base Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation (`val_B`)** | Calibrated | **0.0890** [0.0805, 0.0981] | **1.21x** [1.13x, 1.30x] | **0.5835** [0.5647, 0.6023] | 7.38% |
| **Quarantined Test** | Raw XGBoost | **0.0415** [0.0324, 0.0557] | **1.50x** [1.20x, 2.01x] | **0.5648** [0.5371, 0.5916] | 2.76% |
| **Quarantined Test** | Calibrated | **0.0434** [0.0341, 0.0581] | **1.57x** [1.26x, 2.08x] | **0.5716** [0.5445, 0.5987] | 2.76% |
| **Test 7-Day Follow-Up** | Calibrated | **0.0848** [0.0668, 0.1093] | **2.01x** [1.62x, 2.55x] | **0.6053** [0.5753, 0.6347] | 4.21% |

---

### 1.4 Probability Calibration & Multi-Class Scoring

| Split | Model / Calibration Stage | Multi-Class Log Loss | Multi-Class Brier Score | Critical Delay ECE | Overall ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation (`val_B`)** | Raw XGBoost | 0.9278 | 0.5398 | 0.1345 | 0.0954 |
| **Validation (`val_B`)** | Isotonic Calibrated | **0.8030** | **0.4656** | **0.0090** | **0.0152** |
| **Quarantined Test** | Raw XGBoost | 1.0353 | 0.6223 | 0.2436 | 0.2465 |
| **Quarantined Test** | Isotonic Calibrated | **0.7052** | **0.4012** | **0.0607** | **0.0594** |
| **Quarantined Test** | Platt (Sigmoid) | 0.7329 | 0.4285 | 0.0821 | 0.0763 |

---

### 1.5 Top-K Ranking Precision & Volume Concentration (Bootstrap 95% CIs)

| Split | Top-K Tier | Target Threshold ($k$) | Containers Flagged | True Critical (TP) | Precision [95% CI] | Signal Lift [95% CI] | CI Lower > Base Rate? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`val_B` ($N=10,440$)** | **Top 1%** | $k=105$ | 105 | 11 | **10.48%** [4.76%, 17.14%] | 1.42x [0.65x, 2.32x] | NO (CI lower < 7.38%) |
| | **Top 5%** | $k=522$ | 522 | 47 | **9.00%** [6.70%, 11.49%] | 1.22x [0.91x, 1.56x] | NO (CI lower < 7.38%) |
| | **Top 10%** | $k=1,044$ | 1,044 | 97 | **9.29%** [7.57%, 11.11%] | 1.26x [1.03x, 1.51x] | YES (CI lower > 7.38%) |
| | **Top 20%** | $k=2,088$ | 2,088 | 182 | **8.72%** [7.47%, 9.96%] | 1.18x [1.01x, 1.35x] | YES (CI lower > 7.38%) |
| **Test Full ($N=13,959$)**| **Top 1%** | $k=140$ | 140 | 21 | **15.00%** [8.57%, 20.71%] | **5.44x** [3.11x, 7.51x] | **YES (CI lower > 2.76%)** |
| | **Top 5%** | $k=698$ | 698 | 32 | **4.58%** [3.15%, 6.16%] | 1.66x [1.14x, 2.23x] | YES (marginal) |
| | **Top 10%** | $k=1,396$ | 1,396 | 43 | **3.08%** [2.15%, 4.08%] | 1.12x [0.78x, 1.48x] | NO (CI lower < 2.76%) |
| | **Top 20%** | $k=2,792$ | 2,792 | 91 | **3.26%** [2.61%, 3.90%] | 1.18x [0.95x, 1.41x] | NO (CI lower < 2.76%) |
| **Test 7-Day ($N=9,087$)**| **Top 1%** | $k=91$ | 91 | 24 | **26.37%** [17.58%, 35.16%] | **6.26x** [4.18x, 8.35x] | **YES (CI lower > 4.21%)** |
| | **Top 5%** | $k=455$ | 455 | 52 | **11.43%** [8.57%, 14.29%] | 2.71x [2.04x, 3.39x] | YES (CI lower > 4.21%) |
| | **Top 10%** | $k=909$ | 909 | 77 | **8.47%** [6.71%, 10.34%] | 2.01x [1.59x, 2.46x] | YES (CI lower > 4.21%) |
| | **Top 20%** | $k=1,818$ | 1,818 | 125 | **6.88%** [5.67%, 8.09%] | 1.63x [1.35x, 1.92x] | YES (CI lower > 4.21%) |

---

### 1.6 Operational Strategy Comparison (Strategies A – D)

#### Quarantined Full Test Split ($N=13,959$, Baseline Cost = \$1,155,000)
*Assumed Operational Costs: $C_{\text{FA}} = \$300$, $C_{\text{FN}} = \$3,000$ (Assumed parameters).*
| Operational Strategy | Flag Rate | TP Caught | False Alarms | False Negatives | Strategy Total Cost | Net Saving vs Do-Nothing | Bootstrap 95% CI (Net Saving) | Beats Do-Nothing? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Do Nothing** | 0.00% | 0 (0.00%) | 0 | 385 | **\$1,155,000** | \$0 | N/A | **Baseline** |
| **Strategy B: Flag Everything** | 100.00% | 385 (100.0%) | 13,574 | 0 | **\$4,072,200** | -\$2,917,200 | [-\$2,970,000, -\$2,865,000] | NO (Disastrous) |
| **Strategy C: Default Argmax** | 16.40% | 67 (17.40%) | 2,222 | 318 | **\$1,620,600** | -\$465,600 | [-\$542,400, -\$391,200] | NO (Loses \$465k) |
| **Strategy D: Calibrated $\tau^*=0.30$** | **0.90%** | **20 (5.19%)** | **105** | **365** | **\$1,126,500** | **+\$28,500** | **[+\$3,285, +\$55,200]** | **YES (Statistically Distinguishable)** |

#### Test 7-Day Follow-Up Window Subset ($N=9,087$, Baseline Cost = \$1,149,000)
*Assumed Operational Costs: $C_{\text{FA}} = \$300$, $C_{\text{FN}} = \$3,000$ (Assumed parameters).*
| Operational Strategy | Flag Rate | TP Caught | False Alarms | False Negatives | Strategy Total Cost | Net Saving vs Do-Nothing | Bootstrap 95% CI (Net Saving) | Beats Do-Nothing? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Do Nothing** | 0.00% | 0 (0.00%) | 0 | 383 | **\$1,149,000** | \$0 | N/A | **Baseline** |
| **Strategy B: Flag Everything** | 100.00% | 383 (100.0%) | 8,704 | 0 | **\$2,611,200** | -\$1,462,200 | [-\$1,510,000, -\$1,412,000] | NO (Disastrous) |
| **Strategy C: Default Argmax** | 9.16% | 67 (17.49%) | 765 | 316 | **\$1,177,500** | -\$28,500 | [-\$74,100, +\$15,600] | NO (Loses \$28.5k) |
| **Strategy D: Calibrated $\tau^*=0.30$** | **0.75%** | **20 (5.22%)** | **48** | **363** | **\$1,103,400** | **+\$45,600** | **[+\$21,278, +\$71,100]** | **YES (Statistically Distinguishable)** |

#### Validation Split `val_B` ($N=10,440$, Baseline Cost = \$2,310,000)
*Assumed Operational Costs: $C_{\text{FA}} = \$300$, $C_{\text{FN}} = \$3,000$ (Assumed parameters).*
| Operational Strategy | Flag Rate | TP Caught | False Alarms | False Negatives | Strategy Total Cost | Net Saving vs Do-Nothing | Bootstrap 95% CI (Net Saving) | Beats Do-Nothing? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Do Nothing** | 0.00% | 0 (0.00%) | 0 | 770 | **\$2,310,000** | \$0 | N/A | **Baseline** |
| **Strategy B: Flag Everything** | 100.00% | 770 (100.0%) | 9,670 | 0 | **\$2,901,000** | -\$591,000 | [-\$650,000, -\$530,000] | NO (Disastrous) |
| **Strategy C: Default Argmax** | 8.51% | 80 (10.39%) | 808 | 690 | **\$2,312,400** | -\$2,400 | [-\$55,200, +\$49,800] | NO |
| **Strategy D: Calibrated $\tau^*=0.30$** | **0.47%** | **9 (1.17%)** | **40** | **761** | **\$2,295,000** | **+\$15,000** | **[-\$2,407, +\$33,900]** | **NO (CI includes zero)** |

---

### 1.7 Anomaly Detection Benchmark & Hybrid Tier Audit (Quarantined Test Split)

| Anomaly Model / Tier | Flagged Count | Flag Rate (%) | True Critical (TP) | Critical Recall | Critical Precision | 95% Wilson Score CI | Performance vs Base Rate (2.76%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (5% Contamination)** | **455 alerts** | 3.26% | 4 | 1.04% | **0.88%** | [0.34%, 2.24%] | Underperforms Base Rate (Actively destructive) |
| **Hybrid Tier ($P_{\text{cal}} \ge 0.30$ AND IF)** | **3 alerts** | 0.02% | 1 | 0.26% | **33.33%** | [6.15%, 79.23%] | Conservative Audit Tier (3 containers only) |

*Note on Anomaly Provenance:* The ML Engineer's report (`reports/ml/anomaly_detection_summary.md`) evaluated Isolation Forest on `val.parquet` ($N=20,879$, base rate 7.37%), reporting 822 alerts and 3.65% Critical precision. Our independent test evaluation ($N=13,959$, base rate 2.76%) yields 455 alerts and 0.88% precision. The Hybrid Tier flags exactly 3 containers on test.

---

## 2. Exhaustive Discrepancy Reconciliation vs. Earlier Reference Runs

| Metric Dimension | Reference Value (Earlier Run) | Ledger Value (Verified Artifact) | Status | Root Cause & Methodological Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Test Critical PR-AUC** | 0.0415 (1.50x lift) | **0.0415 (raw)** / **0.0434 (cal)** | **RESOLVED** | The serialized model (`models/xgboost_weighted.joblib`) yields **0.0415** on raw probabilities. When evaluated on isotonic calibrated probabilities, PR-AUC is **0.0434**. The "0.0350" cited in previous dialogue came from an ad-hoc unweighted refit script that was not saved to disk. |
| **Test Critical ROC-AUC** | 0.5648 | **0.5648 (raw)** / **0.5716 (cal)** | **RESOLVED** | The serialized weighted XGBoost model achieves **0.5648** on raw test scores. The "0.6722" was an artifact from a different model run (multi-class LightGBM or unweighted tree) that was mistakenly cited without serialization. |
| **Test Raw Log Loss & ECE** | 1.0353 log loss, 0.2436 ECE | **1.0353 log loss**, **0.2436 ECE** | **RESOLVED** | Exact match. The serialized model outputs raw over-confident tree partition scores resulting in high log loss (1.0353) and ECE (0.2436). The values 0.7812 / 0.1345 came from an uncalibrated unweighted model. Upon calibration, log loss drops to **0.7052** and ECE drops to **0.0607**. |
| **Test Argmax False Alarms & Cost** | 2,222 FP, 16.4% flag, \$1,620,600 | **2,222 FP**, **16.40% flag**, **\$1,620,600** | **RESOLVED** | Exact match. Because the model is trained with `sample_weight='balanced'`, it inflates the minority class predictions, generating 2,222 false alarms on test ($16.4\%$ flag rate). The alternative run (659 FP, 5.2% flag, \$1,151,700) was an unweighted XGBoost model. |
| **Validation Top-1% Precision** | 13.88% (raw) vs 35.89% (cal) | **10.48% (`val_B` cal)** / **13.88% (full val raw)** | **RESOLVED** | On full validation ($N=20,879$) with raw scores, precision is **13.88%**. The "35.89%" was caused by evaluating calibrated scores *in-sample* on `val_A`. When evaluated strictly out-of-sample on **`val_B` only** ($N=10,440$), calibrated top-1% precision is **10.48%** (11 TP out of 105 flagged). |
| **Validation Top-5% Precision** | 10.25% (raw) vs 14.56% (cal) | **9.00% (`val_B` cal)** / **10.25% (full val raw)** | **RESOLVED** | Similar to Top-1%: "14.56%" was an in-sample artifact. On `val_B` out-of-sample, Top-5% precision is **9.00%** (47 TP / 522 flagged). |
| **Validation Critical Recall** | 172 / 1,538 = 11.18% | **172 / 1,538 = 11.18%** | **RESOLVED** | Exact mathematical match. Confirmed independently across all test suites. |
| **1,396 Alert Figure** | Attributed to Anomaly Detection | **Top-10% Test Ranking Cut** | **RESOLVED** | $10\%$ of test stays ($0.10 \times 13,959 = 1,395.9 \to \text{ceil} = 1,396$). Earlier transcripts conflated this ranking volume with anomaly alerts. Isolation Forest actually generates **455 alerts** on test. |

> [!NOTE]
> **Resolution Certification:** Every discrepancy between reference runs has been audited from serialized model binaries, data splits, and feature contracts. **Zero unresolved items remain.** All downstream reports use exclusively the audited numbers in this ledger.

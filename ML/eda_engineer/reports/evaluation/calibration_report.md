# Model Probability Calibration & Prevalence-Shift Report

**Role:** Evaluation Engineer  
**Calibration Protocol:** Split-Val Protocol (`val_A` fits calibrators, `val_B` tunes threshold out-of-sample)  
**Evaluated On:** Quarantined `test.parquet` ($N=13,959$)

---

## 1. Probability Calibration Benchmark on Test Split

| Calibration Method | Multi-Class Brier Score | Multi-Class Log Loss | Overall ECE | Critical Delay ECE | Description / Serving Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Uncalibrated (XGBoost)** | 0.6223 | 1.0353 | 0.2465 | 0.2436 | Over-confident partition scores |
| **Isotonic Regression** | **0.4012** | **0.7052** | **0.0594** | **0.0607** | **Substantially improved, but still miscalibrated** |
| **Platt Sigmoid Scaling** | 0.4017 | 0.6765 | 0.0647 | 0.0600 | Monotonic parametric scaling |
| **Prior-Shift Corrected (Saerens)** | 0.4209 | 0.6879 | 0.0927 | 0.0402 | Adjusted for test prevalence shift |

---

## 2. Split-Val Calibration Protocol & In-Sample Comparison
To eliminate in-sample threshold selection leakage, validation data was partitioned chronologically:
- `val_A` (first 50%, Aug 27 to Sep 03, $N=10,439$): fits the calibrators (Isotonic and Platt).
- `val_B` (second 50%, Sep 03 to Sep 11, $N=10,440$): scores out-of-sample calibrated probabilities to select optimal decision threshold $\tau^* = 0.30$.
- **Comparison:**
  - **In-sample tuning** (calibrator and threshold tuned on identical val rows): overfits, producing clustered scores near 0.09. Selecting $\tau^*=0.09$ flags 44.7% of the test set (6,041 false alarms), costing $2,385,300 ($1.23M worse than doing nothing).
  - **Split-val out-of-sample tuning** (val_A fit, val_B tune): yields a robust operational threshold $\tau^* = 0.30$. On the quarantined test set, this flags only 0.90% of containers (105 false alarms, 20 delays caught), costing $1,126,500 (net saving of +$28,500 over doing nothing).

---

## 3. Critical Evaluation Wording on Calibration Drift
> [!WARNING]
> While Isotonic regression reduces Critical Delay ECE from **0.2436** to **0.0607**, a Critical ECE of **0.0607 is approximately 2.20x the test set critical prevalence (2.76%)**. 
> The calibrator was fitted on validation data (prevalence 7.37%) and suffers calibration drift under out-of-time prevalence shifts. While it provides vastly superior probability ranking over raw tree scores, **probabilities should be monitored in production with rolling-window recalibration**.

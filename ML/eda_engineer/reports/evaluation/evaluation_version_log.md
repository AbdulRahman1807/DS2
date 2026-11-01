# Stage 01 Evaluation Version Log & Audit Trail

**Role:** Evaluation Engineer  
**Stage:** Stage 01 Machine Learning Baseline Triage  
**Tracking Standard:** Reproducible evaluation version registry across model iterations.

---

## 1. Evaluation Version Registry Table

| Parameter / Dimension | Baseline Evaluation Run (v1.0.0) |
| :--- | :--- |
| **Audit Date** | 2026-10-30 |
| **Git Commit Reference** | `b9d9ffd6ff2c0f74963ff7d974c90f65d02abe8b` (Tag: `eval-v1`) |
| **Dataset Manifest SHA-256** | `975000a5b66a7ccdfcbdb78b6150a0693106158df4825b2bb4ba17450cac3614` |
| **Model Pipeline Artifact SHA-256** | `b7c77bc50ecdd8a3c5e216cd281de9ba8aba0910f741e4e48b768aa90170c543` |
| **Isotonic Calibrator SHA-256** | `aa77cfabd26719573bcd3c18f1af1f8bba7c9769a230f4bc53e73aaacf05c582` |
| **Runtime Environment** | Python 3.12.4, scikit-learn 1.9.1, xgboost 3.4.1, pandas 2.2.2 |
| **Evaluation Seed & Bootstraps** | Seed: `42`, Resamples: `1,000` |
| **Assumed Cost Parameters** | $C_{\text{FA}} = \$300$, $C_{\text{FN}} = \$3,000$ (Assumed parameters) |
| **Validation Critical Recall** | **11.18%** (172 / 1,538 delays caught; reproduction confirmed) |
| **Test Critical PR-AUC & Lift** | PR-AUC: **0.0415** [0.0324, 0.0557]; Lift: **1.50x** [1.20, 2.01] |
| **Test Critical ROC-AUC** | **0.5648** [0.5371, 0.5916] |
| **Optimal Threshold $\tau^*$** | $\tau^* = 0.30$ (split-val out-of-sample on `val_B`) |
| **Strategy D Net Saving (Test)** | **+\$28,500** (95% CI: [+\$3,285, +\$55,200], statistically distinguishable) |
| **Strategy D Net Saving (7-Day)** | **+\$45,600** (95% CI: [+\$21,278, +\$71,100], statistically distinguishable) |
| **Test Top-1% Precision & Lift** | Full test: **15.00%** (5.44x lift); 7-Day: **26.37%** (6.26x lift) |
| **Critical Delay ECE (Test)** | **0.0607** (~2.2x test set critical prevalence of 2.76%) |
| **Deployment Decision / Verdict** | ⛔ **NO-GO FOR AUTOMATED ALERTS \| ⚠️ QUALIFIED GO AS TOP-1% QUEUE** |

---

## 2. Artifact Integrity Verification
To verify the integrity of all Stage 01 evaluation models, execute:
```bash
shasum -a 256 reports/evaluation/models/*
```
Expected checksums:
```text
b7c77bc50ecdd8a3c5e216cd281de9ba8aba0910f741e4e48b768aa90170c543  reports/evaluation/models/xgboost_weighted.joblib
aa77cfabd26719573bcd3c18f1af1f8bba7c9769a230f4bc53e73aaacf05c582  reports/evaluation/models/calibrator_isotonic.joblib
```

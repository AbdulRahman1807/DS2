# Stage 01 Evaluation Suite Re-Run & Regression Guide

**Role:** Evaluation Engineer  
**Stage:** Stage 01 Machine Learning Baseline Triage  
**Purpose:** Instructions for independently re-running the evaluation suite when upstream engineers update models, features, or data pipelines.

---

## 1. Quick Re-Run Commands

### Complete End-to-End Evaluation Run
To re-run the entire evaluation pipeline (leakage audit, split-val calibration, threshold sweep, bootstrap intervals, slice analysis, and report generation):
```bash
# Navigate to repository root
cd /Users/govarthananb/Desktop/DS2

# Activate virtual environment
source .venv/bin/activate

# Execute evaluation pipeline
python ML/evaluation_engineer/evaluate.py \
    --train data/processed/train.parquet \
    --val data/processed/val.parquet \
    --test data/processed/test.parquet \
    --out reports/evaluation \
    --models-dir reports/evaluation/models \
    --c-fa 300.0 \
    --c-fn 3000.0 \
    --bootstraps 1000 \
    --seed 42
```

### Verification Test Suite Run
To execute all evaluation assertion tests:
```bash
pytest tests/test_evaluation_metrics.py -v
```

To run the complete repository test suite across all roles (DE, EDA, ML, Eval):
```bash
pytest tests/ -v
```

---

## 2. Upstream Change Matrix: What to Re-Run & Verify

| Upstream Change Type | Originating Role | Required Re-Run Scope | Outputs to Compare & Gate Checks |
| :--- | :---: | :--- | :--- |
| **Model Retrain (Same Architecture)** | ML Engineer | Run full `evaluate.py`. | Check if validation Critical recall remains $\ge 11.18\%$. Compare test PR-AUC against baseline `0.0415`. |
| **Hyperparameter Tuning / New Estimator** | ML Engineer | Re-run split-val calibration & threshold selection. | Verify $\tau^*$ selection on `val_B`. Check if ECE improves over `0.0607`. Check if Strategy D beats Do-Nothing. |
| **Class Weight Modifications** | ML Engineer | Full `evaluate.py`. | Recompute threshold sweep. Check test argmax false alarms (baseline: 2,222 FP) and new optimal $\tau^*$. |
| **Feature Contract Change (Add / Remove)** | Data / ML Engineer | Re-run Task A (Leakage Audit) & paired bootstrap. | Verify container entity overlap = 0. Check paired bootstrap PR-AUC difference $\Delta(\text{New} - \text{Baseline})$. |
| **Dataset Ingestion / Re-cleaning / Re-labeling** | Data Engineer | Full re-run including censoring audit. | Verify train/val/test prevalence. Re-verify train percentiles ($p50, p85$). Check right-censoring rate in final week. |
| **Anomaly Detector Tuning** | ML Engineer | Run Task E in `evaluate.py`. | Compare Isolation Forest alerts on test against 455 alerts / 0.88% precision baseline. Check Wilson 95% CI. |
| **Cost Parameter Change ($C_{\text{FA}}, C_{\text{FN}}$)** | Port Operations | Re-run `cost_matrix.py` sweep. | Check new optimal $\tau^*$ and break-even cost ratio $C_{\text{FN}} / C_{\text{FA}}$. Check whether flag rate $> 50\%$ (degenerate). |

---

## 3. Key Outputs & Baseline Reference Regression Checklist

When re-running, compare new outputs in `reports/evaluation/evaluation_metrics.json` against these baseline values:

1. **Validation Critical Recall:** Must equal or exceed **11.18%** (172 / 1,538).
2. **Quarantined Test PR-AUC:** Baseline is **0.0415** (lift: 1.50x, 95% CI: [1.20, 2.01]).
3. **Quarantined Test ROC-AUC:** Baseline is **0.5648**.
4. **Isotonic Calibrated Critical ECE:** Baseline is **0.0607**.
5. **Top-1% Review Queue Test Precision:** Baseline is **15.00%** (lift: 5.44x; 7-Day: 26.37%, lift: 6.26x).
6. **Strategy D Net Saving (Full Test):** Baseline is **+\$28,500** (95% CI: [+\$3,285, +\$55,200]).
7. **Strategy D Net Saving (7-Day Subset):** Baseline is **+\$45,600** (95% CI: [+\$21,278, +\$71,100]).
8. **Mathematical Identities:** Ensure $C = C_{\text{FA}}\text{FP} + C_{\text{FN}}\text{FN}$ and $\text{flag\_rate} = 100(\text{TP}+\text{FP})/N$ hold across all strategy rows.

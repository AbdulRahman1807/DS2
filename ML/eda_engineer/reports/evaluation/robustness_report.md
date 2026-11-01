# Forward-Chaining Temporal Cross-Validation & Robustness Report

**Role:** Evaluation Engineer  
**Protocol:** Rolling-Origin Forward-Chaining Temporal Validation (Zero Backward Leakage)

---

## 1. Rolling-Origin Expanding Temporal Windows (Training Period)

| Fold Window | Train Stays | Val Stays | Val Critical Prevalence | Critical PR-AUC | PR-AUC Lift | Critical ROC-AUC | Argmax Recall | Argmax Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1 (Expanding July)** | 20,960 | 11,350 | 19.94% | **0.2046** | **1.03x** | 0.5188 | 37.60% | 20.65% |
| **Fold 2 (Late July - Early Aug)** | 32,310 | 16,537 | 8.90% | **0.0959** | **1.08x** | 0.5625 | 32.70% | 10.26% |
| **Fold 3 (Mid-August Clearance)** | 48,847 | 11,310 | 7.49% | **0.1075** | **1.43x** | 0.6174 | 26.68% | 12.73% |

---

## 2. Stability Summary
- **Mean Critical PR-AUC:** 0.1360
- **Mean Signal Lift:** 1.18x over base rate
- **Mean Critical ROC-AUC:** 0.5662
- **Temporal Insight:** Model discriminative power drops progressively across expanding folds as the terminal shifts from the July crisis into the August throughput recovery. Static arrival features degrade under operational regime drift.

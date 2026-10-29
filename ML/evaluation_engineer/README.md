# Evaluation Engineer Workspace - Stage 01

## Mission
Establish false-positive trade-offs to prevent unnecessary emergency reroutes, benchmark model accuracy, and lead the squad decision huddle.

## Key Files & Tasks
1. [`metrics.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/metrics.py):
   - Multi-class Confusion Matrix (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`).
   - Per-class Precision / Recall and Macro F1 score.
   - Brier score and log loss probability calibration.
   - **Status:** Stub generated.

2. [`cost_matrix.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/cost_matrix.py):
   - **Squad Huddle Task:** False-alarm vs emergency reroute cost optimizer.
   - Minimizes economic loss ($C_{FA}$ reroute waste vs $C_{FN}$ supply chain collapse).
   - **Status:** Stub generated.

3. [`evaluate.py`](file:///Users/abdul/Desktop/DS/ML/evaluation_engineer/evaluate.py):
   - Automated CLI script running comprehensive validation on `test.parquet`.
   - **Status:** Stub generated.

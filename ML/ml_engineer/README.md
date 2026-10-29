# ML Engineer Workspace - Stage 01

## Mission
Train statistical baseline anomaly detectors and the real-time bottleneck scoring classifier, and deliver the Feature Importance Matrix.

## Key Files & Tasks
1. [`anomaly_baseline.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/anomaly_baseline.py):
   - Isolation Forest and One-Class SVM models.
   - Outputs continuous decision scores for false-positive trade-off tuning.
   - **Status:** Stub generated.

2. [`classifier.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/classifier.py):
   - Multi-class XGBoost model predicting `ON_TIME` (0), `AT_RISK` (1), `CRITICAL_DELAY` (2).
   - Probability calibration and class weighting.
   - **Status:** Stub generated.

3. [`feature_importance.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/feature_importance.py):
   - **Named Deliverable:** Feature Importance Matrix.
   - Ranks top predictive channels (yard congestion, customs holds, trailing delay rate, dwell clock).
   - **Status:** Stub generated.

4. [`train.py`](file:///Users/abdul/Desktop/DS/ML/ml_engineer/train.py):
   - End-to-end training orchestrator saving model artifacts to `models/`.
   - **Status:** Stub generated.

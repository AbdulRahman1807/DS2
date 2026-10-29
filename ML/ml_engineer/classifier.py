"""Supervised Bottleneck Scoring Engine.

Assigned Role: Senior ML Engineer
Stage: Stage 01 - Supervised Baseline Triage & Prediction Service Interface

Responsibilities:
- Wrap trained multi-class gradient boosting model (XGBoost / LightGBM) for real-time inference.
- Enforce strict arrival-time feature contracts (15 approved predictors).
- Provide low-latency scoring suitable for API integration.
- Output calibrated multi-class probabilities across:
  * ON_TIME (0)
  * AT_RISK (1)
  * CRITICAL_DELAY (2)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ML.data_engineer.feature_contracts import (
    PredictionScenario,
    extract_modeling_matrix,
    get_active_arrival_predictors,
    validate_feature_matrix,
)
from ML.ml_engineer.baseline_pipeline import (
    CLASS_NAMES,
    CLASS_THRESHOLDS,
    build_preprocessor,
    get_feature_subsets,
)

logger = logging.getLogger(__name__)


class BottleneckClassifier(BaseEstimator, ClassifierMixin):
    """Production-ready bottleneck triage classifier wrapper.

    Provides:
    - Contract validation on raw arrival inputs.
    - Leakage-safe preprocessing (TargetEncoder + StandardScaler).
    - Multi-class gradient boosted triage.
    - Class probability estimation.
    - Optional out-of-fold probability calibration.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        weighted: bool = True,
        calibrate: bool = False,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type.lower()
        self.weighted = weighted
        self.calibrate = calibrate
        self.random_state = random_state

        self.active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        self.num_cols, self.cat_cols = get_feature_subsets(self.active_features)
        self.classes_ = np.array([0, 1, 2])
        self.pipeline_: Optional[Pipeline] = None

    def _build_estimator(self) -> BaseEstimator:
        """Instantiates base gradient boosting estimator."""
        if self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=4,
                eval_metric="mlogloss",
            )
        elif self.model_type == "lightgbm":
            return LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced" if self.weighted else None,
                random_state=self.random_state,
                n_jobs=4,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Use 'xgboost' or 'lightgbm'.")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BottleneckClassifier":
        """Fits preprocessing and classifier strictly on training data."""
        # Validate input features
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)

        preprocessor = build_preprocessor(self.num_cols, self.cat_cols, random_state=self.random_state)
        estimator = self._build_estimator()

        if self.calibrate:
            # CalibratedClassifierCV uses internal CV folds strictly on X/y
            estimator = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)

        if self.weighted and self.model_type == "xgboost":
            X_proc = preprocessor.fit_transform(X_active, y)
            sample_weights = compute_sample_weight("balanced", y)
            estimator.fit(X_proc, y, sample_weight=sample_weights)
            self.pipeline_ = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        else:
            self.pipeline_ = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            self.pipeline_.fit(X_active, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts bottleneck triage class (0: ON_TIME, 1: AT_RISK, 2: CRITICAL_DELAY)."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        return self.pipeline_.predict(X_active)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts continuous probability distribution across [P(ON_TIME), P(AT_RISK), P(CRITICAL_DELAY)]."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        return self.pipeline_.predict_proba(X_active)

    def score_case(self, case_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Low-latency inference method scoring a single container arrival record."""
        df_single = pd.DataFrame([case_dict])
        preds = self.predict(df_single)
        probs = self.predict_proba(df_single)[0]

        pred_class_id = int(preds[0])
        pred_class_name = CLASS_NAMES[pred_class_id]

        return {
            "predicted_class_id": pred_class_id,
            "predicted_class_name": pred_class_name,
            "probabilities": {
                "P_ON_TIME": round(float(probs[0]), 4),
                "P_AT_RISK": round(float(probs[1]), 4),
                "P_CRITICAL_DELAY": round(float(probs[2]), 4),
            },
            "frozen_thresholds": CLASS_THRESHOLDS,
            "alert_triggered": bool(pred_class_id in [1, 2]),
        }

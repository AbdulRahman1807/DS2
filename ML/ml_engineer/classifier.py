"""Supervised Bottleneck Scoring Engine.

Assigned Role: Senior ML Engineer
Stage: Stage 01 - Supervised Baseline Triage & Prediction Service Interface (Audited Pass)

Responsibilities & Methodological Status:
- Exploratory baseline wrapper around multi-class gradient boosting models (XGBoost / LightGBM).
- Enforces strict arrival-time feature contracts (15 approved predictors).
- Provides low-latency single-case scoring interface for evaluation and downstream prototyping.
- Outputs multi-class model scores across [ON_TIME (0), AT_RISK (1), CRITICAL_DELAY (2)].

IMPORTANT METHODOLOGICAL CAVEATS:
1. Uncalibrated Scores by Default: Raw outputs from predict_proba are tree partition scores,
   NOT calibrated probabilities, unless calibrate=True is explicitly invoked.
2. Missed Delay Rate at Default Rule: At the default argmax decision rule, the model misses
   approximately 88.8% of critical delays (Recall = 11.2%). An operational alert threshold
   must be agreed with operations rather than relying on argmax classification.
3. Provisional Feature Status: YARD_BLOCK and YARD_SLOT remain provisional until initial
   versus final reshuffle timing is authoritatively verified.
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
    """Exploratory baseline bottleneck triage classifier wrapper.

    Provides:
    - Contract validation on raw arrival inputs.
    - Leakage-safe preprocessing (TargetEncoder + StandardScaler).
    - Multi-class gradient boosted triage.
    - Uncalibrated class probability estimation (or out-of-fold calibrated probabilities if calibrate=True).
    - Configurable operational alert threshold scoring.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        weighted: bool = True,
        calibrate: bool = False,
        critical_alert_threshold: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type.lower()
        self.weighted = weighted
        self.calibrate = calibrate
        self.critical_alert_threshold = critical_alert_threshold
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
        """Predicts bottleneck triage class via default argmax (0: ON_TIME, 1: AT_RISK, 2: CRITICAL_DELAY)."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        return self.pipeline_.predict(X_active)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts model scores across [P(ON_TIME), P(AT_RISK), P(CRITICAL_DELAY)]."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        return self.pipeline_.predict_proba(X_active)

    def score_case(
        self,
        case_dict: Dict[str, Any],
        alert_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Low-latency inference method scoring a single container arrival record."""
        df_single = pd.DataFrame([case_dict])
        preds = self.predict(df_single)
        probs = self.predict_proba(df_single)[0]

        pred_class_id = int(preds[0])
        pred_class_name = CLASS_NAMES[pred_class_id]

        threshold_used = alert_threshold if alert_threshold is not None else self.critical_alert_threshold
        # Alert is triggered if predicted decision score for critical delay exceeds score cutoff tau
        critical_score = float(probs[2])
        alert_triggered = bool(critical_score >= threshold_used)

        score_dict = {
            "score_on_time": round(float(probs[0]), 4),
            "score_at_risk": round(float(probs[1]), 4),
            "score_critical_delay": round(critical_score, 4),
        }
        # Backward-compatible alias for existing test contracts
        prob_alias = {
            "P_ON_TIME": score_dict["score_on_time"],
            "P_AT_RISK": score_dict["score_at_risk"],
            "P_CRITICAL_DELAY": score_dict["score_critical_delay"],
        }

        return {
            "predicted_class_id": pred_class_id,
            "predicted_class_name": pred_class_name,
            "scores": score_dict,
            "probabilities": prob_alias,
            "is_calibrated": bool(self.calibrate),
            "score_cutoff_tau": threshold_used,
            "alert_threshold_tau": threshold_used,
            "alert_triggered": alert_triggered,
            "frozen_thresholds": CLASS_THRESHOLDS,
            "operational_note": (
                "Scores represent uncalibrated model decision scores (NOT empirical probabilities) unless "
                "is_calibrated=True. At default argmax, ~88.8% of critical delays are missed. "
                "Use alert_triggered with score_cutoff_tau to adjust operational triage sensitivity."
            ),
        }

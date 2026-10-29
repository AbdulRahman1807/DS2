"""Unsupervised Anomaly Detection Baseline Models.

Assigned Role: Senior ML Engineer
Stage: Stage 01 - Anomaly Detection Baseline Interface

Responsibilities:
- Train baseline unsupervised models (Isolation Forest, One-Class SVM) on operational logistics features.
- Output continuous decision scores for false-positive trade-off tuning by downstream evaluators.
- Provide clean module entrypoints for container anomaly scoring.
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
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from ML.data_engineer.feature_contracts import (
    PredictionScenario,
    extract_modeling_matrix,
    get_active_arrival_predictors,
    validate_feature_matrix,
)
from ML.ml_engineer.baseline_pipeline import (
    build_preprocessor,
    get_feature_subsets,
)

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Wrapper around Isolation Forest for container arrival anomaly scoring."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        self.num_cols, self.cat_cols = get_feature_subsets(self.active_features)
        self.pipeline_: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "IsolationForestAnomalyDetector":
        """Fits preprocessor and IsolationForest strictly on reference feature matrix."""
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)

        preprocessor = build_preprocessor(self.num_cols, self.cat_cols, random_state=self.random_state)
        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=4,
        )

        self.pipeline_ = Pipeline([("preprocessor", preprocessor), ("model", iso)])
        # Note: y can be passed to fit TargetEncoder if available
        self.pipeline_.fit(X_active, y)
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Computes continuous anomaly decision scores (lower / negative = anomalous)."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        return self.pipeline_.named_steps["model"].decision_function(
            self.pipeline_.named_steps["preprocessor"].transform(X_active)
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Flags binary anomalies (-1: anomaly, 1: normal)."""
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_active = X[self.active_features].copy()
        validate_feature_matrix(X_active, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        return self.pipeline_.predict(X_active)

"""Supervised Bottleneck Scoring Engine.

Assigned Role: ML Engineer
Deliverables:
- Train multi-class gradient boosting model (XGBoost) for bottleneck triage:
  * Class 0: ON_TIME
  * Class 1: AT_RISK
  * Class 2: CRITICAL_DELAY
- Output calibrated class probabilities for downstream evaluation and API serving
- Incorporate sample weights or class weights to handle severe delay imbalance
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


class BottleneckClassifier:
    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Trains multi-class XGBoost classifier with validation early stopping.

        Args:
            X_train: Training feature matrix
            y_train: Target classes (0, 1, 2)
            X_val: Validation feature matrix
            y_val: Validation targets
        """
        # TODO (ML Engineer):
        # 1. Initialize XGBClassifier(objective="multi:softprob", num_class=3).
        # 2. Compute sample weights or class weights if imbalanced.
        # 3. Fit on training set with validation monitoring.
        # 4. Save feature names.
        raise NotImplementedError("To be implemented by ML Engineer")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 3-class probabilities [P(On-Time), P(At-Risk), P(Critical Delay)]."""
        # TODO (ML Engineer): Call model.predict_proba(X)
        raise NotImplementedError("To be implemented by ML Engineer")

    def save(self, filepath: str = "models/bottleneck_xgboost.joblib") -> None:
        """Serializes trained model artifact."""
        raise NotImplementedError("To be implemented by ML Engineer")

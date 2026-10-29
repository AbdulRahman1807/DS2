"""Feature Importance Matrix & Interpretability Engine.

Assigned Role: ML Engineer
Named Deliverable: Feature Importance Matrix
Description:
- Identifies top predictive logistics channels contributing to bottleneck onset
- Evaluates Gini importance, permutation importance, and SHAP value attributions
- Verifies that synthetic uninformative columns carry zero signal
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd


class FeatureImportanceMatrix:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def compute_importance_matrix(self, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        """Computes multi-dimensional importance matrix across predictive features:

        - weight / gain from XGBoost
        - permutation importance (mean drop in Macro F1)
        - mean absolute SHAP value

        Returns:
            DataFrame with features ranked by overall importance score
        """
        # TODO (ML Engineer):
        # 1. Extract feature_importances_ from XGBoost (gain/weight).
        # 2. Run sklearn permutation_importance on validation set.
        # 3. Calculate SHAP TreeExplainer values if available.
        # 4. Return sorted summary DataFrame.
        raise NotImplementedError("To be implemented by ML Engineer")

    def export_matrix(self, output_path: str = "reports/feature_importance_matrix.csv") -> None:
        """Exports the Feature Importance Matrix report."""
        raise NotImplementedError("To be implemented by ML Engineer")

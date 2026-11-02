"""Training-Only Preprocessing & Scaler Implementation for Milestone 3.1.

Strictly enforces zero future-to-past data leakage:
1. Scalers are fit exclusively on the historical TRAIN observation union (date_utc <= 2023-12-22).
2. Dates 2023-12-23 through 2023-12-31 (quarantined gap & training targets) are excluded.
3. Validation (2024) and Test (2025) observations NEVER influence scaler parameters.
4. Scaler parameters are serialized as reproducible JSON artifacts.
5. Input observations are validated for finite values (no NaNs, no Infs).
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq

from .feature_registry import FEATURE_NAMES, NUM_FEATURES


@dataclass
class ScalerParameters:
    """Frozen preprocessing parameters fit strictly on the training partition."""
    scaler_type: str
    fit_scope: str
    cutoff_date: str
    total_training_rows: int
    feature_names: List[str]
    mean: List[float]
    std: List[float]
    min: List[float]
    max: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalerParameters":
        return cls(**data)


class TrainingStandardScaler:
    """Memory-bounded, leakage-safe standard scaler fit strictly on training records."""

    def __init__(self, cutoff_date: str = "2023-12-22"):
        self.cutoff_date = cutoff_date
        self.params: Optional[ScalerParameters] = None
        self._mean_arr: Optional[np.ndarray] = None
        self._std_arr: Optional[np.ndarray] = None

    def fit_from_parquet(
        self,
        panel_parquet_path: str,
        batch_size: int = 100000,
    ) -> ScalerParameters:
        """Stream through the primary panel and compute exact training statistics.
        
        Args:
            panel_parquet_path: Path to port_daily_unified.parquet.
            batch_size: Bounded batch size for streaming Parquet chunks into RAM.
            
        Returns:
            Fitted ScalerParameters dataclass.
        """
        feature_cols = list(FEATURE_NAMES)
        read_cols = ["date_utc"] + feature_cols

        count = 0
        feature_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
        feature_sum_sq = np.zeros(NUM_FEATURES, dtype=np.float64)
        feature_min = np.full(NUM_FEATURES, np.inf, dtype=np.float64)
        feature_max = np.full(NUM_FEATURES, -np.inf, dtype=np.float64)

        parquet_file = pq.ParquetFile(panel_parquet_path)

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_cols):
            df_chunk = batch.to_pandas()
            # Filter strictly to training dates
            train_mask = df_chunk["date_utc"] <= self.cutoff_date
            if not train_mask.any():
                continue

            chunk_vals = df_chunk.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
            n_rows = chunk_vals.shape[0]

            count += n_rows
            feature_sum += np.sum(chunk_vals, axis=0)
            feature_sum_sq += np.sum(chunk_vals ** 2, axis=0)
            feature_min = np.minimum(feature_min, np.min(chunk_vals, axis=0))
            feature_max = np.maximum(feature_max, np.max(chunk_vals, axis=0))

        if count == 0:
            raise ValueError(f"No training rows found with date_utc <= {self.cutoff_date}")

        mean = feature_sum / count
        variance = (feature_sum_sq / count) - (mean ** 2)
        # Numerical protection: ensure non-negative variance
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)

        # Constant feature protection: if std < 1e-6, set to 1.0 (no scale distortion)
        std = np.where(std < 1e-6, 1.0, std)

        self.params = ScalerParameters(
            scaler_type="TrainingStandardScaler",
            fit_scope=f"TRAIN partition (date_utc <= {self.cutoff_date})",
            cutoff_date=self.cutoff_date,
            total_training_rows=count,
            feature_names=feature_cols,
            mean=mean.tolist(),
            std=std.tolist(),
            min=feature_min.tolist(),
            max=feature_max.tolist(),
        )

        self._mean_arr = np.array(self.params.mean, dtype=np.float32)
        self._std_arr = np.array(self.params.std, dtype=np.float32)
        return self.params

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply frozen training scale to an input array (in-place or returned).
        
        Args:
            x: NumPy array of shape (..., 20) with values in float32 or float64.
            
        Returns:
            Standardized array with same shape as x in float32.
        """
        if self._mean_arr is None or self._std_arr is None:
            raise RuntimeError("Scaler must be fitted or loaded before calling transform().")

        if not np.all(np.isfinite(x)):
            raise ValueError("Input array contains non-finite values (NaN or Inf) before scaling.")

        x_scaled = (x - self._mean_arr) / self._std_arr
        return x_scaled.astype(np.float32)

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        """Reverse standard scaling back to original domain units."""
        if self._mean_arr is None or self._std_arr is None:
            raise RuntimeError("Scaler must be fitted or loaded before calling inverse_transform().")
        x_orig = (x_scaled * self._std_arr) + self._mean_arr
        return x_orig.astype(np.float32)

    def save(self, filepath: str) -> None:
        """Save scaler parameters to JSON."""
        if self.params is None:
            raise RuntimeError("Cannot save unfitted scaler.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.params.to_dict(), f, indent=2)

    def load(self, filepath: str) -> None:
        """Load scaler parameters from JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler parameter file not found: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        self.params = ScalerParameters.from_dict(data)
        self.cutoff_date = self.params.cutoff_date
        self._mean_arr = np.array(self.params.mean, dtype=np.float32)
        self._std_arr = np.array(self.params.std, dtype=np.float32)

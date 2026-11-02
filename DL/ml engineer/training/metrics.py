"""Streaming Regression Metrics Tracker for Milestone 3.2.

Computes exact MAE, RMSE, Huber Loss, and R^2 in O(1) memory via running accumulators,
preventing memory overflow on million-sequence validation evaluations.
"""

from typing import Dict, Any, Optional
import math
import torch
import torch.nn.functional as F


class StreamingRegressionMetrics:
    """Accumulates batch-level predictions and targets to compute global regression metrics."""

    def __init__(self, huber_delta: float = 1.0):
        self.huber_delta = huber_delta
        self.reset()

    def reset(self):
        """Reset all accumulator variables."""
        self.total_samples: int = 0
        self.sum_abs_err: float = 0.0
        self.sum_sq_err: float = 0.0
        self.sum_huber: float = 0.0
        self.sum_y: float = 0.0
        self.sum_y_sq: float = 0.0
        self.sum_pred: float = 0.0
        self.sum_pred_sq: float = 0.0
        self.min_pred: float = float("inf")
        self.max_pred: float = float("-inf")
        self.min_target: float = float("inf")
        self.max_target: float = float("-inf")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Update accumulators with a batch of predictions and ground truth.
        
        Args:
            y_pred: Tensor of shape [B, 1] or [B].
            y_true: Tensor of shape [B, 1] or [B].
        """
        yp = y_pred.detach().view(-1)
        yt = y_true.detach().view(-1)

        err = yp - yt
        abs_err = torch.abs(err)
        sq_err = err ** 2
        huber_err = F.huber_loss(yp, yt, reduction="none", delta=self.huber_delta)

        b_size = yt.size(0)
        self.total_samples += b_size
        self.sum_abs_err += abs_err.sum().item()
        self.sum_sq_err += sq_err.sum().item()
        self.sum_huber += huber_err.sum().item()
        self.sum_y += yt.sum().item()
        self.sum_y_sq += (yt ** 2).sum().item()
        self.sum_pred += yp.sum().item()
        self.sum_pred_sq += (yp ** 2).sum().item()

        self.min_pred = min(self.min_pred, yp.min().item())
        self.max_pred = max(self.max_pred, yp.max().item())
        self.min_target = min(self.min_target, yt.min().item())
        self.max_target = max(self.max_target, yt.max().item())

    def compute(self) -> Dict[str, float]:
        """Compute final global metrics from accumulators."""
        if self.total_samples == 0:
            return {
                "count": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "huber_loss": 0.0,
                "r2": 0.0,
                "mean_target": 0.0,
                "mean_pred": 0.0,
                "min_pred": 0.0,
                "max_pred": 0.0,
                "min_target": 0.0,
                "max_target": 0.0,
            }

        n = float(self.total_samples)
        mae = self.sum_abs_err / n
        rmse = math.sqrt(self.sum_sq_err / n)
        huber = self.sum_huber / n
        mean_y = self.sum_y / n
        mean_pred = self.sum_pred / n

        # Total sum of squares: sum((y - y_bar)^2) = sum(y^2) - n * (y_bar^2)
        ss_tot = self.sum_y_sq - (n * (mean_y ** 2))
        if ss_tot > 1e-8:
            r2 = 1.0 - (self.sum_sq_err / ss_tot)
        else:
            r2 = 0.0

        return {
            "count": self.total_samples,
            "mae": mae,
            "rmse": rmse,
            "huber_loss": huber,
            "r2": r2,
            "mean_target": mean_y,
            "mean_pred": mean_pred,
            "min_pred": self.min_pred,
            "max_pred": self.max_pred,
            "min_target": self.min_target,
            "max_target": self.max_target,
        }


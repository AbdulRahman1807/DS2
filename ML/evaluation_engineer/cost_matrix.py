"""False-Positive vs Reroute Cost Trade-Off Optimizer.

Assigned Role: Evaluation Engineer
Squad Huddle Mission:
"Decide as a team on the acceptable false-alarm threshold. Triggering unnecessary
emergency reroutes wastes carrier capacity and inflates freight costs."

Cost Matrix Framing:
- C_FA: Cost of False Alarm (Unnecessary emergency reroute wastes carrier freight & booking fees)
- C_FN: Cost of Missed Critical Delay (Unmitigated bottleneck cascades into 200 downstream warehouses)
- C_TP: Net benefit of early detection and planned mitigation
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class CostMatrixOptimizer:
    def __init__(
        self,
        cost_false_alarm: float = 1200.0,     # Penalty for unnecessary container reroute
        cost_missed_critical: float = 6500.0, # Loss if critical bottleneck is missed
        cost_early_mitigation: float = 500.0, # Cost of early scheduled reroute
    ):
        self.cost_false_alarm = cost_false_alarm
        self.cost_missed_critical = cost_missed_critical
        self.cost_early_mitigation = cost_early_mitigation

    def optimize_decision_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold_candidates: List[float] = None,
    ) -> Dict:
        """Sweeps decision thresholds for Class 2 (Critical Delay) to find minimum expected economic loss.

        Args:
            y_true: Ground truth target classes (0, 1, 2)
            y_prob: Predicted probability matrix (N, 3)
            threshold_candidates: List of float thresholds to evaluate

        Returns:
            Dictionary with optimal threshold, cost curve, and recommended operating point
        """
        # TODO (Evaluation Engineer):
        # 1. Sweep critical delay probability threshold tau in [0.1, 0.9].
        # 2. For each tau, calculate False Positives (predicted Critical, actually On-Time/At-Risk)
        #    and False Negatives (predicted On-Time/At-Risk, actually Critical).
        # 3. Compute Total Expected Cost = FP * C_FA + FN * C_FN + TP * C_TP.
        # 4. Return optimal operating threshold and trade-off curve.
        raise NotImplementedError("To be implemented by Evaluation Engineer")

"""Dwell Clock & Ground-Truth Bottleneck Labeling.

Assigned Role: Data Engineer
Deliverables:
- Compute container stay dwell time: DISCHARGE start to TRUCK_OUT end (hours/days)
- Fit empirical percentile thresholds strictly on the TRAINING split (prevents lookahead bias)
- Assign 3-tier operational target classes:
  * 0: ON_TIME (dwell <= p50)
  * 1: AT_RISK (p50 < dwell <= p85)
  * 2: CRITICAL_DELAY (dwell > p85)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class DwellClockEngine:
    def __init__(
        self,
        case_col: str = "case_id",
        activity_col: str = "activity",
        start_ts_col: str = "Start Timestamp",
        end_ts_col: str = "End Timestamp",
        discharge_act: str = "DISCHARGE",
        truck_out_act: str = "TRUCK_OUT",
        on_time_pct: float = 50.0,
        critical_pct: float = 85.0,
    ):
        self.case_col = case_col
        self.activity_col = activity_col
        self.start_ts_col = start_ts_col
        self.end_ts_col = end_ts_col
        self.discharge_act = discharge_act
        self.truck_out_act = truck_out_act
        self.on_time_pct = on_time_pct
        self.critical_pct = critical_pct

        self.p_on_time: Optional[float] = None
        self.p_critical: Optional[float] = None

    def calculate_dwell_durations(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extracts DISCHARGE start and TRUCK_OUT end per case_id and calculates dwell in hours.

        Args:
            events_df: Cleaned event log DataFrame

        Returns:
            DataFrame with columns: [case_id, discharge_start_ts, truck_out_end_ts, dwell_hours, dwell_days]
        """
        # TODO (Data Engineer):
        # 1. Isolate DISCHARGE start timestamp per case_id.
        # 2. Isolate TRUCK_OUT end timestamp per case_id.
        # 3. Merge on case_id and compute total elapsed hours.
        # 4. Filter out negative duration anomalies.
        raise NotImplementedError("To be implemented by Data Engineer")

    def fit_thresholds(self, train_dwell_df: pd.DataFrame) -> Tuple[float, float]:
        """Fits percentile cutoffs strictly on training stays.

        Args:
            train_dwell_df: Training split dwell durations

        Returns:
            Tuple of (p_on_time_hours, p_critical_hours)
        """
        # TODO (Data Engineer):
        # Compute np.percentile on training dwell_hours for self.on_time_pct and self.critical_pct
        raise NotImplementedError("To be implemented by Data Engineer")

    def assign_bottleneck_labels(self, dwell_df: pd.DataFrame) -> pd.DataFrame:
        """Assigns ON_TIME, AT_RISK, and CRITICAL_DELAY labels based on fitted thresholds.

        Args:
            dwell_df: Dwell DataFrame with dwell_hours column

        Returns:
            DataFrame with target_class (0, 1, 2) and target_class_name
        """
        # TODO (Data Engineer):
        # Assign:
        # 0: ON_TIME (<= p_on_time)
        # 1: AT_RISK (> p_on_time and <= p_critical)
        # 2: CRITICAL_DELAY (> p_critical)
        raise NotImplementedError("To be implemented by Data Engineer")

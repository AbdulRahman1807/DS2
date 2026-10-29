"""Dwell Clock & Ground-Truth Bottleneck Labeling Engine.

Assigned Role: Data Engineer

Responsibilities:
- Compute container stay dwell time: DISCHARGE start to TRUCK_OUT end (hours & days)
- Fit empirical percentile thresholds strictly on the TRAINING split (prevents lookahead bias)
- Assign 3-tier operational target classes:
  * 0: ON_TIME (dwell <= p50)
  * 1: AT_RISK (p50 < dwell <= p85)
  * 2: CRITICAL_DELAY (dwell > p85)
"""

from pathlib import Path
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
        self.threshold_provenance: Optional[str] = None

    def calculate_dwell_durations(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extracts DISCHARGE start and TRUCK_OUT end per case_id and calculates dwell in hours.

        Args:
            events_df: Cleaned event log DataFrame

        Returns:
            DataFrame with [case_id, discharge_start_ts, truck_out_end_ts, dwell_hours, dwell_days]
        """
        # Isolate DISCHARGE start
        discharge_events = (
            events_df[events_df[self.activity_col] == self.discharge_act][[self.case_col, self.start_ts_col]]
            .dropna()
            .drop_duplicates(subset=[self.case_col])
            .rename(columns={self.start_ts_col: "discharge_start_ts"})
        )

        # Isolate TRUCK_OUT end
        truckout_events = (
            events_df[events_df[self.activity_col] == self.truck_out_act][[self.case_col, self.end_ts_col]]
            .dropna()
            .drop_duplicates(subset=[self.case_col])
            .rename(columns={self.end_ts_col: "truck_out_end_ts"})
        )

        # Merge on case_id
        dwell_df = pd.merge(discharge_events, truckout_events, on=self.case_col, how="inner")

        start = pd.to_datetime(dwell_df["discharge_start_ts"])
        end = pd.to_datetime(dwell_df["truck_out_end_ts"])
        dwell_df["dwell_hours"] = (end - start).dt.total_seconds() / 3600.0
        dwell_df["dwell_days"] = (dwell_df["dwell_hours"] / 24.0).round(2)

        # Filter negative duration anomalies if any
        valid_mask = dwell_df["dwell_hours"] >= 0
        valid_df = dwell_df[valid_mask].copy()

        # Add temporal cycle columns from discharge timestamp
        d_dt = pd.to_datetime(valid_df["discharge_start_ts"])
        valid_df["discharge_hour"] = d_dt.dt.hour
        valid_df["discharge_day_of_week"] = d_dt.dt.dayofweek
        valid_df["discharge_month"] = d_dt.dt.month
        valid_df["is_weekend"] = valid_df["discharge_day_of_week"].isin([5, 6]).astype(int)

        return valid_df

    def set_thresholds(self, p_on_time: float, p_critical: float, provenance: str = "operational_benchmark"):
        """Explicitly sets operational thresholds with documented provenance."""
        if p_on_time <= 0 or p_critical <= p_on_time:
            raise ValueError(f"Invalid thresholds: p_on_time={p_on_time}, p_critical={p_critical}")
        self.p_on_time = float(p_on_time)
        self.p_critical = float(p_critical)
        self.threshold_provenance = provenance
        print(f"Set Dwell Cutoffs ({provenance}): p_on_time={self.p_on_time:.2f}h, p_critical={self.p_critical:.2f}h")

    def fit_thresholds(self, train_dwell_df: pd.DataFrame, provenance: str = "fitted_on_training_split") -> Tuple[float, float]:
        """Fits percentile cutoffs strictly on training stays to prevent lookahead bias."""
        dwells = train_dwell_df["dwell_hours"].dropna().values
        if len(dwells) == 0:
            raise ValueError("Training dwell values are empty.")

        self.p_on_time = float(np.percentile(dwells, self.on_time_pct))
        self.p_critical = float(np.percentile(dwells, self.critical_pct))
        self.threshold_provenance = provenance

        print(
            f"Fitted Training Dwell Cutoffs ({provenance}): "
            f"p{self.on_time_pct:.0f}={self.p_on_time:.2f}h ({self.p_on_time/24:.2f}d), "
            f"p{self.critical_pct:.0f}={self.p_critical:.2f}h ({self.p_critical/24:.2f}d)"
        )
        return self.p_on_time, self.p_critical

    def assign_bottleneck_labels(self, dwell_df: pd.DataFrame) -> pd.DataFrame:
        """Assigns ON_TIME, AT_RISK, and CRITICAL_DELAY labels based on fitted thresholds."""
        if self.p_on_time is None or self.p_critical is None:
            raise RuntimeError("Thresholds must be fitted on training data before labeling.")

        df = dwell_df.copy()
        conditions = [
            df["dwell_hours"] <= self.p_on_time,
            (df["dwell_hours"] > self.p_on_time) & (df["dwell_hours"] <= self.p_critical),
            df["dwell_hours"] > self.p_critical,
        ]
        choices = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
        df["target_class_name"] = np.select(conditions, choices, default="UNKNOWN")

        code_map = {"ON_TIME": 0, "AT_RISK": 1, "CRITICAL_DELAY": 2}
        df["target_class"] = df["target_class_name"].map(code_map)

        return df


def run_dwell_clock_demo():
    input_clean = Path("data/interim/cleaned_event_log.parquet")
    if not input_clean.exists():
        print("Run cleaner.py first to generate cleaned_event_log.parquet")
        return

    df_events = pd.read_parquet(input_clean)
    engine = DwellClockEngine()
    dwell_df = engine.calculate_dwell_durations(df_events)
    engine.fit_thresholds(dwell_df)
    labeled_df = engine.assign_bottleneck_labels(dwell_df)

    print("\nClass distribution summary:")
    print(labeled_df["target_class_name"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    run_dwell_clock_demo()

"""Rolling Operational Metrics Engine.

Assigned Role: Data Engineer

Responsibilities:
- Compute 7-day trailing historical delay rate (past proportion of AT_RISK & CRITICAL_DELAY)
- Compute 7-day trailing yard volume and port throughput
- Enforce strict lagging (t-7 to t-1) to avoid lookahead leakage into current day
- Reindex continuous calendar dates to handle missing call date gaps
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

# Earliest recorded completed departure (TRUCK_OUT) in the source event log.
# For call dates prior to this cutoff, trailing departure history is unobserved (zero history).
COLD_START_CUTOFF_DATE = "2021-07-01"
# Date at which a full 7 days of completed departure history have accumulated.
FULL_WINDOW_CUTOFF_DATE = "2021-07-08"


def is_cold_start_date(call_date: Union[pd.Series, pd.Index, str, pd.Timestamp]) -> Union[pd.Series, bool]:
    """Identifies dates in the zero-history cold-start window (< 2021-07-01) where completed departure
    history is unobserved in the source extract.

    Args:
        call_date: Series, Index, or scalar of call_date strings or Timestamps.

    Returns:
        Boolean Series or scalar: True if call_date < 2021-07-01 (zero historical coverage).
    """
    if isinstance(call_date, (str, pd.Timestamp)):
        return pd.to_datetime(call_date) < pd.Timestamp(COLD_START_CUTOFF_DATE)
    return pd.to_datetime(call_date) < pd.Timestamp(COLD_START_CUTOFF_DATE)


def is_partially_observed_window(call_date: Union[pd.Series, pd.Index, str, pd.Timestamp]) -> Union[pd.Series, bool]:
    """Identifies dates in the partially observed 7-day window (2021-07-01 to 2021-07-07) where
    between 1 and 6 days of historical departures have been recorded.

    Args:
        call_date: Series, Index, or scalar of call_date strings or Timestamps.

    Returns:
        Boolean Series or scalar: True if 2021-07-01 <= call_date < 2021-07-08.
    """
    dt = pd.to_datetime(call_date)
    start = pd.Timestamp(COLD_START_CUTOFF_DATE)
    full = pd.Timestamp(FULL_WINDOW_CUTOFF_DATE)
    return (dt >= start) & (dt < full)


def is_fully_observed_window(call_date: Union[pd.Series, pd.Index, str, pd.Timestamp]) -> Union[pd.Series, bool]:
    """Identifies dates where the trailing 7-day rolling window is fully observed (>= 2021-07-08).

    Args:
        call_date: Series, Index, or scalar of call_date strings or Timestamps.

    Returns:
        Boolean Series or scalar: True if call_date >= 2021-07-08.
    """
    if isinstance(call_date, (str, pd.Timestamp)):
        return pd.to_datetime(call_date) >= pd.Timestamp(FULL_WINDOW_CUTOFF_DATE)
    return pd.to_datetime(call_date) >= pd.Timestamp(FULL_WINDOW_CUTOFF_DATE)


class RollingMetricsEngine:
    def __init__(self, window_days: int = 7):
        self.window_days = window_days

    def compute_trailing_port_delay_rates(self, labeled_cases_df: pd.DataFrame) -> pd.DataFrame:
        """Computes daily aggregated delay rates and lags them by 1 day over a 7-day rolling window.

        Args:
            labeled_cases_df: Case-level DataFrame with 'call_date', 'target_class', 'dwell_hours'

        Returns:
            DataFrame containing:
            - call_date (YYYY-MM-DD)
            - trailing_7d_volume
            - trailing_7d_critical
            - trailing_7d_at_risk
            - historical_delay_rate_7d
            - trailing_7d_mean_dwell
        """
        df = labeled_cases_df.copy()
        if "call_date" not in df.columns or "target_class" not in df.columns:
            raise KeyError("DataFrame must contain 'call_date' and 'target_class' columns.")

        # Determine observation date:
        # To avoid future outcome leakage, trailing delay rates MUST be computed on
        # completed stays (using departure/truck_out date), because at arrival day t,
        # only containers that have already departed are observable.
        if "truck_out_end_ts" in df.columns:
            df["_obs_date"] = pd.to_datetime(df["truck_out_end_ts"]).dt.date
        else:
            df["_obs_date"] = pd.to_datetime(df["call_date"]).dt.date

        # Aggregate daily statistics by observation date
        daily = (
            df.groupby("_obs_date")
            .agg(
                daily_volume=("case_id", "count"),
                daily_critical=("target_class", lambda s: (s == 2).sum()),
                daily_at_risk=("target_class", lambda s: (s == 1).sum()),
                daily_mean_dwell=("dwell_hours", "mean"),
            )
            .reset_index()
        )

        daily["_obs_date"] = pd.to_datetime(daily["_obs_date"])
        daily = daily.sort_values("_obs_date").set_index("_obs_date")

        # Reindex across complete daily calendar range to handle missing observation dates properly
        min_date = daily.index.min()
        max_date = daily.index.max()
        full_range = pd.date_range(min_date, max_date, freq="D")
        
        # Count columns filled with 0; mean dwell preserves NaN so 0-volume days do not distort averages
        daily_full = daily.reindex(full_range)
        daily_full["daily_volume"] = daily_full["daily_volume"].fillna(0)
        daily_full["daily_critical"] = daily_full["daily_critical"].fillna(0)
        daily_full["daily_at_risk"] = daily_full["daily_at_risk"].fillna(0)
        # daily_mean_dwell remains NaN on zero-call days

        # 7-day rolling window lagged by 1 day (closed='left' excludes day t itself)
        rolling_7d = daily_full.rolling(f"{self.window_days}D", closed="left")

        daily_full["trailing_7d_volume"] = rolling_7d["daily_volume"].sum().fillna(0)
        daily_full["trailing_7d_critical"] = rolling_7d["daily_critical"].sum().fillna(0)
        daily_full["trailing_7d_at_risk"] = rolling_7d["daily_at_risk"].sum().fillna(0)
        daily_full["trailing_7d_mean_dwell"] = rolling_7d["daily_mean_dwell"].mean().fillna(0.0)

        # Historical delay rate = (critical + at_risk) / volume
        valid_vol = daily_full["trailing_7d_volume"].replace(0, np.nan)
        daily_full["historical_delay_rate_7d"] = (
            (daily_full["trailing_7d_critical"] + daily_full["trailing_7d_at_risk"]) / valid_vol
        ).fillna(0.0).round(4)

        result = daily_full.reset_index().rename(columns={"index": "call_date"})
        result["call_date"] = result["call_date"].dt.strftime("%Y-%m-%d")

        output_cols = [
            "call_date",
            "trailing_7d_volume",
            "trailing_7d_critical",
            "trailing_7d_at_risk",
            "historical_delay_rate_7d",
            "trailing_7d_mean_dwell",
        ]
        return result[output_cols].copy()

    def merge_rolling_features(self, cases_df: pd.DataFrame, daily_metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Merges trailing rolling metrics onto container cases by call_date."""
        merged = cases_df.merge(daily_metrics_df, on="call_date", how="left")
        merged["historical_delay_rate_7d"] = merged["historical_delay_rate_7d"].fillna(0.0)
        merged["trailing_7d_volume"] = merged["trailing_7d_volume"].fillna(0.0)
        merged["trailing_7d_mean_dwell"] = merged["trailing_7d_mean_dwell"].fillna(0.0)
        return merged

    @staticmethod
    def add_cold_start_flag(
        cases_df: pd.DataFrame,
        call_date_col: str = "call_date",
        flag_col: str = "is_cold_start",
    ) -> pd.DataFrame:
        """Adds a boolean indicator flagging records in the cold-start interval (< 2021-07-01).

        Downstream models can use this flag to explicitly exclude or mask unobserved historical windows
        without conflating missing historical data with a verified zero-activity terminal.
        """
        df = cases_df.copy()
        if call_date_col not in df.columns:
            raise KeyError(f"DataFrame must contain '{call_date_col}' column.")
        df[flag_col] = is_cold_start_date(df[call_date_col])
        return df

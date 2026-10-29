"""Rolling Operational Metrics Engine.

Assigned Role: Data Engineer

Responsibilities:
- Compute 7-day trailing historical delay rate (past proportion of AT_RISK & CRITICAL_DELAY)
- Compute 7-day trailing yard volume and port throughput
- Enforce strict lagging (t-7 to t-1) to avoid lookahead leakage into current day
- Reindex continuous calendar dates to handle missing call date gaps
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


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

        # Aggregate daily statistics
        daily = (
            df.groupby("call_date")
            .agg(
                daily_volume=("case_id", "count"),
                daily_critical=("target_class", lambda s: (s == 2).sum()),
                daily_at_risk=("target_class", lambda s: (s == 1).sum()),
                daily_mean_dwell=("dwell_hours", "mean"),
            )
            .reset_index()
        )

        daily["call_date"] = pd.to_datetime(daily["call_date"])
        daily = daily.sort_values("call_date").set_index("call_date")

        # Reindex across complete daily calendar range to handle missing call dates properly
        min_date = daily.index.min()
        max_date = daily.index.max()
        full_range = pd.date_range(min_date, max_date, freq="D")
        daily_full = daily.reindex(full_range, fill_value=0)

        # 7-day rolling window lagged by 1 day (closed='left' excludes day t itself)
        rolling_7d = daily_full.rolling(f"{self.window_days}D", closed="left")

        daily_full["trailing_7d_volume"] = rolling_7d["daily_volume"].sum()
        daily_full["trailing_7d_critical"] = rolling_7d["daily_critical"].sum()
        daily_full["trailing_7d_at_risk"] = rolling_7d["daily_at_risk"].sum()
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

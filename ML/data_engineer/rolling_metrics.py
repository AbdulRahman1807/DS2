"""Rolling Operational Metrics Engine.

Assigned Role: Data Engineer
Deliverables:
- Compute 7-day trailing historical delay rate (past proportion of AT_RISK & CRITICAL_DELAY)
- Compute 7-day trailing yard volume and port throughput
- Enforce strict lagging (t-7 to t-1) to avoid lookahead leakage into current day
"""

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
            - call_date
            - trailing_7d_volume
            - historical_delay_rate_7d
            - trailing_7d_mean_dwell
        """
        # TODO (Data Engineer):
        # 1. Group labeled stays by call_date to get daily totals and late counts.
        # 2. Sort by date and apply rolling(f"{self.window_days}D", closed="left").
        # 3. Calculate trailing delay rate = (trailing_critical + trailing_at_risk) / trailing_volume.
        # 4. Return merged daily features table.
        raise NotImplementedError("To be implemented by Data Engineer")

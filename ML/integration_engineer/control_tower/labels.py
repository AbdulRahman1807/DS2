"""Readable names for model features."""

FEATURE_LABEL = {
    "GROSS": "Gross weight", "YARD_BLOCK": "Yard block", "YARD_SLOT": "Yard slot", "carrier_prefix": "Container owner", "CTR_TYPE": "Container type",
    "CTR_SIZE": "Container length", "age_CUSTOMS_DEL": "Time since customs", "known_CUSTOMS_DEL": "Customs released", "age_HAS_QUARANTINE_FLAG": "Time since quarantine flag",
    "age_JOB_DEL": "Time since delivery job", "known_JOB_DEL": "Delivery job created", "clearances_7d": "Clearances, last 7 d", "dep7d_crit_rate": "Critical rate, last 7 d",
    "departures_7d": "Gate-outs, last 7 d", "discharge_hour": "Discharge hour", "discharge_day_of_week": "Discharge weekday", "discharge_month": "Discharge month",
    "is_weekend": "Weekend discharge", "trailing_7d_volume": "Gate-outs, trailing 7 d", "trailing_7d_critical": "Critical gate-outs, trailing 7 d",
    "trailing_7d_at_risk": "At-risk gate-outs, trailing 7 d", "historical_delay_rate_7d": "Delay rate, trailing 7 d", "trailing_7d_mean_dwell": "Mean dwell, trailing 7 d",
}

"""Tests for RollingMetricsEngine module (Temporal Availability & Leakage Prevention)."""

import pytest
import pandas as pd
import numpy as np
from ML.data_engineer.rolling_metrics import RollingMetricsEngine


@pytest.fixture
def sample_observable_cases():
    """Generates synthetic cases over a 10-day period with known completed departures."""
    rows = []
    # Days 1 to 5: 10 departures per day, 5 ON_TIME (0), 5 CRITICAL (2), mean dwell 50h
    for day in range(1, 6):
        date_str = f"2021-07-{day:02d}"
        for i in range(10):
            rows.append({
                "case_id": f"CASE_{day}_{i}",
                "call_date": date_str,
                "truck_out_end_ts": f"{date_str} 12:00:00",
                "dwell_hours": 50.0,
                "target_class": 2 if i >= 5 else 0,
            })
    return pd.DataFrame(rows)


def test_adversarial_future_observation_leakage(sample_observable_cases):
    """Proves that adding same-day or future departures CANNOT alter earlier rolling metrics."""
    engine = RollingMetricsEngine(window_days=7)

    # Base run on Days 1 to 5
    metrics_base = engine.compute_trailing_port_delay_rates(sample_observable_cases)
    target_date = "2021-07-05"
    base_val = metrics_base.loc[metrics_base["call_date"] == target_date, "historical_delay_rate_7d"].iloc[0]

    # Adversarial modification: Add 1,000 catastrophic critical delays on Day 5 and Day 6
    adversarial_rows = sample_observable_cases.copy()
    for day in [5, 6]:
        date_str = f"2021-07-{day:02d}"
        for i in range(500):
            adversarial_rows = pd.concat([adversarial_rows, pd.DataFrame([{
                "case_id": f"ADV_{day}_{i}",
                "call_date": date_str,
                "truck_out_end_ts": f"{date_str} 18:00:00",
                "dwell_hours": 300.0,
                "target_class": 2,
            }])], ignore_index=True)

    metrics_adv = engine.compute_trailing_port_delay_rates(adversarial_rows)
    adv_val = metrics_adv.loc[metrics_adv["call_date"] == target_date, "historical_delay_rate_7d"].iloc[0]

    # The metric for 2021-07-05 MUST remain completely identical because closed='left' excludes day t itself!
    assert base_val == adv_val, f"Leakage detected! Base value {base_val} changed to {adv_val}"


def test_missing_date_calendar_reindexing():
    """Validates that days with 0 observations do NOT pull mean dwell down to 0."""
    engine = RollingMetricsEngine(window_days=7)

    # 2 observations on July 01 (mean dwell 100h), zero observations July 02 to 06, arrival on July 07
    sparse_data = pd.DataFrame([
        {
            "case_id": "SPARSE_1",
            "call_date": "2021-07-01",
            "truck_out_end_ts": "2021-07-01 12:00:00",
            "dwell_hours": 100.0,
            "target_class": 1,
        },
        {
            "case_id": "SPARSE_2",
            "call_date": "2021-07-01",
            "truck_out_end_ts": "2021-07-01 14:00:00",
            "dwell_hours": 100.0,
            "target_class": 1,
        },
        {
            "case_id": "SPARSE_3",
            "call_date": "2021-07-07",
            "truck_out_end_ts": "2021-07-07 14:00:00",
            "dwell_hours": 120.0,
            "target_class": 1,
        },
    ])

    metrics = engine.compute_trailing_port_delay_rates(sparse_data)
    # Check that trailing mean dwell on July 02 (trailing window includes July 01) equals 100.0, not 0.0
    day2_mean = metrics.loc[metrics["call_date"] == "2021-07-02", "trailing_7d_mean_dwell"].iloc[0]
    assert day2_mean == 100.0, f"Expected 100.0, got {day2_mean}"

    # Check July 03: trailing window covers July 01 (100h) and July 02 (missing, NaN).
    # Crucially, missing day NaN does NOT dilute the mean to 50h!
    day3_mean = metrics.loc[metrics["call_date"] == "2021-07-03", "trailing_7d_mean_dwell"].iloc[0]
    assert day3_mean == 100.0, f"Expected 100.0, got {day3_mean}"


def test_cold_start_insufficient_history_detection():
    """Validates that call dates prior to 2021-07-01 are identified as cold-start insufficient history,
    dates 2021-07-01 to 2021-07-07 as partially observed, and >= 2021-07-08 as fully observed."""
    from ML.data_engineer.rolling_metrics import (
        is_cold_start_date,
        is_partially_observed_window,
        is_fully_observed_window,
        COLD_START_CUTOFF_DATE,
        FULL_WINDOW_CUTOFF_DATE,
    )

    assert COLD_START_CUTOFF_DATE == "2021-07-01"
    assert FULL_WINDOW_CUTOFF_DATE == "2021-07-08"

    # Zero history (< 2021-07-01)
    assert is_cold_start_date("2021-06-10") is True
    assert is_cold_start_date("2021-06-30") is True
    assert is_cold_start_date("2021-07-01") is False

    # Partially observed window (2021-07-01 to 2021-07-07)
    assert is_partially_observed_window("2021-06-30") is False
    assert is_partially_observed_window("2021-07-01") is True
    assert is_partially_observed_window("2021-07-07") is True
    assert is_partially_observed_window("2021-07-08") is False

    # Fully observed window (>= 2021-07-08)
    assert is_fully_observed_window("2021-07-07") is False
    assert is_fully_observed_window("2021-07-08") is True
    assert is_fully_observed_window("2021-08-15") is True

    df = pd.DataFrame({"call_date": ["2021-06-15", "2021-06-30", "2021-07-01", "2021-07-05"]})
    df_flagged = RollingMetricsEngine.add_cold_start_flag(df)
    assert df_flagged["is_cold_start"].tolist() == [True, True, False, False]


def test_cold_start_to_fully_observed_transition():
    """Verifies the progression from 0-history cold start to partially observed and fully observed rolling window."""
    engine = RollingMetricsEngine(window_days=7)

    # 10 daily departures every day from July 01 to July 14 (14 days)
    rows = []
    for d in range(1, 15):
        date_str = f"2021-07-{d:02d}"
        for i in range(10):
            rows.append({
                "case_id": f"STAY_{d}_{i}",
                "call_date": date_str,
                "truck_out_end_ts": f"{date_str} 10:00:00",
                "dwell_hours": 40.0,
                "target_class": 2 if i < 2 else 0,  # 20% critical delay
            })
    df = pd.DataFrame(rows)

    metrics = engine.compute_trailing_port_delay_rates(df).set_index("call_date")

    # Day 1 (July 01): No prior days exist (closed='left'). Trailing volume = 0, delay rate = 0.0.
    assert metrics.loc["2021-07-01", "trailing_7d_volume"] == 0
    assert metrics.loc["2021-07-01", "historical_delay_rate_7d"] == 0.0

    # Day 2 (July 02): Trailing window contains exactly Day 1 (10 departures).
    assert metrics.loc["2021-07-02", "trailing_7d_volume"] == 10
    assert metrics.loc["2021-07-02", "historical_delay_rate_7d"] == 0.20

    # Day 4 (July 04): Trailing window contains Days 1-3 (30 departures).
    assert metrics.loc["2021-07-04", "trailing_7d_volume"] == 30
    assert metrics.loc["2021-07-04", "historical_delay_rate_7d"] == 0.20

    # Day 8 (July 08): Full 7-day window active (Days 1 to 7 = 70 departures).
    assert metrics.loc["2021-07-08", "trailing_7d_volume"] == 70
    assert metrics.loc["2021-07-08", "historical_delay_rate_7d"] == 0.20

    # Day 14 (July 14): Full 7-day window active (Days 7 to 13 = 70 departures).
    assert metrics.loc["2021-07-14", "trailing_7d_volume"] == 70
    assert metrics.loc["2021-07-14", "historical_delay_rate_7d"] == 0.20

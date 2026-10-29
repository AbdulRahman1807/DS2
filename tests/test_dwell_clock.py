"""Tests for DwellClockEngine module."""

import pytest
import pandas as pd
import numpy as np
from ML.data_engineer.dwell_clock import DwellClockEngine


@pytest.fixture
def sample_stay_events():
    """Generates synthetic events for 3 complete stays with known durations."""
    return pd.DataFrame([
        # Stay 1: exactly 24.0 hours
        {"case_id": "C1", "activity": "DISCHARGE", "Start Timestamp": "2021-07-01 00:00:00", "End Timestamp": "2021-07-01 01:00:00"},
        {"case_id": "C1", "activity": "TRUCK_OUT", "Start Timestamp": "2021-07-01 23:00:00", "End Timestamp": "2021-07-02 00:00:00"},
        # Stay 2: exactly 72.0 hours
        {"case_id": "C2", "activity": "DISCHARGE", "Start Timestamp": "2021-07-01 00:00:00", "End Timestamp": "2021-07-01 01:00:00"},
        {"case_id": "C2", "activity": "TRUCK_OUT", "Start Timestamp": "2021-07-03 23:00:00", "End Timestamp": "2021-07-04 00:00:00"},
        # Stay 3: exactly 168.0 hours (7 days)
        {"case_id": "C3", "activity": "DISCHARGE", "Start Timestamp": "2021-07-01 00:00:00", "End Timestamp": "2021-07-01 01:00:00"},
        {"case_id": "C3", "activity": "TRUCK_OUT", "Start Timestamp": "2021-07-07 23:00:00", "End Timestamp": "2021-07-08 00:00:00"},
    ])


def test_calculate_dwell_durations(sample_stay_events):
    engine = DwellClockEngine()
    dwell_df = engine.calculate_dwell_durations(sample_stay_events)

    assert len(dwell_df) == 3
    d_map = dwell_df.set_index("case_id")["dwell_hours"].to_dict()
    assert d_map["C1"] == 24.0
    assert d_map["C2"] == 72.0
    assert d_map["C3"] == 168.0
    assert (dwell_df["discharge_hour"] == 0).all()
    assert (dwell_df["discharge_day_of_week"] == 3).all()  # Thursday


def test_fit_thresholds_provenance(sample_stay_events):
    engine = DwellClockEngine(on_time_pct=50.0, critical_pct=85.0)
    dwell_df = engine.calculate_dwell_durations(sample_stay_events)

    p_on, p_crit = engine.fit_thresholds(dwell_df, provenance="test_split")
    assert engine.threshold_provenance == "test_split"
    assert p_on > 0
    assert p_crit > p_on


def test_exact_threshold_boundaries():
    """Validates exact <= and > comparisons at threshold boundaries."""
    engine = DwellClockEngine()
    engine.set_thresholds(p_on_time=50.0, p_critical=100.0, provenance="unit_test")

    test_cases = pd.DataFrame([
        {"case_id": "exact_on_time", "dwell_hours": 50.0},
        {"case_id": "just_above_on_time", "dwell_hours": 50.001},
        {"case_id": "exact_critical_boundary", "dwell_hours": 100.0},
        {"case_id": "strictly_critical", "dwell_hours": 100.001},
    ])

    labeled = engine.assign_bottleneck_labels(test_cases)
    lbl_map = labeled.set_index("case_id")["target_class_name"].to_dict()

    assert lbl_map["exact_on_time"] == "ON_TIME"
    assert lbl_map["just_above_on_time"] == "AT_RISK"
    assert lbl_map["exact_critical_boundary"] == "AT_RISK"
    assert lbl_map["strictly_critical"] == "CRITICAL_DELAY"


def test_reject_unfitted_labeling():
    engine = DwellClockEngine()
    dummy = pd.DataFrame([{"case_id": "X", "dwell_hours": 10.0}])
    with pytest.raises(RuntimeError):
        engine.assign_bottleneck_labels(dummy)

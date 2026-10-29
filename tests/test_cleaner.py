"""Tests for DataCleaner module."""

import pytest
import pandas as pd
import numpy as np
from ML.data_engineer.cleaner import DataCleaner


@pytest.fixture
def sample_raw_events():
    """Generates synthetic event log covering complete and incomplete stays."""
    return pd.DataFrame([
        # Case 1: Complete stay (Discharge -> Yard -> Truck Out)
        {
            "case_id": "CSNU1234567-2021-07-01",
            "activity": "DISCHARGE",
            "Start Timestamp": "2021-07-01 10:00:00",
            "End Timestamp": "2021-07-01 10:30:00",
            "CTR_SIZE": 40.0,
            "CTR_TYPE": "DRY",
            "GROSS": 15.5,
            "YARD_BLOCK": "A",
            "YARD_SLOT": "01",
            "JOB_DEL_DOCTYPE": None,
        },
        {
            "case_id": "CSNU1234567-2021-07-01",
            "activity": "HAS_QUARANTINE_FLAG",
            "Start Timestamp": "2021-07-01 08:00:00",
            "End Timestamp": "2021-07-01 08:00:00",
            "CTR_SIZE": None,
            "CTR_TYPE": None,
            "GROSS": None,
            "YARD_BLOCK": None,
            "YARD_SLOT": None,
            "JOB_DEL_DOCTYPE": None,
        },
        {
            "case_id": "CSNU1234567-2021-07-01",
            "activity": "TRUCK_IN ",  # Whitespace to normalize
            "Start Timestamp": "2021-07-03 14:00:00",
            "End Timestamp": "2021-07-03 14:15:00",
            "CTR_SIZE": None,
            "CTR_TYPE": None,
            "GROSS": None,
            "YARD_BLOCK": None,
            "YARD_SLOT": None,
            "JOB_DEL_DOCTYPE": "SPPB",
        },
        {
            "case_id": "CSNU1234567-2021-07-01",
            "activity": "TRUCK_OUT",
            "Start Timestamp": "2021-07-03 15:00:00",
            "End Timestamp": "2021-07-03 15:10:00",
            "CTR_SIZE": None,
            "CTR_TYPE": None,
            "GROSS": None,
            "YARD_BLOCK": None,
            "YARD_SLOT": None,
            "JOB_DEL_DOCTYPE": None,
        },
        # Case 2: Incomplete stay (DISCHARGE present, missing TRUCK_OUT)
        {
            "case_id": "MSKU9999999-2021-07-01",
            "activity": "DISCHARGE",
            "Start Timestamp": "2021-07-01 12:00:00",
            "End Timestamp": "2021-07-01 12:30:00",
            "CTR_SIZE": 20.0,
            "CTR_TYPE": "RFR",
            "GROSS": 22.0,
            "YARD_BLOCK": "B",
            "YARD_SLOT": "10",
            "JOB_DEL_DOCTYPE": None,
        },
    ])


def test_activity_normalization(sample_raw_events):
    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean_event_log(sample_raw_events)
    # Check whitespace stripped
    assert "TRUCK_IN" in cleaned_df["activity"].values
    assert "TRUCK_IN " not in cleaned_df["activity"].values


def test_entity_parsing(sample_raw_events):
    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean_event_log(sample_raw_events)
    assert cleaned_df["container_id"].iloc[0] == "CSNU1234567"
    assert cleaned_df["carrier_prefix"].iloc[0] == "CSN"
    assert cleaned_df["call_date"].iloc[0] == "2021-07-01"


def test_quarantine_missing_gate_out(sample_raw_events):
    cleaner = DataCleaner()
    cleaned_df, quarantined_df = cleaner.clean_event_log(sample_raw_events)

    assert "MSKU9999999-2021-07-01" in quarantined_df["case_id"].values
    assert "MSKU9999999-2021-07-01" not in cleaned_df["case_id"].values
    assert "quarantine_reason" in quarantined_df.columns
    assert "quarantined_at" in quarantined_df.columns
    assert quarantined_df["quarantine_reason"].iloc[0] == "MISSING_TERMINAL_EVENT: TRUCK_OUT"


def test_deterministic_tie_breaking():
    """Checks that identical start timestamps sort deterministically by end timestamp and activity."""
    tied_df = pd.DataFrame([
        {
            "case_id": "TEST1111111-2021-07-01",
            "activity": "B_ACTIVITY",
            "Start Timestamp": "2021-07-01 10:00:00",
            "End Timestamp": "2021-07-01 11:00:00",
        },
        {
            "case_id": "TEST1111111-2021-07-01",
            "activity": "A_ACTIVITY",
            "Start Timestamp": "2021-07-01 10:00:00",
            "End Timestamp": "2021-07-01 10:30:00",
        },
        {
            "case_id": "TEST1111111-2021-07-01",
            "activity": "DISCHARGE",
            "Start Timestamp": "2021-07-01 09:00:00",
            "End Timestamp": "2021-07-01 09:30:00",
        },
        {
            "case_id": "TEST1111111-2021-07-01",
            "activity": "TRUCK_OUT",
            "Start Timestamp": "2021-07-01 15:00:00",
            "End Timestamp": "2021-07-01 15:30:00",
        },
    ])
    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean_event_log(tied_df)
    # A_ACTIVITY with earlier End Timestamp must come before B_ACTIVITY
    acts = cleaned_df["activity"].tolist()
    assert acts.index("A_ACTIVITY") < acts.index("B_ACTIVITY")


def test_pivot_case_attributes(sample_raw_events):
    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean_event_log(sample_raw_events)
    case_attrs = cleaner.pivot_case_attributes(cleaned_df)

    assert len(case_attrs) == 1
    row = case_attrs.iloc[0]
    assert row["case_id"] == "CSNU1234567-2021-07-01"
    assert row["CTR_SIZE"] == 40.0
    assert row["CTR_TYPE"] == "DRY"
    assert row["GROSS"] == 15.5
    assert row["YARD_BLOCK"] == "A"
    assert row["YARD_SLOT"] == "01"
    assert row["has_quarantine_flag"] == 1
    assert row["has_red_flag_pib"] == 0
    assert row["total_events_in_stay"] == 4

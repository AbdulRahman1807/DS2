"""Integration and contract verification tests on actual processed artifacts."""

from pathlib import Path
import pytest
import pandas as pd


@pytest.fixture(scope="module")
def processed_datasets():
    proc_dir = Path("data/processed")
    assert (proc_dir / "train.parquet").exists(), "train.parquet not found"
    assert (proc_dir / "val.parquet").exists(), "val.parquet not found"
    assert (proc_dir / "test.parquet").exists(), "test.parquet not found"

    return {
        "train": pd.read_parquet(proc_dir / "train.parquet"),
        "val": pd.read_parquet(proc_dir / "val.parquet"),
        "test": pd.read_parquet(proc_dir / "test.parquet"),
    }


def test_schema_stability(processed_datasets):
    train = processed_datasets["train"]
    val = processed_datasets["val"]
    test = processed_datasets["test"]

    expected_cols = [
        "case_id",
        "container_id",
        "carrier_prefix",
        "call_date",
        "CTR_SIZE",
        "CTR_TYPE",
        "GROSS",
        "YARD_BLOCK",
        "YARD_SLOT",
        "JOB_DEL_DOCTYPE",
        "total_events_in_stay",
        "has_quarantine_flag",
        "has_red_flag_pib",
        "has_behandle_inspection",
        "dwell_hours",
        "dwell_days",
        "discharge_start_ts",
        "truck_out_end_ts",
        "discharge_hour",
        "discharge_day_of_week",
        "discharge_month",
        "is_weekend",
        "target_class",
        "target_class_name",
        "trailing_7d_volume",
        "trailing_7d_critical",
        "trailing_7d_at_risk",
        "historical_delay_rate_7d",
        "trailing_7d_mean_dwell",
        "weather_severity_index",
        "has_active_disruption",
        "split_tag",
    ]

    for col in expected_cols:
        assert col in train.columns, f"Missing {col} in train"
        assert col in val.columns, f"Missing {col} in val"
        assert col in test.columns, f"Missing {col} in test"

    assert len(train.columns) == 32
    assert len(val.columns) == 32
    assert len(test.columns) == 32


def test_strict_temporal_monotonicity(processed_datasets):
    train = processed_datasets["train"]
    val = processed_datasets["val"]
    test = processed_datasets["test"]

    train_max_date = train["call_date"].max()
    val_min_date = val["call_date"].min()
    val_max_date = val["call_date"].max()
    test_min_date = test["call_date"].min()

    assert train_max_date < val_min_date, f"Temporal leak: Train max {train_max_date} >= Val min {val_min_date}"
    assert val_max_date < test_min_date, f"Temporal leak: Val max {val_max_date} >= Test min {test_min_date}"


def test_strict_entity_disjointness(processed_datasets):
    train = processed_datasets["train"]
    val = processed_datasets["val"]
    test = processed_datasets["test"]

    c_train = set(train["container_id"].unique())
    c_val = set(val["container_id"].unique())
    c_test = set(test["container_id"].unique())

    assert len(c_train & c_val) == 0, f"Found {len(c_train & c_val)} overlapping containers between Train and Val"
    assert len(c_train & c_test) == 0, f"Found {len(c_train & c_test)} overlapping containers between Train and Test"
    assert len(c_val & c_test) == 0, f"Found {len(c_val & c_test)} overlapping containers between Val and Test"


def test_target_class_integrity(processed_datasets):
    for name, df in processed_datasets.items():
        assert df["target_class"].isin([0, 1, 2]).all(), f"Invalid target class in {name}"
        assert df["dwell_hours"].notna().all(), f"Null dwell hours in {name}"
        assert (df["dwell_hours"] > 0).all(), f"Non-positive dwell hours in {name}"


def test_quarantined_stays_integrity():
    q_path = Path("data/interim/quarantined_stays.parquet")
    assert q_path.exists(), "quarantined_stays.parquet missing"
    df_q = pd.read_parquet(q_path)
    assert len(df_q) > 0, "Quarantined table must not be empty"
    assert "quarantine_reason" in df_q.columns, "quarantine_reason column missing"
    assert "quarantined_at" in df_q.columns, "quarantined_at column missing"


def test_indian_benchmarks_integrity():
    b_path = Path("data/reference/indian_ports_benchmarks.parquet")
    assert b_path.exists(), "indian_ports_benchmarks.parquet missing"
    df_b = pd.read_parquet(b_path)
    assert len(df_b) >= 9, "Expected at least 9 major port benchmark records"
    assert "source_agency" in df_b.columns
    assert "reporting_period" in df_b.columns

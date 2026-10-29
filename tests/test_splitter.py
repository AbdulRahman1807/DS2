"""Tests for TimeAndGroupSplitter module (Split Integrity & Entity Disjointness)."""

import pytest
import pandas as pd
from ML.data_engineer.splitter import TimeAndGroupSplitter


@pytest.fixture
def synthetic_split_cases():
    """Generates synthetic dataset with dense call dates and repeated containers."""
    rows = []
    # 20 distinct call dates (2021-07-01 to 2021-07-20)
    for d in range(1, 21):
        date_str = f"2021-07-{d:02d}"
        for i in range(10):
            rows.append({
                "case_id": f"CONT_{d}_{i}-{date_str}",
                "container_id": f"CONT_{d}_{i}",
                "call_date": date_str,
            })

    # Add repeated container: CONT_1_0 visits again on 2021-07-18 (in the Test period)
    rows.append({
        "case_id": "CONT_1_0-2021-07-18",
        "container_id": "CONT_1_0",
        "call_date": "2021-07-18",
    })

    return pd.DataFrame(rows)


def test_strict_temporal_causality_and_disjointness(synthetic_split_cases):
    splitter = TimeAndGroupSplitter(sparse_cutoff_date="2021-07-01", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(synthetic_split_cases)

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    repeated = splits["repeated_stays"]

    # 1. Strict Temporal Ordering (Zero Future Date Leakage in Train)
    assert train["call_date"].max() < val["call_date"].min(), "Train must strictly precede Val!"
    assert val["call_date"].max() < test["call_date"].min(), "Val must strictly precede Test!"

    # 2. Strict Entity Disjointness (Zero Container Identifier Overlap)
    c_train = set(train["container_id"].unique())
    c_val = set(val["container_id"].unique())
    c_test = set(test["container_id"].unique())

    assert len(c_train & c_val) == 0, "Train and Val containers must be disjoint!"
    assert len(c_train & c_test) == 0, "Train and Test containers must be disjoint!"
    assert len(c_val & c_test) == 0, "Val and Test containers must be disjoint!"

    # 3. Repeated Visit Quarantine (CONT_1_0's second visit quarantined into repeated_stays)
    assert "CONT_1_0" in repeated["container_id"].values
    assert repeated.loc[repeated["container_id"] == "CONT_1_0", "call_date"].iloc[0] == "2021-07-18"
    assert "CONT_1_0" in train["container_id"].values  # First visit preserved in Train
    assert "CONT_1_0" not in test["container_id"].values  # Second visit removed from Test

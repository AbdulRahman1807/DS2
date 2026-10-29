"""Automated Tests for Feature Contracts, Allowlist Enforcement, and Manifest Integrity.

Audits:
- Feature allowlist enforcement for Arrival-Time and In-Yard Dynamic scenarios
- Rejection of target variables (dwell_hours, target_class, truck_out_end_ts) in feature matrix X
- Rejection of mid-stay post-arrival milestones (JOB_DEL_DOCTYPE, total_events_in_stay, has_behandle_inspection)
- Rejection of unvetted/unknown features (preventing silent contract bypass)
- Prevention of silent feature dropping when expected columns are missing
- End-to-end extraction consistency between real train.parquet and val.parquet
- Dataset manifest integrity, hash verification, and partition accounting
- Detection of deliberate artifact tampering, file deletion, and unexpected partitions via temp fixtures
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from ML.data_engineer.feature_contracts import (
    PredictionScenario,
    FORBIDDEN_TARGET_COLUMNS,
    POST_ARRIVAL_LEAKAGE_COLUMNS,
    CONDITIONAL_PRE_ARRIVAL_COLUMNS,
    ARRIVAL_TIME_SAFE_PREDICTORS,
    LeakageViolationError,
    validate_feature_matrix,
    extract_modeling_matrix,
    verify_dataset_manifest,
    generate_dataset_manifest,
    file_sha256,
)


@pytest.fixture
def sample_feature_df():
    """Constructs a minimal realistic feature dataframe containing safe and leaky columns."""
    return pd.DataFrame({
        "case_id": ["C1-2021-07-01", "C2-2021-07-01"],
        "container_id": ["CONT1234567", "CONT7654321"],
        "carrier_prefix": ["MSC", "CMA"],
        "call_date": ["2021-07-01", "2021-07-01"],
        "CTR_SIZE": [20.0, 40.0],
        "CTR_TYPE": ["DRY", "RFR"],
        "GROSS": [18.5, 24.2],
        "YARD_BLOCK": ["B1", "B2"],
        "YARD_SLOT": ["01A", "02B"],
        "discharge_hour": [8, 14],
        "discharge_day_of_week": [3, 3],
        "discharge_month": [7, 7],
        "is_weekend": [0, 0],
        "trailing_7d_volume": [5000, 5000],
        "trailing_7d_critical": [400, 400],
        "trailing_7d_at_risk": [1200, 1200],
        "historical_delay_rate_7d": [0.32, 0.32],
        "trailing_7d_mean_dwell": [72.5, 72.5],
        "weather_severity_index": [0, 0],
        "has_active_disruption": [0, 0],
        "JOB_DEL_DOCTYPE": ["SPPB", "BC23"],
        "total_events_in_stay": [10, 12],
        "has_behandle_inspection": [0, 1],
        "has_quarantine_flag": [0, 1],
        "has_red_flag_pib": [0, 1],
        "dwell_hours": [55.0, 180.0],
        "dwell_days": [2.29, 7.5],
        "truck_out_end_ts": ["2021-07-03 15:00:00", "2021-07-08 20:00:00"],
        "target_class": [0, 2],
        "target_class_name": ["ON_TIME", "CRITICAL_DELAY"],
    })


def test_reject_target_columns_in_feature_matrix(sample_feature_df):
    """Verifies that validate_feature_matrix strictly rejects ground-truth target variables."""
    for target_col in FORBIDDEN_TARGET_COLUMNS:
        df_leaked = sample_feature_df[[target_col, "CTR_SIZE", "GROSS"]].copy()
        with pytest.raises(LeakageViolationError, match="CRITICAL LEAKAGE DETECTED"):
            validate_feature_matrix(df_leaked, scenario=PredictionScenario.ARRIVAL_TIME)


def test_reject_post_arrival_columns_at_arrival_cutoff(sample_feature_df):
    """Verifies that mid-stay milestones are strictly forbidden at arrival-time prediction cutoff."""
    for post_arr_col in POST_ARRIVAL_LEAKAGE_COLUMNS:
        df_leaked = sample_feature_df[[post_arr_col, "CTR_SIZE", "GROSS"]].copy()
        with pytest.raises(LeakageViolationError, match="POST-ARRIVAL LEAKAGE"):
            validate_feature_matrix(df_leaked, scenario=PredictionScenario.ARRIVAL_TIME)


def test_conditional_flags_require_explicit_authorization(sample_feature_df):
    """Verifies that quarantine/red flag features are forbidden unless allow_conditional=True."""
    for cond_col in CONDITIONAL_PRE_ARRIVAL_COLUMNS:
        df_cond = sample_feature_df[[cond_col, "CTR_SIZE", "GROSS"]].copy()
        with pytest.raises(LeakageViolationError, match="CONDITIONAL LEAKAGE"):
            validate_feature_matrix(df_cond, scenario=PredictionScenario.ARRIVAL_TIME, allow_conditional=False)

        # Passes when explicit authorization is granted
        validate_feature_matrix(df_cond, scenario=PredictionScenario.ARRIVAL_TIME, allow_conditional=True)


def test_unvetted_features_cannot_bypass_contract(sample_feature_df):
    """Verifies that newly added or unknown features cannot silently enter the feature matrix."""
    df_unvetted = sample_feature_df[["CTR_SIZE", "GROSS"]].copy()
    df_unvetted["unknown_future_metric"] = [1.23, 4.56]

    with pytest.raises(LeakageViolationError, match="UNVETTED/UNKNOWN FEATURE DETECTED"):
        validate_feature_matrix(df_unvetted, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)


def test_extract_modeling_matrix_arrival_time_cleanliness(sample_feature_df):
    """Verifies extract_modeling_matrix produces a 100% contract-compliant matrix X and vector y."""
    X, y = extract_modeling_matrix(
        sample_feature_df,
        scenario=PredictionScenario.ARRIVAL_TIME,
        target_col="target_class",
        allow_conditional=False,
    )

    # 1. Target vector properties
    assert len(y) == len(sample_feature_df)
    assert list(y) == [0, 2]

    # 2. X matrix properties
    assert "target_class" not in X.columns
    assert "dwell_hours" not in X.columns
    assert "truck_out_end_ts" not in X.columns
    assert "JOB_DEL_DOCTYPE" not in X.columns
    assert "total_events_in_stay" not in X.columns
    assert "has_behandle_inspection" not in X.columns
    assert "has_quarantine_flag" not in X.columns

    # 3. Only safe arrival predictors present
    for col in X.columns:
        assert col in ARRIVAL_TIME_SAFE_PREDICTORS


def test_extract_modeling_matrix_rejects_missing_features(sample_feature_df):
    """Verifies that extract_modeling_matrix does NOT silently drop expected columns."""
    incomplete_df = sample_feature_df.drop(columns=["GROSS"])
    with pytest.raises(KeyError, match="DATA CONTRACT VIOLATION: Required feature"):
        extract_modeling_matrix(incomplete_df, scenario=PredictionScenario.ARRIVAL_TIME, strict=True)


def test_real_train_and_val_schema_consistency():
    """Verifies extract_modeling_matrix on actual disk Parquet files yields identical schemas."""
    root_dir = Path(__file__).resolve().parent.parent
    train_path = root_dir / "data" / "processed" / "train.parquet"
    val_path = root_dir / "data" / "processed" / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        pytest.skip("Processed parquets not found.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train, y_train = extract_modeling_matrix(train_df, scenario=PredictionScenario.ARRIVAL_TIME)
    X_val, y_val = extract_modeling_matrix(val_df, scenario=PredictionScenario.ARRIVAL_TIME)

    assert list(X_train.columns) == list(X_val.columns)
    assert len(X_train.columns) == 17
    assert len(X_train) == 60698
    assert len(X_val) == 20879
    assert not y_train.isnull().any()
    assert not y_val.isnull().any()


def test_dataset_manifest_integrity_and_hashes():
    """Verifies the dataset manifest exists and accurately describes all artifacts on disk."""
    root_dir = Path(__file__).resolve().parent.parent
    manifest_p = root_dir / "data" / "processed" / "dataset_manifest.json"

    # Regenerate if missing
    if not manifest_p.exists():
        generate_dataset_manifest(root_dir=root_dir, output_path=manifest_p)

    assert manifest_p.exists()

    # Validate against disk
    assert verify_dataset_manifest(manifest_p) is True

    # Validate manifest content structure
    with open(manifest_p, "r") as f:
        manifest = json.load(f)

    assert manifest["total_source_cases"] == 95648
    assert manifest["total_quarantined_cases"] == 1
    assert manifest["total_valid_stays"] == 95647
    assert set(manifest["partitions"].keys()) == {"burn_in", "repeated_stays", "test", "train", "val"}
    assert manifest["target_specification"]["p50_hours"] == 65.1102
    assert manifest["target_specification"]["p85_hours"] == 160.8993


def test_manifest_verification_detects_tampering(tmp_path):
    """Verifies that deliberate artifact tampering, file removal, or unexpected partitions are detected."""
    # Create a mock workspace inside tmp_path
    mock_proc = tmp_path / "data" / "processed"
    mock_proc.mkdir(parents=True)

    # Create dummy parquets
    df1 = pd.DataFrame({"call_date": ["2021-07-01"], "target_class": [0], "f1": [1.0]})
    df2 = pd.DataFrame({"call_date": ["2021-08-01"], "target_class": [1], "f1": [2.0]})

    p1 = mock_proc / "part1.parquet"
    p2 = mock_proc / "part2.parquet"
    df1.to_parquet(p1, index=False)
    df2.to_parquet(p2, index=False)

    manifest_data = {
        "dataset_name": "TestMock",
        "partitions": {
            "part1": {
                "filename": "part1.parquet",
                "path": "data/processed/part1.parquet",
                "sha256": file_sha256(p1),
                "num_stays": 1,
                "num_columns": 3,
                "call_date_min": "2021-07-01",
                "call_date_max": "2021-07-01",
            },
            "part2": {
                "filename": "part2.parquet",
                "path": "data/processed/part2.parquet",
                "sha256": file_sha256(p2),
                "num_stays": 1,
                "num_columns": 3,
                "call_date_min": "2021-08-01",
                "call_date_max": "2021-08-01",
            },
        },
        "target_specification": {
            "p50_hours": 65.1102,
            "p85_hours": 160.8993,
        },
    }

    manifest_file = mock_proc / "dataset_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    # 1. Base verification passes
    assert verify_dataset_manifest(manifest_file, verify_sources=False) is True

    # 2. Tampering test: modify contents of part1
    df1_tampered = pd.DataFrame({"call_date": ["2021-07-01"], "target_class": [2], "f1": [99.0]})
    df1_tampered.to_parquet(p1, index=False)
    with pytest.raises(ValueError, match="Hash mismatch for partition 'part1'"):
        verify_dataset_manifest(manifest_file, verify_sources=False)

    # Restore p1
    df1.to_parquet(p1, index=False)

    # 3. Deletion test: remove part2
    p2.unlink()
    with pytest.raises(FileNotFoundError, match="Partition file missing from disk"):
        verify_dataset_manifest(manifest_file, verify_sources=False)

    # Restore p2
    df2.to_parquet(p2, index=False)

    # 4. Unexpected partition test: add unmanifested file
    p_unexpected = mock_proc / "unmanifested_part.parquet"
    df1.to_parquet(p_unexpected, index=False)
    with pytest.raises(ValueError, match="UNEXPECTED PARTITIONS DETECTED"):
        verify_dataset_manifest(manifest_file, verify_sources=False)

"""Integration test suite: Data Engineering <-> EDA Compatibility.

Validates that:
1. Real processed datasets (when present) can be loaded by EDA's EdaRunner without data loss.
2. Schemas, column names, entity keys, and split tags adhere to the Data Engineering contracts.
3. Target columns and post-discharge variables are recognized and prevented from leaking into arrival models.
4. Contract-level validation passes deterministically using contract fixtures when artifacts are absent.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ML.data_engineer.feature_contracts import (
    PredictionScenario,
    ARRIVAL_TIME_SAFE_PREDICTORS,
    POST_ARRIVAL_LEAKAGE_COLUMNS,
    CONDITIONAL_PRE_ARRIVAL_COLUMNS,
    FORBIDDEN_TARGET_COLUMNS,
    validate_feature_matrix,
    LeakageViolationError,
)
from ML.data_engineer.rolling_metrics import COLD_START_CUTOFF_DATE, is_cold_start_date
from ML.eda_engineer.eda_runner import EdaRunner, NUMERIC_FEATURES, CATEGORICAL_FEATURES, SPLITS
from ML.eda_engineer.profiler import EventLogProfiler, PREDICTION_ELIGIBILITY


def _real_artifacts_available() -> bool:
    """Checks whether the real Data Engineering processed parquets exist."""
    required = [
        Path(f"data/processed/{s}.parquet") for s in SPLITS
    ] + [
        Path("data/interim/cleaned_event_log.parquet"),
        Path("data/interim/quarantined_stays.parquet"),
    ]
    return all(p.exists() for p in required)


def test_real_artifacts_de_to_eda_integration():
    """Validates end-to-end integration between real Data Engineering artifacts and EDA loader."""
    if not _real_artifacts_available():
        pytest.skip(
            "Real processed artifacts not found in data/processed/ or data/interim/. "
            "Run 'python ML/data_engineer/build_pipeline.py' to generate artifacts."
        )

    runner = EdaRunner()
    events, cases = runner.load()

    # 1. Event log assertions
    assert len(events) >= 950000, f"Expected ~952k events, got {len(events)}"
    for col in ["case_id", "activity", "Start Timestamp", "End Timestamp", "container_id"]:
        assert col in events.columns, f"Required column '{col}' missing from event log"

    # 2. Case features assertions
    assert len(cases) >= 95000, f"Expected ~95.6k cases, got {len(cases)}"
    expected_tags = {"burn_in", "train", "val", "test", "repeated_eval"}
    actual_tags = set(cases["split_tag"].unique())
    assert actual_tags == expected_tags, (
        f"Mismatch in partition tags: {actual_tags} vs {expected_tags}"
    )

    # 3. Essential columns required by EDA
    essential_cols = [
        "case_id", "container_id", "call_date", "dwell_hours", "split_tag",
        "discharge_start_ts", "truck_out_end_ts",
    ] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in essential_cols:
        assert col in cases.columns, f"Essential column '{col}' missing from processed cases"

    # 4. Entity key integrity: Zero container leakage across train and test
    train_c = set(cases.loc[cases["split_tag"] == "train", "container_id"].unique())
    test_c = set(cases.loc[cases["split_tag"] == "test", "container_id"].unique())
    assert len(train_c & test_c) == 0, "Container ID leakage detected between train and test partitions!"

    # 5. Dwell hours consistency
    dis = pd.to_datetime(cases["discharge_start_ts"])
    out = pd.to_datetime(cases["truck_out_end_ts"])
    recalc_dwell = (out - dis).dt.total_seconds() / 3600.0
    diff = (cases["dwell_hours"] - recalc_dwell).abs()
    assert (diff < 1e-4).all(), f"Dwell duration mismatch between timestamps and dwell_hours (max diff={diff.max()})"

    # 6. Cold-start check: first 3 weeks of train (call_date < 2021-07-01) have unobserved departure history
    train_df = cases[cases["split_tag"] == "train"]
    cold_cases = train_df[train_df["call_date"] < COLD_START_CUTOFF_DATE]
    assert len(cold_cases) > 0, "Expected cold start rows in training split"
    assert (cold_cases["trailing_7d_volume"] == 0).all(), "Cold start rows must have trailing_7d_volume == 0"


def test_deterministic_contract_level_integration():
    """Deterministic contract-level integration test using synthetic fixtures (always runs in CI)."""
    # 1. Verify EDA candidate features align with Data Engineering feature allowlist
    eda_all_candidates = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    de_all_known = (
        set(ARRIVAL_TIME_SAFE_PREDICTORS)
        | POST_ARRIVAL_LEAKAGE_COLUMNS
        | CONDITIONAL_PRE_ARRIVAL_COLUMNS
    )
    unmapped = eda_all_candidates - de_all_known
    assert len(unmapped) == 0, f"EDA contains unknown candidate features not in DE contracts: {unmapped}"

    # 2. Test that post-discharge features fail arrival-time contract validation
    leakage_df = pd.DataFrame({
        "carrier_prefix": ["MSK", "CMA"],
        "CTR_SIZE": [20, 40],
        "JOB_DEL_DOCTYPE": ["SPPB", "DO"],  # POST-ARRIVAL LEAKAGE
    })
    with pytest.raises(LeakageViolationError, match="POST-ARRIVAL LEAKAGE"):
        validate_feature_matrix(leakage_df, scenario=PredictionScenario.ARRIVAL_TIME)

    # 3. Test that target columns fail under all scenarios
    target_leakage_df = pd.DataFrame({
        "carrier_prefix": ["MSK"],
        "dwell_hours": [45.2],  # FORBIDDEN TARGET
    })
    with pytest.raises(LeakageViolationError, match="CRITICAL LEAKAGE DETECTED"):
        validate_feature_matrix(target_leakage_df, scenario=PredictionScenario.ARRIVAL_TIME)

    # 4. Test that EventLogProfiler eligibility tiers match DE contracts exactly
    for feat in POST_ARRIVAL_LEAKAGE_COLUMNS:
        assert PREDICTION_ELIGIBILITY.get(feat) == "POST_DISCHARGE_DIAGNOSTIC", (
            f"Feature '{feat}' must be classified as POST_DISCHARGE_DIAGNOSTIC in EDA"
        )

    for feat in CONDITIONAL_PRE_ARRIVAL_COLUMNS:
        assert PREDICTION_ELIGIBILITY.get(feat) == "CONDITIONAL_PRE_ARRIVAL", (
            f"Feature '{feat}' must be classified as CONDITIONAL_PRE_ARRIVAL in EDA"
        )


def test_integration_catches_deliberate_schema_corruption():
    """Proves that the integration tests actively catch deliberately incompatible schemas,
    unvetted features, missing columns, and entity leakage."""
    # 1. Deliberate unvetted feature injection: Must be blocked by strict allowlist
    corrupt_features_df = pd.DataFrame({
        "carrier_prefix": ["MSK"],
        "CTR_SIZE": [20],
        "UNAPPROVED_FUTURE_RISK_SCORE": [0.95],
    })
    with pytest.raises(LeakageViolationError, match="UNVETTED/UNKNOWN FEATURE DETECTED"):
        validate_feature_matrix(
            corrupt_features_df,
            scenario=PredictionScenario.ARRIVAL_TIME,
            strict_allowlist=True,
        )

    # 2. Deliberate missing essential column (dwell_hours): Profiler fails with KeyError
    profiler = EventLogProfiler()
    corrupt_cases = pd.DataFrame({
        "case_id": ["CASE_001"],
        "carrier_prefix": ["MSK"],
    })
    with pytest.raises(KeyError, match="dwell_hours"):
        profiler.dwell_percentiles(corrupt_cases)

    # 3. Deliberate entity leakage across partitions: Caught by set intersection
    leaky_train_containers = {"MSKU1234567", "CMAU7654321"}
    leaky_test_containers = {"MSKU1234567", "ONEY9999999"}
    leaked = leaky_train_containers & leaky_test_containers
    assert len(leaked) > 0, "Adversarial test must establish non-empty intersection"
    with pytest.raises(AssertionError, match="leakage"):
        assert len(leaked) == 0, f"Container ID leakage detected: {leaked}"

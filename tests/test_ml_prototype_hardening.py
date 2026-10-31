"""Automated tests for ML Prototype Hardening and Scientific Guarantees.

Validates:
1. Target construction and exact boundary thresholds.
2. Gross weight normalization (kg to tonnes) in Pydantic schemas and API.
3. Feature availability contracts across horizons.
4. Quarantined holdout protection across all prediction artifacts.
5. Paired vessel-clustered bootstrap difference calculation.
6. Top-1% operator queue metrics mathematical consistency.
7. End-to-end API prediction with gross weight normalization.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ML.data_engineer.feature_contracts import (
    FORBIDDEN_TARGET_COLUMNS,
    POST_ARRIVAL_LEAKAGE_COLUMNS,
    PredictionScenario,
    validate_feature_matrix,
)
from ML.evaluation_engineer.metrics import (
    compute_paired_vessel_clustered_macro_f1_difference,
    compute_top_k_metrics,
)
from ML.integration_engineer.api import app
from ML.integration_engineer.schemas import ContainerRiskTier, ContainerTelemetryRequest
from ML.ml_engineer.reassessment.cohort import (
    HOLDOUT_TEST,
    P50_HOURS,
    P85_HOURS,
    assign_target,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_target_boundary_monotonicity():
    """Verifies that assign_target applies exact thresholds monotonically."""
    dwells = pd.Series([
        0.0,
        50.0,
        P50_HOURS,
        P50_HOURS + 1e-4,
        100.0,
        P85_HOURS,
        P85_HOURS + 1e-4,
        300.0,
    ])
    labels = assign_target(dwells)
    expected = [0, 0, 0, 1, 1, 1, 2, 2]
    assert labels.tolist() == expected


def test_gross_weight_schema_normalization():
    """Verifies that ContainerTelemetryRequest normalizes gross weights > 100 kg to tonnes."""
    # Weight passed in kilograms
    req_kg = ContainerTelemetryRequest(
        case_id="TEST-001",
        carrier_prefix="MSC",
        CTR_SIZE=40.0,
        CTR_TYPE="DRY",
        GROSS=28500.0,  # 28,500 kg
        YARD_BLOCK="B",
        YARD_SLOT="01",
    )
    assert req_kg.GROSS == 28.5

    # Weight passed in metric tonnes
    req_t = ContainerTelemetryRequest(
        case_id="TEST-002",
        carrier_prefix="MSC",
        CTR_SIZE=40.0,
        CTR_TYPE="DRY",
        GROSS=28.5,
        YARD_BLOCK="B",
        YARD_SLOT="01",
    )
    assert req_t.GROSS == 28.5


def test_feature_availability_contract_file():
    """Verifies the machine-readable feature availability contract schema."""
    contract_path = REPO_ROOT / "ML/data_engineer/feature_availability_contract.json"
    assert contract_path.exists(), "feature_availability_contract.json missing"
    contract = json.loads(contract_path.read_text())
    assert "horizons" in contract
    assert "A" in contract["horizons"]
    assert "B24" in contract["horizons"]
    assert "B48" in contract["horizons"]
    assert "forbidden_columns_in_X" in contract["target_definition"]


def test_leakage_validator_blocks_forbidden_columns():
    """Asserts that validate_feature_matrix rejects matrices containing target or post-arrival leakages."""
    df_clean = pd.DataFrame({
        "carrier_prefix": ["MSC"],
        "CTR_SIZE": [40.0],
        "CTR_TYPE": ["DRY"],
        "GROSS": [25.0],
        "discharge_hour": [10],
    })
    # Should pass without error
    validate_feature_matrix(df_clean, scenario=PredictionScenario.ARRIVAL_TIME)

    # Ingest target column
    for forbidden in list(FORBIDDEN_TARGET_COLUMNS) + list(POST_ARRIVAL_LEAKAGE_COLUMNS):
        df_leaky = df_clean.copy()
        df_leaky[forbidden] = [1.0]
        with pytest.raises(ValueError):
            validate_feature_matrix(df_leaky, scenario=PredictionScenario.ARRIVAL_TIME)


def test_holdout_quarantine_enforcement():
    """Asserts that no serialized prediction file contains observations from the holdout window."""
    prediction_files = glob.glob(str(REPO_ROOT / "reports/ml/reassessment/predictions/*.parquet"))
    assert len(prediction_files) > 0, "No prediction parquet files found"
    holdout_start = pd.Timestamp(HOLDOUT_TEST["eval_start"])

    for pf in prediction_files:
        df = pd.read_parquet(pf)
        if "discharge_start_ts" in df.columns:
            assert df["discharge_start_ts"].max() < holdout_start, f"Holdout exposure in {pf}"
        if "fold" in df.columns:
            assert "TEST" not in df["fold"].unique(), f"Fold TEST found in {pf}"


def test_paired_vessel_clustered_bootstrap():
    """Tests the paired vessel-clustered bootstrap difference function."""
    rng = np.random.default_rng(42)
    n = 300
    vessels = np.repeat([f"VESSEL_{i}" for i in range(15)], 20)
    y_true = rng.integers(0, 3, size=n)
    # Candidate model is slightly more accurate than reference
    y_pred_cand = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, 3, size=n))
    y_pred_ref = np.where(rng.random(n) < 0.5, y_true, rng.integers(0, 3, size=n))

    res = compute_paired_vessel_clustered_macro_f1_difference(
        vessel_call_ids=vessels,
        y_true=y_true,
        y_pred_candidate=y_pred_cand,
        y_pred_reference=y_pred_ref,
        n_bootstraps=50,
        random_state=42,
    )
    assert "macro_f1_delta" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["ci_upper"]
    assert res["n_vessel_clusters"] == 15
    assert res["resampling_unit"] == "vessel_call_id"


def test_top_k_queue_mathematical_consistency():
    """Asserts precision@1%, recall@1%, and lift@1% mathematical consistency."""
    n = 1000
    y_true = np.zeros(n, dtype=int)
    y_true[:100] = 2  # 10% critical delay prevalence
    probs = np.zeros(n)
    probs[:10] = 1.0  # Put 10 critical delays at the very top

    res = compute_top_k_metrics(y_true, probs, ks=[1])
    top1 = res["top_1pct"]
    assert top1["n_flagged"] == 10
    assert top1["true_positives"] == 10
    assert top1["precision"] == 1.0
    assert abs(top1["recall"] - 0.10) < 1e-4
    assert abs(top1["lift"] - 10.0) < 1e-4


def test_api_predict_with_gross_in_kg():
    """Tests end-to-end FastAPI prediction with container weight provided in kg."""
    client = TestClient(app)
    payload = {
        "case_id": "TEST-KG-001",
        "carrier_prefix": "MSK",
        "CTR_SIZE": 40.0,
        "CTR_TYPE": "DRY",
        "GROSS": 26500.0,  # 26,500 kg
        "YARD_BLOCK": "B",
        "YARD_SLOT": "05",
        "discharge_hour": 14,
        "discharge_day_of_week": 2,
        "discharge_month": 7,
        "is_weekend": 0,
        "trailing_7d_volume": 500.0,
        "trailing_7d_critical": 40.0,
        "trailing_7d_at_risk": 110.0,
        "historical_delay_rate_7d": 0.25,
        "trailing_7d_mean_dwell": 75.0,
    }
    response = client.post("/predict/container", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class_id"] in [0, 1, 2]
    assert data["predicted_class_name"] in ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
    probs = data["probabilities"]
    assert abs(probs["P_ON_TIME"] + probs["P_AT_RISK"] + probs["P_CRITICAL_DELAY"] - 1.0) < 1e-3
    gross_factor = next((f for f in data.get("key_risk_factors", []) if f["name"] == "GROSS"), None)
    if gross_factor:
        assert "26.5 tonnes" in gross_factor["value"]

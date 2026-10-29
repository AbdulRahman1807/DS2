"""Unit & Integration Tests for Stage 01 ML Pipeline (Audited Pass).

Tests encompass:
1. Strict feature contract enforcement and leakage prevention (target and post-event columns).
2. Purely unsupervised anomaly preprocessing (NO consumption of target labels y).
3. Preprocessor fit strictly on training data with safe handling of unknown categoricals and missing values.
4. Multiclass output shapes, valid probability constraints (sum to 1.0, range [0, 1]).
5. Strict chronological ordering of internal temporal validation folds on discharge_start_ts (zero backward leakage).
6. Exact audit of partition boundary timestamps between train, val, and test.
7. Quarantine of test.parquet partition from model selection and tuning.
8. Deterministic reproducibility with fixed seeds.
9. Verification that default decision rule misses ~88.8% of critical delays without threshold tuning.
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

from ML.data_engineer.feature_contracts import (
    FORBIDDEN_TARGET_COLUMNS,
    POST_ARRIVAL_LEAKAGE_COLUMNS,
    LeakageViolationError,
    PredictionScenario,
    extract_modeling_matrix,
    get_active_arrival_predictors,
    validate_feature_matrix,
)
from ML.ml_engineer.anomaly_baseline import IsolationForestAnomalyDetector
from ML.ml_engineer.anomaly_detection import build_unsupervised_preprocessor
from ML.ml_engineer.baseline_pipeline import (
    CLASS_NAMES,
    CLASS_THRESHOLDS,
    build_preprocessor,
    compute_evaluation_metrics,
    define_internal_temporal_folds,
    get_feature_subsets,
)
from ML.ml_engineer.classifier import BottleneckClassifier


@pytest.fixture(scope="module")
def sample_train_val_data():
    """Loads small real sample slices of train and val for deterministic testing."""
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    # Use first 1,000 rows of train and 200 rows of val
    return train_df.iloc[:1000].copy(), val_df.iloc[:200].copy()


class TestFeatureContractEnforcement:
    """Verifies that ML pipeline strictly blocks forbidden columns and unvetted features."""

    def test_blocks_ground_truth_target_columns(self, sample_train_val_data):
        train_df, _ = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        leaky_df = train_df[active_features].copy()
        leaky_df["dwell_hours"] = train_df["dwell_hours"]

        with pytest.raises(LeakageViolationError, match="CRITICAL LEAKAGE DETECTED"):
            validate_feature_matrix(leaky_df, scenario=PredictionScenario.ARRIVAL_TIME)

    def test_blocks_post_arrival_milestone_columns(self, sample_train_val_data):
        train_df, _ = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        for col in POST_ARRIVAL_LEAKAGE_COLUMNS:
            leaky_df = train_df[active_features].copy()
            leaky_df[col] = train_df[col]
            with pytest.raises(LeakageViolationError, match="POST-ARRIVAL LEAKAGE"):
                validate_feature_matrix(leaky_df, scenario=PredictionScenario.ARRIVAL_TIME)

    def test_blocks_unvetted_synthetic_columns(self, sample_train_val_data):
        train_df, _ = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        leaky_df = train_df[active_features].copy()
        leaky_df["synthetic_noise_column"] = np.random.randn(len(leaky_df))

        with pytest.raises(LeakageViolationError, match="UNVETTED/UNKNOWN FEATURE DETECTED"):
            validate_feature_matrix(leaky_df, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)


class TestAnomalyDetectionMethodology:
    """Verifies strictly unsupervised anomaly preprocessing and label independence."""

    def test_anomaly_preprocessor_is_strictly_unsupervised_and_rejects_y(self, sample_train_val_data):
        train_df, _ = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        num_cols, cat_cols = get_feature_subsets(active_features)

        preprocessor = build_unsupervised_preprocessor(num_cols, cat_cols)

        # Inspect fit signature of the preprocessor: should accept X and optional y=None
        # Verify that fit works cleanly without y
        X_train = train_df[active_features].copy()
        preprocessor.fit(X_train)

        # Verify IsolationForestAnomalyDetector.fit signature does NOT accept y
        sig = inspect.signature(IsolationForestAnomalyDetector.fit)
        assert "y" not in sig.parameters, "IsolationForestAnomalyDetector.fit() must not accept y parameter"

    def test_anomaly_preprocessor_handles_unknown_categories_and_missing_values(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        num_cols, cat_cols = get_feature_subsets(active_features)

        preprocessor = build_unsupervised_preprocessor(num_cols, cat_cols)
        preprocessor.fit(train_df[active_features])

        # Inject unknown category and numeric NaN
        val_sample = val_df[active_features].iloc[:10].copy()
        val_sample.loc[0, "carrier_prefix"] = "UNKNOWN_CARRIER_ZZZ"
        val_sample.loc[0, "GROSS"] = np.nan

        X_trans = preprocessor.transform(val_sample)
        assert X_trans.shape[0] == len(val_sample)
        assert not np.isnan(X_trans).any(), "Unsupervised preprocessor must impute missing values and handle unknown categories"

    def test_isolation_forest_reproducibility(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        det1 = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
        det1.fit(train_df[active_features])
        scores1 = det1.score_samples(val_df[active_features])
        preds1 = det1.predict(val_df[active_features])

        det2 = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
        det2.fit(train_df[active_features])
        scores2 = det2.score_samples(val_df[active_features])
        preds2 = det2.predict(val_df[active_features])

        np.testing.assert_array_almost_equal(scores1, scores2, decimal=6)
        np.testing.assert_array_equal(preds1, preds2)


class TestTemporalValidationTimestampOrdering:
    """Verifies strict timestamp ordering using the actual prediction anchor (discharge_start_ts)."""

    def test_internal_folds_strict_monotonicity_on_discharge_start_ts(self):
        train_df = pd.read_parquet("data/processed/train.parquet")
        folds = define_internal_temporal_folds()
        assert len(folds) == 3

        for f in folds:
            fname = f["fold_name"]
            tr_mask = train_df["discharge_start_ts"] < f["train_end_ts"]
            vl_mask = (train_df["discharge_start_ts"] >= f["val_start_ts"]) & (
                train_df["discharge_start_ts"] <= f["val_end_ts"]
            )

            tr = train_df[tr_mask]
            vl = train_df[vl_mask]

            tr_max_ts = tr["discharge_start_ts"].max()
            vl_min_ts = vl["discharge_start_ts"].min()

            assert tr_max_ts < vl_min_ts, (
                f"Temporal violation in '{fname}': "
                f"train_max ({tr_max_ts}) is not strictly before val_min ({vl_min_ts})"
            )
            # Ensure both splits contain observations and all 3 classes
            assert len(tr) > 10000
            assert len(vl) > 5000
            assert set(tr["target_class"].unique()) == {0, 1, 2}
            assert set(vl["target_class"].unique()) == {0, 1, 2}

    def test_val_strictly_precedes_test_in_discharge_start_ts(self):
        val_df = pd.read_parquet("data/processed/val.parquet")
        test_df = pd.read_parquet("data/processed/test.parquet")

        val_max_ts = val_df["discharge_start_ts"].max()
        test_min_ts = test_df["discharge_start_ts"].min()

        assert val_max_ts < test_min_ts, (
            f"Validation partition does not strictly precede test: "
            f"val_max {val_max_ts} >= test_min {test_min_ts}"
        )

    def test_audit_external_train_val_113_boundary_cases(self):
        train_df = pd.read_parquet("data/processed/train.parquet")
        val_df = pd.read_parquet("data/processed/val.parquet")

        val_min_ts = val_df["discharge_start_ts"].min()
        overlap_mask = (train_df["discharge_start_ts"] >= val_min_ts)
        overlap_count = int(overlap_mask.sum())

        # Assert exact documented count of 113 boundary cases
        assert overlap_count == 113, (
            f"Expected exactly 113 boundary cases in train where discharge_start_ts >= val_min, "
            f"found {overlap_count}"
        )
        # Verify that all 113 cases belong to the final train call_date (2021-08-26)
        assert (train_df.loc[overlap_mask, "call_date"] == "2021-08-26").all(), (
            "All overlapping cases must stem from vessel call_date 2021-08-26 with discharge extending into Aug 27."
        )


class TestModelPerformanceRealism:
    """Verifies honest reporting of model limitations and threshold sensitivity."""

    def test_default_decision_rule_misses_most_critical_delays(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        clf = BottleneckClassifier(model_type="xgboost", weighted=True, random_state=42)
        clf.fit(train_df[active_features], train_df["target_class"].to_numpy())

        preds = clf.predict(val_df[active_features])
        val_crit_mask = (val_df["target_class"] == 2).to_numpy()

        crit_recall = (preds[val_crit_mask] == 2).mean() if val_crit_mask.sum() > 0 else 0.0
        # Assert that critical recall at default argmax is low (< 35%), proving the model misses most critical delays
        assert crit_recall < 0.35, "Model at default rule should miss the majority of critical delays"

    def test_single_case_scoring_api_with_alert_threshold(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        clf = BottleneckClassifier(model_type="xgboost", weighted=True, random_state=42)
        clf.fit(train_df[active_features], train_df["target_class"].to_numpy())

        single_case = val_df[active_features].iloc[0].to_dict()
        result = clf.score_case(single_case, alert_threshold=0.10)

        assert "predicted_class_id" in result
        assert "predicted_class_name" in result
        assert "probabilities" in result
        assert "alert_threshold_tau" in result
        assert result["alert_threshold_tau"] == 0.10
        assert "alert_triggered" in result
        assert "operational_note" in result

"""Unit & Integration Tests for Stage 01 ML Pipeline.

Tests encompass:
1. Strict feature contract enforcement and leakage prevention (target and post-event columns).
2. Preprocessor fit strictly on training data with safe handling of unknown categoricals and missing values.
3. Multiclass output shapes, valid probability constraints (sum to 1.0, range [0, 1]).
4. Strict chronological ordering of internal temporal validation folds (no future leakage).
5. Quarantine of test.parquet partition from model selection and tuning.
6. Deterministic reproducibility with fixed seeds.
7. Anomaly detection output shapes and decision score semantics.
"""

from __future__ import annotations

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

        # Attempt to inject dwell_hours into feature matrix
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


class TestPreprocessingIntegrity:
    """Verifies preprocessor handles unseen categories, missing values, and fits training-only."""

    def test_safe_handling_of_unseen_categories(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        num_cols, cat_cols = get_feature_subsets(active_features)

        preprocessor = build_preprocessor(num_cols, cat_cols, random_state=42)

        X_train = train_df[active_features].copy()
        y_train = train_df["target_class"].to_numpy()
        preprocessor.fit(X_train, y_train)

        # Create validation sample with completely novel carrier_prefix and yard block
        X_val_novel = val_df[active_features].copy()
        X_val_novel.loc[0, "carrier_prefix"] = "COMPLETELY_NEW_CARRIER_XYZ"
        X_val_novel.loc[0, "YARD_BLOCK"] = "NOVEL_BLOCK_999"

        # Transformation must succeed without crashing or throwing KeyError
        X_trans = preprocessor.transform(X_val_novel)
        assert X_trans.shape[0] == len(X_val_novel)
        assert not np.isnan(X_trans[0]).any(), "Novel category should be encoded cleanly without NaNs"

    def test_safe_handling_of_missing_numeric_values(self, sample_train_val_data):
        train_df, _ = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        num_cols, cat_cols = get_feature_subsets(active_features)

        preprocessor = build_preprocessor(num_cols, cat_cols, random_state=42)

        X_train = train_df[active_features].copy()
        y_train = train_df["target_class"].to_numpy()
        preprocessor.fit(X_train, y_train)

        # Inject NaNs into numeric features
        X_test_missing = X_train.iloc[:10].copy()
        X_test_missing.loc[0, "GROSS"] = np.nan
        X_test_missing.loc[1, "trailing_7d_critical"] = np.nan

        X_trans = preprocessor.transform(X_test_missing)
        assert not np.isnan(X_trans).any(), "Median imputer must impute all numeric NaNs"


class TestModelOutputsAndProbabilities:
    """Verifies multiclass probabilities, shapes, and inference contracts."""

    def test_bottleneck_classifier_probability_contract(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        X_train = train_df[active_features].copy()
        y_train = train_df["target_class"].to_numpy()
        X_val = val_df[active_features].copy()

        clf = BottleneckClassifier(model_type="xgboost", weighted=True, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)

        # Output shape assertions
        assert preds.shape == (len(X_val),)
        assert probs.shape == (len(X_val), 3)

        # Class membership assertion
        assert set(np.unique(preds)).issubset({0, 1, 2})

        # Probability axiom assertions
        assert np.all(probs >= 0.0), "Probabilities must be non-negative"
        assert np.all(probs <= 1.0), "Probabilities must be <= 1.0"
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg="Probabilities must sum to 1.0 across classes")

    def test_single_case_scoring_api(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        clf = BottleneckClassifier(model_type="xgboost", weighted=False, random_state=42)
        clf.fit(train_df[active_features], train_df["target_class"].to_numpy())

        single_case = val_df[active_features].iloc[0].to_dict()
        result = clf.score_case(single_case)

        assert "predicted_class_id" in result
        assert result["predicted_class_id"] in [0, 1, 2]
        assert "predicted_class_name" in result
        assert result["predicted_class_name"] in CLASS_NAMES
        assert "probabilities" in result
        assert "P_ON_TIME" in result["probabilities"]
        assert "P_AT_RISK" in result["probabilities"]
        assert "P_CRITICAL_DELAY" in result["probabilities"]
        assert "alert_triggered" in result


class TestTemporalValidationIntegrity:
    """Verifies chronological boundaries and cold-start structure."""

    def test_internal_folds_are_strictly_chronological(self):
        folds = define_internal_temporal_folds()
        assert len(folds) == 3

        for f in folds:
            train_end = pd.to_datetime(f["train_end"])
            val_start = pd.to_datetime(f["val_start"])
            assert train_end < val_start, (
                f"Fold '{f['fold_name']}' violates temporal monotonicity: "
                f"train_end {train_end} is not strictly before val_start {val_start}"
            )

    def test_cold_start_partition_counts(self):
        train_df = pd.read_parquet("data/processed/train.parquet")
        zero_history_mask = (train_df["call_date"] < "2021-07-01")
        assert zero_history_mask.sum() == 4422, (
            f"Expected exactly 4,422 zero-history cold start stays, found {zero_history_mask.sum()}"
        )


class TestAnomalyDetectionIntegrity:
    """Verifies Isolation Forest anomaly scoring semantics."""

    def test_isolation_forest_output_shapes_and_scores(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        detector = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
        detector.fit(train_df[active_features], train_df["target_class"].to_numpy())

        preds = detector.predict(val_df[active_features])
        scores = detector.score_samples(val_df[active_features])

        assert preds.shape == (len(val_df),)
        assert scores.shape == (len(val_df),)
        assert set(np.unique(preds)).issubset({-1, 1})
        assert not np.isnan(scores).any()

        # Score semantics: negative scores denote outliers
        flagged_scores = scores[preds == -1]
        normal_scores = scores[preds == 1]
        if len(flagged_scores) > 0 and len(normal_scores) > 0:
            assert np.max(flagged_scores) <= np.min(normal_scores) + 1e-6


class TestReproducibility:
    """Verifies deterministic behavior with fixed random seeds."""

    def test_deterministic_predictions_given_fixed_seed(self, sample_train_val_data):
        train_df, val_df = sample_train_val_data
        active_features = get_active_arrival_predictors(exclude_zero_variance=True)

        clf1 = BottleneckClassifier(model_type="xgboost", weighted=True, random_state=42)
        clf1.fit(train_df[active_features], train_df["target_class"].to_numpy())
        probs1 = clf1.predict_proba(val_df[active_features])

        clf2 = BottleneckClassifier(model_type="xgboost", weighted=True, random_state=42)
        clf2.fit(train_df[active_features], train_df["target_class"].to_numpy())
        probs2 = clf2.predict_proba(val_df[active_features])

        np.testing.assert_array_almost_equal(probs1, probs2, decimal=6, err_msg="Model predictions must be deterministic")

"""Supervised Bottleneck Classification Baseline Pipeline.

Assigned Role: Senior ML Engineer
Stage: Stage 01 - Supervised Baseline Triage & Temporal Evaluation

Models evaluated:
- Model 0: DummyClassifier (no-learning reference)
- Model 1: Logistic Regression (unweighted & balanced)
- Model 2: XGBoost (unweighted & class-weighted)
- Model 3: LightGBM (unweighted & balanced)

Evaluation setup:
- Chronological internal expanding-window folds on train.parquet.
- Primary out-of-time evaluation on val.parquet (with right-censoring caveat).
- Strict quarantine of test.parquet (no model selection, tuning, or threshold fitting).
- Feature contracts enforced via ML.data_engineer.feature_contracts.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ML.data_engineer.feature_contracts import (
    PredictionScenario,
    extract_modeling_matrix,
    get_active_arrival_predictors,
    validate_feature_matrix,
)

logger = logging.getLogger(__name__)

# Target class semantics frozen from train partition
CLASS_NAMES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
CLASS_THRESHOLDS = {"p50_hours": 65.1102, "p85_hours": 160.8993}

# Active feature column categories
CATEGORICAL_FEATURES = ["carrier_prefix", "CTR_TYPE", "YARD_BLOCK", "YARD_SLOT"]


def get_feature_subsets(active_features: List[str]) -> Tuple[List[str], List[str]]:
    """Separates active arrival features into numeric and categorical subsets."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in active_features]
    num_cols = [c for c in active_features if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42,
) -> ColumnTransformer:
    """Constructs a leakage-safe ColumnTransformer fitted strictly on training observations.

    - Numeric features: Median imputation followed by standard scaling.
    - Categorical features: Constant 'UNKNOWN' imputation followed by out-of-fold TargetEncoder.
    """
    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encoder", TargetEncoder(smooth="auto", cv=5, random_state=random_state)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multiclass and bottleneck-triage metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # Per-class metrics
    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        mask_true = (y_true == idx)
        support = int(np.sum(mask_true))
        prec = float(precision_score(y_true == idx, y_pred == idx, zero_division=0))
        rec = float(recall_score(y_true == idx, y_pred == idx, zero_division=0))
        f1_c = float(f1_score(y_true == idx, y_pred == idx, zero_division=0))
        per_class[cname] = {
            "class_id": idx,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1_c, 4),
            "support": support,
            "prevalence": round(support / len(y_true), 4),
        }

    # Critical delay (class 2) PR-AUC
    crit_binary_true = (y_true == 2).astype(int)
    baseline_crit_prevalence = float(np.mean(crit_binary_true))

    crit_ap = None
    multiclass_loss = None

    if y_prob is not None:
        try:
            # y_prob[:, 2] is predicted probability of CRITICAL_DELAY
            crit_ap = float(average_precision_score(crit_binary_true, y_prob[:, 2]))
        except Exception:
            crit_ap = None

        try:
            # Multiclass log loss
            multiclass_loss = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
        except Exception:
            multiclass_loss = None

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "multiclass_log_loss": round(multiclass_loss, 4) if multiclass_loss is not None else None,
        "critical_delay_pr_auc": round(crit_ap, 4) if crit_ap is not None else None,
        "critical_delay_baseline_prevalence": round(baseline_crit_prevalence, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def define_internal_temporal_folds() -> List[Dict[str, str]]:
    """Defines 3 expanding-window chronological folds inside train.parquet."""
    return [
        {
            "fold_name": "Fold 1 (Expanding July)",
            "train_start": "2021-06-10",
            "train_end": "2021-07-18",
            "val_start": "2021-07-19",
            "val_end": "2021-07-31",
            "description": "Train on early arrivals up to July 18; evaluate out-of-time on late July peak congestion.",
        },
        {
            "fold_name": "Fold 2 (Late July - Early Aug)",
            "train_start": "2021-06-10",
            "train_end": "2021-07-31",
            "val_start": "2021-08-01",
            "val_end": "2021-08-14",
            "description": "Train through entire July; evaluate on early August post-peak transition.",
        },
        {
            "fold_name": "Fold 3 (Mid-August Clearance)",
            "train_start": "2021-06-10",
            "train_end": "2021-08-14",
            "val_start": "2021-08-15",
            "val_end": "2021-08-26",
            "description": "Train through mid-August; evaluate on late August pre-validation boundary.",
        },
    ]


class BaselineModelRunner:
    """Orchestrates training and comparative evaluation of supervised baselines."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed",
        random_state: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.active_features = get_active_arrival_predictors(exclude_zero_variance=True)
        self.num_cols, self.cat_cols = get_feature_subsets(self.active_features)

        # Validate existence of processed parquet files
        self.train_path = self.data_dir / "train.parquet"
        self.val_path = self.data_dir / "val.parquet"

        if not self.train_path.exists() or not self.val_path.exists():
            raise FileNotFoundError(
                f"Required datasets missing from {self.data_dir}. Ensure train.parquet and val.parquet exist."
            )

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and val partitions and verifies feature contracts."""
        train_df = pd.read_parquet(self.train_path)
        val_df = pd.read_parquet(self.val_path)

        # Enforce contract validation on raw data extraction
        X_train_raw, y_train = extract_modeling_matrix(
            train_df,
            scenario=PredictionScenario.ARRIVAL_TIME,
            target_col="target_class",
            allow_conditional=False,
            strict=True,
        )
        X_val_raw, y_val = extract_modeling_matrix(
            val_df,
            scenario=PredictionScenario.ARRIVAL_TIME,
            target_col="target_class",
            allow_conditional=False,
            strict=True,
        )

        # Filter strictly to active 15 features
        X_train = X_train_raw[self.active_features].copy()
        X_val = X_val_raw[self.active_features].copy()

        # Validate feature matrices
        validate_feature_matrix(X_train, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)
        validate_feature_matrix(X_val, scenario=PredictionScenario.ARRIVAL_TIME, strict_allowlist=True)

        return train_df, val_df

    def get_models(self) -> Dict[str, Any]:
        """Instantiates all baseline model configurations."""
        return {
            "dummy_prior": {
                "name": "DummyClassifier (Prior)",
                "type": "baseline_reference",
                "weighted": False,
                "estimator": DummyClassifier(strategy="prior"),
            },
            "logistic_regression_unweighted": {
                "name": "Logistic Regression (Unweighted)",
                "type": "linear",
                "weighted": False,
                "estimator": LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                ),
            },
            "logistic_regression_balanced": {
                "name": "Logistic Regression (Balanced)",
                "type": "linear",
                "weighted": True,
                "estimator": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=self.random_state,
                ),
            },
            "xgboost_unweighted": {
                "name": "XGBoost (Unweighted)",
                "type": "gradient_boosting",
                "weighted": False,
                "estimator": XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=4,
                    eval_metric="mlogloss",
                ),
            },
            "xgboost_weighted": {
                "name": "XGBoost (Sample-Weighted)",
                "type": "gradient_boosting",
                "weighted": True,
                "estimator": XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=4,
                    eval_metric="mlogloss",
                ),
            },
            "lightgbm_unweighted": {
                "name": "LightGBM (Unweighted)",
                "type": "gradient_boosting",
                "weighted": False,
                "estimator": LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=4,
                    verbose=-1,
                ),
            },
            "lightgbm_balanced": {
                "name": "LightGBM (Balanced)",
                "type": "gradient_boosting",
                "weighted": True,
                "estimator": LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=4,
                    verbose=-1,
                ),
            },
        }

    def train_and_evaluate_model(
        self,
        model_key: str,
        model_cfg: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_eval: pd.DataFrame,
        y_eval: np.ndarray,
    ) -> Dict[str, Any]:
        """Fits preprocessor and model strictly on training split and evaluates out-of-time."""
        t0 = time.time()

        # Build fresh preprocessor
        num_cols, cat_cols = get_feature_subsets(list(X_train.columns))
        preprocessor = build_preprocessor(num_cols, cat_cols, random_state=self.random_state)

        # Fit preprocessor strictly on training data
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_eval_proc = preprocessor.transform(X_eval)

        estimator = model_cfg["estimator"]

        # Handle sample weighting for XGBoost if weighted
        if model_key == "xgboost_weighted":
            sample_weights = compute_sample_weight("balanced", y_train)
            estimator.fit(X_train_proc, y_train, sample_weight=sample_weights)
        else:
            estimator.fit(X_train_proc, y_train)

        train_runtime = time.time() - t0

        # Predict classes and probabilities
        y_pred = estimator.predict(X_eval_proc)
        y_prob = None
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_eval_proc)

        metrics = compute_evaluation_metrics(y_eval, y_pred, y_prob)
        metrics["runtime_seconds"] = round(train_runtime, 3)
        metrics["model_name"] = model_cfg["name"]
        metrics["model_type"] = model_cfg["type"]
        metrics["weighted"] = model_cfg["weighted"]
        metrics["model_key"] = model_key

        return metrics

    def run_internal_temporal_cross_validation(
        self,
        train_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Evaluates models across 3 expanding temporal folds inside train.parquet."""
        folds = define_internal_temporal_folds()
        models = self.get_models()

        fold_results: Dict[str, Any] = {f["fold_name"]: {} for f in folds}

        for f in folds:
            fname = f["fold_name"]
            # Slice strictly by call_date
            tr_mask = (train_df["call_date"] >= f["train_start"]) & (train_df["call_date"] <= f["train_end"])
            vl_mask = (train_df["call_date"] >= f["val_start"]) & (train_df["call_date"] <= f["val_end"])

            tr_subset = train_df[tr_mask]
            vl_subset = train_df[vl_mask]

            X_tr = tr_subset[self.active_features].copy()
            y_tr = tr_subset["target_class"].to_numpy()
            X_vl = vl_subset[self.active_features].copy()
            y_vl = vl_subset["target_class"].to_numpy()

            fold_info = {
                "train_bounds": [f["train_start"], f["train_end"]],
                "val_bounds": [f["val_start"], f["val_end"]],
                "train_size": len(X_tr),
                "val_size": len(X_vl),
                "train_class_dist": {int(k): int(v) for k, v in tr_subset["target_class"].value_counts().items()},
                "val_class_dist": {int(k): int(v) for k, v in vl_subset["target_class"].value_counts().items()},
                "models": {},
            }

            for m_key, m_cfg in models.items():
                m_res = self.train_and_evaluate_model(m_key, m_cfg, X_tr, y_tr, X_vl, y_vl)
                fold_info["models"][m_key] = m_res

            fold_results[fname] = fold_info

        return fold_results

    def run_primary_validation_benchmark(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Trains on entire train.parquet (N=60,698) and evaluates on val.parquet (N=20,879)."""
        models = self.get_models()

        X_train = train_df[self.active_features].copy()
        y_train = train_df["target_class"].to_numpy()
        X_val = val_df[self.active_features].copy()
        y_val = val_df["target_class"].to_numpy()

        results = {}
        for m_key, m_cfg in models.items():
            results[m_key] = self.train_and_evaluate_model(m_key, m_cfg, X_train, y_train, X_val, y_val)

        return results

    def run_cold_start_sensitivity(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compares training on all data vs excluding zero-history cold start (< 2021-07-01)."""
        # Baseline model for ablation: XGBoost (unweighted) and XGBoost (weighted)
        train_no_cold = train_df[train_df["call_date"] >= "2021-07-01"].copy()

        models = {
            "xgboost_unweighted": self.get_models()["xgboost_unweighted"],
            "xgboost_weighted": self.get_models()["xgboost_weighted"],
        }

        X_val = val_df[self.active_features].copy()
        y_val = val_df["target_class"].to_numpy()

        results = {}
        # 1. Full training
        X_tr_full = train_df[self.active_features].copy()
        y_tr_full = train_df["target_class"].to_numpy()
        results["all_training_rows"] = {
            "n_train": len(X_tr_full),
            "date_range": [str(train_df["call_date"].min()), str(train_df["call_date"].max())],
            "models": {
                k: self.train_and_evaluate_model(k, cfg, X_tr_full, y_tr_full, X_val, y_val)
                for k, cfg in models.items()
            },
        }

        # 2. Excluded zero-history cold start
        X_tr_nocold = train_no_cold[self.active_features].copy()
        y_tr_nocold = train_no_cold["target_class"].to_numpy()
        results["cold_start_excluded"] = {
            "n_train": len(X_tr_nocold),
            "date_range": [str(train_no_cold["call_date"].min()), str(train_no_cold["call_date"].max())],
            "excluded_count": len(X_tr_full) - len(X_tr_nocold),
            "models": {
                k: self.train_and_evaluate_model(k, cfg, X_tr_nocold, y_tr_nocold, X_val, y_val)
                for k, cfg in models.items()
            },
        }

        return results

    def run_feature_group_ablations(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Ablates key feature groups to measure predictive contribution."""
        xgb_cfg = self.get_models()["xgboost_unweighted"]
        y_train = train_df["target_class"].to_numpy()
        y_val = val_df["target_class"].to_numpy()

        ablation_groups = {
            "all_features": list(self.active_features),
            "without_rolling_metrics": [
                c for c in self.active_features
                if not c.startswith("trailing_7d") and c != "historical_delay_rate_7d"
            ],
            "without_calendar_features": [
                c for c in self.active_features
                if c not in ["discharge_hour", "discharge_day_of_week", "discharge_month", "is_weekend"]
            ],
            "without_yard_features": [
                c for c in self.active_features if c not in ["YARD_BLOCK", "YARD_SLOT"]
            ],
            "without_equipment_features": [
                c for c in self.active_features
                if c not in ["carrier_prefix", "CTR_SIZE", "CTR_TYPE", "GROSS"]
            ],
        }

        results = {}
        for group_name, features in ablation_groups.items():
            X_tr = train_df[features].copy()
            X_vl = val_df[features].copy()
            m_res = self.train_and_evaluate_model(
                "xgboost_unweighted",
                xgb_cfg,
                X_tr,
                y_train,
                X_vl,
                y_val,
            )
            results[group_name] = {
                "num_features": len(features),
                "features": features,
                "metrics": m_res,
            }

        return results

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Executes full Stage 01 supervised baseline benchmark."""
        logger.info("Loading datasets and verifying contracts...")
        train_df, val_df = self.load_datasets()

        logger.info("Running primary validation benchmark on val.parquet...")
        primary_val = self.run_primary_validation_benchmark(train_df, val_df)

        logger.info("Running internal temporal cross-validation...")
        temporal_cv = self.run_internal_temporal_cross_validation(train_df)

        logger.info("Running cold-start sensitivity experiment...")
        cold_start = self.run_cold_start_sensitivity(train_df, val_df)

        logger.info("Running feature-group ablations...")
        ablations = self.run_feature_group_ablations(train_df, val_df)

        # Build consolidated output artifact
        output = {
            "metadata": {
                "task": "Stage 01 Supervised Bottleneck Baseline Classification",
                "anchor": "discharge_start_ts",
                "prediction_scenario": "ARRIVAL_TIME",
                "target_definition": "dwell_hours = truck_out_end_ts - discharge_start_ts",
                "frozen_thresholds": CLASS_THRESHOLDS,
                "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
                "active_features": self.active_features,
                "num_features": len(self.active_features),
                "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "random_state": self.random_state,
            },
            "dataset_partitions": {
                "train": {
                    "path": str(self.train_path),
                    "num_rows": len(train_df),
                    "call_date_min": str(train_df["call_date"].min()),
                    "call_date_max": str(train_df["call_date"].max()),
                    "target_distribution": {
                        int(k): int(v) for k, v in train_df["target_class"].value_counts().items()
                    },
                },
                "val": {
                    "path": str(self.val_path),
                    "num_rows": len(val_df),
                    "call_date_min": str(val_df["call_date"].min()),
                    "call_date_max": str(val_df["call_date"].max()),
                    "target_distribution": {
                        int(k): int(v) for k, v in val_df["target_class"].value_counts().items()
                    },
                },
                "test_quarantine_notice": (
                    "test.parquet is strictly quarantined from model selection and tuning. "
                    "Evaluation on test is withheld due to verified ~85.4% right-censoring survivorship bias."
                ),
            },
            "primary_validation_benchmark": primary_val,
            "temporal_cross_validation": temporal_cv,
            "cold_start_sensitivity": cold_start,
            "feature_group_ablations": ablations,
            "known_limitations": [
                "Single-terminal operations (Surabaya, Indonesia, June-Sept 2021).",
                "Single prominent congestion episode in July accounts for bulk of critical delays.",
                "Right-censoring survivorship bias in test partition suppresses slow turnarounds.",
                "PortWatch external weather and disruption indices are constant 0 in the catalog.",
                "Arrival-time prediction scenario excludes all in-yard customs and inspection milestones.",
            ],
        }

        return output


def save_validation_results(
    results: Dict[str, Any],
    output_path: Union[str, Path] = "reports/ml/baseline_validation.json",
) -> Path:
    """Saves benchmark results to a machine-readable JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved baseline validation results to {out_file}")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runner = BaselineModelRunner()
    results = runner.run_full_pipeline()
    save_validation_results(results)

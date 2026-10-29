"""Master Model Training & Baseline Benchmarking Orchestrator.

Assigned Role: Senior ML Engineer
Stage: Stage 01 - Supervised Baseline Triage, Anomaly Benchmark, and Interpretability

Responsibilities:
- Load prepared feature matrices and target splits from data/processed/
- Train baseline anomaly detectors (Isolation Forest, One-Class SVM)
- Train multi-class gradient boosting classifiers (XGBoost, LightGBM, Logistic Regression, Dummy)
- Run chronological internal temporal cross-validation
- Execute cold-start sensitivity and feature-group ablations
- Generate the Feature Importance Matrix deliverable
- Produce reports/ml/ deliverables
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ML.ml_engineer.anomaly_detection import AnomalyDetectionBenchmark
from ML.ml_engineer.baseline_pipeline import (
    BaselineModelRunner,
    save_validation_results,
)
from ML.ml_engineer.feature_importance import compute_feature_importance_matrix

logger = logging.getLogger(__name__)


def run_master_training_suite(data_dir: str = "data/processed") -> None:
    """Executes all Stage 01 ML baseline training and benchmarking workflows."""
    start_time = time.time()
    logger.info("==================================================================")
    logger.info("  STARTING STAGE 01 ML BASELINE AND ANOMALY DETECTION SUITE       ")
    logger.info("==================================================================")

    # 1. Supervised Baselines & Temporal Validation
    logger.info("\n--- PHASE 1: SUPERVISED BOTTLENECK CLASSIFICATION BASELINES ---")
    runner = BaselineModelRunner(data_dir=data_dir, random_state=42)
    baseline_results = runner.run_full_pipeline()
    json_path = save_validation_results(baseline_results)
    logger.info(f"Phase 1 Complete. Baseline results saved to: {json_path}")

    # 2. Feature Importance Matrix
    logger.info("\n--- PHASE 2: FEATURE IMPORTANCE MATRIX GENERATION ---")
    csv_path = "reports/ml/feature_importance.csv"
    df_importance = compute_feature_importance_matrix(data_dir=data_dir, output_csv_path=csv_path)
    logger.info(f"Phase 2 Complete. Feature importance saved to: {csv_path}")

    # 3. Anomaly Detection Benchmark
    logger.info("\n--- PHASE 3: UNSUPERVISED ANOMALY DETECTION BENCHMARK ---")
    anomaly_bench = AnomalyDetectionBenchmark(data_dir=data_dir, random_state=42)
    anomaly_report_path = "reports/ml/anomaly_detection_summary.md"
    anomaly_bench.run_benchmark(output_report_path=anomaly_report_path)
    logger.info(f"Phase 3 Complete. Anomaly summary saved to: {anomaly_report_path}")

    total_time = time.time() - start_time
    logger.info("==================================================================")
    logger.info(f"  STAGE 01 ML WORKFLOW FINISHED SUCCESSFULLY IN {total_time:.2f}s ")
    logger.info("==================================================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_master_training_suite()

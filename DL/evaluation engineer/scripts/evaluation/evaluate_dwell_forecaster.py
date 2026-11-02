"""
Standalone Evaluation Script for Weekly Anchorage Waiting-Time Forecaster
==========================================================================
Milestone 3.6 Phase B: Dwell and Waiting-Time Forecasting

Loads validation and test predictions, prints detailed comparative metrics,
per-port performance breakdowns, and verifies residual improvement.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parents[1]
DL_ROOT = SCRIPT_DIR.parents[2]
ML_ROOT = DL_ROOT / "ml engineer"
ARTIFACTS_DIR = ML_ROOT / "artifacts" / "dwell_forecasting"


def main():
    print("=" * 80)
    print("Milestone 3.6 Phase B: Dwell Forecaster Independent Evaluation")
    print("=" * 80)

    val_path = ARTIFACTS_DIR / "validation_predictions.parquet"
    test_path = ARTIFACTS_DIR / "test_predictions.parquet"
    metrics_path = ARTIFACTS_DIR / "evaluation_metrics.json"

    if not val_path.exists() or not test_path.exists():
        print("Prediction artifacts not found. Please run train_dwell_forecaster.py first.")
        return

    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print(f"Checkpoint SHA256: {metrics['checkpoint_sha256']}")
    print(f"Trainable Parameters: {metrics['trainable_parameters']}")
    print(f"Runtime: {metrics['runtime_seconds']}s")

    print("\n--- 1. Validation 2021 (Crisis Surge Period, N=1,007) ---")
    val_m = metrics["validation_evaluation_2021"]["model_metrics_h1"]
    val_p = metrics["validation_evaluation_2021"]["persistence_metrics_h1"]
    print(f"Naive Persistence -> MAE: {val_p['mae']:.4f}h | RMSE: {val_p['rmse']:.4f}h | Huber: {val_p['huber']:.4f} | R2: {val_p['r2']:.4f}")
    print(f"Residual Dwell GRU -> MAE: {val_m['mae']:.4f}h | RMSE: {val_m['rmse']:.4f}h | Huber: {val_m['huber']:.4f} | R2: {val_m['r2']:.4f}")
    print(f"Relative Improvement: MAE: {metrics['validation_evaluation_2021']['mae_reduction_pct']}% | RMSE: {metrics['validation_evaluation_2021']['rmse_reduction_pct']}% | Huber: {metrics['validation_evaluation_2021']['huber_reduction_pct']}%")

    print("\n--- 2. Held-Out Test 2022 (Stabilization Period, N=931) ---")
    test_m = metrics["test_held_out_evaluation_2022"]["model_metrics_h1"]
    test_p = metrics["test_held_out_evaluation_2022"]["persistence_metrics_h1"]
    print(f"Naive Persistence -> MAE: {test_p['mae']:.4f}h | RMSE: {test_p['rmse']:.4f}h | Huber: {test_p['huber']:.4f} | R2: {test_p['r2']:.4f}")
    print(f"Residual Dwell GRU -> MAE: {test_m['mae']:.4f}h | RMSE: {test_m['rmse']:.4f}h | Huber: {test_m['huber']:.4f} | R2: {test_m['r2']:.4f}")
    print(f"Relative Improvement: MAE: {metrics['test_held_out_evaluation_2022']['mae_reduction_pct']}% | RMSE: {metrics['test_held_out_evaluation_2022']['rmse_reduction_pct']}% | Huber: {metrics['test_held_out_evaluation_2022']['huber_reduction_pct']}%")

    print("\n--- 3. Top Port Breakdown (Validation 2021) ---")
    print(f"{'Port':<18} {'Mean Actual':>12} {'Naive MAE':>12} {'Model MAE':>12} {'Delta MAE':>12} {'Status':>10}")
    print("-" * 72)
    per_port = metrics["validation_evaluation_2021"]["per_port"]
    for p, stats in sorted(per_port.items(), key=lambda x: x[1]["mean_actual"], reverse=True):
        diff = stats["naive_mae"] - stats["model_mae"]
        status = "BETTER" if diff > 0 else "TIED/SLOWER"
        print(f"{p:<18} {stats['mean_actual']:>11.1f}h {stats['naive_mae']:>11.2f}h {stats['model_mae']:>11.2f}h {diff:>+11.2f}h {status:>10}")

    print("\n--- 4. 4-Week Horizon Benchmark (Validation 2021) ---")
    h4_m = metrics["validation_4week_horizon"]["model"]
    h4_p = metrics["validation_4week_horizon"]["naive"]
    print(f"Naive Persistence -> MAE: {h4_p['mae']:.4f}h | RMSE: {h4_p['rmse']:.4f}h | R2: {h4_p['r2']:.4f}")
    print(f"Residual Dwell GRU -> MAE: {h4_m['mae']:.4f}h | RMSE: {h4_m['rmse']:.4f}h | R2: {h4_m['r2']:.4f}")
    print(f"4-Week MAE Reduction: {((h4_p['mae'] - h4_m['mae']) / h4_p['mae'] * 100):.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()


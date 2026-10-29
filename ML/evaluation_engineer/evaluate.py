"""Model Evaluation CLI.

Assigned Role: Evaluation Engineer
Usage:
    python ML/evaluation_engineer/evaluate.py --model models/bottleneck_classifier.joblib --test data/processed/test.parquet
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline and classifier models on test split.")
    parser.add_argument("--model", type=str, default="models/bottleneck_classifier.joblib")
    parser.add_argument("--test", type=str, default="data/processed/test.parquet")
    parser.add_argument("--output-report", type=str, default="reports/evaluation_report.json")
    args = parser.parse_args()

    print(f"Evaluation Engineer running test benchmark on {args.test}...")
    # TODO (Evaluation Engineer):
    # 1. Load serialized model artifact and test dataset.
    # 2. Compute EvaluationMetricsEngine reports (macro F1, confusion matrix).
    # 3. Run CostMatrixOptimizer to recommend the squad false-alarm operating threshold.
    # 4. Save evaluation summary to args.output_report.
    raise NotImplementedError("To be implemented by Evaluation Engineer")


if __name__ == "__main__":
    main()

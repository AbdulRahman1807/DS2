"""Extraction-Design Simulation: exit-window vs completed-only vs censoring-aware arrival cohorts.

Assigned Role: ML Research (Stage 03 Data Recovery)

Pretends the source extract was taken at an earlier cutoff C (true outcomes are known from the
current data because the selection-aware cohort has >= 600 h follow-up) and measures how well each
extraction design recovers the true class mix of arrivals in [2021-07-01, C).

Usage:
    python ML/ml_engineer/reassessment/simulate_extraction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from ML.ml_engineer.reassessment.censoring import (
    complete_label_arrival_window,
    exit_window_extract,
    kaplan_meier_class_probabilities,
    label_status,
)
from ML.ml_engineer.reassessment.cohort import COHORT_START

CUTOFFS = ["2021-08-10", "2021-08-20", "2021-08-30"]
OUT_PATH = REPO_ROOT / "reports/ml/reassessment/extraction_simulation.json"


def class_mix(labels: pd.Series) -> list:
    return [round(float((labels == k).mean()), 4) for k in (0, 1, 2)]


def simulate(cases: pd.DataFrame, cutoff: str) -> dict:
    C = pd.Timestamp(cutoff)
    d0, out = cases["discharge_start_ts"], cases["truck_out_end_ts"]
    arrivals = (d0 >= COHORT_START) & (d0 < C)
    recent = arrivals & (d0 > complete_label_arrival_window(C))
    truth = cases["target_class"]

    status = label_status(d0, out, C)
    exit_window = exit_window_extract(d0, out, COHORT_START, C)
    completed_only = arrivals & status["observed_exit"]
    complete_window = arrivals & ~recent

    km_recent = kaplan_meier_class_probabilities(
        status.loc[recent, "censored_dwell_h"].to_numpy(), status.loc[recent, "observed_exit"].to_numpy()
    )
    return {
        "cutoff": cutoff,
        "true_mix_all_arrivals": class_mix(truth[arrivals]),
        "true_mix_recent_arrivals": class_mix(truth[recent]),
        "n_arrivals": int(arrivals.sum()),
        "n_recent_arrivals_label_incomplete": int(recent.sum()),
        "exit_window_published_design": {
            "n": int(exit_window.sum()),
            "share_rows_discharged_before_window": round(float((d0[exit_window] < COHORT_START).mean()), 4),
            "mix": class_mix(truth[exit_window]),
            "mix_recent_arrivals": class_mix(truth[exit_window & recent]),
        },
        "completed_only_arrival_cohort": {
            "n": int(completed_only.sum()),
            "mix": class_mix(truth[completed_only]),
            "mix_recent_arrivals": class_mix(truth[completed_only & recent]),
        },
        "censoring_aware_arrival_cohort": {
            "n_complete_labels": int(complete_window.sum()),
            "mix_complete_label_window": class_mix(status.loc[complete_window, "label"]),
            "true_mix_complete_label_window": class_mix(truth[complete_window]),
            "recent_label_breakdown": {
                "exact_or_critical": int(status.loc[recent, "label_known"].sum()),
                "partial_at_risk_or_critical": int((status.loc[recent, "label"] == -2).sum()),
                "unknown": int((status.loc[recent, "label"] == -1).sum()),
            },
            "recent_kaplan_meier_mix": [round(float(p), 4) for p in km_recent],
            "unfinished_at_cutoff": int((arrivals & ~status["observed_exit"]).sum()),
        },
    }


def main() -> dict:
    cases = pd.read_parquet(REPO_ROOT / "data/interim/reassessment_case_table.parquet")
    results = {c: simulate(cases, c) for c in CUTOFFS}
    OUT_PATH.write_text(json.dumps(results, indent=2))
    for c, r in results.items():
        print(c, "truth", r["true_mix_all_arrivals"], "| exit-window", r["exit_window_published_design"]["mix"],
              "| completed-only", r["completed_only_arrival_cohort"]["mix"],
              "| censoring-aware (complete window)", r["censoring_aware_arrival_cohort"]["mix_complete_label_window"],
              "vs truth", r["censoring_aware_arrival_cohort"]["true_mix_complete_label_window"])
        print("   recent arrivals truth", r["true_mix_recent_arrivals"], "| completed-only",
              r["completed_only_arrival_cohort"]["mix_recent_arrivals"], "| KM", r["censoring_aware_arrival_cohort"]["recent_kaplan_meier_mix"])
    return results


if __name__ == "__main__":
    main()

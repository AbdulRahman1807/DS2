"""Consolidated Experiment Registry Builder.

Assigned Role: ML Research (Stage 03 Forensic Reassessment)

Writes reports/ml/experiment_registry.csv from:
- the pre-reassessment registry, preserved verbatim at reports/ml/archive/experiment_registry_pre_stage03.csv
  and annotated with provenance (code available? reproduced?),
- reports/ml/reassessment/results.json (every Stage 03 row is scored from a saved prediction file).

Usage:
    python ML/ml_engineer/reassessment/build_registry.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
import pandas as pd

REGISTRY = REPO_ROOT / "reports/ml/experiment_registry.csv"
ARCHIVE = REPO_ROOT / "reports/ml/archive/experiment_registry_pre_stage03.csv"
RESULTS = REPO_ROOT / "reports/ml/reassessment/results.json"

REPRODUCED = {"EXP-BASE-01": "R-BASE-01", "EXP-BASE-01-OPT-MACRO": "R-BASE-01-MULT"}
COLUMNS = [
    "exp_id", "stage", "family", "model_name", "formulation", "population", "eval_protocol",
    "macro_f1", "mean_fold_macro_f1", "macro_f1_ci95", "accuracy", "balanced_accuracy", "weighted_f1",
    "ontime_f1", "atrisk_f1", "crit_prec", "crit_rec", "crit_f1", "critical_pr_auc", "confusion_matrix",
    "open_subset_macro_f1", "median_lead_time_h", "production_valid", "code_available", "reproduced",
    "predictions_file", "hypothesis", "notes",
]


def legacy_rows() -> pd.DataFrame:
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REGISTRY, ARCHIVE)
    old = pd.read_csv(ARCHIVE)
    rows = pd.DataFrame({
        "exp_id": old["exp_id"],
        "stage": "stage01-02 (pre-reassessment)",
        "family": old["family"],
        "model_name": old["model_name"],
        "formulation": "A (legacy arrival features)",
        "population": "Stage01 val.parquet (selection-biased)",
        "eval_protocol": "single train->val split; *-OPT-* multipliers tuned on the same val set",
        "macro_f1": old["macro_f1"],
        "accuracy": old["accuracy"],
        "balanced_accuracy": old["balanced_accuracy"],
        "weighted_f1": old["weighted_f1"],
        "ontime_f1": old["ontime_f1"],
        "atrisk_f1": old["atrisk_f1"],
        "crit_prec": old["crit_prec"],
        "crit_rec": old["crit_rec"],
        "crit_f1": old["crit_f1"],
        "critical_pr_auc": old["critical_pr_auc"],
        "confusion_matrix": old["confusion_matrix"],
        "notes": old["notes"],
    })
    rows["code_available"] = rows["exp_id"].map(lambda e: "yes" if e in REPRODUCED else "no")
    rows["reproduced"] = rows["exp_id"].map(lambda e: f"yes (exact, {REPRODUCED[e]})" if e in REPRODUCED else "not reproducible")
    rows["production_valid"] = rows["exp_id"].map(
        lambda e: "no" if ("CEIL" in e or "OPT" in e) else "unverified"
    )
    rows.loc[rows["exp_id"] == "EXP-BASE-01", "production_valid"] = "yes (legacy protocol)"
    return rows


def stage03_rows() -> pd.DataFrame:
    results = json.loads(RESULTS.read_text())
    out = []
    for exp_id, r in results.items():
        p = r["pooled"]
        pc = p["per_class"]
        out.append({
            "exp_id": exp_id,
            "stage": "stage03 reassessment",
            "family": "XGBoost" if exp_id.startswith(("R-BASE", "R-A0")) else "LightGBM",
            "model_name": r.get("model", "xgboost_balanced_depth5"),
            "formulation": r["formulation"],
            "population": r["population"],
            "eval_protocol": "legacy val.parquet" if exp_id.startswith("R-BASE")
            else "selection-aware cohort, 3 embargoed rolling-origin dev folds (holdout TEST untouched)",
            "macro_f1": round(p["macro_f1"], 4),
            "mean_fold_macro_f1": round(r["mean_fold_macro_f1"], 4),
            "macro_f1_ci95": r.get("pooled_macro_f1_ci95_vessel_bootstrap") and
            "[%.3f, %.3f]" % tuple(r["pooled_macro_f1_ci95_vessel_bootstrap"]),
            "accuracy": round(p["accuracy"], 4),
            "balanced_accuracy": round(p["balanced_accuracy"], 4),
            "weighted_f1": round(p["weighted_f1"], 4),
            "ontime_f1": round(pc["ON_TIME"]["f1"], 4),
            "atrisk_f1": round(pc["AT_RISK"]["f1"], 4),
            "crit_prec": round(pc["CRITICAL_DELAY"]["precision"], 4),
            "crit_rec": round(pc["CRITICAL_DELAY"]["recall"], 4),
            "crit_f1": round(pc["CRITICAL_DELAY"]["f1"], 4),
            "critical_pr_auc": p["critical_pr_auc"] and round(p["critical_pr_auc"], 4),
            "confusion_matrix": json.dumps(p["confusion_matrix"]),
            "open_subset_macro_f1": r.get("three_class_open_subset", {}).get("macro_f1") and
            round(r["three_class_open_subset"]["macro_f1"], 4),
            "median_lead_time_h": r.get("median_lead_time_h") and round(r["median_lead_time_h"], 1),
            "production_valid": "yes" if r["production_valid"]
            else "no (uses future information)" if ("ORACLE" in exp_id or "VOI" in exp_id)
            else "no (decision rule tuned on the reporting set)",
            "code_available": "yes",
            "reproduced": "scored from saved predictions",
            "predictions_file": r["predictions"],
            "hypothesis": r["hypothesis"],
            "notes": "dropped: " + ",".join(r.get("dropped", [])) if r.get("dropped") else "",
        })
    return pd.DataFrame(out)


def main() -> Path:
    registry = pd.concat([legacy_rows(), stage03_rows()], ignore_index=True)[COLUMNS]
    registry.to_csv(REGISTRY, index=False)
    print(f"Wrote {len(registry)} rows to {REGISTRY}")
    return REGISTRY


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

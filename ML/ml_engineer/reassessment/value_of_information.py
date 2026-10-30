"""Value-of-Information Analysis and Formulation Feasibility.

Assigned Role: ML Research (Stage 03 Data Recovery)

Answers "how good would a forecast of each missing process event have to be?" without new data:
the realised event time (future information) is injected at prediction time with Gaussian error
of standard deviation sigma hours, emulating a forecaster built from customs / delivery-order /
truck-appointment data. The model (run_reassessment.lgbm_classifier) is fixed; nothing is tuned.
All rows are information bounds, NOT production results.

Information sets (relative to discharge):
- C:     customs release (date + 24 h, the only precision in the extract)
- CJ:    + delivery-job creation (JOB_DEL)
- CJP:   + pickup (TRUCK_IN, proxy for a truck appointment)

Also writes the formulation feasibility table (eligible population, share with all three classes
still open, median lead time) and re-runs R-A1-ASOF to confirm saved predictions reproduce.

Usage:
    python ML/ml_engineer/reassessment/value_of_information.py
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

from ML.ml_engineer.reassessment import run_reassessment as rr
from ML.ml_engineer.reassessment.cohort import DEV_FOLDS, P50_HOURS, P85_HOURS

SIGMAS = [0, 6, 12, 24, 48]
INFO_SETS = {
    "C": ["CUSTOMS_DEL"],
    "CJ": ["CUSTOMS_DEL", "JOB_DEL"],
    "CJP": ["CUSTOMS_DEL", "JOB_DEL", "TRUCK_IN"],
}
LANDMARK_CHECKS = [("B24", "CJ", 12), ("B24", "CJP", 12), ("B48", "CJ", 12)]
FEASIBILITY_PATH = REPO_ROOT / "reports/ml/reassessment/formulation_feasibility.csv"


def noisy_event_columns(rows: pd.DataFrame, events, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = pd.Timedelta(hours=1)
    cols = {}
    for ev in events:
        rel = ((rows[f"avail_{ev}"] - rows["discharge_start_ts"]) / hour).to_numpy()
        cols[f"forecast_{ev}_rel_h"] = rel + rng.normal(0.0, sigma, len(rel)) if sigma else rel
    return pd.DataFrame(cols)


def run_voi(cases: pd.DataFrame, formulation: str, info: str, sigma: float) -> Path:
    exp_id = f"R-VOI-{formulation}-{info}-s{sigma}"
    frames = []
    for i, fold in enumerate(DEV_FOLDS):
        tr_rows, tr_t, X_tr, ev_rows, ev_t, X_ev = rr.fold_matrices(cases, fold, formulation)
        X_tr = pd.concat([X_tr, noisy_event_columns(tr_rows, INFO_SETS[info], sigma, 100 + i)], axis=1)
        X_ev = pd.concat([X_ev, noisy_event_columns(ev_rows, INFO_SETS[info], sigma, 200 + i)], axis=1)
        y_pred, proba, _ = rr.fit_predict_lgbm(X_tr, tr_rows["target_class"].to_numpy(), X_ev)
        frames.append(rr.prediction_frame(ev_rows, ev_t, y_pred, proba, fold["name"]))
    return rr.save_predictions(exp_id, frames)


def formulation_feasibility(cases: pd.DataFrame) -> pd.DataFrame:
    """Population accounting per prediction time on the development evaluation windows."""
    eval_mask = pd.Series(False, index=cases.index)
    for fold in DEV_FOLDS:
        eval_mask |= rr.fold_masks(cases, fold)[1]
    sub = cases[eval_mask]
    d0, out = sub["discharge_start_ts"], sub["truck_out_end_ts"]
    hour = pd.Timedelta(hours=1)
    times = {
        "A: discharge": d0,
        "B12: discharge + 12h": d0 + 12 * hour,
        "B24: discharge + 24h": d0 + 24 * hour,
        "B48: discharge + 48h": d0 + 48 * hour,
        "C: customs release known (date + 24h)": np.maximum(d0, sub["avail_CUSTOMS_DEL"]),
        "E: delivery job created": np.maximum(d0, sub["avail_JOB_DEL"]),
        "T: truck arrival (appointment proxy)": np.maximum(d0, sub["avail_TRUCK_IN"]),
    }
    rows = []
    for name, t in times.items():
        elig = t.notna() & (out > t)
        elapsed = (t - d0) / hour
        open3 = elig & (elapsed < P50_HOURS)
        rows.append({
            "formulation": name,
            "eligible_share_of_arrivals": round(float(elig.mean()), 4),
            "open_three_class_share_of_arrivals": round(float(open3.mean()), 4),
            "open_share_of_eligible": round(float(open3.sum() / max(elig.sum(), 1)), 4),
            "critical_still_possible_share_of_eligible": round(float((elig & (elapsed < P85_HOURS)).sum() / max(elig.sum(), 1)), 4),
            "critical_share_in_open_population": round(float((sub.loc[open3, "target_class"] == 2).mean()), 4),
            "median_lead_time_h_open": round(float(((out - t) / hour)[open3].median()), 1),
        })
    return pd.DataFrame(rows)


def main() -> dict:
    cases = pd.read_parquet(rr.CASE_TABLE_PATH)
    feas = formulation_feasibility(cases)
    feas.to_csv(FEASIBILITY_PATH, index=False)
    print(feas.to_string(index=False))

    results_path = rr.OUT_DIR / "results.json"
    results = json.loads(results_path.read_text())

    # Reproducibility: re-run R-A1-ASOF under a new id and compare with the saved array.
    saved = pd.read_parquet(rr.PRED_DIR / "R-A1-ASOF.parquet")
    rerun = pd.read_parquet(rr.run_asof_experiment(cases, "R-A1-ASOF-RERUN", "A"))
    identical = bool((saved["y_pred"].to_numpy() == rerun["y_pred"].to_numpy()).all())
    max_prob_diff = float(np.abs(saved[["p_0", "p_1", "p_2"]].to_numpy() - rerun[["p_0", "p_1", "p_2"]].to_numpy()).max())
    print(f"R-A1-ASOF rerun identical predictions: {identical}, max |dp| = {max_prob_diff:.2e}")
    (rr.PRED_DIR / "R-A1-ASOF-RERUN.parquet").unlink()
    (rr.OUT_DIR / "importance_R-A1-ASOF-RERUN.csv").unlink(missing_ok=True)

    runs = [("A", info, s) for info in INFO_SETS for s in SIGMAS] + LANDMARK_CHECKS
    for formulation, info, sigma in runs:
        path = run_voi(cases, formulation, info, sigma)
        exp_id = path.stem
        results[exp_id] = {
            "formulation": f"{formulation} + forecast({info}, sigma={sigma}h)",
            "model": "lgbm",
            "population": "selection-aware cohort, dev folds, still in yard at t",
            "production_valid": False,
            "hypothesis": f"Value of information: {info} event times known at {formulation} with {sigma} h forecast error.",
            "predictions": str(path.relative_to(REPO_ROOT)),
            **rr.score_prediction_file(path),
        }
        print(f"{exp_id:24s} mean-fold macro-F1 = {results[exp_id]['mean_fold_macro_f1']:.4f}")
        results_path.write_text(json.dumps(results, indent=2, default=str))
    return {"reproducible": identical, "max_prob_diff": max_prob_diff}


if __name__ == "__main__":
    main()

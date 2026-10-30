"""Ported and Verified Sequential-Transition, Survival and Yard/Vessel Feature Experiments.

Assigned Role: ML Research (Stage 03)

Ports the external investigation (scratch_run_b24_b48.py, test_candidate_features.py) into the
reassessment framework: saved prediction arrays, registry rows, perturbation-tested features.
Each original configuration is reproduced first, then a corrected variant is run:
- Sequential transition model: original stacks *in-sample* stage predictions; corrected variant
  uses purged time-blocked out-of-fold stage predictions (stage_pipeline.OOF_*).
- Survival AFT: original re-conditions on S(elapsed) although the landmark training population
  is already conditional on survival; corrected variant does not re-condition.
- Vessel manifest: original whole-call aggregates use containers discharged after t; corrected
  as-of version (asof_features.manifest_state) also converts kg GROSS values to tonnes.

Usage:
    python ML/ml_engineer/reassessment/sequential_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from scipy.stats import norm

from ML.ml_engineer.reassessment import run_reassessment as rr
from ML.ml_engineer.reassessment import stage_pipeline as sp
from ML.ml_engineer.reassessment.asof_features import (
    build_asof_features,
    manifest_state,
    manifest_state_whole_call,
    yard_state,
    _ns,
)
from ML.ml_engineer.reassessment.cohort import DEV_FOLDS, P50_HOURS, P85_HOURS, fold_masks

SEED = 42
HOUR = pd.Timedelta(hours=1)
YARD_FAMILIES = {
    "yard_zone": ["yard_zone"],
    "slot_depth": ["slot_depth_tier"],
    "block_inventory": ["block_inventory", "block_inventory_share"],
}


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def transitions(rows: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """Sequential remaining durations from t: to customs, customs->job, job->pickup, pickup->exit."""
    out = rows["truck_out_end_ts"]
    c = rows["avail_CUSTOMS_DEL"].fillna(out)
    j = rows["avail_JOB_DEL"].fillna(out)
    p = rows["avail_TRUCK_IN"].fillna(out)
    t_c = np.maximum(t, c)
    t_j = np.maximum(t_c, j)
    t_p = np.maximum(t_j, p)
    return pd.DataFrame({
        "delta_cust_log": np.log1p(np.maximum(0.0, (c - t) / HOUR)),
        "delta_job_log": np.log1p(np.maximum(0.0, (j - t_c) / HOUR)),
        "delta_pickup_log": np.log1p(np.maximum(0.0, (p - t_j) / HOUR)),
        "delta_exit_log": np.log1p(np.maximum(0.0, (out - t_p) / HOUR)),
    }).reset_index(drop=True)


def _transition_regressor() -> LGBMRegressor:
    return LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED, n_jobs=8, verbose=-1)


def _transition_predictions(X_tr, T_tr, X_ev) -> pd.DataFrame:
    return pd.DataFrame({f"pred_{c}": _transition_regressor().fit(X_tr, T_tr[c]).predict(X_ev) for c in T_tr.columns},
                        index=X_ev.index)


def _oof_transition_predictions(X_tr, T_tr, d0_tr) -> pd.DataFrame:
    d0 = pd.to_datetime(pd.Series(d0_tr)).reset_index(drop=True)
    edges = d0.quantile(np.linspace(0, 1, sp.OOF_BLOCKS + 1)).to_numpy()
    edges[-1] = edges[-1] + np.timedelta64(1, "s")
    parts = []
    for k in range(sp.OOF_BLOCKS):
        lo, hi = pd.Timestamp(edges[k]), pd.Timestamp(edges[k + 1])
        held = ((d0 >= lo) & (d0 < hi)).to_numpy()
        fit = ((d0 < lo - sp.OOF_PURGE) | (d0 >= hi + sp.OOF_PURGE)).to_numpy()
        parts.append(_transition_predictions(X_tr[fit], T_tr[fit], X_tr[held]))
    return pd.concat(parts).sort_index()


def _implied(S: pd.DataFrame, elapsed: np.ndarray) -> pd.DataFrame:
    S = S.copy()
    rem = sum(np.expm1(S[c].clip(lower=0)) for c in S.columns if c.startswith("pred_delta"))
    S["implied_rem_dwell_h"] = rem
    S["implied_total_dwell_h"] = elapsed + rem
    S["implied_dist_p50"] = S["implied_total_dwell_h"] - P50_HOURS
    S["implied_dist_p85"] = S["implied_total_dwell_h"] - P85_HOURS
    return S


def run_sequential(X_tr, rows_tr, t_tr, X_ev, rows_ev, t_ev, y_tr, stacking: str = "insample"):
    """Sequential transition model. `stacking`: 'insample' (original) or 'oof' (purged time-blocked)."""
    T_tr = transitions(rows_tr, t_tr)
    X_tr, X_ev = X_tr.reset_index(drop=True), X_ev.reset_index(drop=True)
    if stacking == "insample":
        S_tr = _transition_predictions(X_tr, T_tr, X_tr)
    elif stacking == "oof":
        S_tr = _oof_transition_predictions(X_tr, T_tr, rows_tr["discharge_start_ts"].to_numpy())
    else:
        raise ValueError(stacking)
    S_ev = _transition_predictions(X_tr, T_tr, X_ev)
    el_tr = ((t_tr - rows_tr["discharge_start_ts"]) / HOUR).to_numpy()
    el_ev = ((t_ev - rows_ev["discharge_start_ts"]) / HOUR).to_numpy()
    X_tr_aug = pd.concat([X_tr, _implied(S_tr.reset_index(drop=True), el_tr)], axis=1)
    X_ev_aug = pd.concat([X_ev, _implied(S_ev.reset_index(drop=True), el_ev)], axis=1)
    proba = rr.lgbm_classifier().fit(X_tr_aug, y_tr).predict_proba(X_ev_aug)
    return proba.argmax(1), proba


def run_aft(X_tr, dwell_tr, X_ev, elapsed_ev, scale: float = 0.75, recondition: bool = True):
    """XGBoost AFT (log-normal). `recondition=True` is the original rule (divides by S(elapsed));
    False uses the predicted distribution directly, since landmark training rows already survived."""
    dtr = xgb.DMatrix(X_tr, enable_categorical=True)
    dtr.set_float_info("label_lower_bound", dwell_tr)
    dtr.set_float_info("label_upper_bound", dwell_tr)
    params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
              "aft_loss_distribution_scale": scale, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "random_state": SEED, "nthread": 8}
    bst = xgb.train(params, dtr, num_boost_round=300)
    mu = np.log(np.maximum(bst.predict(xgb.DMatrix(X_ev, enable_categorical=True)), 1.0))
    s50 = 1.0 - norm.cdf((np.log(P50_HOURS) - mu) / scale)
    s85 = 1.0 - norm.cdf((np.log(P85_HOURS) - mu) / scale)
    if recondition:
        E = np.maximum(elapsed_ev, 0.0)
        sE = np.maximum(np.where(E > 0, 1.0 - norm.cdf((np.log(np.maximum(E, 1e-3)) - mu) / scale), 1.0), 1e-6)
        p_on = np.where(E < P50_HOURS, np.clip((sE - s50) / sE, 0, 1), 0.0)
        p_cr = np.clip(s85 / sE, 0, 1)
    else:
        p_on, p_cr = 1.0 - s50, s85
    p_ar = np.clip(1.0 - p_on - p_cr, 0, 1)
    proba = np.column_stack([p_on, p_ar, p_cr])
    proba = proba / np.maximum(proba.sum(1, keepdims=True), 1e-6)
    return proba.argmax(1), proba


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------
_CACHE: Dict[tuple, tuple] = {}


def fold_inputs(cases, fold, formulation, features: str):
    """features: 'base' | 'yard_zone' | 'slot_depth' | 'block_inventory' | 'manifest_wholecall'
    | 'manifest_asof' | 'yard_manifest_asof' | 'all_wholecall' (original 'all_combined')."""
    key = (fold["name"], formulation)
    if key not in _CACHE:
        tr_mask, ev_mask = fold_masks(cases, fold)
        rows_tr, t_tr = rr.prediction_rows(cases, tr_mask, formulation, before=pd.Timestamp(fold["eval_start"]))
        rows_ev, t_ev = rr.prediction_rows(cases, ev_mask, formulation)
        parts = {}
        for split, rows, t in (("tr", rows_tr, t_tr), ("ev", rows_ev, t_ev)):
            t_ns = _ns(pd.Series(pd.to_datetime(t.to_numpy())))
            parts[split] = dict(
                base=build_asof_features(cases, rows, t),
                yard=yard_state(cases, rows.reset_index(drop=True), t_ns),
                man_asof=manifest_state(cases, rows.reset_index(drop=True), t_ns),
                man_whole=manifest_state_whole_call(cases, rows.reset_index(drop=True)),
            )
        _CACHE[key] = (rows_tr, t_tr, rows_ev, t_ev, parts)
    rows_tr, t_tr, rows_ev, t_ev, parts = _CACHE[key]

    def assemble(p):
        blocks = [p["base"]]
        if features in YARD_FAMILIES:
            blocks.append(p["yard"][YARD_FAMILIES[features]])
        if features in ("yard_manifest_asof", "all_wholecall"):
            blocks.append(p["yard"])
        if features in ("manifest_asof", "yard_manifest_asof"):
            blocks.append(p["man_asof"])
        if features in ("manifest_wholecall", "all_wholecall"):
            blocks.append(p["man_whole"])
        return pd.concat(blocks, axis=1)

    X_tr, X_ev = rr.encode_categoricals(assemble(parts["tr"]), assemble(parts["ev"]))
    return rows_tr, t_tr, X_tr, rows_ev, t_ev, X_ev


EXPERIMENTS = [
    # exp_id, formulation, features, model, production_valid, original value reproduced, hypothesis
    ("V-A-BASE", "A", "base", "lgbm", True, 0.3983, "Reproduce the discharge baseline."),
    ("V-A-YARDZONE", "A", "yard_zone", "lgbm", True, 0.4029, "Reproduce yard-zone family."),
    ("V-A-SLOTDEPTH", "A", "slot_depth", "lgbm", True, 0.4014, "Reproduce slot-depth family."),
    ("V-A-BLOCKINV", "A", "block_inventory", "lgbm", True, 0.4090, "Reproduce block-inventory family."),
    ("V-A-MANIFEST-WHOLECALL", "A", "manifest_wholecall", "lgbm", False, 0.4119,
     "Reproduce vessel-manifest family (whole-call: uses containers discharged after t)."),
    ("V-A-MANIFEST-ASOF", "A", "manifest_asof", "lgbm", True, None, "As-of manifest with kg->t correction."),
    ("V-A-ALL-WHOLECALL", "A", "all_wholecall", "lgbm", False, 0.4119, "Reproduce all-combined family."),
    ("V-A-YARD-MANIFEST-ASOF", "A", "yard_manifest_asof", "lgbm", True, None, "All yard features + as-of manifest."),
    ("V-B24-YARD-MANIFEST-ASOF", "B24", "yard_manifest_asof", "lgbm", True, None, "Same at the 24h landmark."),
    ("V-B24-SEQ-INSAMPLE", "B24", "base", "seq_insample", True, 0.5188, "Reproduce sequential model at 24h."),
    ("V-B24-SEQ-OOF", "B24", "base", "seq_oof", True, None, "Sequential model with purged OOF stacking."),
    ("V-B24-AFT", "B24", "base", "aft", True, 0.5203, "Reproduce AFT at 24h."),
    ("V-B24-AFT-NORECOND", "B24", "base", "aft_norecond", True, None, "AFT without double conditioning."),
    ("V-B48-SEQ-INSAMPLE", "B48", "base", "seq_insample", True, 0.5500, "Reproduce sequential model at 48h."),
    ("V-B48-SEQ-OOF", "B48", "base", "seq_oof", True, None, "Sequential model with purged OOF stacking."),
    ("V-B48-AFT", "B48", "base", "aft", True, 0.3094, "Reproduce AFT collapse at 48h."),
    ("V-B48-AFT-NORECOND", "B48", "base", "aft_norecond", True, None, "AFT without double conditioning at 48h."),
    ("V-B48-SEQ-OOF-YARD-MANIFEST-ASOF", "B48", "yard_manifest_asof", "seq_oof", True, None,
     "Best corrected model with as-of yard and manifest features."),
]


def predict(model, rows_tr, t_tr, X_tr, rows_ev, t_ev, X_ev):
    y_tr = rows_tr["target_class"].to_numpy()
    if model == "lgbm":
        proba = rr.lgbm_classifier().fit(X_tr, y_tr).predict_proba(X_ev)
        return proba.argmax(1), proba
    if model.startswith("seq"):
        return run_sequential(X_tr, rows_tr, t_tr, X_ev, rows_ev, t_ev, y_tr, "insample" if model == "seq_insample" else "oof")
    if model.startswith("aft"):
        el_ev = ((t_ev - rows_ev["discharge_start_ts"]) / HOUR).to_numpy()
        return run_aft(X_tr, rows_tr["dwell_hours"].to_numpy(), X_ev, el_ev, recondition=(model == "aft"))
    raise ValueError(model)


def main(only: Optional[list] = None) -> None:
    cases = pd.read_parquet(rr.CASE_TABLE_PATH)
    results_path = rr.OUT_DIR / "results.json"
    results = json.loads(results_path.read_text())
    for exp_id, F, features, model, valid, original, hyp in EXPERIMENTS:
        if only and exp_id not in only:
            continue
        t0 = time.time()
        frames = []
        for fold in DEV_FOLDS:
            rows_tr, t_tr, X_tr, rows_ev, t_ev, X_ev = fold_inputs(cases, fold, F, features)
            y_pred, proba = predict(model, rows_tr, t_tr, X_tr, rows_ev, t_ev, X_ev)
            frames.append(rr.prediction_frame(rows_ev, t_ev, y_pred, proba, fold["name"]))
        path = rr.save_predictions(exp_id, frames)
        sp.record(results, exp_id, path, {
            "formulation": F, "model": model, "features": features,
            "population": "selection-aware cohort, dev folds, still in yard at t",
            "production_valid": valid, "hypothesis": hyp, "original_reported_mean_fold": original,
            "runtime_s": round(time.time() - t0, 1)})
        if original is not None:
            got = results[exp_id]["mean_fold_macro_f1"]
            print(f"    reproduction check: reported {original:.4f}, got {got:.4f}, diff {got - original:+.4f}", flush=True)
        results_path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or None)

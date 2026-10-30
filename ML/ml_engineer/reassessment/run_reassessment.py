"""Stage 03 Diagnostic Experiment Runner.

Assigned Role: ML Research (Stage 03 Forensic Reassessment)

Every experiment:
- trains on pre-registered folds (cohort.DEV_FOLDS); HOLDOUT_TEST is never scored here,
- saves its prediction array to reports/ml/reassessment/predictions/<exp_id>.parquet,
- is scored only from that saved file, so confusion matrix, per-class report, accuracy and
  Macro-F1 are computed from one array.

Formulations (prediction timestamp t, eligible population = stays still in the yard at t):
- A:   t = discharge_start_ts
- B-L: t = discharge_start_ts + L hours (L < 65.11h keeps all three classes open)
- C:   t = max(discharge, customs release known)  (ON_TIME is already impossible for some rows)
- E_JOBDEL: t = max(discharge, delivery job created)
- ORACLE-*: retrospective, uses future information, NOT production-valid (information bounds only)

Usage:
    python ML/ml_engineer/reassessment/run_reassessment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ML.ml_engineer.baseline_pipeline import build_preprocessor, get_feature_subsets
from ML.ml_engineer.reassessment.asof_features import CATEGORICAL, CUSTOMS_GROUP, build_asof_features
from ML.ml_engineer.reassessment.cohort import (
    CLASS_NAMES,
    DEV_FOLDS,
    P50_HOURS,
    P85_HOURS,
    fold_masks,
    load_case_table,
)

SEED = 42
OUT_DIR = REPO_ROOT / "reports/ml/reassessment"
PRED_DIR = OUT_DIR / "predictions"
CASE_TABLE_PATH = REPO_ROOT / "data/interim/reassessment_case_table.parquet"
LEGACY_FEATURES = [
    "carrier_prefix", "CTR_SIZE", "CTR_TYPE", "GROSS", "discharge_hour", "discharge_day_of_week",
    "discharge_month", "is_weekend", "trailing_7d_volume", "trailing_7d_critical",
    "trailing_7d_at_risk", "historical_delay_rate_7d", "trailing_7d_mean_dwell",
]
LEGACY_MULTIPLIERS = np.array([1.4, 1.2, 1.3])


# --------------------------------------------------------------------------------------------
# Scoring (always from a saved prediction file)
# --------------------------------------------------------------------------------------------
def score_arrays(y_true: np.ndarray, y_pred: np.ndarray, proba: Optional[np.ndarray]) -> Dict[str, Any]:
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    # Consistency guard: Macro-F1 recomputed from the confusion matrix must match sklearn.
    tp = np.diag(cm).astype(float)
    p_cm = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    r_cm = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f_cm = np.divide(2 * p_cm * r_cm, p_cm + r_cm, out=np.zeros(3), where=(p_cm + r_cm) > 0)
    assert abs(f_cm.mean() - macro) < 1e-9, "Macro-F1 inconsistent with confusion matrix"
    assert abs(tp.sum() / cm.sum() - accuracy_score(y_true, y_pred)) < 1e-12
    res = {
        "n": int(len(y_true)),
        "macro_f1": macro,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "critical_pr_auc": None,
        "critical_prevalence": float(np.mean(y_true == 2)),
    }
    if proba is not None and (y_true == 2).any():
        res["critical_pr_auc"] = float(average_precision_score(y_true == 2, proba[:, 2]))
    return res


def cluster_bootstrap_macro_f1(pred: pd.DataFrame, n_boot: int = 300, seed: int = SEED) -> List[float]:
    """95% interval for pooled Macro-F1, resampling whole vessel calls (outcomes cluster by call)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in pred.groupby("vessel_call_id")]
    scores = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        scores.append(f1_score(sample["y_true"], sample["y_pred"], labels=[0, 1, 2], average="macro", zero_division=0))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def score_prediction_file(path: Path) -> Dict[str, Any]:
    pred = pd.read_parquet(path)
    proba = pred[["p_0", "p_1", "p_2"]].to_numpy() if "p_0" in pred else None
    pooled = score_arrays(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy(), proba)
    per_fold = {}
    for fold, g in pred.groupby("fold"):
        gp = g[["p_0", "p_1", "p_2"]].to_numpy() if proba is not None else None
        per_fold[fold] = score_arrays(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), gp)
    out = {
        "pooled": pooled,
        "per_fold": per_fold,
        "mean_fold_macro_f1": float(np.mean([v["macro_f1"] for v in per_fold.values()])),
        "pooled_macro_f1_ci95_vessel_bootstrap": cluster_bootstrap_macro_f1(pred) if "vessel_call_id" in pred else None,
    }
    if "lead_time_h" in pred:
        out["median_lead_time_h"] = float(pred["lead_time_h"].median())
    if "label_open" in pred and not pred["label_open"].all():
        g = pred[pred["label_open"]]
        gp = g[["p_0", "p_1", "p_2"]].to_numpy() if proba is not None else None
        out["three_class_open_subset"] = score_arrays(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), gp)
    return out


# --------------------------------------------------------------------------------------------
# Formulations
# --------------------------------------------------------------------------------------------
def prediction_rows(cases: pd.DataFrame, mask: pd.Series, formulation: str, before: Optional[pd.Timestamp] = None):
    """Eligible rows and prediction timestamps for a formulation (stay must still be in the yard)."""
    sub = cases[mask]
    d0 = sub["discharge_start_ts"]
    if formulation == "A" or formulation.startswith("ORACLE"):
        t = d0
    elif formulation.startswith("B"):
        t = d0 + pd.Timedelta(hours=float(formulation[1:]))
    elif formulation == "C":
        t = np.maximum(d0, sub["avail_CUSTOMS_DEL"])
    elif formulation == "E_JOBDEL":
        t = np.maximum(d0, sub["avail_JOB_DEL"])
    else:
        raise ValueError(formulation)
    eligible = t.notna() & (sub["truck_out_end_ts"] > t)
    if before is not None:
        eligible &= t < before
    return sub[eligible].reset_index(drop=True), t[eligible].reset_index(drop=True)


def add_oracle_columns(feats: pd.DataFrame, rows: pd.DataFrame, formulation: str) -> pd.DataFrame:
    """Future information injected for information-bound experiments only."""
    hour = pd.Timedelta(hours=1)
    if formulation == "ORACLE_CUSTOMS":
        feats["oracle_customs_rel_h"] = ((rows["avail_CUSTOMS_DEL"] - rows["discharge_start_ts"]) / hour).to_numpy()
    if formulation == "ORACLE_CUSTOMS_JOBDEL":
        feats["oracle_customs_rel_h"] = ((rows["avail_CUSTOMS_DEL"] - rows["discharge_start_ts"]) / hour).to_numpy()
        feats["oracle_jobdel_rel_h"] = ((rows["avail_JOB_DEL"] - rows["discharge_start_ts"]) / hour).to_numpy()
    return feats


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def encode_categoricals(train: pd.DataFrame, other: pd.DataFrame, min_count: int = 30):
    train, other = train.copy(), other.copy()
    for c in [c for c in CATEGORICAL if c in train]:
        counts = train[c].astype(str).value_counts()
        keep = sorted(counts[counts >= min_count].index)
        cats = keep + ["OTHER"]
        for df in (train, other):
            v = df[c].astype(str)
            df[c] = pd.Categorical(v.where(v.isin(keep), "OTHER"), categories=cats)
    return train, other


def lgbm_classifier() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced", random_state=SEED, n_jobs=8, verbose=-1,
    )


def fit_predict_lgbm(X_tr, y_tr, X_ev, dwell_tr=None):
    model = lgbm_classifier().fit(X_tr, y_tr)
    proba = model.predict_proba(X_ev)
    return proba.argmax(1), proba, model


def fit_predict_ordinal(X_tr, y_tr, X_ev, dwell_tr=None):
    """Frank & Hall cumulative binary decomposition: P(y>0), P(y>1)."""
    w = compute_sample_weight("balanced", y_tr)
    cum = []
    for k in (0, 1):
        m = LGBMClassifier(**{**lgbm_classifier().get_params(), "class_weight": None})
        m.fit(X_tr, (y_tr > k).astype(int), sample_weight=w)
        cum.append(m.predict_proba(X_ev)[:, 1])
    a, b = cum[0], np.minimum(cum[0], cum[1])
    proba = np.column_stack([1 - a, a - b, b])
    return proba.argmax(1), proba, None


def fit_predict_regression(X_tr, y_tr, X_ev, dwell_tr=None):
    """Regress log dwell, then apply the documented thresholds (no probability output)."""
    m = LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=8, verbose=-1,
    )
    m.fit(X_tr, np.log(dwell_tr), sample_weight=compute_sample_weight("balanced", y_tr))
    pred_dwell = np.exp(m.predict(X_ev))
    y_pred = np.select([pred_dwell <= P50_HOURS, pred_dwell <= P85_HOURS], [0, 1], 2)
    return y_pred, None, m


def fit_predict_unweighted(X_tr, y_tr, X_ev, dwell_tr=None):
    model = LGBMClassifier(**{**lgbm_classifier().get_params(), "class_weight": None}).fit(X_tr, y_tr)
    proba = model.predict_proba(X_ev)
    return proba.argmax(1), proba, model


MULTIPLIER_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]


def fit_predict_inner_tuned(X_tr, y_tr, X_ev, dwell_tr=None, d0_tr=None):
    """Balanced LightGBM whose class multipliers are chosen on the last 7 days of the training
    window (after a further 7-day embargo), then refit on the full training window.
    The evaluation fold is never used for the choice."""
    d0_tr = pd.to_datetime(pd.Series(d0_tr))
    end = d0_tr.max()
    fit_mask = (d0_tr < end - pd.Timedelta(days=14)).to_numpy()
    tune_mask = (d0_tr >= end - pd.Timedelta(days=7)).to_numpy()
    inner = lgbm_classifier().fit(X_tr[fit_mask], y_tr[fit_mask])
    p_tune = inner.predict_proba(X_tr[tune_mask])
    best, best_w = -1.0, np.ones(3)
    for w1 in MULTIPLIER_GRID:
        for w2 in MULTIPLIER_GRID:
            w = np.array([1.0, w1, w2])
            s = f1_score(y_tr[tune_mask], (p_tune * w).argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)
            if s > best:
                best, best_w = s, w
    model = lgbm_classifier().fit(X_tr, y_tr)
    proba = model.predict_proba(X_ev)
    model.inner_multipliers_ = best_w.tolist()
    return (proba * best_w).argmax(1), proba, model


MODEL_FNS: Dict[str, Callable] = {
    "lgbm": fit_predict_lgbm,
    "ordinal": fit_predict_ordinal,
    "regression": fit_predict_regression,
    "unweighted": fit_predict_unweighted,
    "inner_tuned": fit_predict_inner_tuned,
}


# --------------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------------
def save_predictions(exp_id: str, frames: List[pd.DataFrame]) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{exp_id}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def prediction_frame(rows, t, y_pred, proba, fold_name) -> pd.DataFrame:
    frame = pd.DataFrame({
        "case_id": rows["case_id"].to_numpy(),
        "fold": fold_name,
        "vessel_call_id": rows["vessel_call_id"].to_numpy(),
        "discharge_start_ts": rows["discharge_start_ts"].to_numpy(),
        "t_pred": pd.to_datetime(t.to_numpy()),
        "truck_out_end_ts": rows["truck_out_end_ts"].to_numpy(),
        "y_true": rows["target_class"].to_numpy(),
        "y_pred": np.asarray(y_pred).astype(int),
    })
    frame["lead_time_h"] = (frame["truck_out_end_ts"] - frame["t_pred"]).dt.total_seconds() / 3600
    frame["label_open"] = (frame["t_pred"] - frame["discharge_start_ts"]).dt.total_seconds() / 3600 < P50_HOURS
    if proba is not None:
        for k in range(3):
            frame[f"p_{k}"] = proba[:, k]
    return frame


_FEATURE_CACHE: Dict[tuple, tuple] = {}


def fold_matrices(cases, fold, formulation):
    key = (fold["name"], formulation)
    if key not in _FEATURE_CACHE:
        tr_mask, ev_mask = fold_masks(cases, fold)
        eval_start = pd.Timestamp(fold["eval_start"])
        tr_rows, tr_t = prediction_rows(cases, tr_mask, formulation, before=eval_start)
        ev_rows, ev_t = prediction_rows(cases, ev_mask, formulation)
        X_tr = add_oracle_columns(build_asof_features(cases, tr_rows, tr_t), tr_rows, formulation)
        X_ev = add_oracle_columns(build_asof_features(cases, ev_rows, ev_t), ev_rows, formulation)
        X_tr, X_ev = encode_categoricals(X_tr, X_ev)
        _FEATURE_CACHE[key] = (tr_rows, tr_t, X_tr, ev_rows, ev_t, X_ev)
    return _FEATURE_CACHE[key]


def run_asof_experiment(cases, exp_id, formulation, model="lgbm", drop: Optional[List[str]] = None):
    frames, importances = [], []
    for fold in DEV_FOLDS:
        tr_rows, tr_t, X_tr, ev_rows, ev_t, X_ev = fold_matrices(cases, fold, formulation)
        cols = [c for c in X_tr.columns if c not in set(drop or [])]
        extra = {"d0_tr": tr_rows["discharge_start_ts"].to_numpy()} if model == "inner_tuned" else {}
        y_pred, proba, fitted = MODEL_FNS[model](
            X_tr[cols], tr_rows["target_class"].to_numpy(), X_ev[cols], tr_rows["dwell_hours"].to_numpy(), **extra
        )
        if hasattr(fitted, "inner_multipliers_"):
            print(f"  {exp_id} {fold['name']} inner multipliers {fitted.inner_multipliers_}")
        frames.append(prediction_frame(ev_rows, ev_t, y_pred, proba, fold["name"]))
        if isinstance(fitted, LGBMClassifier):
            importances.append(pd.Series(fitted.booster_.feature_importance("gain"), index=cols))
    path = save_predictions(exp_id, frames)
    if importances:
        imp = pd.concat(importances, axis=1).mean(1).sort_values(ascending=False)
        (imp / imp.sum()).round(4).rename("gain_share").to_csv(OUT_DIR / f"importance_{exp_id}.csv")
    return path


def run_legacy_baseline_reproduction() -> Dict[str, Path]:
    """EXP-BASE-01 re-run from the repository's own preprocessing on the Stage 01 partitions."""
    train = pd.read_parquet(REPO_ROOT / "data/processed/train.parquet")
    val = pd.read_parquet(REPO_ROOT / "data/processed/val.parquet")
    num, cat = get_feature_subsets(LEGACY_FEATURES)
    pre = build_preprocessor(num, cat, random_state=SEED)
    X_tr = pre.fit_transform(train[LEGACY_FEATURES], train["target_class"])
    X_va = pre.transform(val[LEGACY_FEATURES])
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=4, eval_metric="mlogloss",
    )
    model.fit(X_tr, train["target_class"], sample_weight=compute_sample_weight("balanced", train["target_class"]))
    proba = model.predict_proba(X_va)
    paths = {}
    for exp_id, scores in [("R-BASE-01", proba), ("R-BASE-01-MULT", proba * LEGACY_MULTIPLIERS)]:
        frame = pd.DataFrame({
            "case_id": val["case_id"].to_numpy(), "fold": "STAGE01_VAL",
            "y_true": val["target_class"].to_numpy(), "y_pred": scores.argmax(1),
        })
        for k in range(3):
            frame[f"p_{k}"] = proba[:, k]
        paths[exp_id] = save_predictions(exp_id, [frame])
    return paths


def run_legacy_features_on_cohort(cases) -> Path:
    """Same 13 features and XGBoost config as EXP-BASE-01, but on the selection-aware cohort folds."""
    legacy = pd.concat(
        [pd.read_parquet(REPO_ROOT / f"data/processed/{p}.parquet") for p in ("burn_in", "train", "val", "test", "repeated_stays")]
    ).set_index("case_id")[LEGACY_FEATURES]
    frames = []
    num, cat = get_feature_subsets(LEGACY_FEATURES)
    for fold in DEV_FOLDS:
        tr_mask, ev_mask = fold_masks(cases, fold)
        tr, ev = cases[tr_mask], cases[ev_mask]
        pre = build_preprocessor(num, cat, random_state=SEED)
        X_tr = pre.fit_transform(legacy.loc[tr["case_id"]], tr["target_class"])
        X_ev = pre.transform(legacy.loc[ev["case_id"]])
        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=4, eval_metric="mlogloss",
        )
        model.fit(X_tr, tr["target_class"], sample_weight=compute_sample_weight("balanced", tr["target_class"]))
        proba = model.predict_proba(X_ev)
        frames.append(prediction_frame(ev.reset_index(drop=True), ev["discharge_start_ts"].reset_index(drop=True),
                                       proba.argmax(1), proba, fold["name"]))
    return save_predictions("R-A0-LEGACY13-COHORT", frames)


EXPERIMENTS = [
    # exp_id, formulation, model, dropped feature groups, production_valid, hypothesis
    ("R-A1-ASOF", "A", "lgbm", None, True,
     "Pre-discharge customs/quarantine status, vessel-call and yard state carry signal ignored in Stage 01-02."),
    ("R-A1-ASOF-NOCUSTOMS", "A", "lgbm", CUSTOMS_GROUP, True,
     "Ablation: how much of R-A1 comes from customs/quarantine status known at discharge."),
    ("R-A1-ORDINAL", "A", "ordinal", None, True, "Ordinal (Frank-Hall) structure improves the middle class."),
    ("R-A1-REGRESSION", "A", "regression", None, True, "Regressing log dwell then thresholding beats direct classification."),
    ("R-B12-ASOF", "B12", "lgbm", None, True, "12h landmark adds early customs/release milestones."),
    ("R-B24-ASOF", "B24", "lgbm", None, True, "24h landmark: most clearances for fast boxes are observed."),
    ("R-B48-ASOF", "B48", "lgbm", None, True, "48h landmark: last landmark where all three classes remain open."),
    ("R-B24-ORDINAL", "B24", "ordinal", None, True, "Ordinal structure at 24h landmark."),
    ("R-C-CUSTOMS", "C", "lgbm", None, True, "Event-driven prediction when customs release becomes known."),
    ("R-A1-UNWEIGHTED", "A", "unweighted", None, True, "Balanced weights over-predict delay classes under prior shift."),
    ("R-A1-INNER-TUNED", "A", "inner_tuned", None, True,
     "Class multipliers chosen on an embargoed inner slice of training fix the prior-shift decision bias."),
    ("R-B24-INNER-TUNED", "B24", "inner_tuned", None, True, "Inner-tuned decision rule at the 24h landmark."),
    ("R-B48-INNER-TUNED", "B48", "inner_tuned", None, True, "Inner-tuned decision rule at the 48h landmark."),
    ("R-E-JOBDEL", "E_JOBDEL", "lgbm", None, True,
     "Event-driven prediction when the delivery job (JOB_DEL) is created."),
    ("R-ORACLE-CUSTOMS", "ORACLE_CUSTOMS", "lgbm", None, False,
     "Information bound: exact customs release time known at discharge (future information)."),
    ("R-ORACLE-CUSTOMS-JOBDEL", "ORACLE_CUSTOMS_JOBDEL", "lgbm", None, False,
     "Information bound: exact customs release and delivery-order times known at discharge."),
]


def main(only: Optional[List[str]] = None) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CASE_TABLE_PATH.exists():
        cases = pd.read_parquet(CASE_TABLE_PATH)
    else:
        cases = load_case_table(str(REPO_ROOT / "data/interim/cleaned_event_log.parquet"))
        cases.to_parquet(CASE_TABLE_PATH, index=False)

    results_path = OUT_DIR / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def record(exp_id, path, meta):
        t0 = time.time()
        results[exp_id] = {**meta, "predictions": str(path.relative_to(REPO_ROOT)), **score_prediction_file(path)}
        print(f"{exp_id:28s} macro_f1(mean folds)={results[exp_id]['mean_fold_macro_f1']:.4f} "
              f"pooled={results[exp_id]['pooled']['macro_f1']:.4f} ({time.time() - t0:.1f}s scoring)")

    if not only or "R-BASE-01" in only:
        for exp_id, path in run_legacy_baseline_reproduction().items():
            record(exp_id, path, {"formulation": "Stage01 arrival", "population": "Stage01 val.parquet",
                                  "production_valid": exp_id == "R-BASE-01",
                                  "hypothesis": "Reproduce EXP-BASE-01 / EXP-BASE-01-OPT-MACRO from code."})
    if not only or "R-A0-LEGACY13-COHORT" in only:
        record("R-A0-LEGACY13-COHORT", run_legacy_features_on_cohort(cases),
               {"formulation": "A", "population": "selection-aware cohort, dev folds", "production_valid": True,
                "hypothesis": "Stage 01 features under the corrected cohort and embargoed rolling-origin folds."})
    for exp_id, formulation, model, drop, valid, hypothesis in EXPERIMENTS:
        if only and exp_id not in only:
            continue
        t0 = time.time()
        path = run_asof_experiment(cases, exp_id, formulation, model, drop)
        record(exp_id, path, {"formulation": formulation, "model": model, "dropped": drop or [],
                              "population": "selection-aware cohort, dev folds, still in yard at t",
                              "production_valid": valid, "hypothesis": hypothesis,
                              "runtime_s": round(time.time() - t0, 1)})
    results_path.write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main(sys.argv[1:] or None)

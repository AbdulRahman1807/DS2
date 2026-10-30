"""End-to-End Stage-Time Stacking Pipeline (classical ML only).

Assigned Role: ML Research (Stage 03)

Replaces the oracle event times of value_of_information.py with *predicted* event times:
1. Stage models (LightGBM / XGBoost / CatBoost regressors on log1p hours) predict, at prediction
   time t, the hours until customs release is known, the delivery job is created, the truck
   arrives and the container exits; quantile models give q10/q90 intervals for exit.
2. Training rows receive out-of-fold stage predictions from purged, time-blocked K-fold inside
   the training window; evaluation rows receive predictions from stage models fit on the whole
   training window. No evaluation-fold information reaches any stage model.
3. The final three-class classifier uses as-of features + stage predictions.
4. Nested temporal validation: inside each outer fold the final configuration is chosen on the
   last 7 days of the training window (after a 7-day embargo), never on the evaluation fold.

Feature matrices pass the Data Engineer contract (validate_feature_matrix, IN_YARD_DYNAMIC).

Usage:
    python ML/ml_engineer/reassessment/stage_pipeline.py dev        # development folds
    python ML/ml_engineer/reassessment/stage_pipeline.py holdout    # single frozen holdout run
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from ML.data_engineer.feature_contracts import PredictionScenario, validate_feature_matrix
from ML.ml_engineer.reassessment import run_reassessment as rr
from ML.ml_engineer.reassessment.asof_features import CATEGORICAL, build_asof_features
from ML.ml_engineer.reassessment.cohort import DEV_FOLDS, HOLDOUT_TEST, P50_HOURS, P85_HOURS, fold_masks

SEED = 42
FEATURE_DIR = REPO_ROOT / "data/interim/reassessment_features"
STAGES = {
    "customs": "avail_CUSTOMS_DEL",
    "jobdel": "avail_JOB_DEL",
    "pickup": "avail_TRUCK_IN",
    "exit": "truck_out_end_ts",
}
EXIT_QUANTILES = (0.1, 0.9)
OOF_BLOCKS = 5
OOF_PURGE = pd.Timedelta(days=3)
INNER_EMBARGO = pd.Timedelta(days=7)
SIBLING_PREFIX, CALENDAR_COLS, PORTWATCH_PREFIX = "sib_", ("ppkm_level", "is_holiday_today", "working_days_next_1d",
                                                            "working_days_next_2d", "working_days_next_3d",
                                                            "days_since_ppkm_change"), "pw_"


# --------------------------------------------------------------------------------------------
# Features (cached)
# --------------------------------------------------------------------------------------------
def fold_data(cases: pd.DataFrame, fold: Dict[str, str], formulation: str):
    """Rows, prediction times and extended as-of features for one fold, cached on disk."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    tr_mask, ev_mask = fold_masks(cases, fold)
    eval_start = pd.Timestamp(fold["eval_start"])
    for split, mask, before in (("train", tr_mask, eval_start), ("eval", ev_mask, None)):
        path = FEATURE_DIR / f"{formulation}_{fold['name']}_{split}.parquet"
        rows, t = rr.prediction_rows(cases, mask, formulation, before=before)
        if path.exists():
            X = pd.read_parquet(path)
        else:
            X = build_asof_features(cases, rows, t, extended=True)
            X.to_parquet(path, index=False)
        validate_feature_matrix(X, scenario=PredictionScenario.IN_YARD_DYNAMIC, strict_allowlist=False)
        out[split] = (rows, t, X)
    return out


def stage_targets(rows: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """log1p hours from t until each stage (0 if already observed; exit if the stage never occurs)."""
    hour = pd.Timedelta(hours=1)
    exit_h = ((rows["truck_out_end_ts"] - t) / hour).clip(lower=0)
    y = {}
    for stage, col in STAGES.items():
        h = ((rows[col] - t) / hour).clip(lower=0)
        y[stage] = np.log1p(h.fillna(exit_h).to_numpy())
    return pd.DataFrame(y)


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def _to_catboost(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    X = X.copy()
    cats = [c for c in CATEGORICAL if c in X]
    for c in cats:
        X[c] = X[c].astype(str)
    return X, cats


def make_regressor(family: str, objective: str = "l2", alpha: Optional[float] = None):
    if family == "lgbm":
        kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=8, verbose=-1)
        return LGBMRegressor(objective="quantile", alpha=alpha, **kw) if objective == "quantile" else LGBMRegressor(**kw)
    if family == "xgb":
        kw = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, n_jobs=8, tree_method="hist", enable_categorical=True)
        if objective == "quantile":
            return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, **kw)
        return XGBRegressor(**kw)
    if family == "catboost":
        kw = dict(iterations=500, learning_rate=0.08, depth=6, random_seed=SEED, verbose=0, thread_count=8)
        loss = f"Quantile:alpha={alpha}" if objective == "quantile" else "RMSE"
        return CatBoostRegressor(loss_function=loss, **kw)
    raise ValueError(family)


def fit_predict_reg(family, X_tr, y_tr, X_ev, objective="l2", alpha=None):
    model = make_regressor(family, objective, alpha)
    if family == "catboost":
        X_tr, cats = _to_catboost(X_tr)
        X_ev, _ = _to_catboost(X_ev)
        model.fit(X_tr, y_tr, cat_features=cats)
    else:
        model.fit(X_tr, y_tr)
    return model.predict(X_ev)


def stage_prediction_columns(family: str, X_tr, Y_tr, X_ev) -> pd.DataFrame:
    cols = {}
    for stage in STAGES:
        cols[f"pred_{stage}_log1p_h"] = fit_predict_reg(family, X_tr, Y_tr[stage].to_numpy(), X_ev)
    for q in EXIT_QUANTILES:
        cols[f"pred_exit_q{int(q * 100)}_log1p_h"] = fit_predict_reg(
            family, X_tr, Y_tr["exit"].to_numpy(), X_ev, objective="quantile", alpha=q)
    return pd.DataFrame(cols, index=X_ev.index)


def oof_stage_predictions(family: str, X: pd.DataFrame, Y: pd.DataFrame, d0: pd.Series) -> pd.DataFrame:
    """Purged, time-blocked K-fold predictions for training rows (each row predicted by models
    that never saw rows discharged within OOF_PURGE of its block)."""
    d0 = pd.to_datetime(d0).reset_index(drop=True)
    edges = d0.quantile(np.linspace(0, 1, OOF_BLOCKS + 1)).to_numpy()
    edges[-1] = edges[-1] + np.timedelta64(1, "s")
    parts = []
    for k in range(OOF_BLOCKS):
        lo, hi = pd.Timestamp(edges[k]), pd.Timestamp(edges[k + 1])
        held = ((d0 >= lo) & (d0 < hi)).to_numpy()
        fit = ((d0 < lo - OOF_PURGE) | (d0 >= hi + OOF_PURGE)).to_numpy()
        parts.append(stage_prediction_columns(family, X[fit], Y[fit], X[held]))
    return pd.concat(parts).sort_index()


def add_derived(S: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Implied total dwell and its position relative to the class thresholds."""
    S = S.copy()
    elapsed = X["elapsed_h"].to_numpy()
    for col in [c for c in S.columns if c.startswith("pred_exit")]:
        total = elapsed + np.expm1(S[col].to_numpy())
        S[col.replace("log1p_h", "total_h")] = total
        S[col.replace("log1p_h", "minus_p50")] = total - P50_HOURS
        S[col.replace("log1p_h", "minus_p85")] = total - P85_HOURS
    S["pred_exit_interval_log_width"] = S["pred_exit_q90_log1p_h"] - S["pred_exit_q10_log1p_h"]
    S["pred_pickup_total_h"] = elapsed + np.expm1(S["pred_pickup_log1p_h"].to_numpy())
    return S


def class_weights(y: np.ndarray, scheme: str) -> np.ndarray:
    w = compute_sample_weight("balanced", y)
    return np.sqrt(w) if scheme == "sqrt" else w


def fit_predict_classifier(family: str, X_tr, y_tr, X_ev, weighting: str = "balanced") -> np.ndarray:
    w = class_weights(y_tr, weighting)
    if family == "lgbm":
        m = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=8, verbose=-1)
        m.fit(X_tr, y_tr, sample_weight=w)
        return m.predict_proba(X_ev)
    if family == "xgb":
        m = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          random_state=SEED, n_jobs=8, tree_method="hist", enable_categorical=True, eval_metric="mlogloss")
        m.fit(X_tr, y_tr, sample_weight=w)
        return m.predict_proba(X_ev)
    if family == "catboost":
        Xt, cats = _to_catboost(X_tr)
        Xe, _ = _to_catboost(X_ev)
        m = CatBoostClassifier(iterations=800, learning_rate=0.06, depth=6, loss_function="MultiClass",
                               random_seed=SEED, verbose=0, thread_count=8)
        m.fit(Xt, y_tr, cat_features=cats, sample_weight=w)
        return m.predict_proba(Xe)
    if family == "ens":
        return np.mean([fit_predict_classifier(f, X_tr, y_tr, X_ev, weighting) for f in ("lgbm", "xgb", "catboost")], axis=0)
    raise ValueError(family)


# --------------------------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------------------------
def select_columns(X: pd.DataFrame, blocks: str) -> List[str]:
    cols = list(X.columns)
    drop = set()
    if "nosib" in blocks:
        drop |= {c for c in cols if c.startswith(SIBLING_PREFIX)}
    if "nocal" in blocks:
        drop |= set(CALENDAR_COLS)
    if "nopw" in blocks:
        drop |= {c for c in cols if c.startswith(PORTWATCH_PREFIX)}
    if "base" in blocks:
        drop |= {c for c in cols if c.startswith((SIBLING_PREFIX, PORTWATCH_PREFIX))} | set(CALENDAR_COLS)
    return [c for c in cols if c not in drop]


_STAGE_CACHE: Dict[tuple, tuple] = {}


def stage_features(fold_name, formulation, family, X_tr, rows_tr, t_tr, X_ev):
    key = (fold_name, formulation, family, tuple(X_tr.columns))
    if key not in _STAGE_CACHE:
        path = FEATURE_DIR / f"stage_{formulation}_{fold_name}_{family}_{len(X_tr.columns)}.parquet"
        if path.exists():
            both = pd.read_parquet(path)
            S_tr = both[both["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
            S_ev = both[both["_split"] == "eval"].drop(columns="_split").reset_index(drop=True)
        else:
            Y_tr = stage_targets(rows_tr, t_tr)
            S_tr = oof_stage_predictions(family, X_tr, Y_tr, rows_tr["discharge_start_ts"])
            S_ev = stage_prediction_columns(family, X_tr, Y_tr, X_ev)
            pd.concat([S_tr.assign(_split="train"), S_ev.reset_index(drop=True).assign(_split="eval")],
                      ignore_index=True).to_parquet(path, index=False)
        _STAGE_CACHE[key] = (add_derived(S_tr.reset_index(drop=True), X_tr), add_derived(S_ev.reset_index(drop=True), X_ev))
    return _STAGE_CACHE[key]


def run_config(cases, folds, formulation, blocks="ext", stage_family=None, final="lgbm", weighting="balanced",
               nested_grid: Optional[List[Tuple[str, str]]] = None, exp_id: str = "", verbose=True):
    """One pipeline configuration over the given outer folds; returns saved prediction path and choices."""
    frames, choices = [], {}
    for fold in folds:
        d = fold_data(cases, fold, formulation)
        rows_tr, t_tr, X_tr_all = d["train"]
        rows_ev, t_ev, X_ev_all = d["eval"]
        cols = select_columns(X_tr_all, blocks)
        X_tr, X_ev = rr.encode_categoricals(X_tr_all[cols], X_ev_all[cols])
        if stage_family:
            S_tr, S_ev = stage_features(fold["name"], formulation, stage_family, X_tr, rows_tr, t_tr, X_ev)
            X_tr = pd.concat([X_tr.reset_index(drop=True), S_tr], axis=1)
            X_ev = pd.concat([X_ev.reset_index(drop=True), S_ev], axis=1)
        y_tr = rows_tr["target_class"].to_numpy()

        chosen = (final, weighting)
        if nested_grid:
            d0 = pd.to_datetime(rows_tr["discharge_start_ts"]).reset_index(drop=True)
            end = d0.max()
            inner_val = (d0 >= end - pd.Timedelta(days=7)).to_numpy()
            inner_fit = (d0 < end - pd.Timedelta(days=7) - INNER_EMBARGO).to_numpy()
            scores = {}
            for fam, wt in nested_grid:
                p = fit_predict_classifier(fam, X_tr[inner_fit], y_tr[inner_fit], X_tr[inner_val], wt)
                scores[(fam, wt)] = f1_score(y_tr[inner_val], p.argmax(1), average="macro")
            chosen = max(scores, key=scores.get)
            choices[fold["name"]] = {"chosen": list(chosen), "inner_scores": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in scores.items()}}
        proba = fit_predict_classifier(chosen[0], X_tr, y_tr, X_ev, chosen[1])
        frames.append(rr.prediction_frame(rows_ev, t_ev, proba.argmax(1), proba, fold["name"]))
        if verbose:
            print(f"  {exp_id} {fold['name']} chosen={chosen} macroF1={f1_score(rows_ev['target_class'], proba.argmax(1), average='macro'):.4f}", flush=True)
    return rr.save_predictions(exp_id, frames), choices


def record(results, exp_id, path, meta, choices=None):
    results[exp_id] = {**meta, "predictions": str(path.relative_to(REPO_ROOT)), **rr.score_prediction_file(path)}
    if choices:
        results[exp_id]["nested_choices"] = choices
    r = results[exp_id]
    pc = r["pooled"]["per_class"]
    print(f"{exp_id:34s} mean-fold={r['mean_fold_macro_f1']:.4f} pooled={r['pooled']['macro_f1']:.4f} "
          f"F1(ON/AR/CR)={pc['ON_TIME']['f1']:.3f}/{pc['AT_RISK']['f1']:.3f}/{pc['CRITICAL_DELAY']['f1']:.3f}", flush=True)


NESTED_GRID = [("lgbm", "balanced"), ("lgbm", "sqrt"), ("catboost", "balanced"), ("xgb", "balanced"), ("ens", "balanced")]


def dev_experiments() -> List[dict]:
    def core(F):
        return [
            dict(exp_id=f"P-{F}-BASE", formulation=F, blocks="base", hyp="Reproduce the as-of baseline in this pipeline."),
            dict(exp_id=f"P-{F}-EXT", formulation=F, blocks="ext", hyp="Add consignment siblings, calendar/PPKM and PortWatch."),
            dict(exp_id=f"P-{F}-EXT-STAGE-LGBM", formulation=F, blocks="ext", stage_family="lgbm",
                 hyp="Stack OOF predicted stage times (LightGBM) and exit intervals."),
            dict(exp_id=f"P-{F}-EXT-STAGE-NESTED", formulation=F, blocks="ext", stage_family="lgbm", nested=True,
                 hyp="Nested temporal selection of the final classifier family and class weighting."),
        ]
    exps = core("B24") + [
        dict(exp_id="P-B24-EXT-NOSIB", formulation="B24", blocks="ext-nosib", hyp="Ablation: without siblings."),
        dict(exp_id="P-B24-EXT-NOCAL", formulation="B24", blocks="ext-nocal", hyp="Ablation: without calendar/PPKM."),
        dict(exp_id="P-B24-EXT-NOPW", formulation="B24", blocks="ext-nopw", hyp="Ablation: without PortWatch."),
        dict(exp_id="P-B24-EXT-STAGE-XGB", formulation="B24", blocks="ext", stage_family="xgb", hyp="XGBoost stage models."),
        dict(exp_id="P-B24-EXT-STAGE-CATBOOST", formulation="B24", blocks="ext", stage_family="catboost", hyp="CatBoost stage models."),
        dict(exp_id="P-B24-EXT-STAGE-LGBM-CAT", formulation="B24", blocks="ext", stage_family="lgbm", final="catboost", hyp="CatBoost final."),
        dict(exp_id="P-B24-EXT-STAGE-LGBM-XGB", formulation="B24", blocks="ext", stage_family="lgbm", final="xgb", hyp="XGBoost final."),
        dict(exp_id="P-B24-EXT-STAGE-LGBM-ENS", formulation="B24", blocks="ext", stage_family="lgbm", final="ens", hyp="Three-family ensemble final."),
        dict(exp_id="P-B24-EXT-STAGE-LGBM-SQRT", formulation="B24", blocks="ext", stage_family="lgbm", weighting="sqrt", hyp="Softer class weights."),
    ]
    return exps + core("A") + core("B12") + core("B48")


def main(mode: str = "dev", only: Optional[List[str]] = None) -> None:
    cases = pd.read_parquet(rr.CASE_TABLE_PATH)
    results_path = rr.OUT_DIR / "results.json"
    results = json.loads(results_path.read_text())
    if mode == "dev":
        for e in dev_experiments():
            if only and e["exp_id"] not in only:
                continue
            t0 = time.time()
            path, choices = run_config(
                cases, DEV_FOLDS, e["formulation"], e.get("blocks", "ext"), e.get("stage_family"),
                e.get("final", "lgbm"), e.get("weighting", "balanced"),
                NESTED_GRID if e.get("nested") else None, exp_id=e["exp_id"])
            record(results, e["exp_id"], path, {
                "formulation": e["formulation"], "model": f"stage={e.get('stage_family')} final={e.get('final', 'lgbm')}",
                "population": "selection-aware cohort, dev folds, still in yard at t", "production_valid": True,
                "hypothesis": e["hyp"], "runtime_s": round(time.time() - t0, 1)}, choices)
            results_path.write_text(json.dumps(results, indent=2, default=str))
    elif mode == "holdout":
        frozen = json.loads((rr.OUT_DIR / "frozen_pipeline.json").read_text())
        for exp_id, cfg in frozen["configs"].items():
            hold_id = f"{exp_id}-HOLDOUT"
            path, choices = run_config(
                cases, [HOLDOUT_TEST], cfg["formulation"], cfg["blocks"], cfg.get("stage_family"), cfg.get("final", "lgbm"),
                cfg.get("weighting", "balanced"), NESTED_GRID if cfg.get("nested") else None, exp_id=hold_id)
            record(results, hold_id, path, {
                "formulation": cfg["formulation"], "model": f"frozen {exp_id}",
                "population": "selection-aware cohort, HOLDOUT TEST (discharges 2021-08-27..09-05), scored once",
                "production_valid": True, "hypothesis": "Single confirmatory run of the frozen pipeline."}, choices)
            results_path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev", sys.argv[2:] or None)

"""Hybrid model build: real stays + synthetic training stays (never fully synthetic).

Scores each horizon on real stays and on a labelled hybrid test set (real + synthetic), then
refits on all discharges before the holdout training cutoff and saves runnable artifacts.
Synthetic rows never enter yard/call/terminal aggregates; the holdout is not read or scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from ML.data_engineer.build_master_dataset import synthesize
from ML.ml_engineer.reassessment import run_reassessment as rr
from ML.ml_engineer.reassessment.asof_features import build_asof_features
from ML.ml_engineer.reassessment.cohort import DEV_FOLDS, HOLDOUT_TEST, fold_masks

N_SYNTHETIC = 60_000
OUT = REPO_ROOT / "ML/ml_engineer/artifacts/hybrid"
HORIZONS = ["A", "B12", "B24", "B48"]


def rows(frame, mask, f, before=None):
    r, t = rr.prediction_rows(frame, mask, f, before=before)
    return r, t, build_asof_features(REAL, r, t)


def syn_mask(end, start=None):
    m = pd.to_datetime(SYN["synth_donor_max_discharge"]) < pd.Timestamp(end)
    return m if start is None else (SYN["discharge_start_ts"] >= pd.Timestamp(start))


def fit(Xr, yr, Xs, ys, Xe):
    Xtr, Xev = rr.encode_categoricals(pd.concat([Xr, Xs], ignore_index=True), Xe)
    return rr.lgbm_classifier().fit(Xtr, np.r_[yr, ys]), Xtr, Xev


REAL = pd.read_parquet(rr.CASE_TABLE_PATH)
SYN = synthesize(REAL, n=N_SYNTHETIC, seed=43).reset_index(drop=True)
OUT.mkdir(parents=True, exist_ok=True)
SYN.to_parquet(REPO_ROOT / "data/master/master_synthetic_v2.parquet", index=False)
report = {}
for f in HORIZONS:
    real_scores, hybrid_scores, ys_all, ps_all = [], [], [], []
    for fold in DEV_FOLDS:
        tr, ev = fold_masks(REAL, fold)
        start = pd.Timestamp(fold["eval_start"])
        rr_, _, Xr = rows(REAL, tr, f, start)
        sr, _, Xs = rows(SYN, syn_mask(fold["train_end"]), f, start)
        re_, _, Xe = rows(REAL, ev, f)
        model, _, Xev = fit(Xr, rr_["target_class"].to_numpy(), Xs, sr["target_class"].to_numpy(), Xe)
        p = model.predict(Xev)
        y = re_["target_class"].to_numpy()
        real_scores.append(f1_score(y, p, average="macro"))
        ys_all.append(y); ps_all.append(p)
        # Hybrid test: real eval stays + synthetic stays discharged in the window (donors may overlap it).
        d = SYN["discharge_start_ts"]
        se, _, Xse = rows(SYN, (d >= start) & (d < pd.Timestamp(fold["eval_end"])), f)
        if len(se):
            _, Xev2 = rr.encode_categoricals(Xr, Xse)[0], rr.encode_categoricals(pd.concat([Xr, Xs], ignore_index=True), Xse)[1]
            hybrid_scores.append(f1_score(np.r_[y, se["target_class"]], np.r_[p, model.predict(Xev2)], average="macro"))
    y, p = np.concatenate(ys_all), np.concatenate(ps_all)
    report[f] = {"real_macro_f1": round(float(np.mean(real_scores)), 4), "real_critical_recall": round(float(recall_score(y == 2, p == 2)), 4),
                 "real_critical_f1": round(float(f1_score(y == 2, p == 2)), 4), "hybrid_test_macro_f1": round(float(np.mean(hybrid_scores)), 4)}
    print(f, report[f], flush=True)
    d0 = REAL["discharge_start_ts"]
    end = pd.Timestamp(HOLDOUT_TEST["train_end"])
    rf, _, Xr = rows(REAL, REAL["in_cohort"] & (d0 < end), f)
    sf, _, Xs = rows(SYN, syn_mask(end), f)
    model, Xtr, _ = fit(Xr, rf["target_class"].to_numpy(), Xs, sf["target_class"].to_numpy(), Xr.head(1))
    joblib.dump({"estimator": model, "features": list(Xtr.columns), "categories": {c: list(Xtr[c].cat.categories) for c in Xtr.select_dtypes("category")},
                 "horizon": f, "n_real": len(rf), "n_synthetic": len(sf)}, OUT / f"hybrid_{f}.joblib")
(OUT / "hybrid_report.json").write_text(json.dumps(report, indent=2))

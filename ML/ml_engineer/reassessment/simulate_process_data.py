"""What-if scenario: simulated customs, delivery-order, payment and truck-booking records.

SIMULATED, NOT OBSERVED. Fields are generated backwards from each stay's real later events using
ASSUMED lead times, so they carry outcome information by construction. Scores estimate the value of
obtaining such data; they are not model performance on available data. Fields are visible only once
their simulated timestamp has passed. The holdout is not read.
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
from sklearn.metrics import f1_score, recall_score

from ML.ml_engineer.reassessment import run_reassessment as rr
from ML.ml_engineer.reassessment.cohort import DEV_FOLDS

HOUR = pd.Timedelta(hours=1)
# Assumed median lead (hours) before the real event, lognormal spread sigma. Not measured anywhere.
LEADS = {"sim_pib_filed": ("avail_CUSTOMS_DEL", 48, .8), "sim_do_requested": ("avail_JOB_DEL", 18, .8),
         "sim_payment_done": ("avail_JOB_DEL", 5, .7), "sim_truck_booked": ("avail_TRUCK_IN", 12, .8)}
SLOT_ERROR_H = 4.0


def simulate(cases: pd.DataFrame, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, out = len(cases), cases["truck_out_end_ts"]
    sim = pd.DataFrame(index=cases.index)
    for name, (event, median, sigma) in LEADS.items():
        lead = median * np.exp(rng.normal(0, sigma, n))
        sim[name] = cases[event].fillna(out) - pd.to_timedelta(lead, unit="h")
    inspected = cases["avail_FIRST_JOB_CUSTOMS_BEHANDLE"].notna() | cases["avail_SP_RED_FLAG_PIB"].notna()
    sim["sim_lane"] = np.where(inspected, 2, np.where(rng.random(n) < .10, 1, 0))
    sim["sim_truck_slot"] = cases["avail_TRUCK_IN"].fillna(out) + pd.to_timedelta(rng.normal(0, SLOT_ERROR_H, n), unit="h")
    return sim


def visible(sim_rows: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    x = {}
    t = pd.to_datetime(t.to_numpy())
    for name in LEADS:
        ts = pd.to_datetime(sim_rows[name].to_numpy())
        known = ts <= t
        x[name + "_known"] = known.astype(int)
        x[name + "_age_h"] = np.where(known, (t - ts) / HOUR, np.nan)
    filed = x["sim_pib_filed_known"] == 1
    x["sim_lane"] = np.where(filed, sim_rows["sim_lane"].to_numpy(), -1)
    booked = x["sim_truck_booked_known"] == 1
    x["sim_slot_in_h"] = np.where(booked, (pd.to_datetime(sim_rows["sim_truck_slot"].to_numpy()) - t) / HOUR, np.nan)
    return pd.DataFrame(x)


cases = pd.read_parquet(rr.CASE_TABLE_PATH)
sim = simulate(cases)
pd.concat([cases[["case_id"]], sim], axis=1).assign(is_simulated=True).to_parquet(REPO_ROOT / "data/master/simulated_process_fields.parquet", index=False)
report = {}
for f in ["A", "B12", "B24", "B48"]:
    scores, ys, ps = [], [], []
    for fold in DEV_FOLDS:
        tr_rows, tr_t, X_tr, ev_rows, ev_t, X_ev = rr.fold_matrices(cases, fold, f)
        s = sim.copy()
        s.insert(0, "case_id", cases["case_id"])
        s = s.set_index("case_id")
        Xa = pd.concat([X_tr.reset_index(drop=True), visible(s.loc[tr_rows["case_id"]], tr_t)], axis=1)
        Xb = pd.concat([X_ev.reset_index(drop=True), visible(s.loc[ev_rows["case_id"]], ev_t)], axis=1)
        p = rr.lgbm_classifier().fit(Xa, tr_rows["target_class"].to_numpy()).predict(Xb)
        y = ev_rows["target_class"].to_numpy()
        scores.append(f1_score(y, p, average="macro")); ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    report[f] = {"simulated_macro_f1": round(float(np.mean(scores)), 4), "per_class_f1": [round(float(v), 3) for v in f1_score(y, p, average=None)],
                 "critical_recall": round(float(recall_score(y == 2, p == 2)), 3), "accuracy": round(float((y == p).mean()), 3)}
    print(f, report[f], flush=True)
(REPO_ROOT / "reports/ml/reassessment/simulated_process_scenario.json").write_text(json.dumps({"label": "SIMULATED what-if; not real performance", "assumed_leads": LEADS, "slot_error_h": SLOT_ERROR_H, "results": report}, indent=2))

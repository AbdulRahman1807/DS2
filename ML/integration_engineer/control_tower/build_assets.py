"""Control Tower Asset Builder.

Assigned Role: Integration Engineer

Precomputes every number the Streamlit control tower displays from verified project artifacts,
so the deployed app needs neither the gitignored data/ folder nor recomputation at runtime.

Sources: data/master/master_real.parquet (real stays only), data/processed/* (Stage 01 inputs),
reports/ml/reassessment/ (saved validation predictions, results.json, importance files),
reports/evaluation/ (calibration report), data/reference and data/raw/portwatch_daily_ports.parquet.

Excluded by construction: synthetic stays, oracle rows as model performance, and any model
prediction for the quarantined holdout window (only development-fold arrays are read).

Usage:
    python ML/integration_engineer/control_tower/build_assets.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ASSETS = Path(__file__).resolve().parent / "assets"
RESULTS = REPO_ROOT / "reports/ml/reassessment"
P50, P85 = 65.1102, 160.8993
HOUR = pd.Timedelta(hours=1)
HORIZONS = {"A": "R-A1-ASOF", "B12": "R-B12-ASOF", "B24": "R-B24-ASOF", "B48": "R-B48-ASOF"}
HORIZON_LABEL = {"A": "At discharge", "B12": "+12 h", "B24": "+24 h", "B48": "+48 h"}
CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
REPLAY_START, REPLAY_END = pd.Timestamp("2021-07-05"), pd.Timestamp("2021-08-09")
REPLAY_SAMPLE_EVERY = 8


def _json(name: str, obj) -> None:
    (ASSETS / name).write_text(json.dumps(obj, separators=(",", ":"), default=str))


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def load_real() -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / "data/master/master_real.parquet")


# --------------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------------
def build_overview(real: pd.DataFrame, results: dict) -> None:
    cohort = real[real["in_cohort"]]
    d0, out = real["discharge_start_ts"], real["truck_out_end_ts"]
    grid = pd.date_range("2021-07-01", "2021-09-06", freq="6h")
    d0_ns, out_ns = d0.to_numpy(), out.to_numpy()
    tide = []
    for t in grid:
        t64 = np.datetime64(t)
        in_yard = (d0_ns <= t64) & (out_ns > t64)
        age = (t64 - d0_ns[in_yard]) / np.timedelta64(1, "h")
        tide.append({"t": t.isoformat(), "fresh": int((age <= P50).sum()),
                     "aging": int(((age > P50) & (age <= P85)).sum()), "over": int((age > P85).sum())})
    days = pd.date_range("2021-07-01", "2021-09-05", freq="D")
    arrivals = d0.dt.normalize().value_counts().reindex(days, fill_value=0)
    departures = out.dt.normalize().value_counts().reindex(days, fill_value=0)
    flow = [{"d": str(x.date()), "in": int(arrivals[x]), "out": int(departures[x])} for x in days]

    cal = pd.read_csv(REPO_ROOT / "data/reference/indonesia_calendar_2021.csv", parse_dates=["date"])
    window = cal[(cal["date"] >= "2021-07-01") & (cal["date"] <= "2021-09-06")]
    holidays = [{"d": str(r.date.date()), "name": r.holiday_name} for r in window[window["is_public_holiday"] == 1].itertuples()]
    ppkm = []
    for label, g in window.groupby((window["ppkm_label"] != window["ppkm_label"].shift()).cumsum()):
        ppkm.append({"start": str(g["date"].min().date()), "end": str(g["date"].max().date()),
                     "label": g["ppkm_label"].iloc[0], "level": int(g["ppkm_level"].iloc[0])})

    weeks = cohort.assign(wk=cohort["discharge_start_ts"].dt.to_period("W").dt.start_time)
    bins = np.linspace(np.log10(4), np.log10(800), 121)
    centers = (bins[:-1] + bins[1:]) / 2
    ridges = []
    for wk, g in weeks.groupby("wk"):
        if len(g) < 200:
            continue
        h, _ = np.histogram(np.log10(g["dwell_hours"].clip(4, 800)), bins=bins, density=True)
        h = np.convolve(h, np.ones(5) / 5, mode="same")
        ridges.append({"wk": str(wk.date()), "n": int(len(g)), "median": _r(g["dwell_hours"].median(), 1),
                       "crit": _r((g["target_class"] == 2).mean(), 3), "density": [round(float(v), 3) for v in h]})

    feas = pd.read_csv(RESULTS / "formulation_feasibility.csv").set_index("formulation")
    coverage = {"A": feas.loc["A: discharge", "open_three_class_share_of_arrivals"],
                "B12": feas.loc["B12: discharge + 12h", "open_three_class_share_of_arrivals"],
                "B24": feas.loc["B24: discharge + 24h", "open_three_class_share_of_arrivals"],
                "B48": feas.loc["B48: discharge + 48h", "open_three_class_share_of_arrivals"]}
    _json("overview.json", {
        "kpi": {"stays_total": int(len(real)), "stays_cohort": int(len(cohort)),
                "first_discharge": str(cohort["discharge_start_ts"].min().date()),
                "last_discharge": str(cohort["discharge_start_ts"].max().date()),
                "mix": [_r((cohort["target_class"] == k).mean()) for k in range(3)],
                "dwell_mean": _r(cohort["dwell_hours"].mean(), 1), "dwell_median": _r(cohort["dwell_hours"].median(), 1),
                "vessel_calls": int(cohort["vessel_call_id"].nunique()),
                "peak_week": max(ridges, key=lambda r: r["crit"]), "calm_week": min(ridges, key=lambda r: r["crit"])},
        "coverage": {k: _r(v, 4) for k, v in coverage.items()},
        "tide": tide, "flow": flow, "holidays": holidays, "ppkm": ppkm,
        "ridge_axis": [round(float(10 ** c), 2) for c in centers], "ridges": ridges,
    })


def build_ports() -> None:
    p = pd.read_parquet(REPO_ROOT / "data/reference/portwatch_ports.parquet")
    p = p[p["vessel_count_container"] > 0]
    rows = [[r.portname, r.country, round(r.lat, 3), round(r.lon, 3), int(r.vessel_count_container)] for r in p.itertuples()]
    _json("ports.json", rows)


# --------------------------------------------------------------------------------------------
# Terminal replay (real event log, sampled)
# --------------------------------------------------------------------------------------------
def build_replay(real: pd.DataFrame) -> None:
    d0, out = real["discharge_start_ts"], real["truck_out_end_ts"]
    sub = real[(d0 < REPLAY_END) & (out > REPLAY_START)].sort_values("case_id").iloc[::REPLAY_SAMPLE_EVERY]
    rel = lambda s: ((s - REPLAY_START) / HOUR).round(1)
    o = sub["truck_out_end_ts"]
    c = sub["avail_CUSTOMS_DEL"].fillna(o).clip(upper=o)
    j = np.maximum(sub["avail_JOB_DEL"].fillna(o).clip(upper=o), c)
    t_in = np.maximum(sub["avail_TRUCK_IN"].fillna(o).clip(upper=o), j)
    rows = [[float(a), float(b), float(e), float(f), float(g), 1 if sz >= 40 else 0]
            for a, b, e, f, g, sz in zip(rel(sub["discharge_start_ts"]), rel(c), rel(j), rel(t_in), rel(o), sub["CTR_SIZE"].fillna(20))]
    days = pd.date_range(REPLAY_START, REPLAY_END, freq="D")
    cal = pd.read_csv(REPO_ROOT / "data/reference/indonesia_calendar_2021.csv", parse_dates=["date"]).set_index("date")
    notes = {}
    for x in days:
        row = cal.loc[x]
        tag = [str(row["ppkm_label"])] if pd.notna(row["ppkm_label"]) else []
        if row["is_public_holiday"] == 1:
            tag.append(str(row["holiday_name"]))
        notes[str(x.date())] = " · ".join(tag)
    _json("replay.json", {"start": str(REPLAY_START.date()), "hours": (REPLAY_END - REPLAY_START) / HOUR,
                          "sample_every": REPLAY_SAMPLE_EVERY, "p50": P50, "p85": P85, "notes": notes, "stays": rows})


# --------------------------------------------------------------------------------------------
# Port intelligence
# --------------------------------------------------------------------------------------------
def build_port(real: pd.DataFrame) -> None:
    cohort = real[real["in_cohort"]].copy()
    out, d0 = cohort["truck_out_end_ts"], cohort["discharge_start_ts"]
    clock_out = pd.crosstab(out.dt.dayofweek, out.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)
    clock_in = pd.crosstab(d0.dt.dayofweek, d0.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)

    cal = pd.read_csv(REPO_ROOT / "data/reference/indonesia_calendar_2021.csv", parse_dates=["date"]).set_index("date")
    daily = cohort.groupby(d0.dt.normalize()).agg(n=("case_id", "size"), crit=("target_class", lambda s: (s == 2).mean()),
                                                   median=("dwell_hours", "median"))
    calendar = [{"d": str(i.date()), "n": int(r.n), "crit": _r(r.crit, 3), "median": _r(r["median"], 1),
                 "ppkm": str(cal.loc[i, "ppkm_label"]), "holiday": None if pd.isna(cal.loc[i, "holiday_name"]) else cal.loc[i, "holiday_name"]}
                for i, r in daily.iterrows()]

    vessels = cohort.groupby("vessel_call_id").agg(
        n=("case_id", "size"), first=("discharge_start_ts", "min"), median=("dwell_hours", "median"),
        crit=("target_class", lambda s: (s == 2).mean()), ontime=("target_class", lambda s: (s == 0).mean()),
        cleared=("avail_CUSTOMS_DEL", "count"))
    pre = cohort.assign(pre=cohort["avail_CUSTOMS_DEL"] <= cohort["discharge_start_ts"]).groupby("vessel_call_id")["pre"].mean()
    vessels["precleared"] = pre
    vessels = vessels[vessels["n"] >= 30]
    vessel_rows = [{"call": i, "t": r["first"].isoformat(), "n": int(r["n"]), "median": _r(r["median"], 1),
                    "crit": _r(r["crit"], 3), "ontime": _r(r["ontime"], 3), "precleared": _r(r["precleared"], 3)}
                   for i, r in vessels.iterrows()]

    blocks = cohort["YARD_BLOCK"].value_counts()
    keep = blocks[blocks >= 400].index.tolist()
    wk = cohort["discharge_start_ts"].dt.to_period("W").dt.start_time
    wk_order = sorted(wk.unique())
    block_grid = []
    for b in keep:
        g = cohort[cohort["YARD_BLOCK"] == b]
        gw = g.groupby(wk.loc[g.index])
        cell = {str(k.date()): {"n": int(len(v)), "crit": _r((v["target_class"] == 2).mean(), 3)} for k, v in gw if len(v) >= 15}
        block_grid.append({"block": b, "n": int(len(g)), "crit": _r((g["target_class"] == 2).mean(), 3),
                           "median": _r(g["dwell_hours"].median(), 1), "weeks": cell})

    grid = np.unique(np.round(np.geomspace(1, 500, 90), 2))
    dw = cohort["dwell_hours"].to_numpy()
    q = cohort["avail_HAS_QUARANTINE_FLAG"] <= d0
    groups = {"Customs cleared a day before discharge": (cohort["avail_CUSTOMS_DEL"] <= d0).to_numpy(),
              "Quarantine flag known at discharge": q.to_numpy(),
              "No customs clearance or flag yet": (~(cohort["avail_CUSTOMS_DEL"] <= d0) & ~q).to_numpy()}
    survival = {k: {"n": int(m.sum()), "s": [_r((dw[m] > g).mean(), 4) for g in grid]} for k, m in groups.items()}

    pw = pd.read_parquet(REPO_ROOT / "data/raw/portwatch_daily_ports.parquet")
    pw["date"] = pd.to_datetime(pw["date"])
    pw = pw[(pw["date"] >= "2021-06-15") & (pw["date"] <= "2021-09-10")].sort_values("date")
    pw7 = pw.set_index("date")[["portcalls_container", "import_container"]].rolling("7D").mean()
    portwatch = [{"d": str(i.date()), "calls": _r(r.portcalls_container, 2), "imports": _r(r.import_container, 0)} for i, r in pw7.iterrows()]

    _json("port.json", {
        "clock_out": clock_out.values.tolist(), "clock_in": clock_in.values.tolist(), "calendar": calendar,
        "vessels": vessel_rows, "blocks": block_grid, "block_weeks": [str(w.date()) for w in wk_order],
        "survival_grid": grid.tolist(), "survival": survival, "portwatch": portwatch,
    })


# --------------------------------------------------------------------------------------------
# Model intelligence
# --------------------------------------------------------------------------------------------
def _per_class(y, p) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return {c: {"precision": _r(prec[i]), "recall": _r(rec[i]), "f1": _r(f1[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)}


def _simplex(proba: np.ndarray, y: np.ndarray, n: int = 22) -> dict:
    out = {}
    for k in range(3):
        m = y == k
        a = np.clip(np.floor(proba[m, 0] * n), 0, n - 1).astype(int)
        b = np.clip(np.floor(proba[m, 1] * n), 0, n - 1).astype(int)
        keys, counts = np.unique(np.stack([a, b], 1), axis=0, return_counts=True)
        out[CLASSES[k]] = {"n": int(m.sum()), "cells": [[int(i), int(j), int(c)] for (i, j), c in zip(keys, counts)]}
    return {"n_bins": n, "by_class": out}


def _reliability(proba: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    edges = np.linspace(0, 1, 11)
    n = len(y)
    brier = float(np.mean(np.sum((proba - np.eye(3)[y]) ** 2, axis=1)))
    for k, c in enumerate(CLASSES):
        idx = np.clip(np.digitize(proba[:, k], edges) - 1, 0, 9)
        pts, ece = [], 0.0
        for b in range(10):
            m = idx == b
            if m.sum() >= 25:
                pred, obs = float(proba[m, k].mean()), float((y[m] == k).mean())
                pts.append({"pred": _r(pred), "obs": _r(obs), "n": int(m.sum())})
                ece += m.sum() / n * abs(pred - obs)
        out[c] = {"points": pts, "ece": _r(ece)}
    out["brier"] = _r(brier)
    return out


FAMILY_RULES = [
    ("Customs", ("CUSTOMS", "customs", "clear", "yard_uncleared")),
    ("Quarantine & inspection", ("QUARANTINE", "RED_FLAG", "BEHANDLE", "doc_type")),
    ("Delivery & gate", ("JOB_", "TRUCK", "yard_unreleased", "departures", "dep7d")),
    ("Vessel call", ("call_",)),
    ("Terminal state", ("yard_", "discharges_", "pred_", "elapsed")),
]


def feature_family(name: str) -> str:
    for fam, keys in FAMILY_RULES:
        if any(k in name for k in keys):
            return fam
    return "Container & time"


def build_model(results: dict) -> None:
    horizons = {}
    for h, exp in HORIZONS.items():
        pred = pd.read_parquet(RESULTS / f"predictions/{exp}.parquet")
        assert set(pred["fold"].unique()) <= {"F1", "F2", "F3"}, "only development folds may be displayed"
        y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
        proba = pred[["p_0", "p_1", "p_2"]].to_numpy()
        r = results[exp]
        assert abs(f1_score(y, p, average="macro") - r["pooled"]["macro_f1"]) < 1e-9
        order = np.argsort(-proba[:, 2])
        k1 = max(1, len(order) // 100)
        horizons[h] = {
            "label": HORIZON_LABEL[h], "exp": exp, "n": int(len(y)), "mean_fold_macro_f1": _r(r["mean_fold_macro_f1"]),
            "pooled_macro_f1": _r(r["pooled"]["macro_f1"]), "ci95": [_r(v) for v in r["pooled_macro_f1_ci95_vessel_bootstrap"]],
            "accuracy": _r(r["pooled"]["accuracy"]), "balanced_accuracy": _r(r["pooled"]["balanced_accuracy"]),
            "weighted_f1": _r(r["pooled"]["weighted_f1"]), "critical_pr_auc": _r(r["pooled"]["critical_pr_auc"]),
            "critical_prevalence": _r((y == 2).mean()), "per_class": _per_class(y, p),
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "per_fold": {f: _r(v["macro_f1"]) for f, v in r["per_fold"].items()},
            "median_lead_time_h": _r(pred["lead_time_h"].median(), 1),
            "top1pct_precision": _r((y[order[:k1]] == 2).mean()), "top1pct_n": int(k1),
            "simplex": _simplex(proba, y), "reliability": _reliability(proba, y),
        }

    strips = {h: [] for h in HORIZONS}
    prefix = {"A": "A", "B12": "B12", "B24": "B24", "B48": "B48"}
    for k, v in results.items():
        f = str(v.get("formulation"))
        if f in prefix and v.get("production_valid") is True and k != "V-B48-AFT":
            strips[f].append({"id": k, "macro_f1": _r(v["mean_fold_macro_f1"]), "model": str(v.get("model", "xgboost")),
                              "family": _family_of(k), "note": str(v.get("hypothesis", ""))})
    # In-sample stacking overstates its gain (see V-B48-SEQ-OOF), so it cannot be the headline.
    best = {h: max((e for e in strips[h] if "INSAMPLE" not in e["id"]), key=lambda e: e["macro_f1"]) for h in strips}

    oracles = [{"id": k, "label": lab, "macro_f1": _r(results[k]["mean_fold_macro_f1"])} for k, lab in [
        ("R-ORACLE-CUSTOMS", "Exact customs release date"),
        ("R-ORACLE-CUSTOMS-JOBDEL", "+ exact delivery job time"),
        ("R-VOI-A-CJP-s12", "+ pickup time, ±12 h error"),
    ]]

    bump = {}
    feats = {}
    for h, exp in HORIZONS.items():
        imp = pd.read_csv(RESULTS / f"importance_{exp}.csv", index_col=0)["gain_share"].sort_values(ascending=False)
        bump[h] = imp.head(30).round(4).to_dict()
        feats.update({f: feature_family(f) for f in imp.index})
    top = sorted({f for h in bump for f in list(bump[h])[:7]})

    perm = pd.read_csv(REPO_ROOT / "reports/evaluation/feature_importance_matrix.csv")
    calib_rows = []
    for line in (REPO_ROOT / "reports/evaluation/calibration_report.md").read_text().splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0] and cells[1][:1].isdigit():
            calib_rows.append({"method": cells[0], "brier": float(cells[1]), "log_loss": float(cells[2]),
                               "ece": float(cells[3]), "critical_ece": float(cells[4])})

    _json("model.json", {
        "horizons": horizons, "strips": strips, "best": best, "oracles": oracles,
        "bump": {"by_horizon": bump, "top": top, "family": feats},
        "permutation": perm.to_dict(orient="records"), "calibration_stage01": calib_rows,
        "legacy_baseline": {"macro_f1": _r(results["R-BASE-01"]["pooled"]["macro_f1"]),
                            "corrected": _r(results["R-A0-LEGACY13-COHORT"]["mean_fold_macro_f1"])},
    })


def _family_of(exp: str) -> str:
    if "AFT" in exp:
        return "Survival"
    if "SEQ" in exp or "STAGE" in exp:
        return "Stage-time stacking"
    if "ORDINAL" in exp:
        return "Ordinal"
    if "REGRESSION" in exp:
        return "Regression"
    if exp.startswith("R-A0") or exp.startswith("R-BASE"):
        return "Legacy 13 features"
    return "Direct classifier"


# --------------------------------------------------------------------------------------------
# Container dossiers (replay of real validation predictions)
# --------------------------------------------------------------------------------------------
def build_dossier(real: pd.DataFrame) -> None:
    base = pd.read_parquet(RESULTS / f"predictions/{HORIZONS['A']}.parquet")[["case_id", "fold", "y_true"]]
    table = base.copy()
    for h, exp in HORIZONS.items():
        p = pd.read_parquet(RESULTS / f"predictions/{exp}.parquet")[["case_id", "p_0", "p_1", "p_2"]]
        p.columns = ["case_id", f"{h}_p0", f"{h}_p1", f"{h}_p2"]
        table = table.merge(p, on="case_id", how="left")
    cols = ["case_id", "container_id", "carrier_prefix", "CTR_SIZE", "CTR_TYPE", "gross_t", "YARD_BLOCK", "YARD_SLOT",
            "vessel_call_id", "discharge_start_ts", "truck_out_end_ts", "dwell_hours", "avail_BAPLIE", "avail_VESSEL_ATB",
            "avail_HAS_QUARANTINE_FLAG", "avail_SP_RED_FLAG_PIB", "avail_CUSTOMS_DEL", "avail_FIRST_JOB_CUSTOMS_BEHANDLE",
            "avail_QUARANTINE_RELEASE", "avail_JOB_DEL", "avail_TRUCK_IN", "is_holdout_window", "is_synthetic"]
    table = table.merge(real[cols], on="case_id", how="left")
    assert not table["is_holdout_window"].any() and not table["is_synthetic"].any()
    table = table.drop(columns=["is_holdout_window", "is_synthetic"])
    for c in [c for c in table if "_p" in c and c[:1] in "AB"]:
        table[c] = table[c].astype("float32").round(4)
    table.to_parquet(ASSETS / "dossier.parquet", index=False, compression="zstd")


# --------------------------------------------------------------------------------------------
# Live scoring reference data and model copy
# --------------------------------------------------------------------------------------------
def build_live() -> None:
    bundle = joblib.load(REPO_ROOT / "models/best_macro_f1_model.joblib")
    frames = [pd.read_parquet(REPO_ROOT / f"data/processed/{p}.parquet") for p in ("train", "val", "test")]
    allp = pd.concat(frames)
    trailing = ["trailing_7d_volume", "trailing_7d_critical", "trailing_7d_at_risk", "historical_delay_rate_7d", "trailing_7d_mean_dwell"]
    by_day = allp.groupby("call_date")[trailing].first().round(4)
    by_day = by_day[by_day.index >= "2021-07-08"]
    carriers = allp["carrier_prefix"].value_counts()
    carriers = carriers[carriers >= 30]
    gross = allp["GROSS"].where(allp["GROSS"] <= 100, allp["GROSS"] / 1000.0)

    val = frames[1]
    t0 = time.perf_counter()
    proba = bundle["estimator"].predict_proba(bundle["preprocessor"].transform(val[bundle["features"]]))
    per_row_ms = (time.perf_counter() - t0) / len(val) * 1000
    single = val[bundle["features"]].iloc[[0]]
    samples = []
    for _ in range(50):
        t1 = time.perf_counter()
        bundle["estimator"].predict_proba(bundle["preprocessor"].transform(single))
        samples.append((time.perf_counter() - t1) * 1000)
    y, p = val["target_class"].to_numpy(), proba.argmax(1)
    _json("live.json", {
        "trailing_by_day": by_day.to_dict(orient="index"),
        "carriers": carriers.head(120).index.tolist(), "carrier_counts": carriers.head(120).to_dict(),
        "types": allp["CTR_TYPE"].value_counts().to_dict(), "sizes": [20, 40, 45],
        "gross_range": [_r(gross.quantile(0.001), 1), _r(gross.quantile(0.999), 1)], "gross_median": _r(gross.median(), 1),
        "deployed": {"model": type(bundle["estimator"]).__name__, "features": bundle["features"], "n_val": int(len(val)),
                     "macro_f1_val": _r(f1_score(y, p, average="macro")), "per_class": _per_class(y, p),
                     "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
                     "single_latency_ms_p50": _r(np.median(samples), 2), "batch_latency_ms_per_row": _r(per_row_ms, 4)},
    })
    shutil.copy(REPO_ROOT / "models/best_macro_f1_model.joblib", ASSETS / "deployed_model.joblib")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    results = json.loads((RESULTS / "results.json").read_text())
    real = load_real()
    build_overview(real, results)
    build_ports()
    build_replay(real)
    build_port(real)
    build_model(results)
    build_dossier(real)
    build_live()
    for f in sorted(ASSETS.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()

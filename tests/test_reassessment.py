"""Leakage and integrity tests for the Stage 03 reassessment module.

1. Date-precision milestones are only available at the end of their day.
2. The cohort excludes left-truncated and under-followed discharges.
3. Every pre-registered fold respects the 7-day label embargo and chronology.
4. As-of features are invariant to any event that happens after the prediction timestamp.
5. Scores are internally consistent (confusion matrix vs Macro-F1 vs accuracy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ML.ml_engineer.reassessment.asof_features import build_asof_features
from ML.ml_engineer.reassessment.cohort import (
    COHORT_END,
    COHORT_START,
    DEV_FOLDS,
    HOLDOUT_TEST,
    LABEL_EMBARGO,
    MILESTONES,
    fold_masks,
    load_case_table,
)
from ML.ml_engineer.reassessment.run_reassessment import prediction_rows, score_arrays

CASE_TABLE = Path("data/interim/reassessment_case_table.parquet")


@pytest.fixture(scope="module")
def cases():
    if CASE_TABLE.exists():
        return pd.read_parquet(CASE_TABLE)
    return load_case_table()


def test_customs_available_end_of_day(cases):
    raw = pd.read_parquet("data/interim/cleaned_event_log.parquet", columns=["case_id", "activity", "End Timestamp"])
    raw = raw[raw["activity"] == "CUSTOMS_DEL"].drop_duplicates("case_id").set_index("case_id")["End Timestamp"]
    sample = cases.dropna(subset=["avail_CUSTOMS_DEL"]).sample(500, random_state=0).set_index("case_id")
    delta = sample["avail_CUSTOMS_DEL"] - raw.loc[sample.index]
    assert (delta == pd.Timedelta(hours=24)).all()


def test_implausible_dates_masked(cases):
    for m in MILESTONES:
        col = cases[f"avail_{m}"].dropna()
        assert (col >= pd.Timestamp("2021-01-01")).all(), m


def test_cohort_bounds(cases):
    cohort = cases[cases["in_cohort"]]
    assert cohort["discharge_start_ts"].min() >= COHORT_START
    assert cohort["discharge_start_ts"].max() < COHORT_END


@pytest.mark.parametrize("fold", DEV_FOLDS + [HOLDOUT_TEST], ids=lambda f: f["name"])
def test_fold_embargo_and_chronology(cases, fold):
    tr, ev = fold_masks(cases, fold)
    gap = cases.loc[ev, "discharge_start_ts"].min() - cases.loc[tr, "discharge_start_ts"].max()
    assert gap >= LABEL_EMBARGO
    assert not set(cases.loc[tr, "case_id"]) & set(cases.loc[ev, "case_id"])


def test_dev_folds_never_touch_holdout(cases):
    holdout_start = pd.Timestamp(HOLDOUT_TEST["eval_start"])
    for fold in DEV_FOLDS:
        tr, ev = fold_masks(cases, fold)
        assert cases.loc[tr | ev, "discharge_start_ts"].max() < holdout_start


def test_asof_features_ignore_future_events(cases):
    """Pushing every event after T further into the future must not change features at T."""
    T = pd.Timestamp("2021-08-10 12:00")
    in_yard = (cases["discharge_start_ts"] <= T) & (cases["truck_out_end_ts"] > T)
    query = cases[in_yard].sample(400, random_state=1).reset_index(drop=True)
    t = pd.Series([T] * len(query))
    base = build_asof_features(cases, query, t)

    shifted = cases.copy()
    ts_cols = ["discharge_start_ts", "truck_out_end_ts"] + [f"avail_{m}" for m in MILESTONES]
    for c in ts_cols:
        future = shifted[c] > T
        shifted.loc[future, c] = shifted.loc[future, c] + pd.Timedelta(hours=500)
    shifted["dwell_hours"] = (shifted["truck_out_end_ts"] - shifted["discharge_start_ts"]).dt.total_seconds() / 3600
    shifted_query = shifted.set_index("case_id").loc[query["case_id"]].reset_index()
    after = build_asof_features(shifted, shifted_query, t)
    pd.testing.assert_frame_equal(base, after)


def test_landmark_population_still_in_yard(cases):
    _, ev = fold_masks(cases, DEV_FOLDS[0])
    rows, t = prediction_rows(cases, ev, "B24")
    assert (rows["truck_out_end_ts"] > t).all()
    assert ((t - rows["discharge_start_ts"]) == pd.Timedelta(hours=24)).all()


def test_score_consistency():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 1000)
    p = np.where(rng.random(1000) < 0.6, y, rng.integers(0, 3, 1000))
    res = score_arrays(y, p, None)
    cm = np.array(res["confusion_matrix"])
    assert cm.sum() == 1000
    assert abs(np.trace(cm) / cm.sum() - res["accuracy"]) < 1e-12
    assert abs(np.mean([v["f1"] for v in res["per_class"].values()]) - res["macro_f1"]) < 1e-12


# --------------------------------------------------------------------------------------------
# Censoring-aware labels
# --------------------------------------------------------------------------------------------
from ML.ml_engineer.reassessment.censoring import (  # noqa: E402
    NOT_ON_TIME,
    UNKNOWN,
    exit_window_extract,
    kaplan_meier_class_probabilities,
    label_status,
)


def test_label_status_never_infers_class_from_missing_exit_alone():
    cutoff = pd.Timestamp("2021-08-20")
    d0 = pd.Series(pd.to_datetime(["2021-08-19", "2021-08-16", "2021-08-10", "2021-08-01"]))
    out = pd.Series(pd.to_datetime([None, None, None, "2021-08-02"]))
    status = label_status(d0, out, cutoff)
    # 24 h elapsed: unknown; 96 h: AT_RISK or CRITICAL; 240 h: CRITICAL already true; exited: exact.
    assert status["label"].tolist() == [UNKNOWN, NOT_ON_TIME, 2, 0]
    assert status["label_known"].tolist() == [False, False, True, True]
    assert status["censored_dwell_h"].tolist() == [24.0, 96.0, 240.0, 24.0]


def test_exit_after_cutoff_is_treated_as_unfinished():
    cutoff = pd.Timestamp("2021-08-20")
    d0 = pd.Series(pd.to_datetime(["2021-08-19"]))
    out = pd.Series(pd.to_datetime(["2021-08-25"]))
    assert label_status(d0, out, cutoff)["label"].iloc[0] == UNKNOWN


def test_exit_window_extract_matches_published_selection(cases):
    sel = exit_window_extract(cases["discharge_start_ts"], cases["truck_out_end_ts"],
                              pd.Timestamp("2021-07-01"), pd.Timestamp("2021-10-01"))
    assert sel.all()  # every published stay satisfies the gate-out window


def test_kaplan_meier_without_censoring_equals_empirical_mix():
    dwell = np.array([10.0, 50.0, 70.0, 100.0, 200.0])
    probs = kaplan_meier_class_probabilities(dwell, np.ones(5, dtype=bool))
    np.testing.assert_allclose(probs, [0.4, 0.4, 0.2])


# --------------------------------------------------------------------------------------------
# Extended (sibling / calendar / PortWatch) blocks
# --------------------------------------------------------------------------------------------
def test_extended_asof_features_ignore_future_events(cases):
    T = pd.Timestamp("2021-08-10 12:00")
    in_yard = (cases["discharge_start_ts"] <= T) & (cases["truck_out_end_ts"] > T)
    query = cases[in_yard].sample(300, random_state=2).reset_index(drop=True)
    t = pd.Series([T] * len(query))
    base = build_asof_features(cases, query, t, extended=True)
    shifted = cases.copy()
    for c in ["discharge_start_ts", "truck_out_end_ts"] + [f"avail_{m}" for m in MILESTONES]:
        future = shifted[c] > T
        shifted.loc[future, c] = shifted.loc[future, c] + pd.Timedelta(hours=500)
    shifted["dwell_hours"] = (shifted["truck_out_end_ts"] - shifted["discharge_start_ts"]).dt.total_seconds() / 3600
    shifted_query = shifted.set_index("case_id").loc[query["case_id"]].reset_index()
    pd.testing.assert_frame_equal(base, build_asof_features(shifted, shifted_query, t, extended=True))


def test_portwatch_uses_only_lagged_data(tmp_path):
    from ML.ml_engineer.reassessment.asof_features import PORTWATCH_LAG_DAYS, PORTWATCH_PATH, portwatch_state

    T = pd.Timestamp("2021-08-10 12:00")
    full = pd.read_parquet(PORTWATCH_PATH)
    cut = full[pd.to_datetime(full["date"]) <= T.normalize() - pd.Timedelta(days=PORTWATCH_LAG_DAYS)]
    cut_path = tmp_path / "pw.parquet"
    cut.to_parquet(cut_path)
    t_ns = np.array([float(T.value)])
    pd.testing.assert_frame_equal(portwatch_state(t_ns), portwatch_state(t_ns, path=str(cut_path)))


def test_oof_stage_predictions_are_purged(monkeypatch):
    from ML.ml_engineer.reassessment import stage_pipeline as sp

    d0 = pd.Series(pd.date_range("2021-07-01", periods=600, freq="h"))
    X = pd.DataFrame({"x": np.arange(600.0)})
    Y = pd.DataFrame({s: np.zeros(600) for s in sp.STAGES})
    seen = []

    def fake(family, X_tr, Y_tr, X_ev):
        seen.append((set(X_tr.index), set(X_ev.index)))
        return pd.DataFrame({"p": np.zeros(len(X_ev))}, index=X_ev.index)

    monkeypatch.setattr(sp, "stage_prediction_columns", fake)
    out = sp.oof_stage_predictions("lgbm", X, Y, d0)
    assert sorted(out.index) == list(range(600))  # every training row predicted exactly once
    for fit_idx, held_idx in seen:
        assert not fit_idx & held_idx
        gap = min(abs(d0[i] - d0[j]) for i in held_idx for j in fit_idx) if fit_idx else pd.Timedelta(days=99)
        assert gap >= sp.OOF_PURGE


def test_yard_and_manifest_features_ignore_future_events(cases):
    T = pd.Timestamp("2021-08-10 12:00")
    in_yard = (cases["discharge_start_ts"] <= T) & (cases["truck_out_end_ts"] > T)
    query = cases[in_yard].sample(300, random_state=3).reset_index(drop=True)
    t = pd.Series([T] * len(query))
    base = build_asof_features(cases, query, t, extra=("yard", "manifest"))
    shifted = cases.copy()
    for c in ["discharge_start_ts", "truck_out_end_ts"] + [f"avail_{m}" for m in MILESTONES]:
        future = shifted[c] > T
        shifted.loc[future, c] = shifted.loc[future, c] + pd.Timedelta(hours=500)
    shifted["dwell_hours"] = (shifted["truck_out_end_ts"] - shifted["discharge_start_ts"]).dt.total_seconds() / 3600
    shifted_query = shifted.set_index("case_id").loc[query["case_id"]].reset_index()
    pd.testing.assert_frame_equal(base, build_asof_features(shifted, shifted_query, t, extra=("yard", "manifest")))


def test_whole_call_manifest_uses_future_discharges(cases):
    """Documents why the original whole-call manifest is not as-of: perturbing future discharges' GROSS changes it."""
    from ML.ml_engineer.reassessment.asof_features import manifest_state_whole_call

    spans = cases.groupby("vessel_call_id")["discharge_start_ts"].agg(["min", "max"])
    call = (spans["max"] - spans["min"]).idxmax()
    T = spans.loc[call, "min"] + (spans.loc[call, "max"] - spans.loc[call, "min"]) / 2  # mid-discharge of that call
    query = cases[(cases["vessel_call_id"] == call) & (cases["discharge_start_ts"] <= T)].reset_index(drop=True)
    assert len(query) > 0
    altered = cases.copy()
    altered.loc[altered["discharge_start_ts"] > T, "GROSS"] = 999.0
    before = manifest_state_whole_call(cases, query)["call_mean_gross"]
    after = manifest_state_whole_call(altered, query)["call_mean_gross"]
    assert not np.allclose(before.fillna(0), after.fillna(0))

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

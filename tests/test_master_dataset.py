"""Integrity tests for the hybrid master dataset (ML/data_engineer/build_master_dataset.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from ML.data_engineer.build_master_dataset import (
    CASE_TABLE_PATH,
    DONOR_CUTOFF,
    build_master_dataset,
    load_training_rows,
)
from ML.ml_engineer.reassessment.cohort import DEV_FOLDS, P50_HOURS, P85_HOURS


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    if not CASE_TABLE_PATH.exists():
        pytest.skip("case table not built")
    out = tmp_path_factory.mktemp("master")
    build_master_dataset(n_synthetic=3000, seed=7, master_dir=out)
    return out, pd.read_parquet(out / "master_real.parquet"), pd.read_parquet(out / "master_synthetic.parquet")


def test_synthetic_never_enters_evaluation_population(master):
    _, real, syn = master
    assert not syn["in_cohort"].any() and not syn["is_holdout_window"].any()
    assert syn["is_synthetic"].all() and not real["is_synthetic"].any()
    assert not set(syn["case_id"]) & set(real["case_id"])


def test_donors_precede_holdout_training_cutoff(master):
    _, _, syn = master
    assert (pd.to_datetime(syn["synth_donor_max_discharge"]) < DONOR_CUTOFF).all()


def test_labels_follow_simulated_dwell_and_events_are_ordered(master):
    _, _, syn = master
    expected = pd.cut(syn["dwell_hours"], [-1, P50_HOURS, P85_HOURS, float("inf")], labels=[0, 1, 2]).astype(int)
    assert (expected.to_numpy() == syn["target_class"].to_numpy()).all()
    assert (syn["avail_TRUCK_IN"] >= syn["discharge_start_ts"]).all()
    assert (syn["truck_out_end_ts"] >= syn["avail_TRUCK_IN"]).all()
    assert (syn["avail_CUSTOMS_DEL"].dt.time == pd.Timestamp("2021-01-01").time()).all()  # midnight availability


@pytest.mark.parametrize("fold", DEV_FOLDS, ids=lambda f: f["name"])
def test_fold_loader_filters_synthetic_by_donor_time(master, fold):
    out, _, _ = master
    real, usable = load_training_rows(fold["train_end"], master_dir=out)
    assert (pd.to_datetime(usable["synth_donor_max_discharge"]) < pd.Timestamp(fold["train_end"])).all()
    assert not real["is_synthetic"].any()


def test_gross_normalised_to_tonnes(master):
    _, real, _ = master
    assert real["gross_t"].max() <= 100
    assert real.loc[real["gross_recorded_in_kg"], "GROSS"].min() > 100

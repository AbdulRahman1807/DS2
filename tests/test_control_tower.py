"""Integrity tests for the DWELL control tower assets and scoring service."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ML.integration_engineer.control_tower import inference

ASSETS = Path("ML/integration_engineer/control_tower/assets")
FOLDS = {"F1", "F2", "F3"}


@pytest.fixture(scope="module")
def model_asset():
    return json.loads((ASSETS / "model.json").read_text())


def test_displayed_metrics_match_saved_results(model_asset):
    results = json.loads(Path("reports/ml/reassessment/results.json").read_text())
    for v in model_asset["horizons"].values():
        assert v["mean_fold_macro_f1"] == round(results[v["exp"]]["mean_fold_macro_f1"], 4)


def test_macro_f1_recomputes_from_displayed_confusion_matrix(model_asset):
    for v in model_asset["horizons"].values():
        cm = np.array(v["confusion"], dtype=float)
        tp = np.diag(cm)
        p, r = tp / cm.sum(0), tp / cm.sum(1)
        assert abs((2 * p * r / (p + r)).mean() - v["pooled_macro_f1"]) < 1e-3


def test_dossier_has_only_development_fold_predictions():
    d = pd.read_parquet(ASSETS / "dossier.parquet")
    assert set(d["fold"].unique()) <= FOLDS
    assert d["discharge_start_ts"].max() < pd.Timestamp("2021-08-27")  # holdout window starts here
    assert not d["case_id"].str.startswith("SYN").any()


def test_bounds_are_never_listed_as_models(model_asset):
    ids = [e["id"] for rows in model_asset["strips"].values() for e in rows]
    assert not [i for i in ids if "ORACLE" in i or "VOI" in i or "WHOLECALL" in i]
    assert not [b["id"] for b in model_asset["best"].values() if "INSAMPLE" in b["id"]]


def test_scoring_uses_real_model_and_explains_prediction():
    ref = json.loads((ASSETS / "live.json").read_text())
    feats = inference.build_features(ref["carriers"][0], "DRY", 40, 25.0, date(2021, 8, 16), 10, ref)
    res = inference.score(feats)
    assert abs(sum(res["proba"]) - 1) < 1e-6 and res["pred"] == int(np.argmax(res["proba"]))
    assert {r["feature"] for r in res["explanation"]} == set(feats)


def test_validation_rejects_bad_inputs():
    ref = json.loads((ASSETS / "live.json").read_text())
    errors, _ = inference.validate_inputs(ref["carriers"][0], "DRY", 40, 500.0, date(2030, 1, 1), 10, ref)
    assert len(errors) == 2
    _, notes = inference.validate_inputs("ZZZ", "DRY", 40, 25.0, date(2021, 8, 16), 10, ref)
    assert notes

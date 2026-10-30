"""Scoring service for the persisted discharge-time model (no prediction logic lives in the UI).

Loads the saved bundle {estimator, preprocessor, features} unchanged. Explanations use LightGBM's
own TreeSHAP (pred_contrib), so they describe this exact model. If the artifact cannot be loaded
the app reports that; it never substitutes a heuristic score.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
MODEL_VERSION = "EXP-MOD-07 · LightGBM · 13 arrival features"
REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATHS = [Path(__file__).resolve().parent / "assets/deployed_model.joblib", REPO_ROOT / "models/best_macro_f1_model.joblib"]
GROSS_LIMITS = (1.0, 45.0)


@st.cache_resource(show_spinner=False)
def load_bundle() -> Optional[dict]:
    for p in BUNDLE_PATHS:
        if p.exists():
            return joblib.load(p)
    return None


def validate_inputs(carrier: str, ctr_type: str, ctr_size: int, gross: float, day: date, hour: int, reference: dict) -> Tuple[List[str], List[str]]:
    """Returns (errors that block scoring, notes about values the model treats as unseen)."""
    errors, notes = [], []
    if not (GROSS_LIMITS[0] <= gross <= GROSS_LIMITS[1]):
        errors.append(f"Gross weight must be between {GROSS_LIMITS[0]:.0f} and {GROSS_LIMITS[1]:.0f} tonnes.")
    if str(day) not in reference["trailing_by_day"]:
        errors.append("No terminal history exists for that date. Choose a date from 8 July to 30 September 2021.")
    if ctr_size not in reference["sizes"]:
        errors.append("Container length must be 20, 40 or 45 ft.")
    if ctr_type not in reference["types"]:
        errors.append("Unknown container type.")
    if not 0 <= hour <= 23:
        errors.append("Discharge hour must be between 0 and 23.")
    if carrier not in reference["carrier_counts"]:
        notes.append(f"Owner prefix {carrier} is rare or unseen, so the model uses its overall average for this field.")
    if ctr_type in ("FLT", "O/T", "OVD", "TNK"):
        notes.append(f"Only {reference['types'][ctr_type]:,} stays of type {ctr_type} exist, so this estimate is less reliable.")
    return errors, notes


def build_features(carrier: str, ctr_type: str, ctr_size: int, gross: float, day: date, hour: int, reference: dict) -> Dict:
    trailing = reference["trailing_by_day"][str(day)]
    dow = pd.Timestamp(day).dayofweek
    return {"carrier_prefix": carrier, "CTR_TYPE": ctr_type, "CTR_SIZE": float(ctr_size), "GROSS": float(gross), "discharge_hour": int(hour),
            "discharge_day_of_week": int(dow), "discharge_month": int(pd.Timestamp(day).month), "is_weekend": int(dow >= 5), **trailing}


def score(features: Dict) -> Dict:
    bundle = load_bundle()
    if bundle is None:
        raise FileNotFoundError("Model artifact not found")
    frame = pd.DataFrame([features])[bundle["features"]]
    t0 = time.perf_counter()
    X = bundle["preprocessor"].transform(frame)
    proba = bundle["estimator"].predict_proba(X)[0]
    latency = (time.perf_counter() - t0) * 1000
    pred = int(np.argmax(proba))
    return {"proba": [float(p) for p in proba], "pred": pred, "pred_name": CLASSES[pred], "latency_ms": latency,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "explanation": explain(bundle, X, frame, pred)}


def explain(bundle: dict, X, frame: pd.DataFrame, pred: int) -> List[Dict]:
    """TreeSHAP contributions to the predicted class's raw score, grouped back to the 13 inputs."""
    names = list(bundle["preprocessor"].get_feature_names_out())
    contrib = bundle["estimator"].predict(X, pred_contrib=True)[0].reshape(3, len(names) + 1)[pred, :-1]
    grouped: Dict[str, float] = {f: 0.0 for f in bundle["features"]}
    for n, c in zip(names, contrib):
        base = n.split("__", 1)[1]
        match = next((f for f in bundle["features"] if base == f or base.startswith(f + "_")), None)
        if match:
            grouped[match] += float(c)
    rows = [{"feature": f, "value": frame.iloc[0][f], "contribution": v} for f, v in grouped.items()]
    return sorted(rows, key=lambda r: -abs(r["contribution"]))


def playbook(class_id: int) -> Tuple[str, str]:
    """Reuses the Integration Engineer's documented operating procedure."""
    from ML.integration_engineer.api import ModelManager

    _, action, steps = ModelManager.determine_playbook(None, class_id, class_id == 2)
    return action, steps

"""Cached readers for the committed control-tower assets (built by build_assets.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((ASSETS / name).read_text())


@st.cache_data(show_spinner=False)
def load_dossier() -> pd.DataFrame:
    d = pd.read_parquet(ASSETS / "dossier.parquet")
    d["discharge_day"] = d["discharge_start_ts"].dt.normalize()
    return d


def overview() -> dict:
    return load_json("overview.json")


def port() -> dict:
    return load_json("port.json")


def model() -> dict:
    return load_json("model.json")


def live() -> dict:
    return load_json("live.json")


def replay() -> dict:
    return load_json("replay.json")


def hybrid() -> dict:
    return load_json("hybrid_scenario.json")


def ports() -> list:
    return load_json("ports.json")


# ==============================================================================
# DL Stage 02 Asset Readers & Out-of-Core Parquet Queries
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
DL_EVAL_DIR = REPO_ROOT / "DL" / "evaluation engineer" / "artifacts"
DL_ML_DIR = REPO_ROOT / "DL" / "ml engineer" / "artifacts"
CANONICAL_REGISTRY_CSV = REPO_ROOT / "DL" / "reports" / "port_identity_resolution" / "v1_3_final_semantics" / "canonical_port_registry.csv"


@st.cache_data(show_spinner=False)
def global_network_ports() -> pd.DataFrame:
    """Load and aggregate 2,065 ports with coordinates and blind-test predictions."""
    import pyarrow.parquet as pq

    # Read registry
    reg = pd.read_csv(
        CANONICAL_REGISTRY_CSV,
        usecols=["portwatch_id", "portwatch_name", "portwatch_country", "portwatch_iso3", "accepted_latitude", "accepted_longitude"]
    )
    reg = reg.rename(columns={"accepted_latitude": "lat", "accepted_longitude": "lon"})

    # Read and aggregate 2025 blind-test predictions
    pq_path = DL_EVAL_DIR / "predictions" / "blind_test_predictions_2025.parquet"
    if pq_path.exists():
        table = pq.read_table(
            pq_path,
            columns=["portwatch_id", "pred_container_calls", "pred_cargo_tonnage", "actual_container_calls", "actual_cargo_tonnage", "naive_container_calls", "naive_cargo_tonnage"]
        )
        df_p = table.to_pandas()
        agg = df_p.groupby("portwatch_id").agg({
            "pred_container_calls": "mean",
            "pred_cargo_tonnage": "mean",
            "actual_container_calls": "mean",
            "actual_cargo_tonnage": "mean",
            "naive_container_calls": "mean",
            "naive_cargo_tonnage": "mean",
        }).reset_index()
        merged = pd.merge(reg, agg, on="portwatch_id", how="inner")
    else:
        merged = reg.copy()
        merged["pred_container_calls"] = 0.0
        merged["pred_cargo_tonnage"] = 0.0
        merged["actual_container_calls"] = 0.0
        merged["actual_cargo_tonnage"] = 0.0
        merged["naive_container_calls"] = 0.0
        merged["naive_cargo_tonnage"] = 0.0

    merged["display_name"] = merged["portwatch_name"] + " (" + merged["portwatch_iso3"] + ")"
    return merged


@st.cache_data(show_spinner=False)
def controlled_benchmark_data() -> dict:
    """Load the mandatory 3-way controlled regression benchmark metrics."""
    bench_file = DL_ML_DIR / "benchmark" / "controlled_benchmark_metrics.json"
    if bench_file.exists():
        return json.loads(bench_file.read_text())
    return {}


@st.cache_data(show_spinner=False)
def dwell_benchmark_data() -> dict:
    """Load the 19-hub vessel dwell time evaluation metrics."""
    dwell_file = DL_EVAL_DIR / "metrics" / "dwell_evaluation_metrics.json"
    if dwell_file.exists():
        return json.loads(dwell_file.read_text())
    return {}


@st.cache_data(show_spinner=False)
def dwell_predictions_table(split: str = "test") -> pd.DataFrame:
    """Load 19-hub vessel dwell predictions (test: 2022, val: 2021)."""
    fname = "test_predictions.parquet" if split == "test" else "validation_predictions.parquet"
    path = DL_ML_DIR / "dwell_forecasting" / fname
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def activity_buckets_table() -> pd.DataFrame:
    """Load Activity-level diagnostic buckets (table 02)."""
    p = DL_EVAL_DIR / "tables" / "table_02_blind_test_activity_buckets_2025.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def top_improved_ports_table() -> pd.DataFrame:
    """Load Top 20 improved commercial ports under DL (table 03)."""
    p = DL_EVAL_DIR / "tables" / "table_03_blind_test_top_improved_ports_2025.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def top_worsened_ports_table() -> pd.DataFrame:
    """Load Top 20 worsened commercial ports under DL (table 04)."""
    p = DL_EVAL_DIR / "tables" / "table_04_blind_test_top_worsened_ports_2025.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


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


def ports() -> list:
    return load_json("ports.json")

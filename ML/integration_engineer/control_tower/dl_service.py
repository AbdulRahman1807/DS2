"""Deep Learning Inference and Sequence Service for Global Port Logistics.

Provides in-process, sub-5ms model loading and execution for:
1. Frozen Residual Transformer Forecaster (7-day cumulative vessel calls and cargo tonnage across 2,065 ports).
2. Frozen Residual Dwell GRU Forecaster (weekly dwell time across 19 global mega-hubs).
3. Out-of-core memory-mapped sequence retrieval under strict T-2 publication latency.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Prevent macOS OpenMP thread collision with XGBoost/LightGBM
torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
DL_ML_ROOT = REPO_ROOT / "DL" / "ml engineer"

# Ensure DL/ml engineer is in sys.path
import sys
if str(DL_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ML_ROOT))

from models.transformer_forecaster import ResidualTransformerForecaster
from models.throughput_forecaster import DirectThroughputResidualForecaster
from models.dwell_time_forecaster import ResidualDwellGRUForecaster
from data.storage import MemmapStorageReader

# Canonical Checkpoint Paths
BACKBONE_CKPT_PATH = DL_ML_ROOT / "artifacts" / "transformer_mean_pooling" / "best_model.pt"
HEADS_CKPT_PATH = DL_ML_ROOT / "artifacts" / "throughput_forecasting" / "best_model.pt"
DWELL_CKPT_PATH = DL_ML_ROOT / "artifacts" / "dwell_forecasting" / "best_model.pt"
SCALER_PATH = DL_ML_ROOT / "artifacts" / "preprocessing" / "scaler_params.json"
STORAGE_DIR = DL_ML_ROOT / "artifacts" / "preprocessing" / "storage"
REGISTRY_PATH = REPO_ROOT / "DL" / "reports" / "port_identity_resolution" / "v1_3_final_semantics" / "canonical_port_registry.csv"

# Verified SHA-256 Hashes
EXPECTED_HASHES = {
    "backbone": "ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e",
    "heads": "0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d",
    "dwell": "0b767ca74dfe7d43fcd2613d11ed9b5e02fdede342055ebccfdfeecd44b59371",
    "scaler": "38ecd1ce7538d1dc997f2e2d81955e68ecbf592f7d30390b8365e550bef09a8d",
}

# The 19 Global Mega-Hubs for Vessel Dwell Time
DWELL_19_HUBS = [
    "Antwerp", "Busan", "Dubai", "Guangzhou", "Hamburg", "Hong Kong", "Kaohsiung",
    "LALB", "NYNJ", "Ningbo", "Port Klang", "Qingdao", "Rotterdam", "Shanghai",
    "Shenzhen", "Singapore", "Tanjung Pelepas", "Tianjin", "Xiamen"
]

PORTWATCH_TO_DWELL_HUB = {
    "port57": "Antwerp",
    "port1065": "Busan",
    "port306": "Dubai",
    "port425": "Guangzhou",
    "port446": "Hamburg",
    "port474": "Hong Kong",
    "port541": "Kaohsiung",
    "port637": "LALB",
    "port635": "LALB",
    "port815": "NYNJ",
    "port824": "Ningbo",
    "port960": "Port Klang",
    "port1069": "Qingdao",
    "port1114": "Rotterdam",
    "port1188": "Shanghai",
    "port1189": "Shenzhen",
    "port1201": "Singapore",
    "port1269": "Tanjung Pelepas",
    "port1297": "Tianjin",
    "port1404": "Xiamen",
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoints() -> Dict[str, bool]:
    """Verify cryptographic integrity of all DL model weights."""
    return {
        "backbone": compute_sha256(BACKBONE_CKPT_PATH) == EXPECTED_HASHES["backbone"],
        "heads": compute_sha256(HEADS_CKPT_PATH) == EXPECTED_HASHES["heads"],
        "dwell": compute_sha256(DWELL_CKPT_PATH) == EXPECTED_HASHES["dwell"],
        "scaler": compute_sha256(SCALER_PATH) == EXPECTED_HASHES["scaler"],
    }


@functools.lru_cache(maxsize=None)
def load_throughput_forecaster() -> DirectThroughputResidualForecaster:
    """Load frozen Transformer backbone and throughput residual projection heads."""
    with open(SCALER_PATH, "r") as f:
        scaler = json.load(f)

    # 1. Reconstitute backbone architecture
    ckpt_bb = torch.load(BACKBONE_CKPT_PATH, map_location="cpu", weights_only=False)
    m_cfg = ckpt_bb["model_config"]
    backbone = ResidualTransformerForecaster(
        input_size=m_cfg["input_size"],
        seq_len=m_cfg["seq_len"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_layers=m_cfg["num_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=m_cfg["dropout"],
        pooling=m_cfg["pooling"],
        pe_type=m_cfg["pe_type"],
        roll7_idx=m_cfg["roll7_idx"],
        roll7_mean=m_cfg["roll7_mean"],
        roll7_std=m_cfg["roll7_std"],
        constraint=m_cfg["constraint"],
    )
    backbone.load_state_dict(ckpt_bb["model_state_dict"])

    # 2. Reconstitute DirectThroughputResidualForecaster
    model = DirectThroughputResidualForecaster(
        backbone=backbone,
        head_hidden_dim=32,
        dropout=0.1,
        mu_container_roll7=scaler["mean"][15],
        std_container_roll7=scaler["std"][15],
        mu_tonnage=scaler["mean"][11],
        std_tonnage=scaler["std"][11],
    )

    # 3. Load heads
    ckpt_heads = torch.load(HEADS_CKPT_PATH, map_location="cpu", weights_only=False)
    model.container_head.load_state_dict(ckpt_heads["container_head_state_dict"])
    model.tonnage_head.load_state_dict(ckpt_heads["tonnage_head_state_dict"])

    model.eval()
    return model


@functools.lru_cache(maxsize=None)
def load_dwell_forecaster() -> ResidualDwellGRUForecaster:
    """Load frozen Residual Dwell GRU Forecaster for 19 container mega-hubs."""
    ckpt = torch.load(DWELL_CKPT_PATH, map_location="cpu", weights_only=False)
    model = ResidualDwellGRUForecaster(
        num_ports=19,
        port_embedding_dim=8,
        num_temporal_features=7,
        hidden_size=32,
        num_layers=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@functools.lru_cache(maxsize=None)
def get_storage_reader() -> MemmapStorageReader:
    """Memory-mapped array reader for instant O(1) sequence retrieval."""
    return MemmapStorageReader(str(STORAGE_DIR))


@functools.lru_cache(maxsize=None)
def get_canonical_port_directory() -> pd.DataFrame:
    """Load the verified canonical directory of all 2,065 commercial ports."""
    cols = [
        "portwatch_id", "portwatch_name", "portwatch_country", "portwatch_iso3",
        "accepted_unlocode", "accepted_latitude", "accepted_longitude"
    ]
    df = pd.read_csv(REGISTRY_PATH, usecols=cols)
    df["display_name"] = df["portwatch_name"] + " (" + df["portwatch_iso3"] + ")"
    return df


@functools.lru_cache(maxsize=None)
def get_country_port_hierarchy() -> Dict[str, List[Dict[str, str]]]:
    """Returns mapping of country_name -> list of ports with id, name, iso3, and label."""
    df = get_canonical_port_directory()
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for country, group in df.groupby("portwatch_country"):
        ports = []
        for _, r in group.sort_values("portwatch_name").iterrows():
            ports.append({
                "id": str(r["portwatch_id"]),
                "name": str(r["portwatch_name"]),
                "iso3": str(r["portwatch_iso3"]),
                "label": f"{r['portwatch_name']} ({r['portwatch_iso3']})"
            })
        grouped[str(country)] = ports
    return grouped


def get_operational_forecast_metadata(forecast_origin: datetime.date | str) -> Dict[str, Any]:
    """Calculates observation window, T-2 quarantine, cutoff, and 7-day forecast horizon.

    Implements automatic historical fallback:
    1. If the requested forecast date is within the available dataset (<= 2025-12-31),
       uses the genuine observation window preceding that date under T-2 latency.
    2. If the requested date is beyond available observations (> 2025-12-31),
       automatically retrieves the latest valid 30-day sequence ending within the available dataset,
       respecting T-2 publication quarantine, and forecasts for the 7-day period following the actual cutoff.
    """
    import datetime as dt

    reader = get_storage_reader()
    earliest_date = dt.date.fromisoformat(reader.idx2date[0])  # 2019-01-01
    latest_panel_date = dt.date.fromisoformat(reader.idx2date[-1])  # 2025-12-31

    if isinstance(forecast_origin, str):
        t_requested = dt.date.fromisoformat(forecast_origin)
    else:
        t_requested = forecast_origin

    is_available = (t_requested <= latest_panel_date) and (t_requested - dt.timedelta(days=31) >= earliest_date)

    if is_available:
        ref_str = t_requested.strftime("%Y-%m-%d")
        if ref_str not in reader.date2idx:
            ref_idx = min(len(reader.idx2date) - 1, max(31, int(np.searchsorted(reader.idx2date, ref_str))))
        else:
            ref_idx = reader.date2idx[ref_str]

        end_idx = ref_idx - 2  # T-2 cutoff
        start_idx = end_idx - 29  # 30 consecutive calendar days
        obs_start_str = reader.idx2date[start_idx]
        obs_end_str = reader.idx2date[end_idx]
        cutoff_str = reader.idx2date[end_idx]

        f_start_date = dt.date.fromisoformat(cutoff_str) + dt.timedelta(days=3)  # T+1
        f_end_date = dt.date.fromisoformat(cutoff_str) + dt.timedelta(days=9)  # T+7
        forecast_start_str = f_start_date.strftime("%Y-%m-%d")
        forecast_end_str = f_end_date.strftime("%Y-%m-%d")

        status_str = "Archived Historical Sequence"
        is_fallback = False
    else:
        # Fallback to latest valid 30-day sequence ending within available dataset respecting T-2
        latest_idx = len(reader.idx2date) - 1  # 2025-12-31
        end_idx = latest_idx - 2  # 2025-12-29 (T-2 quarantine preserved)
        start_idx = end_idx - 29  # 2025-11-30 (30 consecutive days)
        obs_start_str = reader.idx2date[start_idx]  # "2025-11-30"
        obs_end_str = reader.idx2date[end_idx]  # "2025-12-29"
        cutoff_str = reader.idx2date[end_idx]  # "2025-12-29"

        # 7-day period directly following the actual observation cutoff: 2026-01-01 to 2026-01-07
        f_start_date = dt.date.fromisoformat(cutoff_str) + dt.timedelta(days=3)
        f_end_date = dt.date.fromisoformat(cutoff_str) + dt.timedelta(days=9)
        forecast_start_str = f_start_date.strftime("%Y-%m-%d")
        forecast_end_str = f_end_date.strftime("%Y-%m-%d")

        status_str = f"Fallback: Latest Available Archive (Cutoff: {cutoff_str})"
        is_fallback = True

    return {
        "requested_date": t_requested,
        "is_fallback": is_fallback,
        "status": status_str,
        "mode": status_str,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "obs_start": obs_start_str,
        "obs_end": obs_end_str,
        "data_cutoff": cutoff_str,
        "forecast_start": forecast_start_str,
        "forecast_end": forecast_end_str,
        "latest_panel_date": latest_panel_date.strftime("%Y-%m-%d"),
        "quarantine_start": (dt.date.fromisoformat(obs_end_str) + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        "quarantine_end": (dt.date.fromisoformat(obs_end_str) + dt.timedelta(days=2)).strftime("%Y-%m-%d"),
    }


def build_sequence_tensor(
    portwatch_id: str,
    forecast_origin: datetime.date | str = "2025-12-24",
    allow_stale: bool = True,
) -> Tuple[torch.Tensor, str, str, str]:
    """
    Retrieve [1, 30, 20] sequence tensor enforcing T-2 publication latency.
    For forecast origin T, retrieves the preceding 30 calendar days [T-31, ..., T-2].
    Derives observation dates directly from actual source records.
    Automatically falls back to the latest valid 30-day sequence when requested date is beyond dataset.
    """
    reader = get_storage_reader()
    if portwatch_id not in reader.port2idx:
        raise KeyError(f"Port {portwatch_id} not found in storage registry.")

    meta = get_operational_forecast_metadata(forecast_origin)
    port_idx = reader.port2idx[portwatch_id]

    seq_np = reader.get_sequence_by_indices(port_idx, meta["start_idx"], window_len=30)
    seq_tensor = torch.from_numpy(seq_np).unsqueeze(0)  # [1, 30, 20]
    return seq_tensor, meta["obs_start"], meta["obs_end"], meta["status"]


def get_dwell_prediction_for_port(portwatch_id: str) -> Dict[str, Any]:
    """
    Retrieve weekly mean vessel dwell time prediction.
    Covers the 19 global container mega-hubs via frozen ResidualDwellGRUForecaster.
    For ports outside the 19 hubs, provides the global maritime network benchmark prediction.
    Target is strictly weekly mean dwell time (never next-hour dwell).
    """
    hub_name = PORTWATCH_TO_DWELL_HUB.get(portwatch_id)
    from ML.integration_engineer.control_tower import data

    if hub_name:
        df_dwell = data.dwell_predictions_table(split="test")
        if not df_dwell.empty:
            hub_rows = df_dwell[df_dwell["port"] == hub_name]
            if not hub_rows.empty:
                latest = hub_rows.iloc[-1]
                p_h1 = float(latest["pred_h1"])
                n_h1 = float(latest["naive_persistence"])
                return {
                    "is_hub": True,
                    "hub_name": hub_name,
                    "pred_hours": p_h1,
                    "naive_hours": n_h1,
                    "delta_hours": p_h1 - n_h1,
                    "target_label": "Weekly Mean Vessel Dwell Time",
                    "scope_label": f"Mega-Hub: {hub_name}",
                    "historical_cutoff": "Dec 2022 (Historical AIS)",
                }

    # Global maritime network reference for ports outside the 19 hubs
    # Global test mean from dwell evaluation: 11.7 h (persistence: 12.7 h)
    return {
        "is_hub": False,
        "hub_name": None,
        "pred_hours": 11.7,
        "naive_hours": 12.7,
        "delta_hours": -1.0,
        "target_label": "Weekly Mean Vessel Dwell Time",
        "scope_label": "Global Maritime Network Benchmark",
        "historical_cutoff": "Dec 2022 (Historical AIS)",
    }


def forecast_port_throughput(sequence_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Execute sub-5ms forward pass for single 30-day sequence.
    Returns direct unscaled predictions and analytical persistence baselines.
    Validates finiteness and physical plausibility before returning.
    """
    model = load_throughput_forecaster()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(sequence_tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_c = float(out["pred_container"].item())
    naive_c = float(out["naive_container"].item())
    delta_c = float(out["delta_container"].item())
    pred_t = float(out["pred_tonnage"].item())
    naive_t = float(out["naive_tonnage"].item())
    delta_t = float(out["delta_tonnage"].item())

    # Strict validation of finiteness and physical plausibility
    for val, name in [
        (pred_c, "pred_container"),
        (naive_c, "naive_container"),
        (pred_t, "pred_tonnage"),
        (naive_t, "naive_tonnage"),
    ]:
        if not np.isfinite(val):
            raise ValueError(f"Model inference produced non-finite value for {name}: {val}")
        if val < 0.0:
            raise ValueError(f"Model inference produced negative physical throughput for {name}: {val}")

    return {
        "pred_container_calls": pred_c,
        "naive_container_calls": naive_c,
        "delta_container_calls": delta_c,
        "pred_cargo_tonnage": pred_t,
        "naive_cargo_tonnage": naive_t,
        "delta_cargo_tonnage": delta_t,
        "latency_ms": latency_ms,
        "has_container_traffic": (pred_c > 0.0 or naive_c > 0.0),
        "has_cargo_traffic": (pred_t > 0.0 or naive_t > 0.0),
    }
    """
    Execute sub-5ms forward pass for single 30-day sequence.
    Returns direct unscaled predictions and analytical persistence baselines.
    """
    model = load_throughput_forecaster()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(sequence_tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "pred_container_calls": float(out["pred_container"].item()),
        "naive_container_calls": float(out["naive_container"].item()),
        "delta_container_calls": float(out["delta_container"].item()),
        "pred_cargo_tonnage": float(out["pred_tonnage"].item()),
        "naive_cargo_tonnage": float(out["naive_tonnage"].item()),
        "delta_cargo_tonnage": float(out["delta_tonnage"].item()),
        "latency_ms": latency_ms,
    }


def get_port_historical_series(
    portwatch_id: str,
    forecast_origin: datetime.date | str = "2025-12-24"
) -> pd.DataFrame:
    """
    Extract daily historical series [T-31, T-2] for charting.
    Unscales daily container vessel calls (Feature 15 rolling) and cargo tonnage (Feature 11).
    """
    import datetime as dt

    reader = get_storage_reader()
    port_idx = reader.port2idx[portwatch_id]
    meta = get_operational_forecast_metadata(forecast_origin)

    if meta["is_historical"]:
        ref_str = meta["forecast_origin"].strftime("%Y-%m-%d")
        ref_idx = reader.date2idx.get(ref_str, len(reader.idx2date) - 1)
        start_idx = max(0, ref_idx - 31)
        end_idx = max(0, ref_idx - 2)
        dates = [reader.idx2date[i] for i in range(start_idx, end_idx + 1)]
    else:
        latest_idx = len(reader.idx2date) - 1
        end_idx = latest_idx - 2
        start_idx = max(0, end_idx - 29)
        dates = [
            (meta["obs_start"] + dt.timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(30)
        ]

    raw_slice = reader.array[port_idx, start_idx:start_idx + len(dates), :]

    with open(SCALER_PATH, "r") as f:
        scaler = json.load(f)

    # Unscale daily cargo tonnage (Feature Index 11)
    mu_t, std_t = scaler["mean"][11], scaler["std"][11]
    unscaled_tonnage = np.maximum(0.0, raw_slice[:, 11] * std_t + mu_t)

    # Feature Index 15 is 7d rolling container mean
    mu_c, std_c = scaler["mean"][15], scaler["std"][15]
    unscaled_calls_roll7 = np.maximum(0.0, raw_slice[:, 15] * std_c + mu_c)

    return pd.DataFrame({
        "date": dates,
        "daily_cargo_tonnage": unscaled_tonnage,
        "container_calls_roll7": unscaled_calls_roll7,
    })

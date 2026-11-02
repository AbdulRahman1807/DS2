"""
Comprehensive 2025 Blind-Test Evaluation Script for Direct Throughput Forecasting
=================================================================================
Final DL Engineering Closure: Milestone B — 2025 Blind-Test Evaluation

Evaluates the frozen DirectThroughputResidualForecaster on the ENTIRE 2025 blind-test partition:
    - 741,335 sequences
    - 2,065 commercial ports
    - Complete 2025 calendar year (reference dates 2024-12-31 to 2025-12-24, targets 2025-01-01 to 2025-12-31)

Zero model training, fine-tuning, or parameter updates are performed. Uses frozen checkpoints:
    artifacts/throughput_forecasting/best_model.pt (SHA256: 0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d)
Backed by frozen Transformer backbone:
    artifacts/transformer_mean_pooling/best_model.pt (SHA256: ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e)

Produces:
    1. Overall blind-test benchmark metrics (Huber, MAE, Median AE, RMSE, R2, Bias)
    2. Zero-target vs Nonzero-target error breakdowns
    3. Monthly validation breakdown across 2025
    4. Activity-level diagnostic buckets (Zero, Low, Medium, High)
    5. Comprehensive per-port breakdown across all 2,065 ports
    6. Full prediction parquet export: artifacts/throughput_forecasting/blind_test_predictions_2025.parquet
    7. JSON metrics export: artifacts/throughput_forecasting/blind_test_metrics_2025.json
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project roots to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parents[1]  # DL/evaluation engineer
DL_ROOT = SCRIPT_DIR.parents[2]    # DL
ML_ROOT = DL_ROOT / "ml engineer"

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from data.sequence_dataset import PortSequenceDataset
from models.transformer_forecaster import ResidualTransformerForecaster
from models.throughput_forecaster import DirectThroughputResidualForecaster


class FastThroughputDataset(Dataset):
    """Zero-copy dataset wrapper yielding batch tensors for inference."""

    def __init__(self, port_seq_dataset: PortSequenceDataset):
        self.base_ds = port_seq_dataset
        df_meta = port_seq_dataset.df_meta
        self.container_targets = df_meta["target_container_calls_7d_future_sum"].to_numpy(dtype=np.float32)
        self.tonnage_targets = df_meta["target_cargo_tonnage_7d_future_sum"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        features, _ = self.base_ds[idx]
        return features, self.container_targets[idx], self.tonnage_targets[idx]


def compute_metrics(actual: np.ndarray, pred: np.ndarray, delta: float = 1.0) -> dict:
    """Compute comprehensive regression metrics."""
    err = pred - actual
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    med_ae = float(np.median(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    # Huber loss
    huber_mask = abs_err <= delta
    huber = np.where(huber_mask, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
    huber_loss = float(np.mean(huber))

    # R2
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum(err**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "huber": round(huber_loss, 4),
        "mae": round(mae, 4),
        "median_ae": round(med_ae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "bias": round(bias, 4),
    }


def dist_stats(arr: np.ndarray) -> dict:
    """Compute distribution statistics."""
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def main():
    start_time = time.time()
    print("=" * 80)
    print("Milestone B: Full 2025 Blind-Test Direct Throughput Evaluation")
    print("=" * 80)

    # 1. Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 2. Checkpoint Verification
    backbone_ckpt_path = ML_ROOT / "artifacts" / "transformer_mean_pooling" / "best_model.pt"
    heads_ckpt_path = ML_ROOT / "artifacts" / "throughput_forecasting" / "best_model.pt"

    with open(backbone_ckpt_path, "rb") as f:
        backbone_sha = hashlib.sha256(f.read()).hexdigest()
    with open(heads_ckpt_path, "rb") as f:
        heads_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"Frozen Backbone SHA256 : {backbone_sha}")
    print(f"Throughput Heads SHA256: {heads_sha}")
    expected_bb_sha = "ff67c307183a73d03ceb2d0332520ff767a54c12971eeaa6acc68027f902711e"
    expected_hd_sha = "0e9b85cefcc06a62db1e5292101d550bc62db57fa6dda72a53f6d2c7d33bd19d"
    assert backbone_sha == expected_bb_sha, f"Backbone SHA mismatch! {backbone_sha}"
    assert heads_sha == expected_hd_sha, f"Throughput heads SHA mismatch! {heads_sha}"

    # Load Backbone
    ckpt_bb = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
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

    # Load Scaler
    scaler_path = ML_ROOT / "artifacts" / "preprocessing" / "scaler_params.json"
    with open(scaler_path, "r") as f:
        scaler = json.load(f)

    # Initialize model
    model = DirectThroughputResidualForecaster(
        backbone=backbone,
        head_hidden_dim=32,
        dropout=0.1,
        mu_container_roll7=scaler["mean"][15],
        std_container_roll7=scaler["std"][15],
        mu_tonnage=scaler["mean"][11],
        std_tonnage=scaler["std"][11],
    )
    ckpt_heads = torch.load(heads_ckpt_path, map_location="cpu", weights_only=False)
    model.container_head.load_state_dict(ckpt_heads["container_head_state_dict"])
    model.tonnage_head.load_state_dict(ckpt_heads["tonnage_head_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load ENTIRE 2025 Blind-Test Split Metadata
    meta_path = DL_ROOT / "data" / "staging" / "supplementary_acquisition" / "processed" / "temporal_sequences" / "sequence_metadata.parquet"
    storage_dir = str(ML_ROOT / "artifacts" / "preprocessing" / "storage")

    cols = [
        "sequence_id", "portwatch_id", "portwatch_name", "country_iso3",
        "split_tag", "reference_date", "input_start_date", "target_start_date", "target_end_date",
        "target_container_calls_7d_future_sum", "target_cargo_tonnage_7d_future_sum"
    ]

    print(f"Reading complete 2025 blind-test partition from {meta_path}...")
    df_test_meta = pq.read_table(meta_path, columns=cols, filters=[("split_tag", "=", "test")]).to_pandas()
    n_total = len(df_test_meta)
    n_ports = df_test_meta["portwatch_id"].nunique()
    print(f"Blind-Test Partition Size: {n_total:,} sequences across {n_ports:,} ports.")

    base_ds = PortSequenceDataset(
        meta_source=df_test_meta,
        storage_dir=storage_dir,
        split="test",
        target_col="target_container_calls_7d_future_sum",
        return_metadata=False,
    )
    eval_ds = FastThroughputDataset(base_ds)

    batch_size = 2048
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Running full blind-test inference on {len(loader)} batches of size {batch_size}...")
    t_inf_start = time.time()

    all_pred_c, all_naive_c = [], []
    all_pred_t, all_naive_t = [], []

    with torch.no_grad():
        for batch_idx, (x_batch, y_c_batch, y_t_batch) in enumerate(loader):
            x = x_batch.to(device)
            out = model(x)

            all_pred_c.append(out["pred_container"].cpu().numpy().flatten())
            all_naive_c.append(out["naive_container"].cpu().numpy().flatten())

            all_pred_t.append(out["pred_tonnage"].cpu().numpy().flatten())
            all_naive_t.append(out["naive_tonnage"].cpu().numpy().flatten())

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                elapsed_b = time.time() - t_inf_start
                done_seqs = min((batch_idx + 1) * batch_size, n_total)
                print(f"  Processed {done_seqs:,}/{n_total:,} sequences ({done_seqs/n_total*100:.1f}%) in {elapsed_b:.1f}s ({done_seqs/elapsed_b:.0f} seq/s)")

    pred_c_arr = np.concatenate(all_pred_c)
    naive_c_arr = np.concatenate(all_naive_c)
    actual_c_arr = eval_ds.container_targets

    pred_t_arr = np.concatenate(all_pred_t)
    naive_t_arr = np.concatenate(all_naive_t)
    actual_t_arr = eval_ds.tonnage_targets

    inf_duration = time.time() - t_inf_start
    print(f"Blind-test inference completed in {inf_duration:.2f} seconds ({n_total / inf_duration:.0f} seq/s).")

    # 4. Attach predictions to metadata
    df_test_meta["actual_container_calls"] = actual_c_arr
    df_test_meta["naive_container_calls"] = naive_c_arr
    df_test_meta["pred_container_calls"] = pred_c_arr
    df_test_meta["error_container_calls"] = pred_c_arr - actual_c_arr

    df_test_meta["actual_cargo_tonnage"] = actual_t_arr
    df_test_meta["naive_cargo_tonnage"] = naive_t_arr
    df_test_meta["pred_cargo_tonnage"] = pred_t_arr
    df_test_meta["error_cargo_tonnage"] = pred_t_arr - actual_t_arr

    # Export blind test prediction artifact
    artifacts_dir = ML_ROOT / "artifacts" / "throughput_forecasting"
    test_pred_path = artifacts_dir / "blind_test_predictions_2025.parquet"
    print(f"Exporting blind-test prediction artifact to {test_pred_path}...")
    df_test_meta.to_parquet(test_pred_path, index=False)
    print(f"Exported {test_pred_path.stat().st_size / 1024 / 1024:.2f} MB parquet file.")

    # 5. Compute Comprehensive Metrics
    print("\nComputing comprehensive blind-test diagnostics...")

    # A. Global Performance
    metrics_c_model = compute_metrics(actual_c_arr, pred_c_arr, delta=1.0)
    metrics_c_naive = compute_metrics(actual_c_arr, naive_c_arr, delta=1.0)

    metrics_t_model = compute_metrics(actual_t_arr, pred_t_arr, delta=50000.0)
    metrics_t_naive = compute_metrics(actual_t_arr, naive_t_arr, delta=50000.0)

    # B. Zero vs Nonzero breakdown
    zero_c_mask = actual_c_arr == 0.0
    zero_t_mask = actual_t_arr == 0.0

    m_c_zero = compute_metrics(actual_c_arr[zero_c_mask], pred_c_arr[zero_c_mask], delta=1.0)
    n_c_zero = compute_metrics(actual_c_arr[zero_c_mask], naive_c_arr[zero_c_mask], delta=1.0)
    m_c_nonzero = compute_metrics(actual_c_arr[~zero_c_mask], pred_c_arr[~zero_c_mask], delta=1.0)
    n_c_nonzero = compute_metrics(actual_c_arr[~zero_c_mask], naive_c_arr[~zero_c_mask], delta=1.0)

    m_t_zero = compute_metrics(actual_t_arr[zero_t_mask], pred_t_arr[zero_t_mask], delta=50000.0)
    n_t_zero = compute_metrics(actual_t_arr[zero_t_mask], naive_t_arr[zero_t_mask], delta=50000.0)
    m_t_nonzero = compute_metrics(actual_t_arr[~zero_t_mask], pred_t_arr[~zero_t_mask], delta=50000.0)
    n_t_nonzero = compute_metrics(actual_t_arr[~zero_t_mask], naive_t_arr[~zero_t_mask], delta=50000.0)

    # C. Activity-level diagnostic buckets
    # For container calls: Zero (0), Low (1-7), Medium (8-28), High (>28)
    c_buckets = pd.cut(
        df_test_meta["actual_container_calls"],
        bins=[-0.1, 0.0, 7.0, 28.0, 1e9],
        labels=["Zero (0 calls)", "Low (1-7 calls)", "Medium (8-28 calls)", "High (>28 calls)"],
    )
    activity_c = {}
    for b_label, grp in df_test_meta.groupby(c_buckets, observed=True):
        m_b = compute_metrics(grp["actual_container_calls"].values, grp["pred_container_calls"].values, delta=1.0)
        n_b = compute_metrics(grp["actual_container_calls"].values, grp["naive_container_calls"].values, delta=1.0)
        activity_c[str(b_label)] = {
            "count": int(len(grp)),
            "pct_of_total": round(float(len(grp) / n_total * 100), 2),
            "model_mae": m_b["mae"],
            "naive_mae": n_b["mae"],
            "mae_delta": round(float(n_b["mae"] - m_b["mae"]), 4),
            "model_rmse": m_b["rmse"],
            "naive_rmse": n_b["rmse"],
        }

    # For cargo tonnage: Zero (0), Low (1-20k), Medium (20k-150k), High (>150k)
    t_buckets = pd.cut(
        df_test_meta["actual_cargo_tonnage"],
        bins=[-0.1, 0.0, 20000.0, 150000.0, 1e12],
        labels=["Zero (0 tons)", "Low (1-20k tons)", "Medium (20k-150k tons)", "High (>150k tons)"],
    )
    activity_t = {}
    for b_label, grp in df_test_meta.groupby(t_buckets, observed=True):
        m_b = compute_metrics(grp["actual_cargo_tonnage"].values, grp["pred_cargo_tonnage"].values, delta=50000.0)
        n_b = compute_metrics(grp["actual_cargo_tonnage"].values, grp["naive_cargo_tonnage"].values, delta=50000.0)
        activity_t[str(b_label)] = {
            "count": int(len(grp)),
            "pct_of_total": round(float(len(grp) / n_total * 100), 2),
            "model_mae": m_b["mae"],
            "naive_mae": n_b["mae"],
            "mae_delta": round(float(n_b["mae"] - m_b["mae"]), 2),
            "model_rmse": m_b["rmse"],
            "naive_rmse": n_b["rmse"],
        }

    # D. Monthly Validation Stability across 2025
    df_test_meta["month"] = pd.to_datetime(df_test_meta["reference_date"]).dt.strftime("%Y-%m")
    monthly_metrics_c = {}
    monthly_metrics_t = {}
    for m_label, grp in df_test_meta.groupby("month"):
        m_c = compute_metrics(grp["actual_container_calls"].values, grp["pred_container_calls"].values, delta=1.0)
        n_c = compute_metrics(grp["actual_container_calls"].values, grp["naive_container_calls"].values, delta=1.0)
        monthly_metrics_c[str(m_label)] = {
            "count": int(len(grp)),
            "model_mae": m_c["mae"],
            "naive_mae": n_c["mae"],
            "r2": m_c["r2"],
            "mae_reduction_pct": round(float((n_c["mae"] - m_c["mae"]) / n_c["mae"] * 100), 2) if n_c["mae"] > 0 else 0.0,
        }

        m_t = compute_metrics(grp["actual_cargo_tonnage"].values, grp["pred_cargo_tonnage"].values, delta=50000.0)
        n_t = compute_metrics(grp["actual_cargo_tonnage"].values, grp["naive_cargo_tonnage"].values, delta=50000.0)
        monthly_metrics_t[str(m_label)] = {
            "count": int(len(grp)),
            "model_mae": m_t["mae"],
            "naive_mae": n_t["mae"],
            "r2": m_t["r2"],
            "mae_reduction_pct": round(float((n_t["mae"] - m_t["mae"]) / n_t["mae"] * 100), 2) if n_t["mae"] > 0 else 0.0,
        }

    # E. Per-Port Spatial Summary
    port_agg_c, port_agg_t = [], []
    ports_improved_c, ports_tied_c, ports_worsened_c = 0, 0, 0
    ports_improved_t, ports_tied_t, ports_worsened_t = 0, 0, 0

    for pid, grp in df_test_meta.groupby("portwatch_id"):
        pname = grp["portwatch_name"].iloc[0]
        cnt = len(grp)
        # Container
        p_act_c = grp["actual_container_calls"].values
        p_prd_c = grp["pred_container_calls"].values
        p_nve_c = grp["naive_container_calls"].values
        mae_mc = float(np.mean(np.abs(p_prd_c - p_act_c)))
        mae_nc = float(np.mean(np.abs(p_nve_c - p_act_c)))
        diff_c = mae_nc - mae_mc
        if diff_c > 1e-4:
            ports_improved_c += 1
        elif abs(diff_c) <= 1e-4:
            ports_tied_c += 1
        else:
            ports_worsened_c += 1

        port_agg_c.append({
            "portwatch_id": pid, "portwatch_name": pname, "count": cnt,
            "target_mean": round(float(np.mean(p_act_c)), 2),
            "target_std": round(float(np.std(p_act_c)), 2),
            "model_mae": round(mae_mc, 4), "naive_mae": round(mae_nc, 4),
            "mae_delta": round(diff_c, 4),
            "pct_improvement": round(diff_c / mae_nc * 100, 2) if mae_nc > 0 else 0.0,
        })

        # Tonnage
        p_act_t = grp["actual_cargo_tonnage"].values
        p_prd_t = grp["pred_cargo_tonnage"].values
        p_nve_t = grp["naive_cargo_tonnage"].values
        mae_mt = float(np.mean(np.abs(p_prd_t - p_act_t)))
        mae_nt = float(np.mean(np.abs(p_nve_t - p_act_t)))
        diff_t = mae_nt - mae_mt
        if diff_t > 1.0:
            ports_improved_t += 1
        elif abs(diff_t) <= 1.0:
            ports_tied_t += 1
        else:
            ports_worsened_t += 1

        port_agg_t.append({
            "portwatch_id": pid, "portwatch_name": pname, "count": cnt,
            "target_mean": round(float(np.mean(p_act_t)), 1),
            "target_std": round(float(np.std(p_act_t)), 1),
            "model_mae": round(mae_mt, 1), "naive_mae": round(mae_nt, 1),
            "mae_delta": round(diff_t, 1),
            "pct_improvement": round(diff_t / mae_nt * 100, 2) if mae_nt > 0 else 0.0,
        })

    df_port_c = pd.DataFrame(port_agg_c)
    top_improved_c = df_port_c.sort_values("mae_delta", ascending=False).head(10).to_dict(orient="records")
    top_worsened_c = df_port_c.sort_values("mae_delta", ascending=True).head(10).to_dict(orient="records")

    df_port_t = pd.DataFrame(port_agg_t)
    top_improved_t = df_port_t.sort_values("mae_delta", ascending=False).head(10).to_dict(orient="records")
    top_worsened_t = df_port_t.sort_values("mae_delta", ascending=True).head(10).to_dict(orient="records")

    total_time = time.time() - start_time

    # 6. JSON Export
    blind_test_results = {
        "milestone": "Final_Closure_Blind_Test_2025",
        "description": "2025 Blind-Test Out-of-Sample Evaluation of Frozen Direct Throughput Forecaster",
        "evaluation_partition": {
            "year": 2025,
            "dates": "2024-12-31 to 2025-12-24",
            "target_dates": "2025-01-01 to 2025-12-31",
            "total_sequences": n_total,
            "total_ports": n_ports,
            "sequences_per_port": int(n_total / n_ports),
        },
        "checkpoints": {
            "frozen_backbone_path": str(backbone_ckpt_path),
            "frozen_backbone_sha256": backbone_sha,
            "throughput_heads_path": str(heads_ckpt_path),
            "throughput_heads_sha256": heads_sha,
        },
        "runtime_seconds": round(total_time, 2),
        "inference_throughput_seq_per_sec": round(n_total / inf_duration, 1),
        "container_vessel_calls_7d": {
            "target_name": "target_container_calls_7d_future_sum",
            "unit": "vessel_calls",
            "distribution_target": dist_stats(actual_c_arr),
            "distribution_prediction": dist_stats(pred_c_arr),
            "global_model_metrics": metrics_c_model,
            "global_naive_metrics": metrics_c_naive,
            "improvements": {
                "mae_reduction_pct": round(float((metrics_c_naive["mae"] - metrics_c_model["mae"]) / metrics_c_naive["mae"] * 100), 2),
                "rmse_reduction_pct": round(float((metrics_c_naive["rmse"] - metrics_c_model["rmse"]) / metrics_c_naive["rmse"] * 100), 2),
                "huber_reduction_pct": round(float((metrics_c_naive["huber"] - metrics_c_model["huber"]) / metrics_c_naive["huber"] * 100), 2),
            },
            "zero_target_metrics": {"model": m_c_zero, "naive": n_c_zero, "count": int(zero_c_mask.sum())},
            "nonzero_target_metrics": {"model": m_c_nonzero, "naive": n_c_nonzero, "count": int((~zero_c_mask).sum())},
            "activity_buckets": activity_c,
            "monthly_performance": monthly_metrics_c,
            "port_summary": {
                "ports_improved": ports_improved_c,
                "ports_improved_pct": round(ports_improved_c / n_ports * 100, 2),
                "ports_tied": ports_tied_c,
                "ports_tied_pct": round(ports_tied_c / n_ports * 100, 2),
                "ports_worsened": ports_worsened_c,
                "ports_worsened_pct": round(ports_worsened_c / n_ports * 100, 2),
                "top_10_improved": top_improved_c,
                "top_10_worsened": top_worsened_c,
            },
        },
        "cargo_tonnage_7d": {
            "target_name": "target_cargo_tonnage_7d_future_sum",
            "unit": "metric_tons",
            "distribution_target": dist_stats(actual_t_arr),
            "distribution_prediction": dist_stats(pred_t_arr),
            "global_model_metrics": metrics_t_model,
            "global_naive_metrics": metrics_t_naive,
            "improvements": {
                "mae_reduction_pct": round(float((metrics_t_naive["mae"] - metrics_t_model["mae"]) / metrics_t_naive["mae"] * 100), 2),
                "rmse_reduction_pct": round(float((metrics_t_naive["rmse"] - metrics_t_model["rmse"]) / metrics_t_naive["rmse"] * 100), 2),
                "huber_reduction_pct": round(float((metrics_t_naive["huber"] - metrics_t_model["huber"]) / metrics_t_naive["huber"] * 100), 2),
            },
            "zero_target_metrics": {"model": m_t_zero, "naive": n_t_zero, "count": int(zero_t_mask.sum())},
            "nonzero_target_metrics": {"model": m_t_nonzero, "naive": n_t_nonzero, "count": int((~zero_t_mask).sum())},
            "activity_buckets": activity_t,
            "monthly_performance": monthly_metrics_t,
            "port_summary": {
                "ports_improved": ports_improved_t,
                "ports_improved_pct": round(ports_improved_t / n_ports * 100, 2),
                "ports_tied": ports_tied_t,
                "ports_tied_pct": round(ports_tied_t / n_ports * 100, 2),
                "ports_worsened": ports_worsened_t,
                "ports_worsened_pct": round(ports_worsened_t / n_ports * 100, 2),
                "top_10_improved": top_improved_t,
                "top_10_worsened": top_worsened_t,
            },
        },
    }

    json_path = artifacts_dir / "blind_test_metrics_2025.json"
    with open(json_path, "w") as f:
        json.dump(blind_test_results, f, indent=2)
    print(f"Exported metrics JSON to {json_path}")

    # Mirror to Evaluation Engineer metrics artifacts
    eval_metrics_dir = EVAL_ROOT / "artifacts" / "metrics"
    eval_metrics_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = eval_metrics_dir / "blind_test_metrics_2025.json"
    with open(eval_json_path, "w") as f:
        json.dump(blind_test_results, f, indent=2)
    print(f"Exported metrics JSON to {eval_json_path}")

    print("\n" + "=" * 80)
    print("2025 BLIND TEST EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Container Calls (7d Sum):")
    print(f"  Naive Model: Huber={metrics_c_naive['huber']:.4f} | MAE={metrics_c_naive['mae']:.4f} calls | RMSE={metrics_c_naive['rmse']:.4f} | R2={metrics_c_naive['r2']:.4f}")
    print(f"  Ours Model : Huber={metrics_c_model['huber']:.4f} | MAE={metrics_c_model['mae']:.4f} calls | RMSE={metrics_c_model['rmse']:.4f} | R2={metrics_c_model['r2']:.4f}")
    print(f"  Improvement: MAE {blind_test_results['container_vessel_calls_7d']['improvements']['mae_reduction_pct']:+.2f}% | RMSE {blind_test_results['container_vessel_calls_7d']['improvements']['rmse_reduction_pct']:+.2f}% | Huber {blind_test_results['container_vessel_calls_7d']['improvements']['huber_reduction_pct']:+.2f}%")
    print(f"\nCargo Tonnage (7d Sum):")
    print(f"  Naive Model: Huber={metrics_t_naive['huber']:.1f} | MAE={metrics_t_naive['mae']:,.1f} t | RMSE={metrics_t_naive['rmse']:,.1f} | R2={metrics_t_naive['r2']:.4f}")
    print(f"  Ours Model : Huber={metrics_t_model['huber']:.1f} | MAE={metrics_t_model['mae']:,.1f} t | RMSE={metrics_t_model['rmse']:,.1f} | R2={metrics_t_model['r2']:.4f}")
    print(f"  Improvement: MAE {blind_test_results['cargo_tonnage_7d']['improvements']['mae_reduction_pct']:+.2f}% | RMSE {blind_test_results['cargo_tonnage_7d']['improvements']['rmse_reduction_pct']:+.2f}% | Huber {blind_test_results['cargo_tonnage_7d']['improvements']['huber_reduction_pct']:+.2f}%")
    print(f"Total Evaluation Time: {total_time:.2f}s")


if __name__ == "__main__":
    main()


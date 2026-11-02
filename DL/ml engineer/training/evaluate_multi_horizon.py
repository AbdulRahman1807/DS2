"""Multi-Horizon Forecasting Evaluation (Milestone 3.5 — Phase C).

Evaluates cumulative port-call forecasting across three operational planning horizons:
1. H = 1 day cumulative forecast (t+1)
2. H = 3 days cumulative forecast (t+1 .. t+3)
3. H = 7 days cumulative forecast (t+1 .. t+7, primary benchmark)

Compares the Residual Transformer against Horizon-Scaled Naive Persistence:
    y_naive(H) = H * (roll7 * sigma + mu)
    y_hat(H)   = ReLU( y_naive(H) + Delta_H )
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psutil

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models.transformer_forecaster import ResidualTransformerForecaster
from data.storage import MemmapStorageReader


class MultiHorizonSequenceDataset(Dataset):
    """Dataset serving 30-day sequences alongside 1-day, 3-day, and 7-day targets."""

    def __init__(
        self,
        df_meta: pd.DataFrame,
        storage_dir: str,
        window_len: int = 30,
        availability_lag: int = 2,
    ):
        self.df_meta = df_meta.reset_index(drop=True)
        self.storage_reader = MemmapStorageReader(storage_dir)
        self.window_len = window_len
        self.availability_lag = availability_lag

        date2idx = self.storage_reader.date2idx
        port2idx = self.storage_reader.port2idx
        num_panel_days = len(self.storage_reader.idx2date)

        ref_day_indices = self.df_meta["reference_date"].map(date2idx).to_numpy(dtype=np.int32)
        start_offset = self.window_len + self.availability_lag - 1
        obs_start_day_indices = ref_day_indices - start_offset
        obs_end_day_indices = ref_day_indices - self.availability_lag

        # Filter out-of-bounds dates matching PortSequenceDataset
        valid_mask = (obs_start_day_indices >= 0) & (obs_end_day_indices < num_panel_days)
        if (~valid_mask).sum() > 0:
            self.df_meta = self.df_meta[valid_mask].copy().reset_index(drop=True)
            ref_day_indices = self.df_meta["reference_date"].map(date2idx).to_numpy(dtype=np.int32)
            obs_start_day_indices = ref_day_indices - start_offset

        self.port_indices = self.df_meta["portwatch_id"].map(port2idx).to_numpy(dtype=np.int32)
        self.start_day_indices = obs_start_day_indices

        self.targets_1d = self.df_meta["target_1d"].to_numpy(dtype=np.float32)
        self.targets_3d = self.df_meta["target_3d"].to_numpy(dtype=np.float32)
        self.targets_7d = self.df_meta["target_7d"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df_meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p_idx = int(self.port_indices[idx])
        d_start = int(self.start_day_indices[idx])

        x_np = self.storage_reader.get_sequence_by_indices(p_idx, d_start, self.window_len)
        x_tensor = torch.from_numpy(x_np).float()

        y1 = torch.tensor([self.targets_1d[idx]], dtype=torch.float32)
        y3 = torch.tensor([self.targets_3d[idx]], dtype=torch.float32)
        y7 = torch.tensor([self.targets_7d[idx]], dtype=torch.float32)

        return x_tensor, y1, y3, y7


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    delta = y_pred - y_true
    abs_err = np.abs(delta)
    sq_err = delta ** 2
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    huber = float(np.mean(np.where(abs_err <= 1.0, 0.5 * sq_err, abs_err - 0.5)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (np.sum(sq_err) / ss_tot)) if ss_tot > 0 else 0.0
    bias = float(np.mean(delta))
    return {
        "huber": round(huber, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "bias": round(bias, 4),
    }


def run_multi_horizon_benchmark():
    t_start = time.time()
    out_dir = WORKSPACE_DIR / "artifacts/multi_horizon"
    out_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = str(WORKSPACE_DIR / "artifacts/preprocessing/storage")
    meta_p = str(WORKSPACE_DIR / "../data/staging/supplementary_acquisition/processed/temporal_sequences/sequence_metadata.parquet")
    panel_p = str(WORKSPACE_DIR / "../data/staging/supplementary_acquisition/processed/unified_temporal/port_daily_unified.parquet")
    trans_ckpt_p = str(WORKSPACE_DIR / "artifacts/transformer_mean_pooling/best_model.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    print("=" * 80)
    print(" MILESTONE 3.5: PHASE C — MULTI-HORIZON FORECASTING EVALUATION")
    print(f" Artifacts Dir: {out_dir}")
    print(f" Device:        {device}")
    print("=" * 80)

    # 1. Ingest Panel and Targets
    print("\n[Step 1/5] Ingesting multi-horizon target labels from unified panel...")
    t0 = time.time()
    panel = pd.read_parquet(
        panel_p,
        columns=["portwatch_id", "date_utc", "target_portcalls_lead_1d", "target_portcalls_lead_2d", "target_portcalls_lead_3d"]
    ).rename(columns={"date_utc": "reference_date"})
    panel["target_1d"] = panel["target_portcalls_lead_1d"]
    panel["target_3d"] = panel["target_portcalls_lead_1d"] + panel["target_portcalls_lead_2d"] + panel["target_portcalls_lead_3d"]
    print(f"  Loaded panel ({len(panel):,} rows) in {time.time()-t0:.2f}s")

    # Ingest Sequence Metadata
    print("\n[Step 2/5] Merging validation & training sequence partitions...")
    meta = pd.read_parquet(
        meta_p,
        columns=["sequence_id", "portwatch_id", "reference_date", "split_tag", "target_portcalls_7d_future_sum"]
    )
    val_meta = meta[meta["split_tag"] == "val"].copy()
    val_merged = val_meta.merge(panel[["portwatch_id", "reference_date", "target_1d", "target_3d"]], on=["portwatch_id", "reference_date"], how="left")
    val_merged["target_7d"] = val_merged["target_portcalls_7d_future_sum"]
    print(f"  Validation sequences: {len(val_merged):,} (Nulls 1d: {val_merged['target_1d'].isna().sum()}, 3d: {val_merged['target_3d'].isna().sum()})")

    # Sample 500k training sequences
    train_meta = meta[meta["split_tag"] == "train"].sample(n=500000, random_state=42).copy()
    train_merged = train_meta.merge(panel[["portwatch_id", "reference_date", "target_1d", "target_3d"]], on=["portwatch_id", "reference_date"], how="left")
    train_merged["target_7d"] = train_merged["target_portcalls_7d_future_sum"]
    print(f"  Training sample:      {len(train_merged):,} sequences")

    # 2. Datasets and Loaders
    train_ds = MultiHorizonSequenceDataset(train_merged, storage_dir)
    val_ds = MultiHorizonSequenceDataset(val_merged, storage_dir)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    # 3. Model Setup: Frozen Backbone + Multi-Horizon Heads
    print("\n[Step 3/5] Loading frozen M3.4 Transformer backbone...")
    transformer = ResidualTransformerForecaster(
        input_size=20,
        seq_len=30,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        pooling="mean",
        pe_type="sinusoidal",
        roll7_idx=12,
        roll7_mean=2.4068180554238894,
        roll7_std=6.3483944613159915,
        constraint="relu",
    ).to(device)

    ckpt = torch.load(trans_ckpt_p, map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt["model_state_dict"])
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False
    print("  Transformer backbone and reference 7-day head successfully loaded and FROZEN.")

    # Dedicated heads for H=1 and H=3
    head_1d = nn.Linear(64, 1).to(device)
    head_3d = nn.Linear(64, 1).to(device)
    # Initialize near zero residual
    nn.init.xavier_uniform_(head_1d.weight, gain=0.01)
    nn.init.constant_(head_1d.bias, 0.0)
    nn.init.xavier_uniform_(head_3d.weight, gain=0.01)
    nn.init.constant_(head_3d.bias, 0.0)

    # 4. Train Heads on 500k Training Sequences
    print("\n[Step 4/5] Training 1-day and 3-day forecasting heads (500k training steps)...")
    optimizer = torch.optim.AdamW(list(head_1d.parameters()) + list(head_3d.parameters()), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)

    t_tr0 = time.time()
    for b_idx, (bx, by1, by3, _) in enumerate(train_loader):
        bx = bx.to(device)
        by1 = by1.to(device)
        by3 = by3.to(device)

        # Unscaled 7-day rolling mean at t-2
        roll7 = bx[:, -1, 12:13] * 6.3483944613159915 + 2.4068180554238894
        naive_1d = 1.0 * roll7
        naive_3d = 3.0 * roll7

        with torch.no_grad():
            proj = transformer.input_proj(bx)
            pos = transformer.pos_encoder(proj)
            h = pos
            for layer in transformer.layers:
                h, _ = layer(h)
            h = transformer.final_norm(h)
            h_pooled = h.mean(dim=1)

        delta_1d = head_1d(h_pooled)
        delta_3d = head_3d(h_pooled)

        pred_1d = F.relu(naive_1d + delta_1d)
        pred_3d = F.relu(naive_3d + delta_3d)

        l1 = loss_fn(pred_1d, by1)
        l3 = loss_fn(pred_3d, by3)
        loss = l1 + l3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (b_idx + 1) % 250 == 0 or (b_idx + 1) == len(train_loader):
            print(f"  Batch {b_idx+1:4d}/{len(train_loader)} | Loss 1d: {l1.item():.4f} | Loss 3d: {l3.item():.4f}")

    print(f"  Multi-horizon heads trained in {time.time()-t_tr0:.2f}s.")

    # 5. Full Validation Evaluation (743,400 sequences)
    print("\n[Step 5/5] Running multi-horizon validation inference on all 743,400 sequences...")
    head_1d.eval()
    head_3d.eval()

    all_pred_1d, all_naive_1d = [], []
    all_pred_3d, all_naive_3d = [], []
    all_pred_7d, all_naive_7d = [], []

    t_val0 = time.time()
    with torch.no_grad():
        for b_idx, (bx, _, _, _) in enumerate(val_loader):
            bx = bx.to(device)

            roll7 = bx[:, -1, 12:13] * 6.3483944613159915 + 2.4068180554238894
            n_1d = (1.0 * roll7).cpu().numpy().reshape(-1)
            n_3d = (3.0 * roll7).cpu().numpy().reshape(-1)
            n_7d = (7.0 * roll7).cpu().numpy().reshape(-1)

            proj = transformer.input_proj(bx)
            pos = transformer.pos_encoder(proj)
            h = pos
            for layer in transformer.layers:
                h, _ = layer(h)
            h = transformer.final_norm(h)
            h_pooled = h.mean(dim=1)

            p_1d = F.relu(1.0 * roll7 + head_1d(h_pooled)).cpu().numpy().reshape(-1)
            p_3d = F.relu(3.0 * roll7 + head_3d(h_pooled)).cpu().numpy().reshape(-1)
            p_7d = F.relu(7.0 * roll7 + transformer.head(h_pooled)).cpu().numpy().reshape(-1)

            all_naive_1d.append(n_1d)
            all_naive_3d.append(n_3d)
            all_naive_7d.append(n_7d)

            all_pred_1d.append(p_1d)
            all_pred_3d.append(p_3d)
            all_pred_7d.append(p_7d)

    print(f"  Inference completed across 743,400 sequences in {time.time()-t_val0:.2f}s ({743400/(time.time()-t_val0):.1f} seqs/sec).")

    pred_1d = np.concatenate(all_pred_1d)
    naive_1d = np.concatenate(all_naive_1d)
    true_1d = val_merged["target_1d"].to_numpy(dtype=np.float32)

    pred_3d = np.concatenate(all_pred_3d)
    naive_3d = np.concatenate(all_naive_3d)
    true_3d = val_merged["target_3d"].to_numpy(dtype=np.float32)

    pred_7d = np.concatenate(all_pred_7d)
    naive_7d = np.concatenate(all_naive_7d)
    true_7d = val_merged["target_7d"].to_numpy(dtype=np.float32)

    # Compute Global Horizon Metrics
    m_naive_1d = compute_metrics(true_1d, naive_1d)
    m_trans_1d = compute_metrics(true_1d, pred_1d)

    m_naive_3d = compute_metrics(true_3d, naive_3d)
    m_trans_3d = compute_metrics(true_3d, pred_3d)

    m_naive_7d = compute_metrics(true_7d, naive_7d)
    m_trans_7d = compute_metrics(true_7d, pred_7d)

    # Summary Table
    horizons_table = [
        {
            "Horizon": "H = 1 Day (t+1)",
            "Target Mean": round(float(np.mean(true_1d)), 2),
            "Naive Huber": m_naive_1d["huber"],
            "Trans Huber": m_trans_1d["huber"],
            "Naive MAE": m_naive_1d["mae"],
            "Trans MAE": m_trans_1d["mae"],
            "Delta MAE": round(m_naive_1d["mae"] - m_trans_1d["mae"], 4),
            "MAE Imp %": round((m_naive_1d["mae"] - m_trans_1d["mae"]) / m_naive_1d["mae"] * 100.0, 2),
            "Trans RMSE": m_trans_1d["rmse"],
            "Trans R2": m_trans_1d["r2"],
            "Trans Bias": m_trans_1d["bias"],
        },
        {
            "Horizon": "H = 3 Days (t+1..t+3)",
            "Target Mean": round(float(np.mean(true_3d)), 2),
            "Naive Huber": m_naive_3d["huber"],
            "Trans Huber": m_trans_3d["huber"],
            "Naive MAE": m_naive_3d["mae"],
            "Trans MAE": m_trans_3d["mae"],
            "Delta MAE": round(m_naive_3d["mae"] - m_trans_3d["mae"], 4),
            "MAE Imp %": round((m_naive_3d["mae"] - m_trans_3d["mae"]) / m_naive_3d["mae"] * 100.0, 2),
            "Trans RMSE": m_trans_3d["rmse"],
            "Trans R2": m_trans_3d["r2"],
            "Trans Bias": m_trans_3d["bias"],
        },
        {
            "Horizon": "H = 7 Days (t+1..t+7)",
            "Target Mean": round(float(np.mean(true_7d)), 2),
            "Naive Huber": m_naive_7d["huber"],
            "Trans Huber": m_trans_7d["huber"],
            "Naive MAE": m_naive_7d["mae"],
            "Trans MAE": m_trans_7d["mae"],
            "Delta MAE": round(m_naive_7d["mae"] - m_trans_7d["mae"], 4),
            "MAE Imp %": round((m_naive_7d["mae"] - m_trans_7d["mae"]) / m_naive_7d["mae"] * 100.0, 2),
            "Trans RMSE": m_trans_7d["rmse"],
            "Trans R2": m_trans_7d["r2"],
            "Trans Bias": m_trans_7d["bias"],
        },
    ]

    h_df = pd.DataFrame(horizons_table)
    print("\n" + "=" * 80)
    print("PHASE C: MULTI-HORIZON FORECASTING BENCHMARK (N = 743,400 SEQUENCES)")
    print("=" * 80)
    print(h_df.to_string(index=False))

    # Volume Quartile Analysis across Horizons
    q25, q50, q75 = np.percentile(true_7d, [25, 50, 75])
    quartile_masks = [
        ("Q1: Sparse / Zero (0 - 1 calls/wk)", true_7d <= q25),
        ("Q2: Low Volume (2 - 5 calls/wk)", (true_7d > q25) & (true_7d <= q50)),
        ("Q3: Medium Volume (6 - 17 calls/wk)", (true_7d > q50) & (true_7d <= q75)),
        ("Q4: High-Volume Hubs (18 - 1188 calls/wk)", true_7d > q75),
    ]

    vol_rows = []
    for q_label, mask in quartile_masks:
        vol_rows.append({
            "Quartile": q_label,
            "Sequences": int(mask.sum()),
            "H1 Naive MAE": round(float(np.mean(np.abs(naive_1d[mask] - true_1d[mask]))), 4),
            "H1 Trans MAE": round(float(np.mean(np.abs(pred_1d[mask] - true_1d[mask]))), 4),
            "H3 Naive MAE": round(float(np.mean(np.abs(naive_3d[mask] - true_3d[mask]))), 4),
            "H3 Trans MAE": round(float(np.mean(np.abs(pred_3d[mask] - true_3d[mask]))), 4),
            "H7 Naive MAE": round(float(np.mean(np.abs(naive_7d[mask] - true_7d[mask]))), 4),
            "H7 Trans MAE": round(float(np.mean(np.abs(pred_7d[mask] - true_7d[mask]))), 4),
        })

    vol_df = pd.DataFrame(vol_rows)
    print("\n" + "=" * 80)
    print("VOLUME-STRATIFIED MULTI-HORIZON PERFORMANCE")
    print("=" * 80)
    print(vol_df.to_string(index=False))

    # Save full multi-horizon validation predictions Parquet
    val_export = pd.DataFrame({
        "sequence_id": val_merged["sequence_id"],
        "portwatch_id": val_merged["portwatch_id"],
        "reference_date": val_merged["reference_date"],
        "y_true_1d": true_1d,
        "y_pred_1d": pred_1d,
        "y_naive_1d": naive_1d,
        "y_true_3d": true_3d,
        "y_pred_3d": pred_3d,
        "y_naive_3d": naive_3d,
        "y_true_7d": true_7d,
        "y_pred_7d": pred_7d,
        "y_naive_7d": naive_7d,
    })
    val_export_p = out_dir / "validation_predictions_multi_horizon.parquet"
    val_export.to_parquet(str(val_export_p), index=False, compression="SNAPPY")
    print(f"\nSaved Multi-Horizon Validation Predictions: {val_export_p} ({len(val_export):,} rows)")

    # Save checkpoint for multi-horizon heads
    torch.save({
        "head_1d_state_dict": head_1d.state_dict(),
        "head_3d_state_dict": head_3d.state_dict(),
        "transformer_checkpoint": trans_ckpt_p,
        "horizons_metrics": horizons_table,
        "volume_metrics": vol_rows,
    }, str(out_dir / "multi_horizon_transformer.pt"))

    # Save metrics JSON
    with open(str(out_dir / "multi_horizon_metrics.json"), "w") as f:
        json.dump({
            "horizons_table": horizons_table,
            "volume_table": vol_rows,
            "total_duration_sec": time.time() - t_start,
        }, f, indent=2)

    print("=" * 80)
    print("PHASE C BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_multi_horizon_benchmark()

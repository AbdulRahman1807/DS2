"""Training & Evaluation Workflow for Caribbean Regional Residual ST-GNN (Milestone 3.3).

Executes reproducible regional model training, topological ablation benchmarking,
checkpointing, and streaming validation prediction exports.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import psutil

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models.caribbean_stgnn import CaribbeanResidualSTGNN
from data.caribbean_dataset import load_caribbean_topology, CaribbeanSnapshotDataset, caribbean_collate_fn


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_caribbean_stgnn_training(
    config_path: str,
    ablation: str = "none",
    epochs: Optional[int] = None,
    gnn_type: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Caribbean ST-GNN training pipeline and export validation metrics."""
    process = psutil.Process(os.getpid())
    init_mem_mb = process.memory_info().rss / (1024 * 1024)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("random_seed", 42)
    set_seed(seed)

    # Resolve paths
    cfg_p = Path(config_path).resolve()
    base_dir = cfg_p.parents[1]

    def resolve_path(raw_str: str) -> str:
        p = Path(raw_str)
        if p.is_absolute() and p.exists():
            return str(p)
        if p.exists():
            return str(p.resolve())
        for parent in cfg_p.parents:
            if (parent / p).exists():
                return str((parent / p).resolve())
        return str(p)

    edges_p = resolve_path(cfg["paths"]["caribbean_edges_parquet"])
    meta_p = resolve_path(cfg["paths"]["sequence_metadata_parquet"])
    storage_dir = resolve_path(cfg["paths"]["storage_dir"])
    exp03_ckpt = resolve_path(cfg["paths"]["exp03_checkpoint"])

    resolved_ablation = ablation.lower()
    resolved_gnn_type = gnn_type or cfg.get("model", {}).get("gnn_type", "gatv2")

    if artifacts_dir is not None:
        art_dir = Path(artifacts_dir).resolve()
    else:
        art_dir = base_dir / f"artifacts/gnn/caribbean/stgnn_{resolved_gnn_type}_{resolved_ablation}"
    art_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" MILESTONE 3.3: CARIBBEAN RESIDUAL SPATIO-TEMPORAL GNN TRAINING")
    print(f" Config:         {config_path}")
    print(f" Artifacts Dir:  {art_dir}")
    print(f" GNN Layer:      {resolved_gnn_type.upper()}")
    print(f" Ablation Mode:  {resolved_ablation.upper()}")
    print(f" Initial Memory: {init_mem_mb:.2f} MB")
    print("=" * 80)

    # 1. Load Topology
    topo = load_caribbean_topology(edges_p, ablation=resolved_ablation, seed=seed)
    ports = topo["ports"]
    num_ports = topo["num_ports"]
    num_edges = topo["num_edges"]
    print(f"\n[Step 1/5] Loaded Caribbean Topology: {num_ports} ports, {num_edges} directed corridors.")

    # Determine device
    device_name = cfg.get("training", {}).get("device", "mps").lower()
    if device_name == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Active Device: {device}")

    adj_weights = topo["adj_weights"].to(device)
    adj_mask = topo["adj_mask"].to(device)
    norm_adj = topo["norm_adj"].to(device)

    # 2. Datasets & Loaders
    effective_bs = batch_size or cfg.get("training", {}).get("batch_size", 8)
    print(f"\n[Step 2/5] Initializing Daily Snapshot Datasets (batch_size={effective_bs} dates = {effective_bs*num_ports} seqs/step)...")

    train_ds = CaribbeanSnapshotDataset(meta_p, storage_dir, ports, split="train")
    val_ds = CaribbeanSnapshotDataset(meta_p, storage_dir, ports, split="val")

    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=True, collate_fn=caribbean_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=effective_bs, shuffle=False, collate_fn=caribbean_collate_fn)

    print(f"  Train Dates: {len(train_ds):,} dates ({len(train_ds)*num_ports:,} sequences, {len(train_loader)} batches)")
    print(f"  Val Dates:   {len(val_ds):,} dates ({len(val_ds)*num_ports:,} sequences, {len(val_loader)} batches)")

    # 3. Model Instantiation & Pretrained Weight Loading
    m_cfg = cfg.get("model", {})
    model = CaribbeanResidualSTGNN(
        num_nodes=num_ports,
        input_size=m_cfg.get("input_size", 20),
        hidden_size=m_cfg.get("hidden_size", 64),
        graph_hidden_size=m_cfg.get("graph_hidden_size", 32),
        num_lstm_layers=m_cfg.get("num_lstm_layers", 2),
        dropout=m_cfg.get("dropout", 0.1),
        gnn_type=resolved_gnn_type,
        ablation=resolved_ablation,
        freeze_temporal=m_cfg.get("freeze_temporal", True),
    ).to(device)

    model.load_temporal_pretrained(exp03_ckpt, device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Params:     {total_params:,}")
    print(f"  Trainable Params: {trainable_params:,} (Temporal weights frozen={m_cfg.get('freeze_temporal', True)})")

    # 4. Training Loop
    total_epochs = epochs or cfg.get("training", {}).get("num_epochs", 10)
    lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    wd = float(cfg.get("training", {}).get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    criterion = nn.HuberLoss(delta=float(cfg.get("training", {}).get("huber_delta", 1.0)))
    patience = int(cfg.get("training", {}).get("early_stopping_patience", 3))

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    t_train_start = time.time()

    print(f"\n[Step 4/5] Executing Spatio-Temporal Training Loop ({total_epochs} epochs)...")

    for ep in range(1, total_epochs + 1):
        model.train()
        sum_train_loss = 0.0
        n_train_samples = 0
        t_ep = time.time()

        for bx, by, _, _, _ in train_loader:
            bx = bx.to(device)  # [B, 51, 30, 20]
            by = by.to(device)  # [B, 51]

            optimizer.zero_grad()
            y_pred, _, _ = model(bx, adj_weights, adj_mask, norm_adj)
            loss = criterion(y_pred, by)
            loss.backward()

            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()

            sum_train_loss += loss.item() * by.numel()
            n_train_samples += by.numel()

        if device.type == "mps":
            torch.mps.synchronize()

        train_huber = sum_train_loss / n_train_samples

        # Validation pass
        model.eval()
        sum_val_loss = 0.0
        sum_abs_err = 0.0
        sum_sq_err = 0.0
        sum_y = 0.0
        sum_y_sq = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for bx, by, _, _, _ in val_loader:
                bx = bx.to(device)
                by = by.to(device)

                y_pred, _, _ = model(bx, adj_weights, adj_mask, norm_adj)
                loss = criterion(y_pred, by)

                err = y_pred - by
                sum_val_loss += loss.item() * by.numel()
                sum_abs_err += torch.abs(err).sum().item()
                sum_sq_err += (err ** 2).sum().item()
                sum_y += by.sum().item()
                sum_y_sq += (by ** 2).sum().item()
                n_val_samples += by.numel()

        if device.type == "mps":
            torch.mps.synchronize()

        val_huber = sum_val_loss / n_val_samples
        val_mae = sum_abs_err / n_val_samples
        val_rmse = np.sqrt(sum_sq_err / n_val_samples)
        y_mean = sum_y / n_val_samples
        ss_tot = sum_y_sq - (n_val_samples * (y_mean ** 2))
        val_r2 = 1.0 - (sum_sq_err / ss_tot)
        ep_sec = time.time() - t_ep

        print(
            f"  [Epoch {ep:2d}/{total_epochs:2d}] Train Huber: {train_huber:.4f} | "
            f"Val Huber: {val_huber:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f} ({ep_sec:.1f}s)"
        )

        history.append({
            "epoch": ep,
            "train_huber": train_huber,
            "val_huber": val_huber,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "duration_sec": ep_sec,
        })

        if val_huber < best_val_loss:
            best_val_loss = val_huber
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_huber": val_huber,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "model_config": model.get_config(),
            }, str(art_dir / "best_model.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered after {patience} epochs without improvement.")
                break

    # 5. Export Streaming Validation Predictions
    print("\n[Step 5/5] Exporting Validation Predictions Parquet...")
    ckpt_best = torch.load(str(art_dir / "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt_best["model_state_dict"])
    model.eval()

    all_seq_ids = []
    all_port_ids = []
    all_dates = []
    all_y_true = []
    all_y_pred = []
    all_delta_temp = []
    all_delta_graph = []

    with torch.no_grad():
        for bx, by, date_strs, seq_ids_batch, ports_batch in val_loader:
            bx = bx.to(device)
            y_pred, d_temp, d_graph = model(bx, adj_weights, adj_mask, norm_adj)

            y_pred_np = y_pred.cpu().numpy()
            d_temp_np = d_temp.cpu().numpy()
            d_graph_np = d_graph.cpu().numpy()
            by_np = by.numpy()

            for b_idx, d_str in enumerate(date_strs):
                for p_idx, p_id in enumerate(ports_batch):
                    all_seq_ids.append(seq_ids_batch[b_idx][p_idx])
                    all_port_ids.append(p_id)
                    all_dates.append(d_str)
                    all_y_true.append(float(by_np[b_idx, p_idx]))
                    all_y_pred.append(float(y_pred_np[b_idx, p_idx]))
                    all_delta_temp.append(float(d_temp_np[b_idx, p_idx]))
                    all_delta_graph.append(float(d_graph_np[b_idx, p_idx]))

    val_df = pd.DataFrame({
        "sequence_id": all_seq_ids,
        "portwatch_id": all_port_ids,
        "reference_date": all_dates,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "delta_temporal": all_delta_temp,
        "delta_graph": all_delta_graph,
    })

    val_parquet_path = art_dir / "validation_predictions.parquet"
    val_df.to_parquet(str(val_parquet_path), index=False, compression="SNAPPY")

    # Save training history and config
    pd.DataFrame(history).to_csv(str(art_dir / "training_history.csv"), index=False)
    with open(str(art_dir / "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    total_time = time.time() - t_train_start
    peak_rss = process.memory_info().rss / (1024 * 1024)

    print("=" * 80)
    print(" CARIBBEAN ST-GNN WORKFLOW COMPLETED!")
    print(f" Best Epoch:           {best_epoch}")
    print(f" Best Val Huber Loss:  {best_val_loss:.4f}")
    print(f" Best Val MAE:         {ckpt_best['val_mae']:.4f}")
    print(f" Best Val RMSE:        {ckpt_best['val_rmse']:.4f}")
    print(f" Best Val R2:          {ckpt_best['val_r2']:.4f}")
    print(f" Total Duration:       {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f" Peak Process RSS:     {peak_rss:.1f} MB")
    print(f" Saved Predictions:    {val_parquet_path} ({len(val_df):,} rows)")
    print("=" * 80)

    return {
        "best_epoch": best_epoch,
        "best_val_huber": best_val_loss,
        "val_mae": ckpt_best["val_mae"],
        "val_rmse": ckpt_best["val_rmse"],
        "val_r2": ckpt_best["val_r2"],
        "total_time_sec": total_time,
        "peak_rss_mb": peak_rss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Caribbean Regional Residual ST-GNN")
    parser.add_argument("--config", type=str, default=str(WORKSPACE_DIR / "configs/caribbean_stgnn.yaml"))
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "shuffled", "no_edges"])
    parser.add_argument("--gnn-type", type=str, default="gatv2", choices=["gatv2", "gcn"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()
    run_caribbean_stgnn_training(
        config_path=args.config,
        ablation=args.ablation,
        epochs=args.epochs,
        gnn_type=args.gnn_type,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
    )

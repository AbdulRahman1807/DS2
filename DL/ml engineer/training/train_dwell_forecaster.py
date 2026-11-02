"""
Weekly Anchorage Waiting-Time Forecaster Training Pipeline
==========================================================
Milestone 3.6 Phase B: Dwell and Waiting-Time Forecasting

Trains a Residual Dwell GRU Forecaster using the ADB AIS weekly dataset across 19 global container ports.
Maintains strict chronological splits:
    - Train: 2019-W01 to 2020-W53 (1,843 sequences)
    - Validation: 2021-W01 to 2021-W53 (1,007 sequences, supply chain crisis surge)
    - Test / Held-Out: 2022-W01 to 2022-W52 (931 sequences, crisis stabilization)

All feature scalers are fitted strictly on the Training split.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dwell_time_forecaster import ResidualDwellGRUForecaster


class DwellSequenceDataset(Dataset):
    """PyTorch Dataset for weekly port waiting time sequences."""

    def __init__(
        self,
        features: np.ndarray,
        port_ids: np.ndarray,
        naive_waiting: np.ndarray,
        target_h1: np.ndarray,
        target_h4: np.ndarray,
        metadata: list,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.port_ids = torch.tensor(port_ids, dtype=torch.long)
        self.naive_waiting = torch.tensor(naive_waiting, dtype=torch.float32)
        self.target_h1 = torch.tensor(target_h1, dtype=torch.float32)
        self.target_h4 = torch.tensor(target_h4, dtype=torch.float32)
        self.metadata = metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "x_seq": self.features[idx],
            "port_id": self.port_ids[idx],
            "naive_waiting": self.naive_waiting[idx],
            "target_h1": self.target_h1[idx],
            "target_h4": self.target_h4[idx],
            "meta": self.metadata[idx],
        }


def compute_metrics(actual: np.ndarray, pred: np.ndarray, delta: float = 3.0) -> dict:
    """Compute regression metrics: Huber, MAE, RMSE, R2, Bias."""
    err = pred - actual
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
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
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "bias": round(bias, 4),
    }


def main():
    start_time = time.time()
    print("=" * 80)
    print("Milestone 3.6 Phase B: Weekly Anchorage Waiting-Time Forecaster Training")
    print("=" * 80)

    # 1. Load config
    config_path = PROJECT_ROOT / "configs" / "dwell_forecaster.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(cfg["training"]["random_seed"])
    np.random.seed(cfg["training"]["random_seed"])

    # Device
    device_str = cfg["training"]["device"]
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # 2. Ingest and prepare ADB AIS weekly data
    raw_path = Path("/Users/abdul/Desktop/DS") / cfg["data"]["raw_path"]
    df = pd.read_csv(raw_path).dropna(subset=["Port", "Year", "Week"])
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(["Port", "Year", "Week"]).reset_index(drop=True)

    ports = sorted(df["Port"].unique())
    port_to_id = {p: i for i, p in enumerate(ports)}
    print(f"Loaded {len(df)} records across {len(ports)} ports: {ports}")

    # Feature definitions
    feature_cols = [
        "avgwaiting_time",
        "avgservice_time",
        "numofcall",
        "count_entrance",
        "share_neopan",
    ]
    # Add calendar harmonic features
    df["week_sin"] = np.sin(2 * np.pi * df["Week"] / 53.0)
    df["week_cos"] = np.cos(2 * np.pi * df["Week"] / 53.0)
    all_feature_cols = feature_cols + ["week_sin", "week_cos"]

    # 3. Fit scaler ONLY on training split (Year <= 2020)
    train_mask = df["Year"] <= 2020
    train_df = df[train_mask]
    means = train_df[feature_cols].mean().to_dict()
    stds = train_df[feature_cols].std().replace(0, 1.0).to_dict()

    # Normalize feature columns
    for col in feature_cols:
        df[col + "_norm"] = (df[col] - means[col]) / stds[col]

    norm_cols = [c + "_norm" for c in feature_cols] + ["week_sin", "week_cos"]

    # 4. Construct sequence datasets with lookback L=8
    L = cfg["data"]["lookback_weeks"]
    sequences_train, sequences_val, sequences_test = [], [], []

    for p in ports:
        pdf = df[df["Port"] == p].sort_values(["Year", "Week"]).reset_index(drop=True)
        feat_matrix = pdf[norm_cols].values
        unscaled_waiting = pdf["avgwaiting_time"].values
        years = pdf["Year"].values
        weeks = pdf["Week"].values

        for i in range(L - 1, len(pdf) - 4):
            # Input window: [i - L + 1 : i + 1] (length L)
            seq_feat = feat_matrix[i - L + 1 : i + 1]
            p_id = port_to_id[p]
            y_naive = unscaled_waiting[i]
            y_target_h1 = unscaled_waiting[i + 1]
            y_target_h4 = np.mean(unscaled_waiting[i + 1 : i + 5])
            target_year = int(years[i + 1])
            target_week = int(weeks[i + 1])

            meta = {
                "port": p,
                "port_id": p_id,
                "target_year": target_year,
                "target_week": target_week,
                "y_naive": float(y_naive),
                "y_target_h1": float(y_target_h1),
                "y_target_h4": float(y_target_h4),
            }

            item = (seq_feat, p_id, y_naive, y_target_h1, y_target_h4, meta)

            if target_year <= 2020:
                sequences_train.append(item)
            elif target_year == 2021:
                sequences_val.append(item)
            elif target_year == 2022:
                sequences_test.append(item)

    print(f"Constructed sequence datasets: Train={len(sequences_train)}, Val={len(sequences_val)}, Test={len(sequences_test)}")

    def collate_seqs(seq_list):
        feats = np.stack([s[0] for s in seq_list])
        p_ids = np.array([s[1] for s in seq_list], dtype=np.int64)
        naives = np.array([s[2] for s in seq_list], dtype=np.float32)
        h1s = np.array([s[3] for s in seq_list], dtype=np.float32)
        h4s = np.array([s[4] for s in seq_list], dtype=np.float32)
        metas = [s[5] for s in seq_list]
        return DwellSequenceDataset(feats, p_ids, naives, h1s, h4s, metas)

    ds_train = collate_seqs(sequences_train)
    ds_val = collate_seqs(sequences_val)
    ds_test = collate_seqs(sequences_test)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    model = ResidualDwellGRUForecaster(
        num_ports=len(ports),
        port_embedding_dim=cfg["model"]["port_embedding_dim"],
        num_temporal_features=len(norm_cols),
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        predict_service_time=False,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {cfg['model']['model_type']} with {total_params:,} trainable parameters.")

    # 6. Loss and Optimizer
    huber_delta = cfg["training"]["huber_delta"]
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"], eta_min=1e-5
    )

    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = artifacts_dir / "best_model.pt"

    # 7. Training Loop
    num_epochs = cfg["training"]["num_epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    print("\nStarting Training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_h1_losses = []

        for batch in train_loader:
            x_seq = batch["x_seq"].to(device)
            port_ids = batch["port_id"].to(device)
            naive_waiting = batch["naive_waiting"].to(device)
            target_h1 = batch["target_h1"].to(device)
            target_h4 = batch["target_h4"].to(device)

            optimizer.zero_grad()
            preds = model(x_seq, port_ids, naive_waiting)

            loss_h1 = criterion(preds["waiting_pred_h1"], target_h1)
            loss_h4 = criterion(preds["waiting_pred_h4"], target_h4)
            loss = loss_h1 + 0.5 * loss_h4

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()

            train_loss_total += loss.item() * len(x_seq)
            train_h1_losses.append(loss_h1.item())

        lr_scheduler.step()
        train_loss = train_loss_total / len(ds_train)

        # Validation
        model.eval()
        val_preds_h1, val_targets_h1 = [], []
        val_preds_h4, val_targets_h4 = [], []

        with torch.no_grad():
            for batch in val_loader:
                x_seq = batch["x_seq"].to(device)
                port_ids = batch["port_id"].to(device)
                naive_waiting = batch["naive_waiting"].to(device)
                target_h1 = batch["target_h1"].to(device)
                target_h4 = batch["target_h4"].to(device)

                preds = model(x_seq, port_ids, naive_waiting)
                val_preds_h1.append(preds["waiting_pred_h1"].cpu().numpy())
                val_targets_h1.append(target_h1.cpu().numpy())
                val_preds_h4.append(preds["waiting_pred_h4"].cpu().numpy())
                val_targets_h4.append(target_h4.cpu().numpy())

        v_act_h1 = np.concatenate(val_targets_h1)
        v_pred_h1 = np.concatenate(val_preds_h1)
        val_metrics_h1 = compute_metrics(v_act_h1, v_pred_h1, delta=huber_delta)

        v_act_h4 = np.concatenate(val_targets_h4)
        v_pred_h4 = np.concatenate(val_preds_h4)
        val_metrics_h4 = compute_metrics(v_act_h4, v_pred_h4, delta=huber_delta)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_mae_h1": val_metrics_h1["mae"],
            "val_rmse_h1": val_metrics_h1["rmse"],
            "val_r2_h1": val_metrics_h1["r2"],
            "val_mae_h4": val_metrics_h4["mae"],
            "val_rmse_h4": val_metrics_h4["rmse"],
            "val_r2_h4": val_metrics_h4["r2"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val H1 MAE: {val_metrics_h1['mae']:.4f} (RMSE: {val_metrics_h1['rmse']:.4f}, R2: {val_metrics_h1['r2']:.4f}) | "
            f"Val H4 MAE: {val_metrics_h4['mae']:.4f}"
        )

        # Early stopping and checkpointing on Val MAE H1
        if val_metrics_h1["mae"] < best_val_mae:
            best_val_mae = val_metrics_h1["mae"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics_h1": val_metrics_h1,
                    "val_metrics_h4": val_metrics_h4,
                    "ports": ports,
                    "port_to_id": port_to_id,
                    "feature_cols": all_feature_cols,
                    "scaler_means": means,
                    "scaler_stds": stds,
                    "config": cfg,
                },
                best_model_path,
            )
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val MAE: {best_val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = artifacts_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    # 8. Final Comprehensive Evaluation with Best Checkpoint
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Function to run inference and export prediction records
    def evaluate_split(loader, split_name):
        records = []
        with torch.no_grad():
            for batch in loader:
                x_seq = batch["x_seq"].to(device)
                port_ids = batch["port_id"].to(device)
                naive_waiting = batch["naive_waiting"].to(device)
                target_h1 = batch["target_h1"].to(device)
                target_h4 = batch["target_h4"].to(device)
                metas = batch["meta"]

                preds = model(x_seq, port_ids, naive_waiting)
                p_h1 = preds["waiting_pred_h1"].cpu().numpy()
                p_h4 = preds["waiting_pred_h4"].cpu().numpy()
                t_h1 = target_h1.cpu().numpy()
                t_h4 = target_h4.cpu().numpy()
                nv = naive_waiting.cpu().numpy()

                for b in range(len(p_h1)):
                    records.append({
                        "split": split_name,
                        "port": metas["port"][b],
                        "port_id": int(metas["port_id"][b]),
                        "target_year": int(metas["target_year"][b]),
                        "target_week": int(metas["target_week"][b]),
                        "naive_persistence": float(nv[b]),
                        "actual_h1": float(t_h1[b]),
                        "pred_h1": float(p_h1[b]),
                        "error_h1": float(p_h1[b] - t_h1[b]),
                        "actual_h4": float(t_h4[b]),
                        "pred_h4": float(p_h4[b]),
                        "error_h4": float(p_h4[b] - t_h4[b]),
                    })
        return pd.DataFrame(records)

    val_preds_df = evaluate_split(val_loader, "val")
    test_preds_df = evaluate_split(test_loader, "test")

    # Save prediction parquet files
    val_pred_path = artifacts_dir / "validation_predictions.parquet"
    test_pred_path = artifacts_dir / "test_predictions.parquet"
    val_preds_df.to_parquet(val_pred_path, index=False)
    test_preds_df.to_parquet(test_pred_path, index=False)

    # Compute baseline comparisons
    def summarize_benchmark(df_eval):
        act_h1 = df_eval["actual_h1"].values
        pred_h1 = df_eval["pred_h1"].values
        naive_h1 = df_eval["naive_persistence"].values

        model_m = compute_metrics(act_h1, pred_h1)
        naive_m = compute_metrics(act_h1, naive_h1)

        # Per port metrics
        per_port = {}
        for p, pdf in df_eval.groupby("port"):
            p_act = pdf["actual_h1"].values
            p_pred = pdf["pred_h1"].values
            p_naive = pdf["naive_persistence"].values
            per_port[p] = {
                "count": len(pdf),
                "model_mae": round(float(np.mean(np.abs(p_pred - p_act))), 4),
                "naive_mae": round(float(np.mean(np.abs(p_naive - p_act))), 4),
                "model_rmse": round(float(np.sqrt(np.mean((p_pred - p_act)**2))), 4),
                "naive_rmse": round(float(np.sqrt(np.mean((p_naive - p_act)**2))), 4),
                "mean_actual": round(float(np.mean(p_act)), 2),
            }

        return {
            "model_metrics_h1": model_m,
            "persistence_metrics_h1": naive_m,
            "mae_reduction_pct": round(float((naive_m["mae"] - model_m["mae"]) / naive_m["mae"] * 100), 2),
            "rmse_reduction_pct": round(float((naive_m["rmse"] - model_m["rmse"]) / naive_m["rmse"] * 100), 2),
            "huber_reduction_pct": round(float((naive_m["huber"] - model_m["huber"]) / naive_m["huber"] * 100), 2),
            "per_port": per_port,
        }

    val_summary = summarize_benchmark(val_preds_df)
    test_summary = summarize_benchmark(test_preds_df)

    # 4-week horizon summary
    val_h4_model = compute_metrics(val_preds_df["actual_h4"].values, val_preds_df["pred_h4"].values)
    val_h4_naive = compute_metrics(val_preds_df["actual_h4"].values, val_preds_df["naive_persistence"].values)

    # Checkpoint sha256
    with open(best_model_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    elapsed = time.time() - start_time

    # Metadata and summary json
    eval_metrics = {
        "milestone": "3.6_Phase_B",
        "task": "Weekly Anchorage Waiting-Time Forecasting",
        "model_type": "ResidualDwellGRUForecaster",
        "trainable_parameters": total_params,
        "checkpoint_path": str(best_model_path),
        "checkpoint_sha256": ckpt_sha256,
        "runtime_seconds": round(elapsed, 2),
        "sample_counts": {
            "train": len(ds_train),
            "val": len(ds_val),
            "test": len(ds_test),
            "ports": len(ports),
        },
        "validation_evaluation_2021": val_summary,
        "validation_4week_horizon": {
            "model": val_h4_model,
            "naive": val_h4_naive,
        },
        "test_held_out_evaluation_2022": test_summary,
    }

    metrics_path = artifacts_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    metadata_path = artifacts_dir / "checkpoint_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "model_type": "ResidualDwellGRUForecaster",
            "sha256": ckpt_sha256,
            "best_epoch": checkpoint["epoch"],
            "best_val_mae": best_val_mae,
            "config": cfg,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL PHASE B DWELL FORECASTING RESULTS")
    print("=" * 80)
    print(f"Checkpoint SHA256: {ckpt_sha256}")
    print(f"\n[Validation 2021 - Crisis Period (N={len(ds_val)})]")
    print(f"  Persistence Baseline -> Huber: {val_summary['persistence_metrics_h1']['huber']}, MAE: {val_summary['persistence_metrics_h1']['mae']:.4f}h, RMSE: {val_summary['persistence_metrics_h1']['rmse']:.4f}h, R2: {val_summary['persistence_metrics_h1']['r2']:.4f}")
    print(f"  Residual Dwell GRU   -> Huber: {val_summary['model_metrics_h1']['huber']}, MAE: {val_summary['model_metrics_h1']['mae']:.4f}h, RMSE: {val_summary['model_metrics_h1']['rmse']:.4f}h, R2: {val_summary['model_metrics_h1']['r2']:.4f}")
    print(f"  --> MAE Reduction: {val_summary['mae_reduction_pct']}% | RMSE Reduction: {val_summary['rmse_reduction_pct']}% | Huber Reduction: {val_summary['huber_reduction_pct']}%")

    print(f"\n[Test 2022 - Post-Crisis Held-Out (N={len(ds_test)})]")
    print(f"  Persistence Baseline -> Huber: {test_summary['persistence_metrics_h1']['huber']}, MAE: {test_summary['persistence_metrics_h1']['mae']:.4f}h, RMSE: {test_summary['persistence_metrics_h1']['rmse']:.4f}h, R2: {test_summary['persistence_metrics_h1']['r2']:.4f}")
    print(f"  Residual Dwell GRU   -> Huber: {test_summary['model_metrics_h1']['huber']}, MAE: {test_summary['model_metrics_h1']['mae']:.4f}h, RMSE: {test_summary['model_metrics_h1']['rmse']:.4f}h, R2: {test_summary['model_metrics_h1']['r2']:.4f}")
    print(f"  --> MAE Reduction: {test_summary['mae_reduction_pct']}% | RMSE Reduction: {test_summary['rmse_reduction_pct']}% | Huber Reduction: {test_summary['huber_reduction_pct']}%")

    print(f"\n[4-Week Horizon Validation (N={len(ds_val)})]")
    print(f"  Persistence Baseline -> MAE: {val_h4_naive['mae']:.4f}h, RMSE: {val_h4_naive['rmse']:.4f}h, R2: {val_h4_naive['r2']:.4f}")
    print(f"  Residual Dwell GRU   -> MAE: {val_h4_model['mae']:.4f}h, RMSE: {val_h4_model['rmse']:.4f}h, R2: {val_h4_model['r2']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()


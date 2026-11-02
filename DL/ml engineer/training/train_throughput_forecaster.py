"""
Direct Throughput Forecaster Training Pipeline
==============================================
Milestone 3.6 Phase C: Direct Throughput Forecasting

Trains dedicated residual throughput prediction heads on top of the frozen
Milestone 3.4/3.5 Transformer encoder backbone.

Targets predicted:
1. Cumulative 7-day container-carrying vessel calls (target_container_calls_7d_future_sum)
2. Cumulative 7-day metric cargo tonnage (target_cargo_tonnage_7d_future_sum)

Zero backbone fine-tuning is performed; backbone weights remain 100% frozen.
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
import yaml

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sequence_dataset import PortSequenceDataset
from models.transformer_forecaster import ResidualTransformerForecaster
from models.throughput_forecaster import DirectThroughputResidualForecaster


class ThroughputMultiTargetDataset(Dataset):
    """Wraps PortSequenceDataset to yield input sequences and both throughput targets."""

    def __init__(self, port_seq_dataset: PortSequenceDataset):
        self.base_ds = port_seq_dataset
        df_meta = port_seq_dataset.df_meta
        self.container_targets = torch.tensor(
            df_meta["target_container_calls_7d_future_sum"].values, dtype=torch.float32
        ).unsqueeze(1)
        self.tonnage_targets = torch.tensor(
            df_meta["target_cargo_tonnage_7d_future_sum"].values, dtype=torch.float32
        ).unsqueeze(1)
        self.ports = df_meta["portwatch_id"].values
        self.names = df_meta["portwatch_name"].values
        self.dates = df_meta["reference_date"].values

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        features, _, _ = self.base_ds[idx]
        return {
            "x": features,
            "y_container": self.container_targets[idx],
            "y_tonnage": self.tonnage_targets[idx],
            "port": self.ports[idx],
            "name": self.names[idx],
            "date": self.dates[idx],
        }


def compute_regression_metrics(actual: np.ndarray, pred: np.ndarray, delta: float = 1.0) -> dict:
    """Compute standard regression metrics."""
    err = pred - actual
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    # Huber
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
    print("Milestone 3.6 Phase C: Direct Throughput Forecaster Training")
    print("=" * 80)

    # 1. Load config
    config_path = PROJECT_ROOT / "configs" / "throughput_forecaster.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

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

    # 2. Verify and Load Frozen Backbone
    backbone_ckpt_path = PROJECT_ROOT / cfg["backbone"]["checkpoint_path"]
    if not backbone_ckpt_path.exists():
        raise FileNotFoundError(f"Frozen backbone checkpoint missing: {backbone_ckpt_path}")

    with open(backbone_ckpt_path, "rb") as f:
        backbone_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Loaded frozen Transformer backbone from: {backbone_ckpt_path}")
    print(f"Backbone SHA256: {backbone_sha256}")

    ckpt = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_config"]

    backbone = ResidualTransformerForecaster(
        input_size=model_cfg["input_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        pooling=model_cfg["pooling"],
        pe_type=model_cfg["pe_type"],
        roll7_idx=model_cfg["roll7_idx"],
        roll7_mean=model_cfg["roll7_mean"],
        roll7_std=model_cfg["roll7_std"],
        constraint=model_cfg["constraint"],
    )
    backbone.load_state_dict(ckpt["model_state_dict"])

    # Scaler constants
    scaler_path = PROJECT_ROOT / cfg["paths"]["scaler_params"]
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    mu_c = scaler["mean"][15]
    std_c = scaler["std"][15]
    mu_t = scaler["mean"][11]
    std_t = scaler["std"][11]

    # Initialize model
    model = DirectThroughputResidualForecaster(
        backbone=backbone,
        head_hidden_dim=cfg["model"]["head_hidden_dim"],
        dropout=cfg["model"]["dropout"],
        mu_container_roll7=mu_c,
        std_container_roll7=std_c,
        mu_tonnage=mu_t,
        std_tonnage=std_t,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Model initialized: {trainable_params:,} trainable head parameters | {frozen_params:,} frozen backbone parameters.")

    # 3. Load Sequence Datasets
    meta_path = PROJECT_ROOT.parent / cfg["paths"]["sequence_metadata"]
    storage_dir = str(PROJECT_ROOT / cfg["paths"]["storage_dir"])

    cols = [
        "sequence_id", "portwatch_id", "portwatch_name", "country_iso3",
        "split_tag", "reference_date", "input_start_date", "target_start_date", "target_end_date",
        "target_container_calls_7d_future_sum", "target_cargo_tonnage_7d_future_sum"
    ]

    print("Loading sequence metadata...")
    df_train_meta = pq.read_table(meta_path, columns=cols, filters=[("split_tag", "=", "train")]).to_pandas()
    df_val_meta = pq.read_table(meta_path, columns=cols, filters=[("split_tag", "=", "val")]).to_pandas()

    train_limit = cfg["training"]["train_sample_limit"]
    val_limit = cfg["training"]["val_sample_limit"]

    if len(df_train_meta) > train_limit:
        # Balanced strided sampling across ports
        df_train_sample = df_train_meta.iloc[np.linspace(0, len(df_train_meta) - 1, train_limit, dtype=int)].reset_index(drop=True)
    else:
        df_train_sample = df_train_meta

    if len(df_val_meta) > val_limit:
        df_val_sample = df_val_meta.iloc[np.linspace(0, len(df_val_meta) - 1, val_limit, dtype=int)].reset_index(drop=True)
    else:
        df_val_sample = df_val_meta

    print(f"Sampled datasets: Train={len(df_train_sample):,} sequences | Val={len(df_val_sample):,} sequences")

    base_ds_train = PortSequenceDataset(
        meta_source=df_train_sample,
        storage_dir=storage_dir,
        split="train",
        target_col="target_container_calls_7d_future_sum",
        return_metadata=True,
    )
    base_ds_val = PortSequenceDataset(
        meta_source=df_val_sample,
        storage_dir=storage_dir,
        split="val",
        target_col="target_container_calls_7d_future_sum",
        return_metadata=True,
    )

    ds_train = ThroughputMultiTargetDataset(base_ds_train)
    ds_val = ThroughputMultiTargetDataset(base_ds_val)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    num_epochs = cfg["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    artifacts_dir = PROJECT_ROOT / cfg["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = artifacts_dir / "best_model.pt"

    # 5. Training Loop
    history = []
    best_val_loss = float("inf")

    print("\nStarting Throughput Heads Training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            y_c = batch["y_container"].to(device)
            y_t = batch["y_tonnage"].to(device)

            optimizer.zero_grad()
            out = model(x)

            loss_c = nn.functional.huber_loss(out["pred_container"], y_c, delta=1.0)
            loss_t = nn.functional.huber_loss(
                out["pred_tonnage"] / std_t, y_t / std_t, delta=0.5
            )
            loss = loss_c + 2.0 * loss_t

            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg["training"]["grad_clip_norm"],
            )
            optimizer.step()
            train_loss_total += loss.item() * len(x)

        scheduler.step()
        train_loss = train_loss_total / len(ds_train)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_actual_c, val_pred_c, val_naive_c = [], [], []
        val_actual_t, val_pred_t, val_naive_t = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y_c = batch["y_container"].to(device)
                y_t = batch["y_tonnage"].to(device)

                out = model(x)
                loss_c = nn.functional.huber_loss(out["pred_container"], y_c, delta=1.0)
                loss_t = nn.functional.huber_loss(
                    out["pred_tonnage"] / std_t, y_t / std_t, delta=0.5
                )
                val_loss_total += (loss_c + 2.0 * loss_t).item() * len(x)

                val_actual_c.append(y_c.cpu().numpy())
                val_pred_c.append(out["pred_container"].cpu().numpy())
                val_naive_c.append(out["naive_container"].cpu().numpy())

                val_actual_t.append(y_t.cpu().numpy())
                val_pred_t.append(out["pred_tonnage"].cpu().numpy())
                val_naive_t.append(out["naive_tonnage"].cpu().numpy())

        val_loss = val_loss_total / len(ds_val)

        v_act_c = np.concatenate(val_actual_c).flatten()
        v_pred_c = np.concatenate(val_pred_c).flatten()
        v_naive_c = np.concatenate(val_naive_c).flatten()
        m_c = compute_regression_metrics(v_act_c, v_pred_c, delta=1.0)
        n_c = compute_regression_metrics(v_act_c, v_naive_c, delta=1.0)

        v_act_t = np.concatenate(val_actual_t).flatten()
        v_pred_t = np.concatenate(val_pred_t).flatten()
        v_naive_t = np.concatenate(val_naive_t).flatten()
        m_t = compute_regression_metrics(v_act_t, v_pred_t, delta=50000.0)
        n_t = compute_regression_metrics(v_act_t, v_naive_t, delta=50000.0)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_container_mae": m_c["mae"],
            "val_container_rmse": m_c["rmse"],
            "val_container_r2": m_c["r2"],
            "val_tonnage_mae": m_t["mae"],
            "val_tonnage_rmse": m_t["rmse"],
            "val_tonnage_r2": m_t["r2"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Container MAE: {m_c['mae']:.4f} (Naive: {n_c['mae']:.4f}, R2: {m_c['r2']:.4f}) | "
            f"Tonnage MAE: {m_t['mae']:,.1f}t (Naive: {n_t['mae']:,.1f}t, R2: {m_t['r2']:.4f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "container_head_state_dict": model.container_head.state_dict(),
                    "tonnage_head_state_dict": model.tonnage_head.state_dict(),
                    "backbone_checkpoint_sha256": backbone_sha256,
                    "trainable_params": trainable_params,
                    "frozen_params": frozen_params,
                    "val_metrics_container": m_c,
                    "val_metrics_tonnage": m_t,
                    "config": cfg,
                },
                best_model_path,
            )
            print(f"  --> Saved new best throughput checkpoint to {best_model_path}")

    # 6. Save Training History
    history_df = pd.DataFrame(history)
    history_df.to_csv(artifacts_dir / "training_history.csv", index=False)

    # 7. Comprehensive Prediction Export and Evaluation
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.container_head.load_state_dict(best_ckpt["container_head_state_dict"])
    model.tonnage_head.load_state_dict(best_ckpt["tonnage_head_state_dict"])
    model.eval()

    print("\nRunning comprehensive evaluation and exporting predictions...")
    pred_records = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y_c = batch["y_container"].to(device)
            y_t = batch["y_tonnage"].to(device)
            ports = batch["port"]
            names = batch["name"]
            dates = batch["date"]

            out = model(x)
            p_c = out["pred_container"].cpu().numpy().flatten()
            n_c_batch = out["naive_container"].cpu().numpy().flatten()
            t_c = y_c.cpu().numpy().flatten()

            p_t = out["pred_tonnage"].cpu().numpy().flatten()
            n_t_batch = out["naive_tonnage"].cpu().numpy().flatten()
            t_t = y_t.cpu().numpy().flatten()

            for b in range(len(p_c)):
                pred_records.append({
                    "portwatch_id": ports[b],
                    "portwatch_name": names[b],
                    "reference_date": dates[b],
                    "actual_container_calls": float(t_c[b]),
                    "naive_container_calls": float(n_c_batch[b]),
                    "pred_container_calls": float(p_c[b]),
                    "error_container_calls": float(p_c[b] - t_c[b]),
                    "actual_cargo_tonnage": float(t_t[b]),
                    "naive_cargo_tonnage": float(n_t_batch[b]),
                    "pred_cargo_tonnage": float(p_t[b]),
                    "error_cargo_tonnage": float(p_t[b] - t_t[b]),
                })

    df_preds = pd.DataFrame(pred_records)
    preds_out_path = artifacts_dir / "validation_predictions.parquet"
    df_preds.to_parquet(preds_out_path, index=False)
    print(f"Exported {len(df_preds):,} validation prediction rows to {preds_out_path}")

    # Final Metric Summaries
    act_c = df_preds["actual_container_calls"].values
    prd_c = df_preds["pred_container_calls"].values
    nve_c = df_preds["naive_container_calls"].values
    final_m_c = compute_regression_metrics(act_c, prd_c, delta=1.0)
    final_n_c = compute_regression_metrics(act_c, nve_c, delta=1.0)

    act_t = df_preds["actual_cargo_tonnage"].values
    prd_t = df_preds["pred_cargo_tonnage"].values
    nve_t = df_preds["naive_cargo_tonnage"].values
    final_m_t = compute_regression_metrics(act_t, prd_t, delta=50000.0)
    final_n_t = compute_regression_metrics(act_t, nve_t, delta=50000.0)

    # Volume quartile stratification
    quantiles_c = pd.qcut(df_preds["actual_container_calls"], q=4, duplicates="drop")
    strat_c = {}
    for q, grp in df_preds.groupby(quantiles_c, observed=True):
        m_grp = compute_regression_metrics(grp["actual_container_calls"].values, grp["pred_container_calls"].values)
        n_grp = compute_regression_metrics(grp["actual_container_calls"].values, grp["naive_container_calls"].values)
        strat_c[str(q)] = {"count": len(grp), "model_mae": m_grp["mae"], "naive_mae": n_grp["mae"]}

    with open(best_model_path, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()

    elapsed = time.time() - start_time

    eval_summary = {
        "milestone": "3.6_Phase_C",
        "task": "Direct Throughput Forecasting",
        "model_type": "DirectThroughputResidualForecaster",
        "frozen_backbone_path": str(backbone_ckpt_path),
        "frozen_backbone_sha256": backbone_sha256,
        "throughput_checkpoint_sha256": model_sha256,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "runtime_seconds": round(elapsed, 2),
        "sample_counts": {
            "train_sample": len(df_train_sample),
            "val_sample": len(df_val_sample),
            "total_val_panel_sequences": len(df_val_meta),
        },
        "container_calls_7d_future_sum": {
            "model_metrics": final_m_c,
            "persistence_metrics": final_n_c,
            "mae_reduction_pct": round(float((final_n_c["mae"] - final_m_c["mae"]) / final_n_c["mae"] * 100), 2),
            "rmse_reduction_pct": round(float((final_n_c["rmse"] - final_m_c["rmse"]) / final_n_c["rmse"] * 100), 2),
            "huber_reduction_pct": round(float((final_n_c["huber"] - final_m_c["huber"]) / final_n_c["huber"] * 100), 2),
            "volume_quartiles": strat_c,
        },
        "cargo_tonnage_7d_future_sum": {
            "model_metrics": final_m_t,
            "persistence_metrics": final_n_t,
            "mae_reduction_pct": round(float((final_n_t["mae"] - final_m_t["mae"]) / final_n_t["mae"] * 100), 2),
            "rmse_reduction_pct": round(float((final_n_t["rmse"] - final_m_t["rmse"]) / final_n_t["rmse"] * 100), 2),
            "huber_reduction_pct": round(float((final_n_t["huber"] - final_m_t["huber"]) / final_n_t["huber"] * 100), 2),
        },
    }

    metrics_out = artifacts_dir / "evaluation_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(eval_summary, f, indent=2)

    meta_out = artifacts_dir / "checkpoint_metadata.json"
    with open(meta_out, "w") as f:
        json.dump({
            "model_type": "DirectThroughputResidualForecaster",
            "sha256": model_sha256,
            "backbone_sha256": backbone_sha256,
            "best_epoch": best_ckpt["epoch"],
            "config": cfg,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL PHASE C DIRECT THROUGHPUT BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Frozen Backbone SHA256 : {backbone_sha256}")
    print(f"Throughput Heads SHA256: {model_sha256}")
    print(f"\n[1. Container Vessel Calls (7-Day Future Cumulative Sum)]")
    print(f"  Naive Persistence -> Huber: {final_n_c['huber']:.4f} | MAE: {final_n_c['mae']:.4f} calls | RMSE: {final_n_c['rmse']:.4f} | R2: {final_n_c['r2']:.4f}")
    print(f"  Residual Heads    -> Huber: {final_m_c['huber']:.4f} | MAE: {final_m_c['mae']:.4f} calls | RMSE: {final_m_c['rmse']:.4f} | R2: {final_m_c['r2']:.4f}")
    print(f"  --> MAE Reduction: {eval_summary['container_calls_7d_future_sum']['mae_reduction_pct']}% | RMSE Reduction: {eval_summary['container_calls_7d_future_sum']['rmse_reduction_pct']}% | Huber Reduction: {eval_summary['container_calls_7d_future_sum']['huber_reduction_pct']}%")

    print(f"\n[2. Cargo Tonnage (7-Day Future Cumulative Metric Tons)]")
    print(f"  Naive Persistence -> Huber: {final_n_t['huber']:.4f} | MAE: {final_n_t['mae']:,.1f} tons | RMSE: {final_n_t['rmse']:,.1f} | R2: {final_n_t['r2']:.4f}")
    print(f"  Residual Heads    -> Huber: {final_m_t['huber']:.4f} | MAE: {final_m_t['mae']:,.1f} tons | RMSE: {final_m_t['rmse']:,.1f} | R2: {final_m_t['r2']:.4f}")
    print(f"  --> MAE Reduction: {eval_summary['cargo_tonnage_7d_future_sum']['mae_reduction_pct']}% | RMSE Reduction: {eval_summary['cargo_tonnage_7d_future_sum']['rmse_reduction_pct']}% | Huber Reduction: {eval_summary['cargo_tonnage_7d_future_sum']['huber_reduction_pct']}%")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""Main Training Entrypoint for Milestone 3.2: LSTM Forecasting Baseline.

Executes reproducible training, validation, checkpointing, and prediction handoff.
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
import torch
import psutil

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models.lstm import LSTMForecaster
from models.lstm_variants import PortEmbeddedLSTMForecaster, ResidualLSTMForecaster
from data.data_module import create_dataloaders
from training.trainer import LSTMTrainer
from training.metrics import StreamingRegressionMetrics
from inference import run_inference_and_export


def set_seed(seed: int = 42):
    """Ensure reproducible execution across torch, numpy, and python."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_training_pipeline(
    config_path: str,
    epochs: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    model_type: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """Execute complete end-to-end training and handoff workflow."""
    process = psutil.Process(os.getpid())
    init_mem_mb = process.memory_info().rss / (1024 * 1024)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("random_seed", 42)
    set_seed(seed)

    # Resolve paths
    base_dir = Path(config_path).resolve().parents[1]
    pipe_cfg_path = base_dir / "configs/sequence_pipeline.yaml"
    if artifacts_dir is not None:
        art_dir = Path(artifacts_dir).resolve()
    else:
        art_dir = base_dir / "artifacts/lstm"
    art_dir.mkdir(parents=True, exist_ok=True)

    # Determine model type
    m_cfg = cfg.get("model", {})
    resolved_model_type = model_type or m_cfg.get("model_type", "LSTMForecaster")
    return_port_idx = (resolved_model_type == "PortEmbeddedLSTMForecaster")

    print("=" * 80)
    print(" MILESTONE 3.2: LSTM FORECASTING TRAINING WORKFLOW")
    print(f" Config:         {config_path}")
    print(f" Artifacts Dir:  {art_dir}")
    print(f" Model Type:     {resolved_model_type}")
    print(f" Initial Memory: {init_mem_mb:.2f} MB")
    print("=" * 80)

    # 1. Initialize DataLoaders
    effective_batch_size = batch_size or cfg.get("training", {}).get("batch_size", 64)
    print("\n[Step 1/5] Initializing Production Sequence DataLoaders...")
    loaders = create_dataloaders(
        config_path=str(pipe_cfg_path),
        batch_size=effective_batch_size,
        val_batch_size=512,
        num_workers=0,
        return_port_idx=return_port_idx,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"  Train Loader Batches: {len(train_loader):,} (batch_size={effective_batch_size})")
    print(f"  Val Loader Batches:   {len(val_loader):,} (batch_size=512)")

    # 2. Instantiate Model
    print(f"\n[Step 2/5] Instantiating {resolved_model_type}...")
    if resolved_model_type == "PortEmbeddedLSTMForecaster":
        model = PortEmbeddedLSTMForecaster(
            num_ports=m_cfg.get("num_ports", 2065),
            embedding_dim=m_cfg.get("embedding_dim", 16),
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            output_size=m_cfg.get("output_size", 1),
            output_activation=m_cfg.get("output_activation", "softplus"),
        )
    elif resolved_model_type == "ResidualLSTMForecaster":
        model = ResidualLSTMForecaster(
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            roll7_idx=m_cfg.get("roll7_idx", 12),
            roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
            roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
            constraint=m_cfg.get("constraint", "relu"),
        )
    else:
        model = LSTMForecaster(
            input_size=m_cfg.get("input_size", 20),
            hidden_size=m_cfg.get("hidden_size", 64),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
            output_size=m_cfg.get("output_size", 1),
            output_activation=m_cfg.get("output_activation", "softplus"),
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Type:        {resolved_model_type}")
    print(f"  Trainable Params:  {num_params:,}")

    # 3. Instantiate Trainer
    print("\n[Step 3/5] Configuring Optimizer, Loss, and Device...")
    trainer = LSTMTrainer(model=model, config=cfg)
    print(f"  Device:            {trainer.device}")
    print(f"  Loss Function:     {trainer.criterion}")
    print(f"  Optimizer:         AdamW (lr={cfg['training']['learning_rate']}, wd={cfg['training']['weight_decay']})")
    print(f"  Grad Clip Norm:    {trainer.grad_clip_norm}")

    # Save training configuration
    cfg_copy = dict(cfg)
    cfg_copy["model"]["model_type"] = resolved_model_type
    cfg_dump_path = art_dir / "training_config.json"
    with open(cfg_dump_path, "w") as f:
        json.dump(cfg_copy, f, indent=2)

    # 4. Execute Training Loop
    total_epochs = epochs or cfg.get("training", {}).get("num_epochs", 1)
    patience = cfg.get("training", {}).get("early_stopping_patience", 2)
    print(f"\n[Step 4/5] Executing Training Schedule ({total_epochs} epoch(s))...")
    t_train_start = time.time()

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=total_epochs,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        artifacts_dir=str(art_dir),
        early_stopping_patience=patience,
    )
    train_total_elapsed = time.time() - t_train_start

    # 5. Export Validation Predictions for Evaluation Engineer
    print("\n[Step 5/5] Generating Validation Predictions Parquet Handoff...")
    best_model_path = art_dir / "best_model.pt"
    val_pred_parquet = art_dir / "validation_predictions.parquet"
    
    handoff_metrics = run_inference_and_export(
        checkpoint_path=str(best_model_path),
        config_path=str(pipe_cfg_path),
        output_parquet_path=str(val_pred_parquet),
        split="val",
        batch_size=1024,
        device=str(trainer.device),
    )

    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    print("=" * 80)
    print(" MILESTONE 3.2 TRAINING PIPELINE SUCCESSFULLY COMPLETED!")
    print(f" Total Training Duration:    {train_total_elapsed:.2f} seconds ({train_total_elapsed/60:.2f} minutes)")
    print(f" Best Validation Huber Loss: {trainer.best_val_loss:.4f}")
    print(f" Best Checkpoint:            {best_model_path}")
    print(f" Validation Predictions:     {val_pred_parquet} ({handoff_metrics['file_size_mb']:.2f} MB)")
    print(f" Peak Process RSS:           {peak_rss_mb:.2f} MB")
    print("=" * 80)

    return {
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "history": history,
        "handoff": handoff_metrics,
        "peak_rss_mb": peak_rss_mb,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Forecasting Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(WORKSPACE_DIR / "configs/lstm_baseline.yaml"),
        help="Path to lstm_baseline.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max batches per train epoch (for testing/calibration)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Max batches per val epoch (for testing/calibration)")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Custom artifacts destination directory")
    parser.add_argument("--model-type", type=str, default=None, help="Model architecture type")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")

    args = parser.parse_args()
    run_training_pipeline(
        config_path=args.config,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        artifacts_dir=args.artifacts_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
    )

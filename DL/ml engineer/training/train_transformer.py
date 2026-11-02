"""Training & Handoff Pipeline for Lightweight Transformer Forecaster (Milestone 3.4).

Executes reproducible training, validation, checkpointing, and Parquet streaming export
for the ResidualTransformerForecaster architecture.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import psutil
import torch

# Add workspace to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from models.transformer_forecaster import ResidualTransformerForecaster
from data.data_module import create_dataloaders
from training.trainer import LSTMTrainer
from inference import run_inference_and_export


def set_seed(seed: int = 42):
    """Ensure reproducible execution across torch, numpy, and python."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_transformer_training(
    config_path: str,
    epochs: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    pooling: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end training and prediction export for Transformer."""
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

    pipe_cfg_path = resolve_path(cfg["paths"]["pipeline_config"])

    if artifacts_dir is not None:
        art_dir = Path(artifacts_dir).resolve()
    else:
        art_dir = base_dir / cfg["paths"].get("artifacts_dir", "artifacts/transformer")
    art_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = cfg.get("model", {})
    resolved_pooling = pooling or m_cfg.get("pooling", "last")

    print("=" * 80)
    print(" MILESTONE 3.4: TRANSFORMER TEMPORAL RESIDUAL FORECASTER TRAINING")
    print(f" Config:         {config_path}")
    print(f" Artifacts Dir:  {art_dir}")
    print(f" Pooling Mode:   {resolved_pooling.upper()}")
    print(f" Initial Memory: {init_mem_mb:.2f} MB")
    print("=" * 80)

    # 1. Initialize DataLoaders
    effective_batch_size = batch_size or cfg.get("training", {}).get("batch_size", 64)
    print("\n[Step 1/5] Initializing Sequence DataLoaders...")
    loaders = create_dataloaders(
        config_path=str(pipe_cfg_path),
        batch_size=effective_batch_size,
        val_batch_size=512,
        num_workers=0,
        return_port_idx=False,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"  Train Loader Batches: {len(train_loader):,} (batch_size={effective_batch_size})")
    print(f"  Val Loader Batches:   {len(val_loader):,} (batch_size=512)")

    # 2. Instantiate Model
    print("\n[Step 2/5] Instantiating ResidualTransformerForecaster...")
    model = ResidualTransformerForecaster(
        input_size=m_cfg.get("input_size", 20),
        seq_len=m_cfg.get("seq_len", 30),
        d_model=m_cfg.get("d_model", 64),
        nhead=m_cfg.get("nhead", 4),
        num_layers=m_cfg.get("num_layers", 2),
        dim_feedforward=m_cfg.get("dim_feedforward", 128),
        dropout=m_cfg.get("dropout", 0.1),
        pooling=resolved_pooling,
        pe_type=m_cfg.get("pe_type", "sinusoidal"),
        roll7_idx=m_cfg.get("roll7_idx", 12),
        roll7_mean=m_cfg.get("roll7_mean", 2.4068180554238894),
        roll7_std=m_cfg.get("roll7_std", 6.3483944613159915),
        constraint=m_cfg.get("constraint", "relu"),
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Type:        ResidualTransformerForecaster (pooling={resolved_pooling})")
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
    cfg_copy["model"]["pooling"] = resolved_pooling
    with open(art_dir / "training_config.json", "w") as f:
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

    # 5. Export Validation Predictions
    print("\n[Step 5/5] Generating Validation Predictions Parquet Handoff...")
    best_model_path = art_dir / "best_model.pt"
    val_pred_parquet = art_dir / "validation_predictions.parquet"

    t_inf_start = time.time()
    handoff_metrics = run_inference_and_export(
        checkpoint_path=str(best_model_path),
        config_path=str(pipe_cfg_path),
        output_parquet_path=str(val_pred_parquet),
        split="val",
        batch_size=1024,
        device=str(trainer.device),
    )
    inf_duration_sec = time.time() - t_inf_start
    inf_throughput = len(val_loader.dataset) / inf_duration_sec

    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    print("=" * 80)
    print(" MILESTONE 3.4 TRANSFORMER TRAINING COMPLETED!")
    print(f" Total Training Duration:    {train_total_elapsed:.2f} seconds ({train_total_elapsed/60:.2f} minutes)")
    print(f" Best Validation Huber Loss: {trainer.best_val_loss:.4f}")
    print(f" Inference Throughput:       {inf_throughput:,.1f} seq/s ({inf_duration_sec:.2f}s for {len(val_loader.dataset):,} sequences)")
    print(f" Validation Predictions:     {val_pred_parquet} ({handoff_metrics['file_size_mb']:.2f} MB)")
    print(f" Peak Process RSS:           {peak_rss_mb:.2f} MB")
    print("=" * 80)

    # Save summary execution metadata
    meta = {
        "model_type": "ResidualTransformerForecaster",
        "pooling": resolved_pooling,
        "trainable_parameters": num_params,
        "training_duration_sec": train_total_elapsed,
        "best_epoch": trainer.best_epoch,
        "best_val_huber": trainer.best_val_loss,
        "inference_duration_sec": inf_duration_sec,
        "inference_throughput_seq_per_sec": inf_throughput,
        "peak_rss_mb": peak_rss_mb,
    }
    with open(art_dir / "execution_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "history": history,
        "handoff": handoff_metrics,
        "peak_rss_mb": peak_rss_mb,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Residual Transformer Forecaster")
    parser.add_argument("--config", type=str, default="DL/ml engineer/configs/transformer.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()
    run_transformer_training(
        config_path=args.config,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        artifacts_dir=args.artifacts_dir,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )


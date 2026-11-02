"""LSTM Model Trainer for Milestone 3.2.

Orchestrates reproducible model training, validation under no_grad, gradient clipping,
streaming metric tracking, early stopping, and atomic checkpoint saving.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import StreamingRegressionMetrics
from models.lstm import LSTMForecaster


class LSTMTrainer:
    """Production trainer for the LSTM forecasting baseline."""

    def __init__(
        self,
        model: LSTMForecaster,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config

        # Determine device
        if device is None:
            cfg_device = self.config.get("training", {}).get("device", "mps").lower()
            if cfg_device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Loss function
        loss_type = self.config.get("training", {}).get("loss_type", "huber").lower()
        huber_delta = float(self.config.get("training", {}).get("huber_delta", 1.0))
        if loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=huber_delta)
        elif loss_type == "mae" or loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type '{loss_type}'.")

        # Optimizer
        lr = float(self.config.get("training", {}).get("learning_rate", 1e-3))
        weight_decay = float(self.config.get("training", {}).get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Gradient clipping
        self.grad_clip_norm = float(self.config.get("training", {}).get("grad_clip_norm", 1.0))

        # Metrics tracker
        self.train_metrics = StreamingRegressionMetrics(huber_delta=huber_delta)
        self.val_metrics = StreamingRegressionMetrics(huber_delta=huber_delta)

        # Training state
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        max_batches: Optional[int] = None,
        log_interval: int = 2000,
    ) -> Dict[str, float]:
        """Execute one complete training epoch over train_loader."""
        self.model.train()
        self.train_metrics.reset()
        t_epoch_start = time.time()

        total_batches = len(train_loader)
        target_batches = min(total_batches, max_batches) if max_batches else total_batches
        t_batch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            if len(batch) == 3:
                bx, bp, by = batch
                bx = bx.to(self.device)
                bp = bp.to(self.device)
                by = by.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(bx, bp)
            else:
                bx, by = batch
                bx = bx.to(self.device)
                by = by.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(bx)

            loss = self.criterion(pred, by)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at batch {batch_idx}: {loss.item()}")

            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            self.optimizer.step()

            # Accumulate streaming metrics
            self.train_metrics.update(pred.detach(), by.detach())

            # Periodic batch progress logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == target_batches:
                batch_elapsed = time.time() - t_batch_start
                rate = (batch_idx + 1) / batch_elapsed
                print(
                    f"  [Epoch {epoch:2d}] Batch {batch_idx + 1:5d}/{target_batches:5d} "
                    f"| Loss: {loss.item():.4f} "
                    f"| Running Huber: {self.train_metrics.compute()['huber_loss']:.4f} "
                    f"| Speed: {rate:.1f} b/s ({rate * bx.size(0):,.0f} seq/s)"
                )

        if self.device.type == "mps":
            torch.mps.synchronize()

        epoch_elapsed = time.time() - t_epoch_start
        metrics = self.train_metrics.compute()
        metrics["epoch_duration_sec"] = epoch_elapsed
        metrics["batches_processed"] = target_batches
        return metrics

    def validate(
        self,
        val_loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Execute validation evaluation under torch.no_grad()."""
        self.model.eval()
        self.val_metrics.reset()
        t_val_start = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches and batch_idx >= max_batches:
                    break

                if len(batch) == 3:
                    bx, bp, by = batch
                    bx = bx.to(self.device)
                    bp = bp.to(self.device)
                    by = by.to(self.device)
                    pred = self.model(bx, bp)
                else:
                    bx, by = batch
                    bx = bx.to(self.device)
                    by = by.to(self.device)
                    pred = self.model(bx)

                self.val_metrics.update(pred, by)

        if self.device.type == "mps":
            torch.mps.synchronize()

        val_elapsed = time.time() - t_val_start
        metrics = self.val_metrics.compute()
        metrics["val_duration_sec"] = val_elapsed
        metrics["val_batches"] = batch_idx + 1
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 3,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None,
        artifacts_dir: Optional[str] = None,
        early_stopping_patience: int = 2,
    ) -> List[Dict[str, Any]]:
        """Fit the model over multiple epochs with checkpoint saving and early stopping."""
        if artifacts_dir is not None:
            os.makedirs(artifacts_dir, exist_ok=True)

        print("=" * 80)
        print(f" STARTING LSTM TRAINING PIPELINE ON DEVICE: {self.device}")
        print(f" Epochs: {num_epochs} | Grad Clip: {self.grad_clip_norm} | LR: {self.optimizer.param_groups[0]['lr']}")
        print(f" Train Batches: {len(train_loader):,} (limit: {max_train_batches or 'Full'})")
        print(f" Val Batches:   {len(val_loader):,} (limit: {max_val_batches or 'Full'})")
        print("=" * 80)

        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            train_m = self.train_epoch(train_loader, epoch, max_batches=max_train_batches)
            val_m = self.validate(val_loader, max_batches=max_val_batches)

            # Record epoch statistics
            record = {
                "epoch": epoch,
                "train_huber": train_m["huber_loss"],
                "train_mae": train_m["mae"],
                "train_rmse": train_m["rmse"],
                "train_r2": train_m["r2"],
                "train_mean_pred": train_m["mean_pred"],
                "train_duration_sec": train_m["epoch_duration_sec"],
                "val_huber": val_m["huber_loss"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
                "val_r2": val_m["r2"],
                "val_mean_pred": val_m["mean_pred"],
                "val_min_pred": val_m["min_pred"],
                "val_max_pred": val_m["max_pred"],
                "val_duration_sec": val_m["val_duration_sec"],
            }
            self.history.append(record)

            print(
                f"Epoch {epoch:2d} Summary: "
                f"Train Huber={train_m['huber_loss']:.4f}, MAE={train_m['mae']:.4f}, R2={train_m['r2']:.4f} ({train_m['epoch_duration_sec']:.1f}s) | "
                f"Val Huber={val_m['huber_loss']:.4f}, MAE={val_m['mae']:.4f}, RMSE={val_m['rmse']:.4f}, R2={val_m['r2']:.4f} ({val_m['val_duration_sec']:.1f}s)"
            )

            # Checkpoint selection (minimize validation Huber loss)
            current_val_loss = val_m["huber_loss"]
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                if artifacts_dir:
                    best_path = Path(artifacts_dir) / "best_model.pt"
                    self.save_checkpoint(str(best_path), is_best=True, epoch=epoch, val_metrics=val_m)
                    print(f"  --> Best model checkpoint saved to {best_path} (Val Huber: {current_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  (Val loss did not improve. Patience: {epochs_without_improvement}/{early_stopping_patience})")

            # Always save last model checkpoint
            if artifacts_dir:
                last_path = Path(artifacts_dir) / "last_model.pt"
                self.save_checkpoint(str(last_path), is_best=False, epoch=epoch, val_metrics=val_m)

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

        # Save history CSV and metadata
        if artifacts_dir:
            hist_path = Path(artifacts_dir) / "training_history.csv"
            pd.DataFrame(self.history).to_csv(hist_path, index=False)
            print(f"Training history saved -> {hist_path}")

        print("=" * 80)
        print(f" TRAINING COMPLETE! Best Epoch: {self.best_epoch} | Best Val Huber Loss: {self.best_val_loss:.4f}")
        print("=" * 80)
        return self.history

    def save_checkpoint(
        self,
        checkpoint_path: str,
        is_best: bool = False,
        epoch: int = 1,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Save a comprehensive, self-contained model checkpoint."""
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model.get_config(),
            "training_config": self.config,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "is_best": is_best,
            "val_metrics": val_metrics or {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": torch.__version__,
            "device": str(self.device),
        }
        torch.save(checkpoint_data, checkpoint_path)

        # Write or update checkpoint metadata JSON
        meta_json_path = Path(checkpoint_path).parent / "checkpoint_metadata.json"
        meta_dict = {
            "architecture": self.model.get_config(),
            "best_epoch": self.best_epoch,
            "best_val_huber": self.best_val_loss,
            "latest_epoch": epoch,
            "val_metrics": val_metrics or {},
            "optimizer": {
                "name": "AdamW",
                "lr": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                "grad_clip_norm": self.grad_clip_norm,
            },
            "device": str(self.device),
            "torch_version": torch.__version__,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(meta_json_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load weights and training state from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        return ckpt

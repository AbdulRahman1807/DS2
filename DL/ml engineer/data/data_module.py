"""DataLoader Factory and DataModule for Milestone 3.1.

Constructs PyTorch DataLoaders for Train, Validation, and Test splits:
- Default batch size: 64
- Default num_workers: 0 (avoids macOS multiprocessing IPC overhead)
- Train shuffle: True (shuffles sequence order, never within a sequence)
- Validation / Test shuffle: False (deterministic evaluation order)
- Drop last: False
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .sequence_dataset import PortSequenceDataset


class PortSequenceDataModule:
    """Manages train, validation, and test PortSequenceDataset and DataLoader pipelines."""

    def __init__(
        self,
        meta_parquet_path: str,
        storage_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        availability_lag: int = 2,
        window_len: int = 30,
        target_col: str = "target_portcalls_7d_future_sum",
        filter_valid_dates: bool = True,
        return_port_idx: bool = False,
    ):
        self.meta_parquet_path = meta_parquet_path
        self.storage_dir = storage_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.availability_lag = availability_lag
        self.window_len = window_len
        self.target_col = target_col
        self.filter_valid_dates = filter_valid_dates
        self.return_port_idx = return_port_idx

        self.train_dataset: Optional[PortSequenceDataset] = None
        self.val_dataset: Optional[PortSequenceDataset] = None
        self.test_dataset: Optional[PortSequenceDataset] = None

    def setup(self):
        """Instantiate the three split datasets efficiently in bounded memory."""
        import pandas as pd
        cols_to_load = ["sequence_id", "portwatch_id", "split_tag", "reference_date", self.target_col]
        df_all = pd.read_parquet(self.meta_parquet_path, columns=cols_to_load)

        df_train = df_all[df_all["split_tag"] == "train"].copy().reset_index(drop=True)
        df_val = df_all[df_all["split_tag"] == "val"].copy().reset_index(drop=True)
        df_test = df_all[df_all["split_tag"] == "test"].copy().reset_index(drop=True)
        del df_all

        self.train_dataset = PortSequenceDataset(
            meta_source=df_train,
            storage_dir=self.storage_dir,
            split="train",
            target_col=self.target_col,
            availability_lag=self.availability_lag,
            window_len=self.window_len,
            filter_valid_dates=self.filter_valid_dates,
            return_metadata=False,
            return_port_idx=self.return_port_idx,
        )

        self.val_dataset = PortSequenceDataset(
            meta_source=df_val,
            storage_dir=self.storage_dir,
            split="val",
            target_col=self.target_col,
            availability_lag=self.availability_lag,
            window_len=self.window_len,
            filter_valid_dates=self.filter_valid_dates,
            return_metadata=False,
            return_port_idx=self.return_port_idx,
        )

        self.test_dataset = PortSequenceDataset(
            meta_source=df_test,
            storage_dir=self.storage_dir,
            split="test",
            target_col=self.target_col,
            availability_lag=self.availability_lag,
            window_len=self.window_len,
            filter_valid_dates=self.filter_valid_dates,
            return_metadata=False,
            return_port_idx=self.return_port_idx,
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self, shuffle: bool = False, batch_size: Optional[int] = None) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self, shuffle: bool = False, batch_size: Optional[int] = None) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


def create_dataloaders(
    meta_parquet_path: Optional[str] = None,
    storage_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    availability_lag: int = 2,
    window_len: int = 30,
    target_col: str = "target_portcalls_7d_future_sum",
    filter_valid_dates: bool = True,
    val_batch_size: Optional[int] = None,
    return_port_idx: bool = False,
) -> Dict[str, DataLoader]:
    """Helper factory function to create {'train': train_loader, 'val': val_loader, 'test': test_loader}.
    
    Accepts either explicit paths or a config_path to sequence_pipeline.yaml.
    """
    if config_path is not None:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        cfg_p = Path(config_path).resolve()
        
        def resolve_cfg_path(raw_str: str) -> str:
            p = Path(raw_str)
            if p.is_absolute() and p.exists():
                return str(p)
            if p.exists():
                return str(p.resolve())
            for parent in cfg_p.parents:
                if (parent / p).exists():
                    return str(parent / p)
            return str(p)

        meta_parquet_path = resolve_cfg_path(cfg["paths"]["sequence_metadata_parquet"])
        storage_dir = resolve_cfg_path(cfg["paths"]["storage_dir"])
        if batch_size is None:
            batch_size = cfg.get("dataloader", {}).get("batch_size", 64)
        num_workers = cfg.get("dataloader", {}).get("num_workers", num_workers)
        pin_memory = cfg.get("dataloader", {}).get("pin_memory", pin_memory)
        drop_last = cfg.get("dataloader", {}).get("drop_last", drop_last)
        availability_lag = cfg.get("availability_policy", {}).get("lag_days", availability_lag)
        window_len = cfg.get("availability_policy", {}).get("lookback_days", window_len)
        target_col = cfg.get("target_contract", {}).get("primary_target", target_col)
        filter_valid_dates = cfg.get("dataloader", {}).get("filter_valid_dates", filter_valid_dates)
    elif meta_parquet_path is None or storage_dir is None:
        raise ValueError("Must provide either config_path or both meta_parquet_path and storage_dir.")

    dm = PortSequenceDataModule(
        meta_parquet_path=meta_parquet_path,
        storage_dir=storage_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        availability_lag=availability_lag,
        window_len=window_len,
        target_col=target_col,
        filter_valid_dates=filter_valid_dates,
        return_port_idx=return_port_idx,
    )
    dm.setup()
    return {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(batch_size=val_batch_size),
        "test": dm.test_dataloader(batch_size=val_batch_size),
    }

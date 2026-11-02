"""Metadata-Driven, Availability-Aware PyTorch Sequence Dataset for Milestone 3.1.

Implements the strict PortWatch T-2 availability policy:
- For reference date t, model consumes observation dates [t - 31, ..., t - 2] (30 days).
- Observation dates t-1 and t are strictly quarantined (two-day publication latency gap).
- Target window is [t + 1, ..., t + 7] (seven days), matching sequence_metadata.parquet.
- High-speed zero-copy integer offset slicing via MemmapStorageReader.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .storage import MemmapStorageReader


class PortSequenceDataset(Dataset):
    """PyTorch Dataset indexing 30-day availability-safe sequences for global ports."""

    def __init__(
        self,
        meta_source: Union[str, pd.DataFrame],
        storage_dir: str,
        split: str = "train",
        target_col: str = "target_portcalls_7d_future_sum",
        availability_lag: int = 2,
        window_len: int = 30,
        filter_valid_dates: bool = True,
        return_metadata: bool = False,
        return_port_idx: bool = False,
    ):
        """Initialize the sequence dataset.
        
        Args:
            meta_source: Path to sequence_metadata.parquet OR a pre-loaded/filtered pandas DataFrame.
            storage_dir: Directory containing memory-mapped features and manifest.
            split: Partition to filter ('train', 'val', 'test', or 'all').
            target_col: Column name of target in sequence_metadata.parquet.
            availability_lag: Publication lag in days (default 2 for PortWatch T-2).
            window_len: Number of historical observation days (default 30).
            filter_valid_dates: If True, drops sequences whose observation window precedes
                               the earliest panel date (2019-01-01).
            return_metadata: If True, __getitem__ returns (x, y, meta_dict) for debugging.
            return_port_idx: If True, __getitem__ returns (x, port_idx, y) for entity embedding.
        """
        super().__init__()
        self.meta_source = meta_source
        self.storage_dir = storage_dir
        self.split = split.lower()
        self.target_col = target_col
        self.availability_lag = availability_lag
        self.window_len = window_len
        self.filter_valid_dates = filter_valid_dates
        self.return_metadata = return_metadata
        self.return_port_idx = return_port_idx

        # 1. Initialize memory-mapped storage reader
        self.storage_reader = MemmapStorageReader(storage_dir)

        # 2. Ingest authoritative sequence metadata (bounded memory loading)
        if isinstance(self.meta_source, pd.DataFrame):
            df_meta = self.meta_source
        else:
            cols_to_load = ["sequence_id", "portwatch_id", "split_tag", "reference_date", self.target_col]
            if self.return_metadata:
                cols_to_load += ["portwatch_name", "country_iso3", "input_start_date", "target_start_date", "target_end_date"]
            seen = set()
            dedup_cols = [c for c in cols_to_load if not (c in seen or seen.add(c))]
            df_meta = pd.read_parquet(self.meta_source, columns=dedup_cols)

        if self.split != "all":
            valid_splits = {"train", "val", "test"}
            if self.split not in valid_splits:
                raise ValueError(f"Invalid split '{self.split}'. Must be one of {valid_splits} or 'all'.")
            if "split_tag" in df_meta.columns and (df_meta["split_tag"] != self.split).any():
                df_meta = df_meta[df_meta["split_tag"] == self.split].copy()

        if len(df_meta) == 0:
            raise ValueError(f"No sequences found for split '{self.split}' in {meta_parquet_path}")

        if self.target_col not in df_meta.columns:
            raise KeyError(
                f"Target column '{self.target_col}' not found in metadata. "
                f"Available: {[c for c in df_meta.columns if c.startswith('target_')]}"
            )

        # 3. Calculate Availability-Safe Observation Dates via Direct Integer Offsets:
        # For window_len=30 and lag=2: [t - 31, t - 2]
        date2idx = self.storage_reader.date2idx
        idx2date = self.storage_reader.idx2date
        num_panel_days = len(idx2date)

        ref_day_series = df_meta["reference_date"].map(date2idx)
        if ref_day_series.isnull().any():
            missing_dates = df_meta.loc[ref_day_series.isnull(), "reference_date"].unique()
            raise KeyError(f"Reference dates not found in storage mapping: {missing_dates[:5]}")
        ref_day_indices = ref_day_series.to_numpy(dtype=np.int32)

        start_day_offset = self.window_len + self.availability_lag - 1
        obs_start_day_indices = ref_day_indices - start_day_offset
        obs_end_day_indices = ref_day_indices - self.availability_lag

        # Check out-of-bounds observation windows (start < 0 or end >= num_panel_days)
        valid_mask = (obs_start_day_indices >= 0) & (obs_end_day_indices < num_panel_days)
        self.num_out_of_bounds = int((~valid_mask).sum())

        if self.filter_valid_dates and self.num_out_of_bounds > 0:
            df_meta = df_meta[valid_mask].copy().reset_index(drop=True)
            obs_start_day_indices = obs_start_day_indices[valid_mask]
            obs_end_day_indices = obs_end_day_indices[valid_mask]
            valid_mask = valid_mask[valid_mask]
        else:
            df_meta = df_meta.reset_index(drop=True)

        self.df_meta = df_meta
        self.num_sequences = len(df_meta)

        # 4. Precompute integer index lookups for ultra-fast __getitem__ (vectorized)
        port2idx = self.storage_reader.port2idx
        port_series = self.df_meta["portwatch_id"].map(port2idx)
        if port_series.isnull().any():
            missing_ports = self.df_meta.loc[port_series.isnull(), "portwatch_id"].unique()
            raise KeyError(f"Ports not found in storage mapping: {missing_ports[:5]}")
        self.port_indices = port_series.to_numpy(dtype=np.int16)
        self.start_day_indices = obs_start_day_indices.astype(np.int16)
        self.end_day_indices = obs_end_day_indices.astype(np.int16)
        self.valid_flags = valid_mask.astype(bool)
        self.targets = self.df_meta[self.target_col].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.num_sequences}")

        if not self.valid_flags[idx]:
            seq_id = self.df_meta.iloc[idx]["sequence_id"]
            p_id = self.df_meta.iloc[idx]["portwatch_id"]
            ref_d = self.df_meta.iloc[idx]["reference_date"]
            start_offset = self.window_len + self.availability_lag - 1
            calc_start_date = (pd.Timestamp(ref_d) - pd.Timedelta(days=start_offset)).strftime("%Y-%m-%d")
            raise ValueError(
                f"Sequence {seq_id} (port {p_id}) requires observation start date {calc_start_date} "
                f"which precedes the earliest panel date ({self.storage_reader.idx2date[0]}) "
                f"under availability lag {self.availability_lag}."
            )

        p_idx = int(self.port_indices[idx])
        d_start = int(self.start_day_indices[idx])
        d_end = int(self.end_day_indices[idx])

        # Slice memory-mapped array [30, 20]
        x_np = self.storage_reader.get_sequence_by_indices(p_idx, d_start, self.window_len)
        x_tensor = torch.from_numpy(x_np).float()

        # Target tensor [1]
        y_val = float(self.targets[idx])
        y_tensor = torch.tensor([y_val], dtype=torch.float32)

        if not self.return_metadata:
            if self.return_port_idx:
                return x_tensor, torch.tensor(p_idx, dtype=torch.long), y_tensor
            return x_tensor, y_tensor

        row = self.df_meta.iloc[idx]
        obs_start_str = self.storage_reader.idx2date[d_start] if d_start >= 0 else "OUT_OF_BOUNDS"
        obs_end_str = self.storage_reader.idx2date[d_end] if d_end >= 0 else "OUT_OF_BOUNDS"
        meta_dict = {
            "sequence_id": str(row["sequence_id"]),
            "portwatch_id": str(row["portwatch_id"]),
            "portwatch_name": str(row["portwatch_name"]),
            "country_iso3": str(row["country_iso3"]),
            "split_tag": str(row["split_tag"]),
            "reference_date": str(row["reference_date"]),
            "nominal_input_start_date": str(row["input_start_date"]),
            "operational_obs_start_date": obs_start_str,
            "operational_obs_end_date": obs_end_str,
            "target_start_date": str(row["target_start_date"]),
            "target_end_date": str(row["target_end_date"]),
            "availability_lag_days": self.availability_lag,
        }
        return x_tensor, y_tensor, meta_dict

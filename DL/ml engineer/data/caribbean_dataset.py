"""Caribbean Maritime Spatio-Temporal Dataset & Topology Builder for Milestone 3.3.

Extracts the 51 canonical Caribbean ports, builds empirical adjacency matrices,
and provides date-snapshot DataLoaders for spatio-temporal message passing.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from .storage import MemmapStorageReader


def load_caribbean_topology(
    edges_parquet_path: str,
    ablation: str = "none",
    seed: int = 42,
) -> Dict[str, Any]:
    """Load and process the Caribbean maritime shipping topology across 51 canonical ports.
    
    Args:
        edges_parquet_path: Path to caribbean_graph_edges.parquet.
        ablation: 'none' (actual edges), 'shuffled' (permuted topology), or 'no_edges' (identity).
        seed: Random seed for topology shuffling.
        
    Returns:
        Dictionary containing ports list, mappings, and adjacency matrices.
    """
    edges_df = pq.read_table(edges_parquet_path).to_pandas()
    inter = edges_df[edges_df["is_inter_canonical_edge"] == 1].copy()

    # Extract 51 canonical ports
    ports = sorted(list(set(inter["source_canonical_id"]).union(set(inter["target_canonical_id"]))))
    num_ports = len(ports)
    port2idx = {p: i for i, p in enumerate(ports)}
    idx2port = {i: p for i, p in enumerate(ports)}

    # Aggregate directed voyage counts
    agg = inter.groupby(["source_canonical_id", "target_canonical_id"])["voyages_total"].sum().reset_index()

    # Empirical Adjacency Matrix
    A_raw = np.zeros((num_ports, num_ports), dtype=np.float32)
    for _, r in agg.iterrows():
        u = port2idx[r["source_canonical_id"]]
        v = port2idx[r["target_canonical_id"]]
        A_raw[u, v] = float(r["voyages_total"])

    # Handle Ablations
    if ablation == "shuffled":
        rng = np.random.RandomState(seed)
        # Extract all non-diagonal coordinates and permute
        non_diag_u, non_diag_v = np.where(~np.eye(num_ports, dtype=bool))
        values = A_raw[non_diag_u, non_diag_v]
        rng.shuffle(values)
        A_active = np.zeros_like(A_raw)
        A_active[non_diag_u, non_diag_v] = values
    elif ablation == "no_edges":
        # Pure identity matrix (no inter-port corridors)
        A_active = np.zeros_like(A_raw)
    else:
        A_active = A_raw.copy()

    # Adjacency with self-loops
    A_self = A_active + np.eye(num_ports, dtype=np.float32)
    adj_mask = (A_self > 0)

    # Degree normalization for GCN: D^(-1/2) (A + I) D^(-1/2)
    deg = A_self.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=(deg > 0))
    deg_inv_sqrt[deg == 0] = 0.0
    D_mat = np.diag(deg_inv_sqrt)
    norm_adj = D_mat @ A_self @ D_mat

    return {
        "ports": ports,
        "port2idx": port2idx,
        "idx2port": idx2port,
        "num_ports": num_ports,
        "num_edges": int(np.count_nonzero(A_active)),
        "adj_weights": torch.from_numpy(A_active).float(),
        "adj_mask": torch.from_numpy(adj_mask).bool(),
        "norm_adj": torch.from_numpy(norm_adj).float(),
    }


class CaribbeanSnapshotDataset(Dataset):
    """Dataset serving daily Caribbean maritime snapshots of all 51 ports simultaneously."""

    def __init__(
        self,
        meta_parquet_path: str,
        storage_dir: str,
        carib_ports: List[str],
        split: str = "train",
        window_len: int = 30,
        availability_lag: int = 2,
    ):
        self.meta_parquet_path = meta_parquet_path
        self.storage_dir = storage_dir
        self.carib_ports = carib_ports
        self.split = split.lower()
        self.window_len = window_len
        self.availability_lag = availability_lag

        # 1. Initialize Storage Reader
        self.sr = MemmapStorageReader(storage_dir)

        # 2. Ingest Metadata for Caribbean Ports
        cols = ["sequence_id", "portwatch_id", "split_tag", "reference_date", "target_portcalls_7d_future_sum"]
        df_meta = pd.read_parquet(meta_parquet_path, columns=cols)
        df_c = df_meta[(df_meta["portwatch_id"].isin(self.carib_ports)) & (df_meta["split_tag"] == self.split)].copy()

        # Fast lookup map: (date, port) -> (sequence_id, target)
        self.lookup = {}
        for _, row in df_c.iterrows():
            self.lookup[(row["reference_date"], row["portwatch_id"])] = (
                str(row["sequence_id"]),
                float(row["target_portcalls_7d_future_sum"]),
            )

        # Precompute date index offsets
        self.start_offsets = self.window_len + self.availability_lag - 1
        num_panel_days = len(self.sr.idx2date)

        # Filter dates to ensure observation window is completely within panel bounds
        all_unique_dates = sorted(df_c["reference_date"].unique().tolist())
        valid_dates = []
        for d in all_unique_dates:
            d_idx = self.sr.date2idx[d]
            s_idx = d_idx - self.start_offsets
            e_idx = d_idx - self.availability_lag
            if s_idx >= 0 and e_idx < num_panel_days:
                valid_dates.append(d)

        self.dates = valid_dates
        self.num_ports = len(self.carib_ports)
        self.storage_indices = [self.sr.port2idx[p] for p in self.carib_ports]

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, List[str], List[str]]:
        date_str = self.dates[idx]
        d_idx = self.sr.date2idx[date_str]
        start_day_idx = d_idx - self.start_offsets

        # Slices: [num_ports=51, window_len=30, num_features=20]
        x_list = []
        y_list = []
        seq_ids = []

        for i, p_name in enumerate(self.carib_ports):
            p_storage_idx = self.storage_indices[i]
            x_seq = self.sr.get_sequence_by_indices(p_storage_idx, start_day_idx, self.window_len)
            x_list.append(x_seq)

            seq_id, y_target = self.lookup[(date_str, p_name)]
            y_list.append(y_target)
            seq_ids.append(seq_id)

        x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float()  # [51, 30, 20]
        y_tensor = torch.tensor(y_list, dtype=torch.float32)            # [51]

        return x_tensor, y_tensor, date_str, seq_ids, self.carib_ports


def caribbean_collate_fn(batch):
    """Collate multiple daily snapshots into a batch [B, 51, 30, 20]."""
    x_batch = torch.stack([item[0] for item in batch], dim=0)  # [B, 51, 30, 20]
    y_batch = torch.stack([item[1] for item in batch], dim=0)  # [B, 51]
    dates = [item[2] for item in batch]
    seq_ids_batch = [item[3] for item in batch]
    ports = batch[0][4]
    return x_batch, y_batch, dates, seq_ids_batch, ports

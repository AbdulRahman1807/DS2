"""Bounded, Memory-Mapped Feature Storage Architecture for Milestone 3.1.

Implements high-performance, memory-bounded array storage:
1. Stores the 20-feature core as a 3D memory-mapped NumPy float32 array [2065, 2557, 20].
2. Eliminates Parquet deserialization overhead during Dataset.__getitem__.
3. Memory footprint: 402.85 MB on disk; O(1) RAM paging on macOS.
4. Uses atomic staging (*.tmp -> rename) and verification manifests to ensure restartability.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pyarrow.parquet as pq

from .feature_registry import FEATURE_NAMES, NUM_FEATURES
from .preprocessing import TrainingStandardScaler


STORAGE_DAT_FILENAME = "features_core20.dat"
MANIFEST_FILENAME = "storage_manifest.json"
MARKER_FILENAME = "STORAGE_COMPLETE"
PORT2IDX_FILENAME = "port2idx.json"
DATE2IDX_FILENAME = "date2idx.json"


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class StorageManifest:
    """Metadata manifest validating the storage array integrity."""

    def __init__(
        self,
        storage_format: str,
        dat_filename: str,
        shape: List[int],
        dtype: str,
        num_ports: int,
        num_days: int,
        num_features: int,
        feature_names: List[str],
        file_size_bytes: int,
        file_sha256: str,
        created_at: str,
    ):
        self.storage_format = storage_format
        self.dat_filename = dat_filename
        self.shape = shape
        self.dtype = dtype
        self.num_ports = num_ports
        self.num_days = num_days
        self.num_features = num_features
        self.feature_names = feature_names
        self.file_size_bytes = file_size_bytes
        self.file_sha256 = file_sha256
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_format": self.storage_format,
            "dat_filename": self.dat_filename,
            "shape": self.shape,
            "dtype": self.dtype,
            "num_ports": self.num_ports,
            "num_days": self.num_days,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "file_size_bytes": self.file_size_bytes,
            "file_sha256": self.file_sha256,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StorageManifest":
        return cls(**d)


class MemmapStorageBuilder:
    """Builds the memory-mapped storage array from primary panel Parquet in bounded RAM."""

    @staticmethod
    def build(
        panel_parquet_path: str,
        scaler: TrainingStandardScaler,
        output_dir: str,
        batch_size: int = 100000,
        overwrite: bool = False,
    ) -> StorageManifest:
        """Stream primary panel, apply frozen scaler, and write to memory-mapped array.
        
        Args:
            panel_parquet_path: Path to port_daily_unified.parquet.
            scaler: Fitted TrainingStandardScaler instance.
            output_dir: Directory where storage artifacts are persisted.
            batch_size: Number of Parquet rows to read per streaming chunk.
            overwrite: Whether to overwrite existing verified storage artifacts.
            
        Returns:
            StorageManifest describing the finalized storage.
        """
        os.makedirs(output_dir, exist_ok=True)
        final_dat_path = os.path.join(output_dir, STORAGE_DAT_FILENAME)
        manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
        marker_path = os.path.join(output_dir, MARKER_FILENAME)
        port2idx_path = os.path.join(output_dir, PORT2IDX_FILENAME)
        date2idx_path = os.path.join(output_dir, DATE2IDX_FILENAME)

        # Check if already built and verified
        if not overwrite and os.path.exists(marker_path) and os.path.exists(manifest_path) and os.path.exists(final_dat_path):
            with open(manifest_path, "r") as f:
                manifest_dict = json.load(f)
            return StorageManifest.from_dict(manifest_dict)

        # 1. Discover unique ports and dates from Parquet metadata
        table_meta = pq.read_table(panel_parquet_path, columns=["portwatch_id", "date_utc"])
        df_index = table_meta.to_pandas()

        unique_ports = sorted(df_index["portwatch_id"].unique().tolist())
        unique_dates = sorted(df_index["date_utc"].unique().tolist())

        num_ports = len(unique_ports)
        num_days = len(unique_dates)
        expected_rows = num_ports * num_days

        if len(df_index) != expected_rows:
            raise ValueError(
                f"Cartesian irregularity: expected {num_ports} x {num_days} = {expected_rows} rows, "
                f"found {len(df_index)}"
            )

        port2idx = {p: i for i, p in enumerate(unique_ports)}
        date2idx = {d: i for i, d in enumerate(unique_dates)}

        # Save port and date mapping artifacts
        with open(port2idx_path, "w") as f:
            json.dump(port2idx, f, indent=2)
        with open(date2idx_path, "w") as f:
            json.dump(date2idx, f, indent=2)

        del df_index, table_meta

        # 2. Stage temporary memory-mapped binary array
        temp_dat_path = final_dat_path + ".tmp"
        if os.path.exists(temp_dat_path):
            os.remove(temp_dat_path)

        shape = (num_ports, num_days, NUM_FEATURES)
        total_elements = num_ports * num_days * NUM_FEATURES
        expected_bytes = total_elements * 4  # float32 = 4 bytes

        # Create memory-mapped array on disk
        memmap_arr = np.memmap(temp_dat_path, dtype=np.float32, mode="w+", shape=shape)

        # 3. Stream through primary panel in bounded chunks
        feature_cols = list(FEATURE_NAMES)
        read_cols = ["portwatch_id", "date_utc"] + feature_cols

        parquet_file = pq.ParquetFile(panel_parquet_path)
        current_flat_idx = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_cols):
            df_chunk = batch.to_pandas()
            n_chunk = len(df_chunk)

            # Extract 20 raw feature values
            raw_vals = df_chunk[feature_cols].to_numpy(dtype=np.float32)

            # Apply frozen training scaler
            scaled_vals = scaler.transform(raw_vals)

            if not np.all(np.isfinite(scaled_vals)):
                raise ValueError(f"Encountered non-finite values during streaming scaling at row {current_flat_idx}")

            # Calculate slice bounds in flat view
            # Flat row index = p * num_days + d
            flat_start = current_flat_idx
            flat_end = current_flat_idx + n_chunk

            # Write directly to memory-mapped flat buffer
            memmap_arr.reshape(-1, NUM_FEATURES)[flat_start:flat_end] = scaled_vals
            current_flat_idx = flat_end

        if current_flat_idx != expected_rows:
            raise ValueError(f"Streaming write incomplete: processed {current_flat_idx} of {expected_rows} rows")

        # Flush buffer to ensure all pages are committed to disk
        memmap_arr.flush()
        del memmap_arr

        # Verify file size matches exact calculation
        actual_bytes = os.path.getsize(temp_dat_path)
        if actual_bytes != expected_bytes:
            raise IOError(f"Byte mismatch on staged memmap: expected {expected_bytes}, got {actual_bytes}")

        # Compute hash
        sha256_hash = compute_file_sha256(temp_dat_path)

        # 4. Atomic rename and manifest generation
        os.replace(temp_dat_path, final_dat_path)

        manifest = StorageManifest(
            storage_format="numpy_memmap",
            dat_filename=STORAGE_DAT_FILENAME,
            shape=list(shape),
            dtype="float32",
            num_ports=num_ports,
            num_days=num_days,
            num_features=NUM_FEATURES,
            feature_names=feature_cols,
            file_size_bytes=expected_bytes,
            file_sha256=sha256_hash,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        with open(marker_path, "w") as f:
            f.write(f"PIPELINE_COMPLETE\ntimestamp={manifest.created_at}\nsha256={sha256_hash}\n")

        return manifest


class MemmapStorageReader:
    """Zero-copy memory-mapped storage reader for high-throughput sequence retrieval."""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        manifest_path = os.path.join(storage_dir, MANIFEST_FILENAME)
        marker_path = os.path.join(storage_dir, MARKER_FILENAME)
        dat_path = os.path.join(storage_dir, STORAGE_DAT_FILENAME)
        port2idx_path = os.path.join(storage_dir, PORT2IDX_FILENAME)
        date2idx_path = os.path.join(storage_dir, DATE2IDX_FILENAME)

        if not os.path.exists(marker_path):
            raise FileNotFoundError(f"Storage completion marker missing in {storage_dir}. Run builder first.")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Storage manifest missing in {storage_dir}.")
        if not os.path.exists(dat_path):
            raise FileNotFoundError(f"Storage binary missing: {dat_path}")

        with open(manifest_path, "r") as f:
            self.manifest = StorageManifest.from_dict(json.load(f))

        with open(port2idx_path, "r") as f:
            self.port2idx: Dict[str, int] = json.load(f)
        with open(date2idx_path, "r") as f:
            self.date2idx: Dict[str, int] = json.load(f)

        # Invert mappings for reverse lookup if needed
        self.idx2port: List[str] = sorted(self.port2idx.keys(), key=lambda p: self.port2idx[p])
        self.idx2date: List[str] = sorted(self.date2idx.keys(), key=lambda d: self.date2idx[d])

        shape = tuple(self.manifest.shape)
        # Open in read-only mode ('r') for memory-mapped access
        self.array: np.memmap = np.memmap(dat_path, dtype=np.float32, mode="r", shape=shape)

    def get_sequence_by_indices(self, port_idx: int, start_day_idx: int, window_len: int = 30) -> np.ndarray:
        """Fast slice by integer indices [window_len, 20]."""
        end_day_idx = start_day_idx + window_len
        # Slice memory-mapped buffer directly (returns copy to ensure safe PyTorch tensor conversion)
        return np.array(self.array[port_idx, start_day_idx:end_day_idx, :], copy=True)

    def get_sequence_by_keys(self, portwatch_id: str, start_date: str, window_len: int = 30) -> np.ndarray:
        """Retrieve sequence slice by portwatch_id and start date string."""
        if portwatch_id not in self.port2idx:
            raise KeyError(f"Unknown portwatch_id: '{portwatch_id}'")
        if start_date not in self.date2idx:
            raise KeyError(f"Unknown observation start_date: '{start_date}'")

        port_idx = self.port2idx[portwatch_id]
        start_day_idx = self.date2idx[start_date]
        return self.get_sequence_by_indices(port_idx, start_day_idx, window_len)

    def close(self):
        """Close memory-mapped view."""
        if hasattr(self, "array"):
            del self.array

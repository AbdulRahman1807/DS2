"""Pipeline Build Runner for Milestone 3.1.

Executes end-to-end building of preprocessing artifacts and memory-mapped storage:
1. Feature contract validation against port_daily_unified.parquet.
2. TRAIN-only scaler fitting (date_utc <= 2023-12-31).
3. Streaming memory-mapped array construction [2065, 2557, 20] float32.
4. Atomic staging, verification, and manifest generation.
"""

import os
import sys
import time
import yaml
import psutil
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from data.feature_registry import validate_feature_registry, FEATURE_NAMES, NUM_FEATURES
from data.preprocessing import TrainingStandardScaler
from data.storage import MemmapStorageBuilder
import pyarrow.parquet as pq


def main():
    t_start = time.time()
    process = psutil.Process(os.getpid())
    init_mem_mb = process.memory_info().rss / (1024 * 1024)

    config_path = SCRIPT_DIR / "configs/sequence_pipeline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    panel_path = PROJECT_ROOT / cfg["paths"]["panel_parquet"]
    meta_path = PROJECT_ROOT / cfg["paths"]["sequence_metadata_parquet"]
    artifacts_dir = SCRIPT_DIR / "artifacts/preprocessing"
    storage_dir = artifacts_dir / "storage"
    scaler_path = artifacts_dir / "scaler_params.json"

    print("=" * 80)
    print("MILESTONE 3.1: BUILDING AVAILABILITY-AWARE SEQUENCE DATA PIPELINE")
    print(f"Panel Parquet:    {panel_path}")
    print(f"Sequence Metadata:{meta_path}")
    print(f"Artifacts Dir:    {artifacts_dir}")
    print(f"Initial Memory:   {init_mem_mb:.2f} MB")
    print("=" * 80)

    # 1. Validate Feature Registry
    print("\n[Step 1/4] Validating 20-Feature Universal Operational Core...")
    panel_file = pq.ParquetFile(str(panel_path))
    schema_cols = panel_file.schema.names
    validation_res = validate_feature_registry(schema_cols)
    print(f"  Feature Registry Status: {validation_res['status']}")
    print(f"  Features Registered:     {validation_res['num_features']}")
    for grp, cols in validation_res["feature_groups"].items():
        print(f"    - {grp} ({len(cols)}): {cols}")

    # 2. Fit Preprocessing Scaler (TRAIN ONLY)
    cutoff_date = cfg["preprocessing"]["fit_cutoff_date"]
    batch_size = cfg["preprocessing"]["streaming_batch_size"]
    print(f"\n[Step 2/4] Fitting TrainingStandardScaler strictly on TRAIN observations (<= {cutoff_date})...")
    scaler = TrainingStandardScaler(cutoff_date=cutoff_date)
    scaler_params = scaler.fit_from_parquet(str(panel_path), batch_size=batch_size)
    scaler.save(str(scaler_path))
    print(f"  Training rows fit:   {scaler_params.total_training_rows:,}")
    print(f"  Scaler parameters saved -> {scaler_path}")
    mem_after_fit = process.memory_info().rss / (1024 * 1024)
    print(f"  Current Memory:      {mem_after_fit:.2f} MB")

    # 3. Build Memory-Mapped Array Storage
    print("\n[Step 3/4] Building Memory-Mapped Array Storage (features_core20.dat)...")
    os.makedirs(storage_dir, exist_ok=True)
    manifest = MemmapStorageBuilder.build(
        panel_parquet_path=str(panel_path),
        scaler=scaler,
        output_dir=str(storage_dir),
        batch_size=batch_size,
        overwrite=True,
    )
    print(f"  Storage Format:      {manifest.storage_format}")
    print(f"  Array Shape:         {manifest.shape}")
    print(f"  Data Type:           {manifest.dtype}")
    print(f"  File Size:           {manifest.file_size_bytes / (1024*1024):.2f} MB ({manifest.file_size_bytes:,} bytes)")
    print(f"  SHA-256 Checksum:    {manifest.file_sha256}")
    mem_after_store = process.memory_info().rss / (1024 * 1024)
    print(f"  Current Memory:      {mem_after_store:.2f} MB")

    # 4. Summary & Verification
    elapsed = time.time() - t_start
    peak_mem_mb = process.memory_info().rss / (1024 * 1024)
    print("\n[Step 4/4] Pipeline Build Verification Complete!")
    print("=" * 80)
    print(f"Total Time Elapsed:    {elapsed:.2f} seconds")
    print(f"Peak Process Memory:   {peak_mem_mb:.2f} MB")
    print(f"Artifacts Directory:   {artifacts_dir}")
    print(f"Storage Complete Flag: {storage_dir / 'STORAGE_COMPLETE'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Mendeley Container Dwell Event Log Ingestion Script.

Responsibilities (Data Engineer):
- Ingest raw Excel event log ('Dataset of Dwelling Time Process Fix.xlsx')
- Handle mixed-type columns (e.g. YARD_BLOCK having both ints and strings like 'N')
- Convert and optimize storage to Parquet for high-speed columnar access
- Standardize column names and format datetimes
"""

from pathlib import Path
import time
import pandas as pd


def ingest_mendeley_excel(
    excel_path: str = "Dataset of Dwelling Time Process Fix.xlsx",
    output_parquet: str = "data/raw/mendeley_event_log.parquet",
) -> Path:
    excel_file = Path(excel_path)
    if not excel_file.exists():
        raise FileNotFoundError(f"Source file {excel_file} not found in project root.")

    out_file = Path(output_parquet)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {excel_file} (63.8 MB Excel format, ~952,000 rows)...")
    start_t = time.time()
    df = pd.read_excel(excel_file, engine="openpyxl")
    elapsed = time.time() - start_t
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns in {elapsed:.1f}s.")

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]

    # Resolve mixed type columns that choke PyArrow
    # (e.g. YARD_BLOCK has both string letters and numeric block IDs)
    object_cols = ["YARD_BLOCK", "YARD_SLOT", "JOB_DEL_DOCTYPE", "Duration", "CTR_TYPE", "Role", "case_id", "activity"]
    for col in object_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"None": None, "nan": None, "<NA>": None})

    # Numeric conversions with coercion
    if "CTR_SIZE" in df.columns:
        df["CTR_SIZE"] = pd.to_numeric(df["CTR_SIZE"], errors="coerce")
    if "GROSS" in df.columns:
        df["GROSS"] = pd.to_numeric(df["GROSS"], errors="coerce")

    # Timestamps
    if "Start Timestamp" in df.columns:
        df["Start Timestamp"] = pd.to_datetime(df["Start Timestamp"], errors="coerce")
    if "End Timestamp" in df.columns:
        df["End Timestamp"] = pd.to_datetime(df["End Timestamp"], errors="coerce")

    # Save to Parquet
    print(f"Saving optimized columnar Parquet to {out_file}...")
    df.to_parquet(out_file, index=False, engine="pyarrow")
    parquet_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Successfully created {out_file} ({parquet_size_mb:.2f} MB).")

    return out_file


if __name__ == "__main__":
    ingest_mendeley_excel()

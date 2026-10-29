"""Leak-Free Temporal and Container-Grouped Splitter.

Assigned Role: Data Engineer

Responsibilities:
- Isolate sparse burn-in period (< 2021-06-10) from the dense evaluation window
- Partition dense call dates chronologically (70% Train, 15% Val, 15% Test) to respect time's arrow
- Group by container ID to guarantee zero container leakage between train and test
- Export partitioned datasets to data/processed/ (train, val, test, burn_in)
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class TimeAndGroupSplitter:
    def __init__(
        self,
        sparse_cutoff_date: str = "2021-06-10",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.sparse_cutoff_date = sparse_cutoff_date
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Partitions the dataset into burn_in, train, val, and test splits with zero container leakage.

        Args:
            df: DataFrame containing 'call_date' and 'container_id'

        Returns:
            Dictionary with 'burn_in', 'train', 'val', 'test' DataFrames
        """
        df = df.copy()
        if "call_date" not in df.columns or "container_id" not in df.columns:
            raise KeyError("DataFrame must contain 'call_date' and 'container_id' columns.")

        # 1. Isolate sparse burn-in period (< 2021-06-10)
        burn_in_mask = df["call_date"] < self.sparse_cutoff_date
        burn_in_df = df[burn_in_mask].copy()
        dense_df = df[~burn_in_mask].copy()

        # 2. Chronological date partition on dense window (104 unique call dates)
        unique_dates = sorted(dense_df["call_date"].dropna().unique())
        n_dates = len(unique_dates)

        train_cutoff_idx = int(n_dates * self.train_ratio)
        val_cutoff_idx = int(n_dates * (self.train_ratio + self.val_ratio))

        train_dates = set(unique_dates[:train_cutoff_idx])
        val_dates = set(unique_dates[train_cutoff_idx:val_cutoff_idx])
        test_dates = set(unique_dates[val_cutoff_idx:])

        print(f"Call date allocation: Train={len(train_dates)} dates, Val={len(val_dates)} dates, Test={len(test_dates)} dates")

        # Initial date-based splits
        train_mask = dense_df["call_date"].isin(train_dates)
        val_mask = dense_df["call_date"].isin(val_dates)
        test_mask = dense_df["call_date"].isin(test_dates)

        train_candidates = dense_df[train_mask].copy()
        val_candidates = dense_df[val_mask].copy()
        test_candidates = dense_df[test_mask].copy()

        # 3. Prevent Container Leakage without Temporal Inversion:
        # Containers in Val or Test that already appeared in Train must NOT be moved into Train
        # (which would introduce future dates into Train). Instead, their subsequent repeated
        # visits in Val/Test are quarantined into repeated_stays so that:
        # 1) Train date range strictly precedes Val date range (temporal causality).
        # 2) Container IDs across Train, Val, and Test remain 100% mutually disjoint.
        train_containers = set(train_candidates["container_id"].unique())
        val_leak = val_candidates["container_id"].isin(train_containers)
        val_clean = val_candidates[~val_leak].copy()
        val_repeated = val_candidates[val_leak].copy()

        combined_train_val = train_containers | set(val_clean["container_id"].unique())
        test_leak = test_candidates["container_id"].isin(combined_train_val)
        test_clean = test_candidates[~test_leak].copy()
        test_repeated = test_candidates[test_leak].copy()

        repeated_stays_df = pd.concat([val_repeated, test_repeated], ignore_index=True)
        if len(repeated_stays_df) > 0:
            print(f"Quarantined {len(repeated_stays_df)} repeated container stays to preserve both strict chronology and entity disjointness.")

        # Add split_tag
        burn_in_df["split_tag"] = "burn_in"
        train_candidates["split_tag"] = "train"
        val_clean["split_tag"] = "val"
        test_clean["split_tag"] = "test"
        repeated_stays_df["split_tag"] = "repeated_eval"

        # Final audit
        c_train = set(train_candidates["container_id"].unique())
        c_val = set(val_clean["container_id"].unique())
        c_test = set(test_clean["container_id"].unique())
        assert len(c_train & c_val) == 0, "Train and Val containers must be strictly disjoint!"
        assert len(c_train & c_test) == 0, "Train and Test containers must be strictly disjoint!"
        assert len(c_val & c_test) == 0, "Val and Test containers must be strictly disjoint!"
        assert train_candidates["call_date"].max() < val_clean["call_date"].min(), "Train must strictly precede Val temporally!"
        assert val_clean["call_date"].max() < test_clean["call_date"].min(), "Val must strictly precede Test temporally!"

        print(f"Split completed successfully:")
        print(f"  - Burn-in:        {len(burn_in_df):,} stays")
        print(f"  - Train:          {len(train_candidates):,} stays ({len(c_train):,} containers, dates {train_candidates['call_date'].min()} to {train_candidates['call_date'].max()})")
        print(f"  - Val:            {len(val_clean):,} stays ({len(c_val):,} containers, dates {val_clean['call_date'].min()} to {val_clean['call_date'].max()})")
        print(f"  - Test:           {len(test_clean):,} stays ({len(c_test):,} containers, dates {test_clean['call_date'].min()} to {test_clean['call_date'].max()})")
        print(f"  - Repeated Stays: {len(repeated_stays_df):,} stays (quarantined longitudinal visits)")

        return {
            "burn_in": burn_in_df,
            "train": train_candidates,
            "val": val_clean,
            "test": test_clean,
            "repeated_stays": repeated_stays_df,
        }


def run_splitting_pipeline(
    case_attrs_path: str = "data/interim/case_attributes.parquet",
    output_dir: str = "data/processed",
) -> Dict[str, Path]:
    """Partitions case features and dwell labels into train/val/test splits."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cases = pd.read_parquet(case_attrs_path)
    splitter = TimeAndGroupSplitter()
    splits = splitter.split(df_cases)

    paths = {}
    for name, split_df in splits.items():
        p = out_dir / f"{name}.parquet"
        split_df.to_parquet(p, index=False)
        paths[name] = p
        print(f"Saved {p.name} ({len(split_df):,} rows)")

    return paths


if __name__ == "__main__":
    run_splitting_pipeline()

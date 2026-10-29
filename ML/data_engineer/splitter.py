"""Leak-Free Temporal and Container-Grouped Splitter.

Assigned Role: Data Engineer
Deliverables:
- Split dataset chronologically to reflect operational time-arrow (no lookahead leakage)
- Group by container_id so recurring containers never appear in both train and test splits
- Isolate sparse burn-in period (< 2021-06-10) from dense evaluation window (2021-06-10 to 2021-09-30)
"""

from typing import Dict
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
        """Partitions the dataset into burn_in, train, val, and test splits.

        Args:
            df: DataFrame containing 'call_date' and 'container_id'

        Returns:
            Dictionary with keys:
            - 'burn_in': Sparse early calls (< 2021-06-10) used for warm-up history
            - 'train': First ~70% of dense call dates (disjoint containers)
            - 'val': Next ~15% of dense call dates
            - 'test': Final ~15% of dense call dates
        """
        # TODO (Data Engineer):
        # 1. Filter out dates < self.sparse_cutoff_date into 'burn_in'.
        # 2. Sort dense call dates chronologically.
        # 3. Partition call dates into train, val, and test ranges.
        # 4. Check for overlapping container_ids between train and test/val.
        # 5. Quarantine or reassign any overlapping containers to guarantee leak-free testing.
        raise NotImplementedError("To be implemented by Data Engineer")

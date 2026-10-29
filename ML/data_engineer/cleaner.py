"""Data Cleaning & Sanitization Module.

Assigned Role: Data Engineer
Deliverables:
- Parse and standardize timestamps across multi-carrier event streams
- Handle messy entries, uncomputed Excel formulas, and text error values
- Pivot container-level attributes (CTR_SIZE, CTR_TYPE, GROSS, YARD_BLOCK) from event rows
- Quarantine incomplete stays (DISCHARGE without TRUCK_OUT) to protect training labels
"""

from pathlib import Path
from typing import Tuple
import pandas as pd


class DataCleaner:
    def __init__(
        self,
        case_col: str = "case_id",
        activity_col: str = "activity",
        start_ts_col: str = "Start Timestamp",
        end_ts_col: str = "End Timestamp",
        discharge_act: str = "DISCHARGE",
        truck_out_act: str = "TRUCK_OUT",
    ):
        self.case_col = case_col
        self.activity_col = activity_col
        self.start_ts_col = start_ts_col
        self.end_ts_col = end_ts_col
        self.discharge_act = discharge_act
        self.truck_out_act = truck_out_act

    def clean_event_log(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans event log records and separates quarantined invalid/incomplete stays.

        Args:
            raw_df: Raw DataFrame loaded from mendeley_event_log.parquet

        Returns:
            Tuple of (cleaned_df, quarantined_df)
        """
        # TODO (Data Engineer):
        # 1. Standardize text columns (strip whitespace, uppercase activity codes).
        # 2. Extract container_id (first 11 chars) and call_date (last 10 chars YYYY-MM-DD).
        # 3. Parse and coerce datetime columns; log unparseable timestamps.
        # 4. Identify cases with DISCHARGE but missing TRUCK_OUT (1 known incomplete case).
        # 5. Quarantine incomplete stays into quarantined_df.
        # 6. Sort chronologically by case_id and Start Timestamp.
        raise NotImplementedError("To be implemented by Data Engineer")

    def pivot_container_attributes(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Pivots container attributes scattered across events into a single case-level table.

        Attributes to extract:
        - CTR_SIZE & CTR_TYPE from 'BAPLIE'
        - GROSS weight & YARD_BLOCK/SLOT from 'DISCHARGE'
        - Customs flags from 'CUSTOMS_DEL', 'SP_RED_FLAG_PIB', 'HAS_QUARANTINE_FLAG'

        Args:
            events_df: Cleaned event log DataFrame

        Returns:
            DataFrame with one row per case_id containing extracted static attributes
        """
        # TODO (Data Engineer): Aggregate / pivot static attributes per case_id
        raise NotImplementedError("To be implemented by Data Engineer")

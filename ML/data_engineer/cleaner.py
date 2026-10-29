"""Data Cleaning & Sanitization Module.

Assigned Role: Data Engineer

Responsibilities:
- Normalize activity names (strip trailing whitespace like 'TRUCK_IN ')
- Extract canonical container IDs, carrier prefixes, and vessel call dates
- Parse timestamps and compute clean event durations
- Identify and quarantine incomplete container stays (missing TRUCK_OUT)
- Pivot scattered static attributes (CTR_SIZE, CTR_TYPE, GROSS, YARD_BLOCK, inspection flags)
"""

from pathlib import Path
from typing import Dict, Tuple
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

    def clean_event_log(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans event log records and isolates quarantined stays.

        Returns:
            Tuple of (cleaned_df, quarantined_df)
        """
        print(f"Cleaning event log with {len(df):,} records...")
        df = df.copy()

        # 1. Standardize string fields
        df[self.case_col] = df[self.case_col].astype(str).str.strip()
        df[self.activity_col] = df[self.activity_col].astype(str).str.strip()

        # 2. Extract container identifier, carrier code, and vessel call date
        df["container_id"] = df[self.case_col].apply(lambda x: x[:11] if len(x) >= 11 else x)
        df["carrier_prefix"] = df["container_id"].apply(lambda x: x[:3] if len(x) >= 3 else "UNK")
        df["call_date"] = df[self.case_col].apply(
            lambda x: x[-10:] if len(x) >= 21 and "-" in x[-10:] else None
        )

        # 3. Ensure timestamps are parsed and valid
        df[self.start_ts_col] = pd.to_datetime(df[self.start_ts_col], errors="coerce")
        df[self.end_ts_col] = pd.to_datetime(df[self.end_ts_col], errors="coerce")

        # Compute clean event duration in minutes
        df["event_duration_minutes"] = (
            (df[self.end_ts_col] - df[self.start_ts_col]).dt.total_seconds() / 60.0
        ).round(2)

        # 4. Quarantine Incomplete Stays (DISCHARGE present but missing TRUCK_OUT)
        discharge_cases = set(df[df[self.activity_col] == self.discharge_act][self.case_col].unique())
        truck_out_cases = set(df[df[self.activity_col] == self.truck_out_act][self.case_col].unique())
        incomplete_cases = discharge_cases - truck_out_cases

        quarantine_mask = df[self.case_col].isin(incomplete_cases)
        quarantined_df = df[quarantine_mask].copy()
        cleaned_df = df[~quarantine_mask].copy()

        if len(quarantined_df) > 0:
            quarantined_df["quarantine_reason"] = "MISSING_TERMINAL_EVENT: TRUCK_OUT"
            quarantined_df["quarantined_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        else:
            quarantined_df["quarantine_reason"] = pd.Series(dtype=str)
            quarantined_df["quarantined_at"] = pd.Series(dtype=str)

        if incomplete_cases:
            print(f"Quarantined {len(incomplete_cases)} incomplete case(s) ({len(quarantined_df)} events): {list(incomplete_cases)}")

        # 5. Sort chronologically by case, start time, end time, and activity for deterministic order
        cleaned_df = cleaned_df.sort_values(
            by=[self.case_col, self.start_ts_col, self.end_ts_col, self.activity_col],
            na_position="last"
        ).reset_index(drop=True)

        print(f"Clean event log ready: {len(cleaned_df):,} rows across {cleaned_df[self.case_col].nunique():,} cases.")
        return cleaned_df, quarantined_df

    def pivot_case_attributes(self, cleaned_events_df: pd.DataFrame) -> pd.DataFrame:
        """Pivots static container attributes scattered across events into a single case table."""
        print("Pivoting case-level static attributes...")
        df = cleaned_events_df

        # Base case list with extracted container_id and call_date
        cases = df[[self.case_col, "container_id", "carrier_prefix", "call_date"]].drop_duplicates(subset=[self.case_col])

        # CTR_SIZE and CTR_TYPE (typically on BAPLIE)
        size_type = (
            df[df["CTR_SIZE"].notna()][[self.case_col, "CTR_SIZE", "CTR_TYPE"]]
            .drop_duplicates(subset=[self.case_col])
        )

        # GROSS weight (typically on DISCHARGE)
        gross = (
            df[df["GROSS"].notna()][[self.case_col, "GROSS"]]
            .drop_duplicates(subset=[self.case_col])
        )

        # YARD_BLOCK and YARD_SLOT
        yard = (
            df[df["YARD_BLOCK"].notna()][[self.case_col, "YARD_BLOCK", "YARD_SLOT"]]
            .drop_duplicates(subset=[self.case_col])
        )

        # Document type
        doc_type = (
            df[df["JOB_DEL_DOCTYPE"].notna()][[self.case_col, "JOB_DEL_DOCTYPE"]]
            .drop_duplicates(subset=[self.case_col])
        )

        # Inspection and event indicators
        event_counts = df.groupby(self.case_col)[self.activity_col].count().rename("total_events_in_stay")
        has_quarantine = (
            df[df[self.activity_col] == "HAS_QUARANTINE_FLAG"]
            .groupby(self.case_col)[self.activity_col]
            .count()
            .rename("has_quarantine_flag")
            .clip(upper=1)
        )
        has_red_flag = (
            df[df[self.activity_col] == "SP_RED_FLAG_PIB"]
            .groupby(self.case_col)[self.activity_col]
            .count()
            .rename("has_red_flag_pib")
            .clip(upper=1)
        )
        has_behandle = (
            df[df[self.activity_col].str.contains("BEHANDLE", na=False)]
            .groupby(self.case_col)[self.activity_col]
            .count()
            .rename("has_behandle_inspection")
            .clip(upper=1)
        )

        # Merge all attributes
        pivoted = cases.merge(size_type, on=self.case_col, how="left")
        pivoted = pivoted.merge(gross, on=self.case_col, how="left")
        pivoted = pivoted.merge(yard, on=self.case_col, how="left")
        pivoted = pivoted.merge(doc_type, on=self.case_col, how="left")
        pivoted = pivoted.merge(event_counts, on=self.case_col, how="left")
        pivoted = pivoted.merge(has_quarantine, on=self.case_col, how="left")
        pivoted = pivoted.merge(has_red_flag, on=self.case_col, how="left")
        pivoted = pivoted.merge(has_behandle, on=self.case_col, how="left")

        # Fill flags
        flag_cols = ["has_quarantine_flag", "has_red_flag_pib", "has_behandle_inspection"]
        pivoted[flag_cols] = pivoted[flag_cols].fillna(0).astype(int)
        pivoted["total_events_in_stay"] = pivoted["total_events_in_stay"].fillna(1).astype(int)

        print(f"Pivoted {len(pivoted):,} case records with {len(pivoted.columns)} attributes.")
        return pivoted


def run_cleaner_pipeline(
    raw_parquet_path: str = "data/raw/mendeley_event_log.parquet",
    output_dir: str = "data/interim",
) -> Tuple[Path, Path, Path]:
    """Executes end-to-end cleaning and saves interim artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_parquet(raw_parquet_path)
    cleaner = DataCleaner()

    cleaned_events, quarantined = cleaner.clean_event_log(df_raw)
    case_attrs = cleaner.pivot_case_attributes(cleaned_events)

    p_clean = out_dir / "cleaned_event_log.parquet"
    p_quarantine = out_dir / "quarantined_stays.parquet"
    p_attrs = out_dir / "case_attributes.parquet"

    cleaned_events.to_parquet(p_clean, index=False)
    quarantined.to_parquet(p_quarantine, index=False)
    case_attrs.to_parquet(p_attrs, index=False)

    print(f"Interim outputs saved to {out_dir}:")
    print(f"  - {p_clean.name} ({p_clean.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - {p_attrs.name} ({p_attrs.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - {p_quarantine.name} ({p_quarantine.stat().st_size / 1024:.2f} KB)")

    return p_clean, p_attrs, p_quarantine


if __name__ == "__main__":
    run_cleaner_pipeline()

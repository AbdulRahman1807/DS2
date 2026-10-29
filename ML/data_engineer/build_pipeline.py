"""End-to-End Data Engineering Pipeline.

Assigned Role: Data Engineer

Executes the complete data transformation lifecycle:
1. Ingests raw event logs and PortWatch disruption feeds
2. Sanitizes events, extracts identifiers, and quarantines incomplete stays
3. Computes the Dwell Clock and generates 3-tier bottleneck triage labels
4. Calculates strictly lagged 7-day trailing historical delay rates and throughput
5. Injects PortWatch weather disruption severity features
6. Partitions into leak-free, container-grouped train, validation, and test datasets
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from ML.data_engineer.cleaner import DataCleaner
from ML.data_engineer.dwell_clock import DwellClockEngine
from ML.data_engineer.rolling_metrics import RollingMetricsEngine
from ML.data_engineer.splitter import TimeAndGroupSplitter


def extract_disruption_features(call_dates: pd.Series, disruptions_path: Path) -> pd.DataFrame:
    """Maps call dates to PortWatch active disruption events to compute weather severity."""
    if not disruptions_path.exists():
        print(f"Warning: Disruptions file not found at {disruptions_path}. Using nominal weather.")
        return pd.DataFrame({"call_date": call_dates.unique(), "weather_severity_index": 0, "has_active_disruption": 0})

    disrupt_df = pd.read_parquet(disruptions_path)
    disrupt_df["start_dt"] = pd.to_datetime(disrupt_df["fromdate"], unit="ms", errors="coerce").dt.date
    disrupt_df["end_dt"] = pd.to_datetime(disrupt_df["todate"], unit="ms", errors="coerce").dt.date

    unique_dates = pd.to_datetime(call_dates.dropna().unique()).date
    weather_records = []

    for d in unique_dates:
        # Check active disruptions on this date
        active = disrupt_df[(disrupt_df["start_dt"] <= d) & (disrupt_df["end_dt"] >= d)]
        if len(active) > 0:
            levels = active["alertlevel"].astype(str).str.upper().tolist()
            if "RED" in levels:
                severity = 10
            elif "ORANGE" in levels:
                severity = 5
            else:
                severity = 3
            has_disrupt = 1
        else:
            severity = 0
            has_disrupt = 0

        weather_records.append({
            "call_date": d.strftime("%Y-%m-%d"),
            "weather_severity_index": severity,
            "has_active_disruption": has_disrupt,
        })

    return pd.DataFrame(weather_records)


def run_full_pipeline(
    raw_event_log_path: str = "data/raw/mendeley_event_log.parquet",
    disruptions_path: str = "data/raw/portwatch_disruptions.parquet",
    output_processed_dir: str = "data/processed",
) -> Dict[str, Path]:
    """Runs the complete Data Engineer pipeline from raw event log to final processed feature matrices."""
    print("=" * 70)
    print("  STAGE 01 DATA ENGINEERING PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Clean and pivot
    print("\n[Step 1/5] Ingesting and sanitizing raw event log...")
    raw_df = pd.read_parquet(raw_event_log_path)
    cleaner = DataCleaner()
    cleaned_events, quarantined = cleaner.clean_event_log(raw_df)
    case_attrs = cleaner.pivot_case_attributes(cleaned_events)

    # 2. Dwell Clock & Ground-Truth Labels
    print("\n[Step 2/5] Computing Dwell Clock and fitting operational percentiles...")
    dwell_engine = DwellClockEngine()
    dwell_df = dwell_engine.calculate_dwell_durations(cleaned_events)
    dwell_engine.fit_thresholds(dwell_df)
    labeled_dwell = dwell_engine.assign_bottleneck_labels(dwell_df)

    # Merge labels and dwell hours onto case attributes
    dwell_cols = [
        "case_id",
        "dwell_hours",
        "dwell_days",
        "discharge_start_ts",
        "truck_out_end_ts",
        "discharge_hour",
        "discharge_day_of_week",
        "discharge_month",
        "is_weekend",
        "target_class",
        "target_class_name",
    ]
    cases_labeled = case_attrs.merge(labeled_dwell[dwell_cols], on="case_id", how="inner")

    # 3. Rolling Metrics (strictly lagged 7-day historical delay rate & throughput)
    print("\n[Step 3/5] Computing strictly lagged 7-day trailing metrics...")
    rolling_engine = RollingMetricsEngine(window_days=7)
    daily_metrics = rolling_engine.compute_trailing_port_delay_rates(cases_labeled)
    cases_with_rolling = rolling_engine.merge_rolling_features(cases_labeled, daily_metrics)

    # 4. Weather & Disruption Features from IMF PortWatch
    print("\n[Step 4/5] Merging IMF PortWatch disruption & weather features...")
    weather_df = extract_disruption_features(cases_with_rolling["call_date"], Path(disruptions_path))
    full_feature_matrix = cases_with_rolling.merge(weather_df, on="call_date", how="left")
    full_feature_matrix["weather_severity_index"] = full_feature_matrix["weather_severity_index"].fillna(0).astype(int)
    full_feature_matrix["has_active_disruption"] = full_feature_matrix["has_active_disruption"].fillna(0).astype(int)

    # 5. Leak-Free Temporal & Container-Grouped Splitting
    print("\n[Step 5/5] Partitioning feature matrix into leak-free splits...")
    splitter = TimeAndGroupSplitter()
    splits = splitter.split(full_feature_matrix)

    # Save final artifacts to data/processed/
    proc_dir = Path(output_processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    print("\nFinal Processed Feature Matrices Saved:")
    for split_name, split_df in splits.items():
        out_p = proc_dir / f"{split_name}.parquet"
        split_df.to_parquet(out_p, index=False)
        saved_paths[split_name] = out_p
        size_mb = out_p.stat().st_size / (1024 * 1024)
        print(f"  • {out_p.name:<20}: {len(split_df):>6,} stays ({len(split_df.columns):>2} features, {size_mb:.2f} MB)")

    print("\nData Engineer workflow completed successfully!")
    print("=" * 70)
    return saved_paths


if __name__ == "__main__":
    run_full_pipeline()

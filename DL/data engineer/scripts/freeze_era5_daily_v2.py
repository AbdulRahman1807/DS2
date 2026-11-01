
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/processed/era5/era5_port_daily.parquet"
OUTPUT_DIR = ROOT / "data/processed/era5"

OUTPUT = OUTPUT_DIR / "era5_port_daily_v2.parquet"
METADATA = OUTPUT_DIR / "era5_port_daily_v2_metadata.json"

EXPECTED_ROWS = 12_785
EXPECTED_PORTS = 5
EXPECTED_DAYS = 2_557

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

DIRECTION_EPSILON = 1e-8


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. Check input and output paths
    # -----------------------------------------------------

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE}"
        )

    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError(
            "A v2 artifact already exists. "
            "Refusing to overwrite existing files."
        )

    print("Reading original daily ERA5 dataset...")

    df = pd.read_parquet(SOURCE)

    original_columns = list(df.columns)

    print("Source shape:", df.shape)

    # -----------------------------------------------------
    # 2. Validate required source columns
    # -----------------------------------------------------

    required_columns = [
        "port_id",
        "port_name",
        "date_utc",
        "u10_mean_ms",
        "v10_mean_ms",
        "precipitation_mm",
        "hourly_observations",
    ]

    missing_columns = (
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Normalize dates for validation.
    df["date_utc"] = pd.to_datetime(
        df["date_utc"],
        utc=True,
    )

    # -----------------------------------------------------
    # 3. Derive resultant daily wind features
    # -----------------------------------------------------

    print("\nCalculating resultant daily wind features...")

    u = df["u10_mean_ms"].to_numpy(dtype=float)
    v = df["v10_mean_ms"].to_numpy(dtype=float)

    resultant_speed = np.hypot(u, v)

    df["wind_speed_from_daily_mean_components_ms"] = (
        resultant_speed
    )

    # Meteorological convention:
    # Direction FROM which the wind blows.
    direction = (
        np.degrees(np.arctan2(-u, -v)) % 360
    )

    # Direction is mathematically undefined when
    # the resultant wind vector is effectively zero.
    direction[resultant_speed < DIRECTION_EPSILON] = np.nan

    df["wind_direction_from_daily_mean_components_deg"] = (
        direction
    )

    # -----------------------------------------------------
    # 4. Validate total row count and port coverage
    # -----------------------------------------------------

    print("\nRunning structural validation...")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
        )

    if df["port_id"].nunique() != EXPECTED_PORTS:
        raise ValueError(
            f"Expected {EXPECTED_PORTS} unique ports."
        )

    if df["port_name"].nunique() != EXPECTED_PORTS:
        raise ValueError(
            "Unexpected port-name count."
        )

    # -----------------------------------------------------
    # 5. Validate unique port-day keys
    # -----------------------------------------------------

    duplicate_count = df.duplicated(
        ["port_id", "date_utc"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate port-day keys."
        )

    # -----------------------------------------------------
    # 6. Validate hourly coverage
    # -----------------------------------------------------

    if not (
        df["hourly_observations"] == 24
    ).all():
        raise ValueError(
            "One or more port-days do not contain "
            "exactly 24 hourly observations."
        )

    
    # -----------------------------------------------------
    # 7. Validate date continuity
    # -----------------------------------------------------

    expected_dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
        tz="UTC",
    )

    for port_id, group in df.groupby("port_id"):

        actual = pd.DatetimeIndex(
            pd.to_datetime(
                group["date_utc"],
                utc=True,
            ).sort_values()
        )

        if not actual.equals(expected_dates):
            raise ValueError(
                f"Date continuity failure: {port_id}"
            )

        if len(group) != EXPECTED_DAYS:
            raise ValueError(
                f"Incorrect day count for {port_id}: "
                f"{len(group)}"
            )

    # -----------------------------------------------------
    # 8. Validate numerical values
    # -----------------------------------------------------

    direction_column = (
        "wind_direction_from_daily_mean_components_deg"
    )

    numeric_without_direction = (
        df.drop(columns=[direction_column])
        .select_dtypes(include="number")
    )

    if not np.isfinite(
        numeric_without_direction.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite numeric values detected."
        )

    # Only mathematically undefined wind directions
    # may be null.
    direction_values = (
        df[direction_column].dropna().to_numpy(dtype=float)
    )

    if not np.isfinite(direction_values).all():
        raise ValueError(
            "Non-finite wind directions detected."
        )

    if not (
        (direction_values >= 0) &
        (direction_values < 360)
    ).all():
        raise ValueError(
            "Wind direction outside valid [0, 360) range."
        )

    # -----------------------------------------------------
    # 9. Validate precipitation
    # -----------------------------------------------------

    if (df["precipitation_mm"] < -1e-6).any():
        raise ValueError(
            "Negative precipitation detected."
        )

    # -----------------------------------------------------
    # 10. Validate resultant wind speed
    # -----------------------------------------------------

    if (
        df["wind_speed_from_daily_mean_components_ms"] < 0
    ).any():
        raise ValueError(
            "Negative resultant wind speed detected."
        )

    # -----------------------------------------------------
    # 11. Confirm original columns are unchanged
    # -----------------------------------------------------

    print("\nChecking original feature preservation...")

    original = pd.read_parquet(SOURCE)

    pd.testing.assert_frame_equal(
        df[original_columns].reset_index(drop=True),
        original[original_columns].reset_index(drop=True),
        check_exact=True,
    )

    print("Original columns preserved:", len(original_columns))

    # -----------------------------------------------------
    # 12. Sort and save versioned dataset
    # -----------------------------------------------------

    df = df.sort_values(
        ["port_id", "date_utc"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nWriting frozen ERA5 daily v2...")

    df.to_parquet(
        OUTPUT,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    # -----------------------------------------------------
    # 13. Build provenance metadata
    # -----------------------------------------------------

    metadata = {

        "dataset": "ERA5 daily port environmental features",

        "version": "2.0",

        "status": "FROZEN",

        "source_artifact": str(
            SOURCE.relative_to(ROOT)
        ),

        "output_artifact": str(
            OUTPUT.relative_to(ROOT)
        ),

        "period_utc": {
            "start": START_DATE,
            "end": END_DATE,
        },

        "frequency": "daily",

        "rows": len(df),

        "columns": len(df.columns),

        "ports": int(
            df["port_id"].nunique()
        ),

        "days_per_port": EXPECTED_DAYS,

        "new_features": {

            "wind_speed_from_daily_mean_components_ms": {
                "formula": (
                    "sqrt(mean(u10)^2 + mean(v10)^2)"
                ),
                "unit": "m/s",
                "interpretation": (
                    "Magnitude of the daily mean wind vector."
                ),
            },

            "wind_direction_from_daily_mean_components_deg": {
                "formula": (
                    "degrees(atan2(-mean(u10), "
                    "-mean(v10))) modulo 360"
                ),
                "unit": "degrees",
                "convention": (
                    "Meteorological direction FROM."
                ),
                "zero_vector": (
                    "Undefined when resultant speed is "
                    "below 1e-8 m/s; stored as null."
                ),
            },
        },

        "preserved_features": [
            "wind_speed_mean_ms",
            "wind_speed_max_ms",
            "wind_direction_mean_deg",
        ],

        "original_columns_preserved": (
            original_columns
        ),

        "validation": {

            "row_count": "passed",

            "five_ports": "passed",

            "date_continuity": "passed",

            "24_hours_per_port_day": "passed",

            "unique_port_day_keys": "passed",

            "finite_numeric_values": "passed",

            "nonnegative_precipitation": "passed",

            "valid_wind_direction": "passed",

            "original_columns_unchanged": "passed",
        },

        "limitations": [

            "ERA5 values are reanalysis grid estimates, "
            "not terminal observations.",

            "Wave grid cells are offshore proxies with "
            "documented distances.",

            "Daily features are environmental predictors, "
            "not dwell-time labels.",

            "This dataset covers five US port complexes, "
            "not global ports.",

            "Undefined resultant wind directions are "
            "represented as null.",

        ],

    }

    # Write metadata only after Parquet succeeds.
    METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # 14. Final report
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("ERA5 DAILY V2 FROZEN SUCCESSFULLY")
    print("=" * 55)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Ports:", df["port_id"].nunique())
    print("Days per port:", EXPECTED_DAYS)

    print("\nAdded features:")
    print(
        " - wind_speed_from_daily_mean_components_ms"
    )
    print(
        " - wind_direction_from_daily_mean_components_deg"
    )

    print(
        "Undefined wind directions:",
        int(df[direction_column].isna().sum()),
    )

    print(
        "Original columns preserved:",
        len(original_columns),
    )

    print("\nOutput:", OUTPUT)
    print("Metadata:", METADATA)

    print(
        "Parquet size:",
        round(OUTPUT.stat().st_size / (1024 ** 2), 2),
        "MB",
    )

    print("=" * 55)


if __name__ == "__main__":
    main()

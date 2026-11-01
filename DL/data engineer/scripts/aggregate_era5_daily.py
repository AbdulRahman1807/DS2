
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/era5/era5_port_hourly.parquet"
OUTPUT_DIR = ROOT / "data/processed/era5"
OUTPUT = OUTPUT_DIR / "era5_port_daily.parquet"

EXPECTED_DAYS = 2557
EXPECTED_ROWS = 5 * EXPECTED_DAYS

VARIABLES = [
    "u10", "v10", "t2m", "msl", "tp",
    "swh", "mwd", "mwp",
]


def circular_mean_degrees(series):
    """Circular mean for angles expressed in degrees."""
    radians = np.deg2rad(series.dropna().to_numpy())

    if len(radians) == 0:
        return np.nan

    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()

    if np.hypot(sin_mean, cos_mean) < 1e-8:
        return np.nan

    return np.rad2deg(
        np.arctan2(sin_mean, cos_mean)
    ) % 360


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. "
            "Refusing to overwrite."
        )

    print("Reading hourly ERA5 dataset...")
    df = pd.read_parquet(INPUT)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    if df[VARIABLES].isna().any().any():
        raise ValueError("Hourly source contains missing values.")

    # Verify exactly 24 hourly observations per port-day.
    df["date_utc"] = df["timestamp_utc"].dt.floor("D")

    hourly_counts = (
        df.groupby(["port_id", "date_utc"])
        .size()
    )

    if not (hourly_counts == 24).all():
        raise ValueError(
            "Incomplete port-days detected. "
            f"Counts: {hourly_counts.value_counts().to_dict()}"
        )

    # Derive hourly wind speed and meteorological direction.
    df["wind_speed_ms"] = np.hypot(
        df["u10"], df["v10"]
    )

    df["wind_direction_deg"] = (
        np.rad2deg(
            np.arctan2(-df["u10"], -df["v10"])
        ) % 360
    )

    group_keys = ["port_id", "port_name", "date_utc"]

    print("Aggregating daily features...")

    daily = (
        df.groupby(group_keys, as_index=False)
        .agg(
            u10_mean_ms=("u10", "mean"),
            v10_mean_ms=("v10", "mean"),

            wind_speed_mean_ms=("wind_speed_ms", "mean"),
            wind_speed_max_ms=("wind_speed_ms", "max"),

            t2m_mean_k=("t2m", "mean"),
            t2m_min_k=("t2m", "min"),
            t2m_max_k=("t2m", "max"),

            msl_mean_pa=("msl", "mean"),
            msl_min_pa=("msl", "min"),
            msl_max_pa=("msl", "max"),

            # Hourly accumulated precipitation in metres.
            tp_sum_m=("tp", "sum"),

            swh_mean_m=("swh", "mean"),
            swh_max_m=("swh", "max"),

            mwp_mean_s=("mwp", "mean"),

            hourly_observations=("timestamp_utc", "count"),
        )
    )

    # Convert units to more interpretable values.
    daily["t2m_mean_c"] = daily["t2m_mean_k"] - 273.15
    daily["t2m_min_c"] = daily["t2m_min_k"] - 273.15
    daily["t2m_max_c"] = daily["t2m_max_k"] - 273.15

    daily["msl_mean_hpa"] = daily["msl_mean_pa"] / 100
    daily["msl_min_hpa"] = daily["msl_min_pa"] / 100
    daily["msl_max_hpa"] = daily["msl_max_pa"] / 100

    daily["precipitation_mm"] = daily["tp_sum_m"] * 1000

    # Circular daily direction calculations.
    directions = (
        df.groupby(group_keys)["wind_direction_deg"]
        .apply(circular_mean_degrees)
        .rename("wind_direction_mean_deg")
        .reset_index()
    )

    wave_directions = (
        df.groupby(group_keys)["mwd"]
        .apply(circular_mean_degrees)
        .rename("wave_direction_mean_deg")
        .reset_index()
    )

    daily = daily.merge(
        directions, on=group_keys, validate="one_to_one"
    )

    daily = daily.merge(
        wave_directions, on=group_keys, validate="one_to_one"
    )

    # Preserve spatial provenance from the hourly source.
    spatial_columns = [
        "port_id",
        "surface_grid_latitude",
        "surface_grid_longitude",
        "wave_grid_latitude",
        "wave_grid_longitude",
        "reference_latitude",
        "reference_longitude",
        "wave_distance_km",
    ]

    spatial = df[spatial_columns].drop_duplicates()

    if spatial["port_id"].duplicated().any():
        raise ValueError("Inconsistent spatial metadata per port.")

    daily = daily.merge(
        spatial, on="port_id", validate="many_to_one"
    )

    daily = daily.sort_values(
        ["port_id", "date_utc"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print("\nValidating daily dataset...")

    if len(daily) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, got {len(daily)}"
        )

    if daily.duplicated(["port_id", "date_utc"]).any():
        raise ValueError("Duplicate port-day records detected.")

    if not (daily["hourly_observations"] == 24).all():
        raise ValueError("A daily record does not contain 24 hours.")

    if daily.isna().any().any():
        missing = daily.isna().sum()
        raise ValueError(
            f"Missing values detected:\n{missing[missing > 0]}"
        )

    expected_dates = pd.date_range(
        "2019-01-01",
        "2025-12-31",
        freq="D",
        tz="UTC",
    )

    for port_id, group in daily.groupby("port_id"):
        actual_dates = pd.DatetimeIndex(
            group["date_utc"].sort_values()
        )

        if not actual_dates.equals(expected_dates):
            raise ValueError(
                f"Daily date continuity failed: {port_id}"
            )

    if (daily["precipitation_mm"] < -1e-6).any():
        raise ValueError("Negative daily precipitation detected.")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_parquet(
        OUTPUT,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    print("\nAll validations passed.")
    print(f"Rows: {len(daily):,}")
    print(f"Ports: {daily['port_id'].nunique()}")
    print(f"Columns: {len(daily.columns)}")
    print(f"Date range: {daily['date_utc'].min()} to "
          f"{daily['date_utc'].max()}")
    print(f"Output: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()

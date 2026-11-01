from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "interim" / "era5"

OUTPUT_FILE = OUTPUT_DIR / "era5_port_hourly.parquet"
METADATA_FILE = OUTPUT_DIR / "era5_port_hourly_metadata.json"

START = "2019-01-01T00:00:00"
END = "2025-12-31T23:00:00"

EXPECTED_HOURS = 61_368
EXPECTED_ROWS = 306_840

SURFACE_URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-002/arco/reanalysis_era5_single_levels/"
    "sfc/geoChunked.zarr"
)

WAVE_URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-003/arco/reanalysis_era5_single_levels/"
    "wav/geoChunked.zarr"
)

SURFACE_VARIABLES = ["u10", "v10", "t2m", "msl", "tp"]
WAVE_VARIABLES = ["swh", "mwd", "mwp"]


# Approximate port reference coordinates.
# Surface and wave coordinates are the previously audited
# nearest grid cells, not necessarily the same location.

PORTS = [
    {
        "port_id": "port481",
        "port_name": "Houston",
        "reference_lat": 29.7,
        "reference_lon": -95.0,
        "surface_lat": 29.75,
        "surface_lon": -95.0,
        "wave_lat": 29.5,
        "wave_lon": -94.5,
        "wave_distance_km": 53.2,
    },
    {
        "port_id": "port664",
        "port_name": "Los Angeles-Long Beach",
        "reference_lat": 33.5,
        "reference_lon": -118.2,
        "surface_lat": 33.5,
        "surface_lon": -118.25,
        "wave_lat": 33.5,
        "wave_lon": -118.0,
        "wave_distance_km": 18.5,
    },
    {
        "port_id": "port815",
        "port_name": "New York-New Jersey",
        "reference_lat": 40.7,
        "reference_lon": -74.0,
        "surface_lat": 40.75,
        "surface_lon": -74.0,
        "wave_lat": 40.0,
        "wave_lon": -74.0,
        "wave_distance_km": 77.8,
    },
    {
        "port_id": "port1170",
        "port_name": "Savannah",
        "reference_lat": 32.1,
        "reference_lon": -81.1,
        "surface_lat": 32.0,
        "surface_lon": -81.0,
        "wave_lat": 32.0,
        "wave_lon": -80.5,
        "wave_distance_km": 57.6,
    },
    {
        "port_id": "port1175",
        "port_name": "Seattle",
        "reference_lat": 47.6,
        "reference_lon": -122.4,
        "surface_lat": 47.5,
        "surface_lon": -122.5,
        "wave_lat": 47.5,
        "wave_lon": -122.5,
        "wave_distance_km": 13.4,
    },
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def open_remote_zarr(url: str, token: str) -> xr.Dataset:
    return xr.open_zarr(
        url,
        consolidated=True,
        storage_options={
            "headers": {
                "Authorization": f"Bearer {token}"
            }
        },
    )


def extract_point(
    ds: xr.Dataset,
    variables: list[str],
    latitude: float,
    longitude: float,
) -> pd.DataFrame:

    selected = ds[variables].sel(
        time=slice(START, END)
    ).sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest",
    )

    # Only the selected point and selected time range
    # are computed and transferred.
    selected = selected.compute()

    actual_lat = float(selected.latitude.values)
    actual_lon = float(selected.longitude.values)

    frame = selected.to_dataframe().reset_index()

    frame = frame.rename(
        columns={
            "time": "timestamp_utc",
            "latitude": "selected_latitude",
            "longitude": "selected_longitude",
        }
    )

    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"],
        utc=True,
    )

    return frame, actual_lat, actual_lon


# ---------------------------------------------------------
# Main extraction
# ---------------------------------------------------------

def main():

    token = os.environ.get("CDS_API_KEY")

    if not token:
        raise RuntimeError(
            "CDS_API_KEY is missing. "
            "Configure it in this Terminal session."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Move or rename it before rerunning. "
            "This script will not overwrite existing output."
        )

    print("Opening ERA5 remote stores...")

    surface_ds = open_remote_zarr(SURFACE_URL, token)
    wave_ds = open_remote_zarr(WAVE_URL, token)

    print("Surface store opened.")
    print("Wave store opened.")

    results = []
    selected_coordinates = {}

    for index, port in enumerate(PORTS, start=1):

        print(
            f"\n[{index}/{len(PORTS)}] "
            f"Extracting {port['port_name']}..."
        )

        surface, surface_lat, surface_lon = extract_point(
            surface_ds,
            SURFACE_VARIABLES,
            port["surface_lat"],
            port["surface_lon"],
        )

        wave, wave_lat, wave_lon = extract_point(
            wave_ds,
            WAVE_VARIABLES,
            port["wave_lat"],
            port["wave_lon"],
        )

        surface = surface.rename(
            columns={
                "selected_latitude": "surface_grid_latitude",
                "selected_longitude": "surface_grid_longitude",
            }
        )

        wave = wave.rename(
            columns={
                "selected_latitude": "wave_grid_latitude",
                "selected_longitude": "wave_grid_longitude",
            }
        )

        surface = surface.drop(
            columns=["port", "number", "step"],
            errors="ignore",
        )

        wave = wave.drop(
            columns=["port", "number", "step"],
            errors="ignore",
        )

        merged = surface.merge(
            wave,
            on="timestamp_utc",
            how="inner",
            validate="one_to_one",
        )

        merged["port_id"] = port["port_id"]
        merged["port_name"] = port["port_name"]

        merged["reference_latitude"] = port["reference_lat"]
        merged["reference_longitude"] = port["reference_lon"]

        merged["wave_distance_km"] = port["wave_distance_km"]

        selected_coordinates[port["port_id"]] = {
            "port_name": port["port_name"],
            "reference_coordinates": [
                port["reference_lat"],
                port["reference_lon"],
            ],
            "surface_grid_coordinates": [
                surface_lat,
                surface_lon,
            ],
            "wave_grid_coordinates": [
                wave_lat,
                wave_lon,
            ],
            "wave_distance_km": port["wave_distance_km"],
        }

        if len(merged) != EXPECTED_HOURS:
            raise ValueError(
                f"{port['port_name']}: expected "
                f"{EXPECTED_HOURS} rows, got {len(merged)}"
            )

        print(
            f"  Surface cell: {surface_lat}, {surface_lon}"
        )
        print(
            f"  Wave cell: {wave_lat}, {wave_lon}"
        )
        print(f"  Extracted rows: {len(merged):,}")

        results.append(merged)

    surface_ds.close()
    wave_ds.close()

    print("\nCombining all ports...")

    final_df = pd.concat(results, ignore_index=True)

    final_df = final_df.sort_values(
        ["port_id", "timestamp_utc"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print("\nRunning final validation...")

    expected_timestamps = pd.date_range(
        START,
        END,
        freq="h",
        tz="UTC",
    )

    assert len(final_df) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, "
        f"got {len(final_df)}"
    )

    assert final_df["port_id"].nunique() == 5

    assert final_df["timestamp_utc"].nunique() == EXPECTED_HOURS

    duplicate_count = final_df.duplicated(
        ["port_id", "timestamp_utc"]
    ).sum()

    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate port-hour keys"
    )

    missing_by_column = (
        final_df[
            SURFACE_VARIABLES + WAVE_VARIABLES
        ].isna().sum()
    )

    if missing_by_column.any():
        raise ValueError(
            "Missing values detected:\n"
            f"{missing_by_column}"
        )

    for port_id, group in final_df.groupby("port_id"):

        actual = pd.DatetimeIndex(
            group["timestamp_utc"].sort_values()
        )

        if not actual.equals(expected_timestamps):
            raise ValueError(
                f"Timestamp continuity failed for {port_id}"
            )

    print("  Total rows: PASS")
    print("  Five ports: PASS")
    print("  Timestamp continuity: PASS")
    print("  Duplicate keys: PASS")
    print("  Missing values: PASS")

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    final_df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    metadata = {
        "dataset": "ERA5 hourly port environmental features",
        "source": "ECMWF ERA5 ARCO Zarr",
        "period_utc": {
            "start": START,
            "end": END,
        },
        "frequency": "hourly",
        "expected_hours_per_port": EXPECTED_HOURS,
        "expected_total_rows": EXPECTED_ROWS,
        "actual_total_rows": len(final_df),
        "ports": selected_coordinates,
        "variables": {
            "u10": "m s**-1",
            "v10": "m s**-1",
            "t2m": "K",
            "msl": "Pa",
            "tp": "m",
            "swh": "m",
            "mwd": "degrees true",
            "mwp": "s",
        },
        "important_limitations": [
            "ERA5 values are reanalysis grid estimates, not "
            "ground-truth terminal observations.",
            "Wave cells are offshore proxies and may be "
            "distant from the port reference location.",
            "Surface and wave values may come from different "
            "grid resolutions and locations.",
            "The five-port extraction does not represent "
            "global operational weather coverage.",
        ],
        "source_urls": {
            "surface": SURFACE_URL,
            "wave": WAVE_URL,
        },
        "validation": {
            "row_count_passed": True,
            "timestamp_continuity_passed": True,
            "duplicate_keys": int(duplicate_count),
            "missing_values": {
                key: int(value)
                for key, value in missing_by_column.items()
            },
        },
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nExtraction completed.")
    print(f"Parquet: {OUTPUT_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print(
        f"Parquet size: "
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )


if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd
import xarray as xr

TOKEN = os.environ.get("CDS_API_KEY")
if not TOKEN:
    sys.exit("CDS_API_KEY is not configured.")

URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-002/arco/reanalysis_era5_single_levels/"
    "sfc/geoChunked.zarr"
)

STORAGE = {"headers": {"Authorization": f"Bearer {TOKEN}"}}

PORTS = {
    "Houston": (29.7, -95.0),
    "Los_Angeles_Long_Beach": (33.5, -118.2),
    "New_York_New_Jersey": (40.7, -74.0),
    "Savannah": (32.1, -81.1),
    "Seattle": (47.6, -122.4),
}

VARIABLES = ["u10", "v10", "t2m", "msl", "tp"]

START = "2019-01-01"
END = "2025-12-31T23:00"

print("Opening ERA5 surface store...")

ds = xr.open_zarr(
    URL,
    consolidated=True,
    storage_options=STORAGE,
)

period = ds.sel(time=slice(START, END))

timestamps = pd.DatetimeIndex(period.time.values)
expected = pd.date_range(START, END, freq="h")

print("\n===== TIME VALIDATION =====")
print("Actual timestamps:", len(timestamps))
print("Expected timestamps:", len(expected))
print("Unique:", timestamps.is_unique)
print("Continuous hourly sequence:", timestamps.equals(expected))

for name, (port_lat, port_lon) in PORTS.items():
    print(f"\n===== {name} =====")

    point = period.sel(
        latitude=port_lat,
        longitude=port_lon,
        method="nearest",
    )

    actual_lat = float(point.latitude.values)
    actual_lon = float(point.longitude.values)

    print(f"Selected cell: {actual_lat}, {actual_lon}")

    for variable in VARIABLES:
        series = point[variable]
        valid = int(series.count().compute().item())
        total = series.size

        print(
            f"{variable}: {valid}/{total} valid "
            f"({100 * valid / total:.4f}%)"
        )

print("\nSurface temporal audit complete.")


import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

TOKEN = os.environ.get("CDS_API_KEY")
if not TOKEN:
    sys.exit("CDS_API_KEY is not configured.")

URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-003/arco/reanalysis_era5_single_levels/"
    "wav/geoChunked.zarr"
)

STORAGE = {"headers": {"Authorization": f"Bearer {TOKEN}"}}

# Approximate port reference coordinates
PORTS = {
    "Houston": (29.7, -95.0),
    "Los Angeles_Long_Beach": (33.5, -118.2),
    "New_York_New_Jersey": (40.7, -74.0),
    "Savannah": (32.1, -81.1),
    "Seattle": (47.6, -122.4),
}

START = "2019-01-01"
END = "2025-12-31T23:00"

print("Opening ERA5 wave store...")
ds = xr.open_zarr(
    URL,
    consolidated=True,
    storage_options=STORAGE,
)

# Restrict time before selecting spatial points.
period = ds.sel(time=slice(START, END))

print("\nExpected hourly timestamps:", len(period.time))
print("Actual first timestamp:", period.time.values[0])
print("Actual last timestamp:", period.time.values[-1])

time_values = pd.DatetimeIndex(period.time.values)
expected = pd.date_range(START, END, freq="h")

print("Timestamp count:", len(time_values))
print("Expected count:", len(expected))
print("Unique timestamps:", time_values.is_unique)
print("Continuous hourly sequence:", time_values.equals(expected))

for name, (port_lat, port_lon) in PORTS.items():
    print(f"\n===== {name} =====")

    region = ds.sel(
        time="2019-01-01T12:00",
        method="nearest",
        ).sel(
        latitude=slice(port_lat - 3, port_lat + 3),
        longitude=slice(port_lon - 3, port_lon + 3),
    )

    mask = region["swh"].notnull().compute()

    lat_grid, lon_grid = np.meshgrid(
        region.latitude.values,
        region.longitude.values,
        indexing="ij",
    )

    valid_lats = lat_grid[mask.values]
    valid_lons = lon_grid[mask.values]

    if len(valid_lats) == 0:
        print("No valid ocean cell found.")
        continue

    lat1 = np.radians(port_lat)
    lat2 = np.radians(valid_lats)
    dlat = lat2 - lat1
    dlon = np.radians(valid_lons - port_lon)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    distances = 6371 * 2 * np.arcsin(np.sqrt(a))
    nearest = np.argmin(distances)

    cell_lat = float(valid_lats[nearest])
    cell_lon = float(valid_lons[nearest])

    print(f"Selected cell: {cell_lat}, {cell_lon}")
    print(f"Distance: {distances[nearest]:.1f} km")

    point = period.sel(
        latitude=cell_lat,
        longitude=cell_lon,
    )

    for variable in ["swh", "mwd", "mwp"]:
        series = point[variable]
        valid_count = int(series.count().compute().item())
        total_count = series.size

        print(
            f"{variable}: {valid_count}/{total_count} valid "
            f"({100 * valid_count / total_count:.4f}%)"
        )

print("\nTemporal availability audit complete.")

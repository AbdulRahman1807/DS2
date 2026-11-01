
import os
import sys
import numpy as np
import xarray as xr

TOKEN = os.environ.get("CDS_API_KEY")
if not TOKEN:
    sys.exit("CDS_API_KEY is not configured.")

URL = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-003/arco/reanalysis_era5_single_levels/"
    "wav/geoChunked.zarr"
)

STORAGE = {
    "headers": {"Authorization": f"Bearer {TOKEN}"}
}

# Approximate port-complex reference coordinates.
PORTS = {
    "Houston": (29.7, -95.0),
    "Los Angeles / Long Beach": (33.5, -118.2),
    "New York / New Jersey": (40.7, -74.0),
    "Savannah": (32.1, -81.1),
    "Seattle": (47.6, -122.4),
}

print("Opening ERA5 wave store...")
ds = xr.open_zarr(
    URL,
    consolidated=True,
    storage_options=STORAGE,
)

# Inspect a broader 6-degree-wide region around each port.
timestamp = "2019-01-01T12:00"

for name, (port_lat, port_lon) in PORTS.items():
    print(f"\n===== {name} =====")

    region = ds.sel(
        time=timestamp,
        method="nearest",
    ).sel(
        latitude=slice(port_lat - 3, port_lat + 3),
        longitude=slice(port_lon - 3, port_lon + 3),
    )

    # A valid SWH cell represents an ocean cell at this timestamp.
    valid = region["swh"].notnull().compute()

    lat_grid, lon_grid = np.meshgrid(
        region.latitude.values,
        region.longitude.values,
        indexing="ij",
    )

    lat_values = lat_grid[valid.values]
    lon_values = lon_grid[valid.values]

    if len(lat_values) == 0:
        print("No valid wave cells within 3 degrees.")
        continue

    # Haversine distance in kilometres.
    lat1 = np.radians(port_lat)
    lat2 = np.radians(lat_values)
    dlat = lat2 - lat1
    dlon = np.radians(lon_values - port_lon)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    distances = 6371 * 2 * np.arcsin(np.sqrt(a))

    order = np.argsort(distances)

    print(f"Valid ocean cells: {len(lat_values)}")
    print("Nearest valid cells:")

    for idx in order[:5]:
        print(
            f"  lat={lat_values[idx]:.2f}, "
            f"lon={lon_values[idx]:.2f}, "
            f"distance={distances[idx]:.1f} km"
        )

print("\nOffshore grid audit complete.")

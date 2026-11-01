
import os
import sys
import xarray as xr

TOKEN = os.environ.get("CDS_API_KEY")
if not TOKEN:
    sys.exit("CDS_API_KEY is not configured.")

STORAGE = {
    "headers": {"Authorization": f"Bearer {TOKEN}"}
}

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

PORTS = {
    "Houston": (30.2, 29.2, -95.5, -94.5),
    "Los Angeles / Long Beach": (34.0, 33.0, -118.7, -117.7),
    "New York / New Jersey": (41.2, 40.2, -74.5, -73.5),
    "Savannah": (32.6, 31.6, -81.6, -80.6),
    "Seattle": (48.1, 47.1, -122.9, -121.9),
}

print("Opening remote ERA5 stores...")

surface = xr.open_zarr(
    SURFACE_URL,
    consolidated=True,
    storage_options=STORAGE,
)

wave = xr.open_zarr(
    WAVE_URL,
    consolidated=True,
    storage_options=STORAGE,
)

timestamp = "2019-01-01T12:00"

for port, (north, south, west, east) in PORTS.items():
    print(f"\n===== {port} =====")

    # ERA5 latitude coordinates are ascending.
    s = surface.sel(
        time=timestamp,
        method="nearest",
    ).sel(
        latitude=slice(south, north),
        longitude=slice(west, east),
    )

    w = wave.sel(
        time=timestamp,
        method="nearest",
    ).sel(
        latitude=slice(south, north),
        longitude=slice(west, east),
    )

    print("Surface grid:", s.sizes)
    print("Wave grid:", w.sizes)

    for variable in ["u10", "tp"]:
        valid = int(s[variable].count().compute().item())
        total = s[variable].size
        print(f"{variable}: {valid}/{total} valid cells")

    for variable in ["swh", "mwd", "mwp"]:
        valid = int(w[variable].count().compute().item())
        total = w[variable].size
        print(f"{variable}: {valid}/{total} valid cells")

print("\nGrid availability audit complete.")

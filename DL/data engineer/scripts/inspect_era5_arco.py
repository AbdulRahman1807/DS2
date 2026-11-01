
import os
import sys
import xarray as xr

surface_geo_url = (
    "https://arco.datastores.ecmwf.int/"
    "cadl-arco-geo-002/arco/reanalysis_era5_single_levels/"
    "sfc/geoChunked.zarr"
)

token = os.environ.get("CDS_API_KEY")

if not token:
    sys.exit("CDS_API_KEY is not configured.")

print("Opening remote ERA5 surface Zarr store...")

ds = xr.open_zarr(
    surface_geo_url,
    consolidated=True,
    storage_options={
        "headers": {
            "Authorization": f"Bearer {token}"
        }
    },
)

print("\n===== DATASET SUMMARY =====")
print(ds)

print("\n===== VARIABLES =====")
for name, variable in ds.data_vars.items():
    print(
        f"{name}: dims={variable.dims}, "
        f"units={variable.attrs.get('units', 'unknown')}"
    )

print("\n===== COORDINATES =====")
for name, coordinate in ds.coords.items():
    print(name, coordinate.dims, coordinate.shape)

print("\n===== TIME COVERAGE =====")
if "time" in ds.coords:
    print("First:", ds.time.values[0])
    print("Last:", ds.time.values[-1])
    print("Count:", ds.sizes.get("time"))

print("\nInspection complete. No full dataset downloaded.")
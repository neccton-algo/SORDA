import numpy as np
import xarray as xr

def apply_fillvalue(ds: xr.Dataset, fill_value: float = -999.0) -> xr.Dataset:
    """
    Replace NaNs with a fixed fill value for all variables in an xarray Dataset.
    Sets the NetCDF _FillValue attribute when saving.
    """
    for v in ds.data_vars:
        arr = ds[v]
        filled = arr.where(~np.isnan(arr), fill_value)
        ds[v] = filled
        enc = ds[v].encoding.copy()
        enc.update({"_FillValue": fill_value})
        ds[v].encoding = enc
    return ds

def normalize_dataset(ds: xr.Dataset, time_val, levels_to_keep=30, fill_value=-999.0) -> xr.Dataset:
    """Normalize OAK NetCDF outputs for concatenation."""

    # Rename generic dims
    rename_map = {}
    for d in ds.dims:
        if d == "dim001":
            rename_map[d] = "x"
        elif d == "dim002":
            rename_map[d] = "y"
        elif d == "dim004":
            rename_map[d] = "depth"
        elif d.startswith("dim"):
            rename_map[d] = f"unknown_{d}"
    ds = ds.rename(rename_map).squeeze()

    # Slice depth
    if "depth" in ds.dims and ds.sizes["depth"] > levels_to_keep:
        ds = ds.isel(depth=slice(0, levels_to_keep))

    # Drop bounds-like variables
    for v in list(ds.variables):
        if "bnds" in v.lower():
            ds = ds.drop_vars(v)

    # Ensure nav_lat/nav_lon are not coords
    for v in ("nav_lat", "nav_lon"):
        if v in ds.coords:
            ds = ds.reset_coords(v, drop=False)

    # Harmonize fill values and attrs
    for v in ds.data_vars:
        arr = ds[v]
        # Replace 1e20 with NaN, then with fill_value
        arr = arr.where(arr != 1.0e20, np.nan)
        arr = arr.where(~np.isnan(arr), fill_value)

        # Clean attributes
        new_attrs = {k: val for k, val in arr.attrs.items() if k not in ["missing_value", "_FillValue"]}

        # Set encoding
        arr.encoding = {"_FillValue": fill_value}

        ds[v] = arr
        ds[v].attrs = new_attrs

    # Add time_counter (always at the very end)
    ds = ds.expand_dims("time_counter")
    ds = ds.assign_coords(time_counter=("time_counter", [np.datetime64(time_val, "ns")]))
    return ds
#!/usr/bin/env python3
"""
Portable replacement for twin_driver.sh.

Prepares a single day's working directory (default: ./current_day)
by:
  - copying static fields needed by OAK
  - copying daily obs + ensemble model files from data/dynamic
  - renaming files to match the OAK naming convention
  - replacing/setting fill values (-999) in NetCDFs via xarray
  - generating assim.YYYYMMDD from a template (assim.date) with placeholders:
      <DATETIME>, <DATE>, <DRADATE>

Usage:
  python src/python/prep.py --date 20160201
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
from utils import apply_fillvalue

import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser(description="Prepare DA working dir for one day.")
    p.add_argument("--date", required=True, help="Date in YYYYMMDD (e.g., 20170101)")
    p.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    p.add_argument("--static-dir", default="data/static", help="Path to static fields")
    p.add_argument("--obs-dir", default="data/dynamic/obs", help="Path to obs (2D) files")
    p.add_argument("--model-dir", default="data/dynamic/model", help="Path to model (3D ensemble) files")
    p.add_argument("--workdir", default="current_day", help="Working directory to (re)create")
    p.add_argument("--assim-template", default="assim.date", help="Path to the assim template file")
    p.add_argument("--ensemble-size", type=int, default=20, help="Number of ensemble members (default 20)")
    p.add_argument("--fill-value", type=float, default=-999.0, help="Fill value to write in NetCDF")
    p.add_argument("--datetime-hour", default="11:30:00.00", help="Time to embed in <DATETIME> (HH:MM:SS.xx)")
    return p.parse_args()


def ymd_parts(yyyymmdd: str):
    try:
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    except ValueError:
        raise SystemExit(f"ERROR: invalid --date '{yyyymmdd}', expected YYYYMMDD")
    y = dt.strftime("%Y")
    m = dt.strftime("%m")
    d = dt.strftime("%d")
    return y, m, d, dt


def clean_and_make_workdir(workdir: Path, date_str: str):
    workdir.mkdir(parents=True, exist_ok=True)
    # Remove old day-specific files to avoid cross-day contamination
    for p in workdir.glob(f"*{date_str}*"):
        if p.is_file():
            p.unlink()
    # You can add more cleanup patterns if needed:
    for pat in ["*.tmp", "*.swp", "slurm*", "assim.2016*"]:
        for p in workdir.glob(pat):
            if p.is_file():
                p.unlink()


def copy_static_fields(static_dir: Path, workdir: Path):
    needed = [
        "domain_cfg.nc",
        "domain_part.nc",
        "gridZ.nc",
        "mesh_mask.nc",
        # REF_stds handling below (month-specific if available)
    ]
    for name in needed:
        src = static_dir / name
        if not src.exists():
            print(f"WARNING: static file missing: {src}")
            continue
        shutil.copy2(src, workdir / name)


def copy_ref_stds(static_dir: Path, workdir: Path, year: str, month: str):
    """
    REF_stds may be a single file (REF_stds.nc), or a month-specific file
    (e.g., REF_stds_YYYYMM.nc). Prefer month-specific if present.
    """
    monthly = static_dir / f"REF_stds_{year}{month}.nc"
    generic = static_dir / "REF_stds.nc"
    if monthly.exists():
        shutil.copy2(monthly, workdir / "REF_stds.nc")
    elif generic.exists():
        shutil.copy2(generic, workdir / "REF_stds.nc")
    else:
        print("WARNING: REF_stds not found (neither monthly nor generic).")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def apply_fillvalue(ds: xr.Dataset, fill_value: float = -999.0) -> xr.Dataset:
    """
    Replace NaNs and 1.e20 with fill_value for all variables
    and set NetCDF _FillValue encoding to the required value (-999).
    """
    for v in ds.data_vars:
        arr = ds[v]

        data = arr.data
        # Replace 1e20 with NaN first
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isclose(data, 1.0e20), np.nan, data)
        # Replace NaNs with fill_value
        data = np.where(np.isnan(data), fill_value, data)

        # Clean attrs
        new_attrs = {k: val for k, val in arr.attrs.items()
                     if k not in ["missing_value", "_FillValue"]}

        # Rebuild variable
        arr_clean = xr.DataArray(
            data,
            dims=arr.dims,
            coords=arr.coords,
            attrs=new_attrs
        )
        arr_clean.encoding["_FillValue"] = fill_value

        ds[v] = arr_clean

    return ds

def load_and_write_with_fill(src, dst, fill_value):
    ds = xr.open_dataset(src, decode_cf=False)
    ds = apply_fillvalue(ds, fill_value)
    ds = enforce_dims(ds)
    ds.to_netcdf(dst)
    ds.close()


def prepare_obs(obs_dir: Path, workdir: Path, y: str, m: str, d: str, fill_value: float):
    # Original REF file name: CREF_y${y}m${m}d${d}.nc
    src_name = f"CREF_y{y}m{m}d{d}.nc"
    src = obs_dir / src_name
    if not src.exists():
        raise FileNotFoundError(f"Missing obs file: {src}")

    # Then produce the _refl variant with fill value adjustments
    dst = workdir / f"REF_y{y}m{m}d{d}.nc"
    load_and_write_with_fill(src, dst, fill_value)


def prepare_model_ensemble(model_dir: Path, workdir: Path, y: str, m: str, d: str, n_ens: int, fill_value: float):
    """
    Copy each ensemble file:
      C0XX_yYYYYmMMdDD.nc  ->  C0XX_yYYYYmMMdDD.nc
    Then write a *_refl.nc with fill handling.
    """
    for ii in range(1, n_ens + 1):
        mem = f"{ii:02d}"
        src_name = f"C0{mem}_y{y}m{m}d{d}.nc"
        src = model_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing model file: {src}")

        refl = workdir / f"C0{mem}_y{y}m{m}d{d}.nc"
        load_and_write_with_fill(src, refl, fill_value)

def prepare_ptrc_ensemble(ptrc_dir: Path, workdir: Path, y: str, m: str, d: str,
                          n_ens: int, fill_value: float):
    """
    Copy each ptrc ensemble file:
      PC0XX_yYYYYmMMdDD.nc -> workdir
    Then apply fill handling (_fill.nc) and overwrite the original.
    """
    for ii in range(1, n_ens + 1):
        mem = f"{ii:02d}"
        src_name = f"PC0{mem}_y{y}m{m}d{d}.nc"
        src = ptrc_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing ptrc file: {src}")

        dst = workdir / src_name
        shutil.copy2(src, dst)

        # Process fill values
        tmp = workdir / f"PC0{mem}_y{y}m{m}d{d}_fill.nc"
        load_and_write_with_fill(dst, tmp, fill_value)

        # Replace the original with the _fill version
        tmp.rename(dst)

def enforce_dims(ds: xr.Dataset) -> xr.Dataset:
    coords2d = ('y', 'x')
    coords3d = ('z', 'y', 'x')

    if "RRS_412nm" in ds:
        ds["RRS_412nm"] = (coords2d, ds["RRS_412nm"].values[0])
        ds["RRS_443nm"] = (coords2d, ds["RRS_443nm"].values[0])
        ds["RRS_490nm"] = (coords2d, ds["RRS_490nm"].values[0])
        ds["RRS_510nm"] = (coords2d, ds["RRS_510nm"].values[0])
        ds["RRS_555nm"] = (coords2d, ds["RRS_555nm"].values[0])
        ds["RRS_670nm"] = (coords2d, ds["RRS_670nm"].values[0])

    for var in ["Eu_490nm", "Ed_490nm", "Es_490nm", "PAR"]:
        if var in ds:
            ds[var] = (coords3d, ds[var].values[0])

    if "nav_lon" in ds:
        ds["nav_lon"] = (coords2d, ds["nav_lon"].values)
    if "nav_lat" in ds:
        ds["nav_lat"] = (coords2d, ds["nav_lat"].values)
    if "deptht" in ds:
        ds["deptht"] = ("z", ds["deptht"].values)

    return ds

def write_assim_namelist(template_path: Path, workdir: Path, y: str, m: str, d: str, time_str: str):
    """
    Create 'assim.YYYYMMDD' from 'assim.date' template,
    replacing <DATETIME>, <DATE>, <DRADATE> as in your bash script.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Assimilation template not found: {template_path}")

    with open(template_path, "r") as f:
        txt = f.read()

    datetime_iso = f"{y}-{m}-{d}T{time_str}"
    date_compact = f"{y}{m}{d}"
    dradate = f"y{y}m{m}d{d}"

    txt = txt.replace("<DATETIME>", datetime_iso)
    txt = txt.replace("<DATE>", date_compact)
    txt = txt.replace("<DRADATE>", dradate)

    out_name = f"assim.{date_compact}"
    with open(workdir / out_name, "w") as f:
        f.write(txt)


def main():
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    static_dir = (repo / args.static_dir).resolve()
    obs_dir = (repo / args.obs_dir).resolve()
    model_dir = (repo / args.model_dir).resolve()
    workdir = (repo / args.workdir).resolve()
    tmpl = (repo / args.assim_template).resolve()

    y, m, d, _ = ymd_parts(args.date)

    print(f"[prep] date={args.date}  workdir={workdir}")
    ensure_dir(workdir)
    clean_and_make_workdir(workdir, args.date)

    print("[prep] copying static fields...")
    copy_static_fields(static_dir, workdir)
    copy_ref_stds(static_dir, workdir, y, m)

    print("[prep] preparing observations (2D)...")
    prepare_obs(obs_dir, workdir, y, m, d, args.fill_value)

    print("[prep] preparing model ensemble (3D)...")
    prepare_model_ensemble(model_dir, workdir, y, m, d, args.ensemble_size, args.fill_value)
    
    print("[prep] preparing biogeochemistry ensemble (3D)...")
    prepare_ptrc_ensemble(repo / "data/dynamic/ptrc", workdir, y, m, d,
                      args.ensemble_size, args.fill_value)

    print("[prep] writing assimilation namelist...")
    write_assim_namelist(tmpl, workdir, y, m, d, args.datetime_hour)

    print("[prep] done ✅")


if __name__ == "__main__":
    sys.exit(main())

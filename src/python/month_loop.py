#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import csv
import time
import sys, subprocess
import shutil

from job import run_assimilation
from utils import normalize_dataset

import warnings

# Suppress the annoying xarray duplicate-dimension warning globally in this module
warnings.filterwarnings(
    "ignore",
    message="Duplicate dimension names present",
    category=UserWarning,
)

# Resolve repo root from this file location:
# this file:   .../WP4_demo/src/python/month_loop.py
# parents[0] = .../WP4_demo/src/python
# parents[1] = .../WP4_demo/src
# parents[2] = .../WP4_demo  ← repo root
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SRC_PY    = REPO_ROOT / "src" / "python"

PREP_PY = SRC_PY / "prep.py"
JOB_PY  = SRC_PY / "job.py"

def date_range(start: str, end: str):
    """Yield YYYYMMDD strings from start to end inclusive."""
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    while d0 <= d1:
        yield d0.strftime("%Y%m%d")
        d0 += timedelta(days=1)


def append_or_create(ds_new: xr.Dataset, out_file: str):
    """Append a dataset to an existing NetCDF along time_counter, or create new."""
    out_path = Path(out_file)
    if out_path.exists():
        with xr.open_dataset(out_path) as ds_old:
            ds_old.load()
        ds_combined = xr.concat([ds_old, ds_new], dim="time_counter")
    else:
        ds_combined = ds_new
    ds_combined.to_netcdf(out_path, mode="w")


def month_loop(start: str, end: str,
               forecast_out="forecast_combined.nc",
               analysis_out="analysis_combined.nc",
               increments_out="increments_combined.nc",
               obs_ref="opti_ref_combined.nc",
               ptrc_ref="ptrc_ref_combined.nc",
               runtime_log="runtime_log.csv",
               ensemble_size=20,
               levels_to_keep=30):

    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    keep_files = {"domain_cfg.nc", "domain_part.nc", "gridZ.nc",
                  "mesh_mask.nc", "REF_stds.nc", "assim-gfortran-double-mpi"}

    t0_total = time.perf_counter()

    with open(runtime_log, "w", newline="") as f:
        csv.writer(f).writerow(["date", "elapsed_seconds"])

    for date in date_range(start, end):
        print(f"\n=== {date} ===")
        t0 = time.perf_counter()
        time_val = datetime.strptime(date, "%Y%m%d") + timedelta(hours=12)
        y, m, d = date[:4], date[4:6], date[6:8]

        # 1) Prep inputs
        subprocess.run(
            [
                sys.executable,
                str(PREP_PY),
                "--date", date,
                "--repo-root", str(REPO_ROOT),
            ],
            check=True,
            cwd=REPO_ROOT,
        )

        # 2) Save FORECAST before assimilation
        members_fcst = []
        for i in range(1, ensemble_size + 1):
            mem = f"{i:03d}"
        
            ds_parts = []
            for prefix in ("C", "PC"):
                f = REPO_ROOT / "current_day" / f"{prefix}{mem}_y{y}m{m}d{d}.nc"
                if f.exists():
                    ds = xr.open_dataset(f, decode_times=False)
                    ds = normalize_dataset(ds, time_val, levels_to_keep=levels_to_keep)
                    ds_parts.append(ds)
        
            if not ds_parts:
                # no data for this member, skip
                continue
        
            # Merge C + PC into a single dataset for this ensemble member
            ds_member = xr.merge(ds_parts)
        
            # Add ensemble dimension
            ds_member = ds_member.expand_dims("ensemble").assign_coords(ensemble=[i])
        
            members_fcst.append(ds_member)
        
        if members_fcst:
            ds_fcst_day = xr.concat(members_fcst, dim="ensemble")
            append_or_create(ds_fcst_day, REPO_ROOT / forecast_out)


        # 3) Run assimilation
        run_assimilation(date, workdir=str(REPO_ROOT / "current_day"))

        # 4) Save ANALYSIS after assimilation
        members_an = []
        for i in range(1, ensemble_size + 1):
            mem = f"{i:03d}"
            fpath = REPO_ROOT / "analysis" / f"{date}_Ea_C{mem}.nc"
            if fpath.exists():
                ds = xr.open_dataset(fpath, decode_times=False)
                ds = normalize_dataset(ds, time_val, levels_to_keep=levels_to_keep)
                ds = ds.expand_dims("ensemble").assign_coords(ensemble=[i])
                members_an.append(ds)
        if members_an:
            ds_an_day = xr.concat(members_an, dim="ensemble")
            append_or_create(ds_an_day, REPO_ROOT / analysis_out)

        # 5) Save INCREMENTS
        incr_file = REPO_ROOT / "analysis" / f"{date}_xa_incr.nc"
        if incr_file.exists():
            ds_incr = xr.open_dataset(incr_file, decode_times=False)
            ds_incr = normalize_dataset(ds_incr, time_val, levels_to_keep=levels_to_keep)
            append_or_create(ds_incr, REPO_ROOT / increments_out)

        # 6) Save REF (observations)
        ref_file = REPO_ROOT / "data" / "dynamic" / "obs" / f"PREF_y{y}m{m}d{d}.nc"
        if ref_file.exists():
            ds_ref = xr.open_dataset(ref_file, decode_times=False)
            ds_ref = normalize_dataset(ds_ref, time_val, levels_to_keep=levels_to_keep)
            append_or_create(ds_ref, REPO_ROOT / ptrc_ref)

        # 7) Save REF (observations)
        obs_file = REPO_ROOT / "current_day" / f"REF_y{y}m{m}d{d}.nc"
        if obs_file.exists():
            ds_obs = xr.open_dataset(obs_file, decode_times=False)
            ds_obs = normalize_dataset(ds_obs, time_val, levels_to_keep=levels_to_keep)
            append_or_create(ds_obs, REPO_ROOT / obs_ref)

        # 8) Timing
        elapsed = time.perf_counter() - t0
        with open(runtime_log, "a", newline="") as f:
            csv.writer(f).writerow([date, f"{elapsed:.3f}"])
        print(f"[{date}] elapsed: {elapsed:.1f}s")
        print(f"[total so far] {(time.perf_counter()-t0_total)/60:.1f} min")

        # 9) Clean current_day except static files
        workdir = REPO_ROOT / "current_day"
        
        if workdir.exists():
            for f in workdir.iterdir():
                if f.name not in keep_files:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        # optional: remove subdirectories if you ever create any
                        # shutil.rmtree(f)
                        pass
        else:
            print(f"[clean] workdir does not exist, skipping cleanup: {workdir}")
        
            total = time.perf_counter() - t0_total
            print(f"\nAll done. Total elapsed: {total/60:.1f} min")


if __name__ == "__main__":
    # Example: run one week
    month_loop("20160208", "20160214")
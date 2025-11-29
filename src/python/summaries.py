import xarray as xr
import numpy as np
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SRC_PY    = REPO_ROOT / "src" / "python"

def _rename_depth_to_standard(da):
    for cand in ("depth", "deptht", "z"):
        if cand in da.dims:
            return da.rename({cand: "depth"})
    return da

def fix_time_axis(ax, max_ticks=6, fmt='%Y-%m-%d'):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=max_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')

def save_mae_summary(forecast_file, analysis_file, obs_file, var, out_file="mae_summary.nc"):
    """
    Compute MAE summary (Hovmöller + timeseries + depth profile)
    and save to a compact NetCDF.
    """

    fcst = xr.open_dataset(REPO_ROOT / forecast_file)[var][:,:,0:26,:,:]
    an   = xr.open_dataset(REPO_ROOT / analysis_file)[var][:,:,0:26,:,:]
    obs  = xr.open_dataset(REPO_ROOT / obs_file)[var][:,0:26,:,:]

    # Normalize depth name
    fcst = _rename_depth_to_standard(fcst)
    an   = _rename_depth_to_standard(an)
    obs  = _rename_depth_to_standard(obs)

    # Match ensemble sizes if forecast > analysis
    if "ensemble" in fcst.dims and "ensemble" in an.dims and fcst.sizes["ensemble"] > an.sizes["ensemble"]:
        fcst = fcst.isel(ensemble=slice(0, an.sizes["ensemble"]))

    # Compute mean absolute error
    reduce_dims = [d for d in ("ensemble", "y", "x") if d in fcst.dims]
    mae_fcst = np.abs(fcst - obs).mean(dim=reduce_dims, skipna=True)
    mae_an   = np.abs(an   - obs).mean(dim=reduce_dims, skipna=True)

    # === Derive summaries ===
    # Hovmöller: time x depth
    mae_fcst_hov = mae_fcst.transpose("time_counter", "depth")
    mae_an_hov   = mae_an.transpose("time_counter", "depth")

    # Timeseries: mean over depth
    mae_fcst_ts = mae_fcst_hov.mean(dim="depth")
    mae_an_ts   = mae_an_hov.mean(dim="depth")

    # Depth profile: mean over time
    mae_fcst_dp = mae_fcst_hov.mean(dim="time_counter")
    mae_an_dp   = mae_an_hov.mean(dim="time_counter")

    # === Save compact NetCDF ===
    ds_out = xr.Dataset(
        {
            "mae_fcst_hovmoller": mae_fcst_hov,
            "mae_an_hovmoller":   mae_an_hov,
            "mae_fcst_timeseries": mae_fcst_ts,
            "mae_an_timeseries":   mae_an_ts,
            "mae_fcst_depth_profile": mae_fcst_dp,
            "mae_an_depth_profile":   mae_an_dp,
        }
    )

    ds_out.to_netcdf(REPO_ROOT / out_file)
    print(f"[OK] Saved MAE summary → {out_file}")

from properscoring import crps_ensemble

def save_crps_summary(fcst_file, an_file, obs_file, var, out_file="crps_summary.nc", levels=(0,5,10,15,20,25)):
    """
    Compute CRPS summary (timeseries and selected depth levels)
    and store in a compact NetCDF.
    """

    fcst = xr.open_dataset(REPO_ROOT / fcst_file)[var][:,:,0:26,:,:]
    an   = xr.open_dataset(REPO_ROOT / an_file)[var][:,:,0:26,:,:]
    obs  = xr.open_dataset(REPO_ROOT / obs_file)[var][:,0:26,:,:]

    fcst = _rename_depth_to_standard(fcst)
    an   = _rename_depth_to_standard(an)
    obs  = _rename_depth_to_standard(obs)

    # Trim ensemble mismatch
    if "ensemble" in fcst.dims and "ensemble" in an.dims:
        if fcst.sizes["ensemble"] > an.sizes["ensemble"]:
            fcst = fcst.isel(ensemble=slice(0, an.sizes["ensemble"]))

    # keep dims so xarray keeps axis names
    fcst = fcst.mean(dim="depth", skipna=True)
    an   = an.mean(dim="depth", skipna=True)
    obs  = obs.mean(dim="depth", skipna=True)

    # Move ensemble last
    fcst_vals = np.moveaxis(fcst.values, 0, -1)
    an_vals   = np.moveaxis(an.values,   0, -1)
    obs_vals  = obs.values  # no ensemble axis

    # === CRPS over full depth 0:25 ===
    crps_fcst_grid = crps_ensemble(obs_vals, fcst_vals, axis=-1)
    crps_an_grid   = crps_ensemble(obs_vals, an_vals,   axis=-1)

    # mean over space (y,x)
    crps_fcst_ts = np.nanmean(crps_fcst_grid, axis=(1,2))
    crps_an_ts   = np.nanmean(crps_an_grid,   axis=(1,2))

    ds_out = xr.Dataset(
        {
            "crps_fcst_timeseries": (("time_counter"), crps_fcst_ts),
            "crps_an_timeseries":   (("time_counter"), crps_an_ts),
        },
        coords={
            "time_counter": obs.time_counter.values,
        }
    )

    ds_out.to_netcdf(REPO_ROOT / out_file)
    print(f"[OK] Saved CRPS summary → {out_file}")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from pathlib import Path


# --------------------------------------------------------------
# Pretty rounding of contour levels
# --------------------------------------------------------------
def smart_levels(vmin, vmax, n=25, digits=4):
    """
    Compute 'nice' contour levels with controlled rounding.
    Ensures rounded vmin/vmax and even number of intervals.
    """
    step = 10 ** -digits
    vmin_r = np.floor(vmin / step) * step
    vmax_r = np.ceil(vmax / step) * step

    # ensure number of intervals is even
    if (n - 1) % 2 != 0:
        n += 1

    return np.linspace(vmin_r, vmax_r, n)


# --------------------------------------------------------------
# Pretty contour label formatter
# --------------------------------------------------------------
class nf(float):
    """Formats contour labels so that trailing .0 disappears."""
    def __repr__(self):
        s = '%.1f' % (self.__float__(),)
        if s.endswith('0'):
            return '%.0f' % self.__float__()
        return s


# --------------------------------------------------------------
# MAIN PLOTTING FUNCTION — your exact preferred style
# --------------------------------------------------------------
import matplotlib.dates as mdates
def plot_mae_hovmoller_from_summary(
    summary_file="mae_summary.nc",
    var="POC",
    cmap_txt=None,
    output="mae_hovmoller_summary.png"
    ):
    """
    Plot MAE Hovmöller diagrams from summarized NetCDF output using
    the full high-quality style from the main notebook.
    """

    ds = xr.open_dataset(summary_file)
    A_fcst = ds["mae_fcst_hovmoller"].values        # (time, depth)
    A_an   = ds["mae_an_hovmoller"].values          # (time, depth)
    time_vals  = ds["time_counter"].values
    depth_vals = ds["depth"].values                 # positive downward

    if cmap_txt:
        with open(cmap_txt) as f:
            color = [list(map(lambda x: int(x)/255., ln.split()[:3]))
                     for ln in f if ln.strip() and not ln.startswith("#")]
        cmap = matplotlib.colors.ListedColormap(color, name="custom")
    else:
        cmap = "viridis"

    vmin = np.nanmin([A_fcst, A_an])
    vmax = np.nanmax([A_fcst, A_an])
    levels = smart_levels(vmin, vmax)

    fig = plt.figure(figsize=(10, 5))
    gs  = GridSpec(nrows=2, ncols=2, height_ratios=[18, 1.3],
                   hspace=0.3, wspace=0.15, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cax = fig.add_subplot(gs[1, :])

    im0 = ax0.pcolormesh(time_vals, depth_vals * -1, A_fcst.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    im1 = ax1.pcolormesh(time_vals, depth_vals * -1, A_an.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # ----------------------------------------------------------
    # Contours
    # ----------------------------------------------------------
    co0 = ax0.contour(time_vals, depth_vals * -1, A_fcst.T, levels=levels[::2], colors='k', linestyles='--', linewidths=1.0)
    co1 = ax1.contour(time_vals, depth_vals * -1, A_an.T, levels=levels[::2], colors='k', linestyles='--', linewidths=1.0)

    # formatted labels
    co0.levels = [nf(val) for val in co0.levels]
    co1.levels = [nf(val) for val in co1.levels]
    plt.clabel(co0, co0.levels, inline=True, fmt='%.3f', fontsize=10)
    plt.clabel(co1, co1.levels, inline=True, fmt='%.3f', fontsize=10)
    fix_time_axis(plt.gca())

    for ax, title in zip(
            [ax0, ax1],
            [f"Forecast {var} MAE", f"Analysis {var} MAE"]):

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time, days")
        if ax is ax0:
            ax.set_ylabel("depth, m")
        ax.tick_params(labelsize=9)
        ax.grid(True)

    # ----------------------------------------------------------
    # Fix time axis tick clutter on BOTH panels
    # ----------------------------------------------------------
    for ax in (ax0, ax1):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        # plt.setp(ax.get_xticklabels(), rotation=25, ha='right')

    cb = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cb.set_label(r"Mean Absolute Error, $mmol/m^3$", fontsize=9)
    # fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()

def plot_mae_global_from_summary(
    summary_file="mae_summary.nc",
    var="POC",
    output="mae_global_summary.png"
):
    """
    Plot domain- and depth-mean MAE as a time series
    using precomputed values from mae_summary.nc.
    """
    ds = xr.open_dataset(summary_file)

    mae_fcst = ds["mae_fcst_timeseries"]
    mae_an   = ds["mae_an_timeseries"]
    t        = ds["time_counter"]

    plt.figure(figsize=(8,4))
    plt.plot(t, mae_fcst, label="Forecast MAE", marker="o")
    plt.plot(t, mae_an,   label="Analysis MAE", marker="o")
    plt.legend()
    plt.title(f"50m mean MAE of {var}")
    plt.xlabel("Time")
    plt.ylabel("MAE")
    plt.grid(True)
    fix_time_axis(plt.gca())
    plt.tight_layout()
    # plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close()

def plot_mae_depth_profile_from_summary(
    summary_file="mae_summary.nc",
    var="POC",
    output="mae_profile_summary.png"
):
    """
    Plot depth profile of MAE (time-mean) using mae_summary.nc.
    """
    ds = xr.open_dataset(summary_file)

    mae_fcst_dp = ds["mae_fcst_depth_profile"]
    mae_an_dp   = ds["mae_an_depth_profile"]
    depth       = ds["depth"]

    plt.figure(figsize=(4,5))
    plt.plot(mae_fcst_dp, depth, label="Forecast MAE")
    plt.plot(mae_an_dp,   depth, label="Analysis MAE")
    plt.gca().invert_yaxis()
    plt.xlabel(f"{var} MAE")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.title(f"Depth profile of MAE ({var})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close()

def plot_crps_from_summary(
    summary_file="crps_summary.nc",
    var="POC",
    output="crps_summary.png"
):
    """
    Plot domain- and depth-mean CRPS as a time series
    using precomputed values from crps_summary.nc.
    """
    ds = xr.open_dataset(summary_file)

    crps_fcst = ds["crps_fcst_timeseries"]
    crps_an   = ds["crps_an_timeseries"]
    t        = ds["time_counter"]

    plt.figure(figsize=(8,4))
    plt.plot(t, crps_fcst, label="Forecast CRPS", marker="o")
    plt.plot(t, crps_an,   label="Analysis CRPS", marker="o")
    plt.legend()
    plt.title(f"50m mean CRPS of {var}")
    plt.xlabel("Time")
    plt.ylabel("CRPS")
    plt.grid(True)
    fix_time_axis(plt.gca())
    plt.tight_layout()
    # plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close()

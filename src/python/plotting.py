import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from cycler import cycler

# pick a colormap and sample N colors
N = 6  # max number of distinct lines you expect
new_colors = plt.cm.Set2(np.linspace(0, 1, N))

plt.rcParams['axes.prop_cycle'] = cycler(color=new_colors)

from pathlib import Path

# Resolve repo root from this file location:
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]


def _load_lonlat_from_domain():
    """
    Load lon/lat from domain_cfg.nc.
    Tries data/static first, then current_day.
    Returns (lon, lat) as xarray DataArrays with dims (y, x).
    """
    candidates = [
        REPO_ROOT / "data" / "static" / "domain_cfg.nc",
        REPO_ROOT / "current_day" / "domain_cfg.nc",
    ]

    domain_path = None
    for c in candidates:
        if c.exists():
            domain_path = c
            break

    if domain_path is None:
        raise FileNotFoundError(
            f"domain_cfg.nc not found in {candidates}"
        )

    ds_dom = xr.open_dataset(domain_path)

    # Try common variable names for lon/lat
    for lon_name in ("nav_lon", "glamt", "glamf", "lon"):
        if lon_name in ds_dom:
            lon = ds_dom[lon_name]
            break
    else:
        raise KeyError("No longitude variable found in domain_cfg.nc")

    for lat_name in ("nav_lat", "gphit", "gphif", "lat"):
        if lat_name in ds_dom:
            lat = ds_dom[lat_name]
            break
    else:
        raise KeyError("No latitude variable found in domain_cfg.nc")

    return lon, lat


# --- MAE ---

def _rename_depth_to_standard(da: xr.DataArray) -> xr.DataArray:
    for cand in ("depth", "deptht", "z"):
        if cand in da.dims:
            return da if cand == "depth" else da.rename({cand: "depth"})
    return da  # surface-only vars

def _align3(a: xr.DataArray, b: xr.DataArray, c: xr.DataArray):
    a1, b1 = xr.align(a, b, join="inner")
    a2, c1 = xr.align(a1, c, join="inner")
    return a2, b1, c1

def _print_da(label: str, da: xr.DataArray):
    print(f"\n[{label}]")
    print("  dims:", da.dims)
    print("  sizes:", {k:int(v) for k,v in da.sizes.items()})
    # show coord names present
    print("  coords:", [k for k in da.coords])
    # sample values along dims if small
    for d in ("time_counter", "depth"):
        if d in da.coords:
            vals = da.coords[d].values
            if vals.size:
                print(f"  {d}[0:{min(3, vals.size)}] ->", vals[:min(3, vals.size)])

def _rename_depth_to_standard(da: xr.DataArray) -> xr.DataArray:
    for cand in ("depth", "deptht", "z"):
        if cand in da.dims:
            return da.rename({cand: "depth"})
    return da

def smart_levels(vmin, vmax, n=25, digits=4):
    """
    Returns rounded vmin/vmax and an even number of levels for contour/colormap.
    Ensures symmetric colorbar spacing and visually neat ticks.
    """
    step = 10 ** -digits
    vmin_r = np.floor(vmin / step) * step
    vmax_r = np.ceil(vmax / step) * step

    # Adjust N so that number of intervals (N-1) is even
    if (n - 1) % 2 != 0:
        n += 1

    levels = np.linspace(vmin_r, vmax_r, n)
    return levels

class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

def plot_mae_hovmoller(
    forecast_file,
    analysis_file,
    obs_file,
    var,
    output="mae_hovmoller.png",
    cmap_txt=None,
    figsize=(9, 4),
):
    # --- Load datasets ---
    dsf = xr.open_dataset(forecast_file, decode_times=False)
    dsa = xr.open_dataset(analysis_file, decode_times=False)
    dso = xr.open_dataset(obs_file,     decode_times=False)

    fcst = dsf[var][:,:,0:26,:,:]
    an   = dsa[var][:,:,0:26,:,:]
    obs  = dso[var][:,  0:26,:,:]

    # Rename depth dimension if needed
    def _to_depth(da):
        for cand in ("depth", "deptht", "z"):
            if cand in da.dims:
                return da.rename({cand: "depth"})
        return da
    fcst = _to_depth(fcst); an = _to_depth(an); obs = _to_depth(obs)

    # Match ensemble sizes if forecast > analysis
    if "ensemble" in fcst.dims and "ensemble" in an.dims and fcst.sizes["ensemble"] > an.sizes["ensemble"]:
        fcst = fcst.isel(ensemble=slice(0, an.sizes["ensemble"]))

    # Compute mean absolute error
    reduce_dims = [d for d in ("ensemble", "y", "x") if d in fcst.dims]
    mae_fcst = np.abs(fcst - obs).mean(dim=reduce_dims, skipna=True)
    mae_an   = np.abs(an   - obs).mean(dim=reduce_dims, skipna=True)

    time_vals = mae_fcst["time_counter"].values
    depth_vals = mae_fcst["depth"].values
    A_fcst = mae_fcst.values.mean(axis=0) if mae_fcst.ndim == 3 else mae_fcst.values
    A_an   = mae_an.values.mean(axis=0)   if mae_an.ndim == 3 else mae_an.values

    # --- Colormap ---
    if cmap_txt:
        with open(cmap_txt) as f:
            color = [list(map(lambda x: int(x)/255., ln.split()[:3]))
                     for ln in f if ln.strip() and not ln.strip().startswith("#")]
        cmap = matplotlib.colors.ListedColormap(color, name="custom")
    else:
        cmap = "viridis"

    vmin = np.nanmin([A_fcst, A_an])
    vmax = np.nanmax([A_fcst, A_an])

    levels = smart_levels(vmin, vmax)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 5))
    gs  = GridSpec(
        nrows=2, ncols=2,
        height_ratios=[18, 1.3],  # make the colorbar row taller and lower
        hspace=0.3,              # add a bit more vertical spacing
        wspace=0.15,
        figure=fig
    )
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cax = fig.add_subplot(gs[1, :])  # colorbar below both plots


    # im0 = ax0.contourf(time_vals, depth_vals*-1, A_fcst.T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    # im1 = ax1.contourf(time_vals, depth_vals*-1, A_an.T,   levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    im0 = ax0.pcolormesh(time_vals, depth_vals*-1, A_fcst.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    im1 = ax1.pcolormesh(time_vals, depth_vals*-1, A_an.T,   shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    co0 = ax0.contour(time_vals, depth_vals*-1, A_fcst.T, levels=levels[::2], colors='k', linestyles='--', linewidths=1.0)
    co1 = ax1.contour(time_vals, depth_vals*-1, A_an.T,   levels=levels[::2], colors='k', linestyles='--', linewidths=1.0)
    co0.levels = [nf(val) for val in co0.levels]
    plt.clabel(co0, co0.levels, inline=True, fmt='%.3f',fontsize=10)
    co1.levels = [nf(val) for val in co1.levels]
    plt.clabel(co1, co1.levels, inline=True, fmt='%.3f',fontsize=10)

    plt.grid(True)

    for ax, title in zip([ax0, ax1], [f"Forecast {var} MAE", f"Analysis {var} MAE"]):
        ax.set_title(title, fontsize=10)
        if ax == ax0:
            ax.set_ylabel("depth, m")
        ax.set_xlabel("time, days")
        # ax.invert_yaxis()  # ocean style
        ax.tick_params(labelsize=9)

    cb = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cb.set_label(r"Mean Absolute Error, $mmol/m^3$", fontsize=9)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close(fig)

def plot_mae_global(fcst_file, an_file, obs_file, var, output="mae_global.png"):
    fcst = xr.open_dataset(fcst_file)[var][:,:,0:26,:,:]
    an = xr.open_dataset(an_file)[var][:,:,0:26,:,:]
    obs = xr.open_dataset(obs_file)[var][:,0:26,:,:]

    # Normalize depth dimension names
    fcst = _rename_depth_to_standard(fcst)
    an   = _rename_depth_to_standard(an)
    obs  = _rename_depth_to_standard(obs)

    mae_fcst = np.abs(fcst.mean("ensemble") - obs).mean(dim=("x","y","depth"))
    mae_an   = np.abs(an.mean("ensemble") - obs).mean(dim=("x","y","depth"))

    plt.figure(figsize=(8,4))
    mae_fcst.plot(label="Forecast MAE")
    mae_an.plot(label="Analysis MAE")
    plt.legend(); plt.title(f"50m mean MAE of {var}")
    plt.grid(True)
    # plt.savefig(output, dpi=150)
    plt.show()
    # plt.close()

def plot_mae_depth_profile(fcst_file, an_file, obs_file, var, output="mae_profile.png"):
    fcst = xr.open_dataset(fcst_file)[var][:,:,0:26,:,:]
    an = xr.open_dataset(an_file)[var][:,:,0:26,:,:]
    obs = xr.open_dataset(obs_file)[var][:,0:26,:,:]

    # Normalize depth dimension names
    fcst = _rename_depth_to_standard(fcst)
    an   = _rename_depth_to_standard(an)
    obs  = _rename_depth_to_standard(obs)

    mae_fcst = np.abs(fcst.mean("ensemble") - obs).mean(dim=("x","y","time_counter"))
    mae_an   = np.abs(an.mean("ensemble") - obs).mean(dim=("x","y","time_counter"))

    plt.figure(figsize=(4,5))
    plt.plot(mae_fcst, mae_fcst.depth, label="Forecast MAE")
    plt.plot(mae_an, mae_an.depth, label="Analysis MAE")
    plt.gca().invert_yaxis()
    plt.xlabel(f"{var} MAE"); plt.ylabel("Depth")
    plt.legend(); plt.title(f"Depth profile of MAE ({var})")
    plt.grid(True)
    # plt.savefig(output, dpi=150)
    plt.show()
    # plt.close()

def _rename_depth_to_standard(da):
    for cand in ("depth", "deptht", "z"):
        if cand in da.dims:
            return da.rename({cand: "depth"})
    return da

def _mean_over_spatial(arr):
    # keep axis 0 (time), average over the rest—works for 2D or 3D vars
    if arr.ndim <= 1:
        return arr
    return arr.mean(axis=tuple(range(1, arr.ndim)))

import warnings
def plot_crps_timeseries(fcst_file, an_file, obs_file, var, output="crps_timeseries.png"):
    fcst = xr.open_dataset(fcst_file, decode_times=False)[var]
    an   = xr.open_dataset(an_file,   decode_times=False)[var]
    obs  = xr.open_dataset(obs_file,  decode_times=False)[var]

    # normalize depth name and select level 26 (27th level)
    fcst = _rename_depth_to_standard(fcst); 
    an   = _rename_depth_to_standard(an);   
    obs  = _rename_depth_to_standard(obs);  

    # --- Depth mean over 0:25 (first 26 levels) ---
    # keep dims so xarray keeps axis names
    fcst = fcst.isel(depth=slice(0, 26)).mean(dim="depth", skipna=True)
    an   = an.isel(depth=slice(0, 26)).mean(dim="depth", skipna=True)
    obs  = obs.isel(depth=slice(0, 26)).mean(dim="depth", skipna=True)

    # move ensemble axis last for forecasts; obs has no ensemble axis
    fcst_vals = np.moveaxis(fcst.values, 0, -1) if "ensemble" in fcst.dims else fcst.values
    an_vals   = np.moveaxis(an.values,   0, -1) if "ensemble" in an.dims   else an.values
    obs_vals  = obs.values
        
    crps_fcst_grid = crps_ensemble(obs_vals, fcst_vals, axis=-1)
    crps_an_grid   = crps_ensemble(obs_vals, an_vals,   axis=-1)

    # --- Now average over space (y,x) → timeseries ---
    crps_fcst = np.nanmean(crps_fcst_grid, axis=(1,2))
    crps_an   = np.nanmean(crps_an_grid,   axis=(1,2))

    times = obs["time_counter"].values

    plt.figure(figsize=(8, 4))
    plt.plot(times, crps_fcst, label="Forecast CRPS", marker="o")
    plt.plot(times, crps_an,   label="Analysis CRPS", marker="o")
    plt.title(f"CRPS Time Series for {var} (0-50 m mean)")
    plt.xlabel("Time")
    plt.ylabel("CRPS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig(output, dpi=150)
    # plt.close()

def plot_crps_levels(fcst_file, an_file, obs_file, var, levels, output="crps_levels.png"):
    fcst = xr.open_dataset(fcst_file)[var][:,:,0:26,:,:]
    an   = xr.open_dataset(an_file)[var][:,:,0:26,:,:]
    obs  = xr.open_dataset(obs_file)[var][:,0:26,:,:]

    # Normalize depth dimension names
    fcst = _rename_depth_to_standard(fcst)
    an   = _rename_depth_to_standard(an)
    obs  = _rename_depth_to_standard(obs)

    cmap = plt.cm.Set1  # or any other colormap you like
    colors = cmap(np.linspace(0, 1, len(levels)))

    plt.figure(figsize=(10, 6))

    for i, lvl in enumerate(levels):
        color = colors[i]

        obs_lvl  = obs.isel(depth=lvl)
        fcst_lvl = fcst.isel(depth=lvl)
        an_lvl   = an.isel(depth=lvl)

        # move ensemble axis last for forecasts; obs has no ensemble axis
        fcst_vals = np.moveaxis(fcst_lvl.values, 0, -1) if "ensemble" in fcst_lvl.dims else fcst_lvl.values
        an_vals   = np.moveaxis(an_lvl.values,   0, -1) if "ensemble" in an_lvl.dims   else an_lvl.values
        obs_vals  = obs_lvl.values

        crps_fcst_grid = crps_ensemble(obs_vals, fcst_vals, axis=-1)
        crps_an_grid   = crps_ensemble(obs_vals, an_vals,   axis=-1)

        # average over space (y,x) → timeseries
        crps_fcst = np.nanmean(crps_fcst_grid, axis=(1, 2))
        crps_an   = np.nanmean(crps_an_grid,   axis=(1, 2))

        times = obs.time_counter.values

        # same color, different linestyle
        plt.plot(times, crps_fcst, "--", color=color, label=f"Forecast CRPS @ {lvl} z-lvl")
        plt.plot(times, crps_an,   "-",  color=color, label=f"Analysis CRPS @ {lvl} z-lvl")

    plt.legend()
    plt.title(f"CRPS at selected levels ({var})")
    plt.xlabel("Time")
    plt.ylabel("CRPS")
    plt.grid(True)
    # plt.savefig(output, dpi=150)
    plt.show()
    # plt.close()

# --- Increments ---

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def setup_map(
    ax,
    extent=(27, 42.5, 40, 46.97111),
    rivers=True,
    land=True,
    show_left_labels=True,
    show_bottom_labels=True,
):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if rivers:
        ax.add_feature(cfeature.RIVERS, zorder=2.2)
    if land:
        ax.add_feature(cfeature.LAND, edgecolor="grey", zorder=2)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = show_bottom_labels
    gl.left_labels = show_left_labels

    gl.xlabel_style = {"size": 12, "color": "black"}
    gl.ylabel_style = {"size": 12, "color": "black"}


def plot_increments_map(incr_file, var, depth=0, date_index=-1, output="increments_map.png"):
    # Open increments, normalize depth name
    ds = xr.open_dataset(incr_file)
    ds = _rename_depth_to_standard(ds)

    da = ds[var]

    # Select time
    if "time_counter" in da.dims:
        da = da.isel(time_counter=date_index)

    # Select vertical level (by index; increments_combined has depth levels, not meters)
    if "depth" in da.dims:
        da = da.isel(depth=depth)

    # Get lon/lat from domain_cfg.nc
    lon, lat = _load_lonlat_from_domain()

    # Time label
    if "time_counter" in ds.coords:
        tval = ds["time_counter"].values[date_index]
        tlabel = str(tval)[:10]
    else:
        tlabel = ""

    # --- Plot on map ---
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    vmin = np.nanmin(da); vmax = np.nanmax(da)
    if np.abs(vmin) > np.abs(vmax):
        vmax = np.abs(vmin)
    else:
        vmin = - np.abs(vmax)
        
    pcm = ax.pcolormesh(lon, lat, da, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap="seismic", shading="auto")
    setup_map(ax)

    cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    cb.set_label(f"{var} increment, $mmol/m^3$")

    ax.set_title(f"Increments {var} @ level {depth} ({tlabel})")

    plt.tight_layout()
    plt.show()
    # plt.savefig(output, dpi=150); plt.close()

def plot_increments_section(incr_file, var, axis="y", index=20, date_index=-1, output="increments_section.png"):
    ds = xr.open_dataset(incr_file)
    ds = _rename_depth_to_standard(ds)
    da = ds[var].isel(time_counter=date_index)
    if axis == "y":  # section in x-z plane
        sec = da.isel(y=index)
        sec.plot(x="x", y="depth", cmap="RdBu_r", robust=True)
        plt.gca().invert_yaxis()
        plt.title(f"Increments {var} section (y={index})")
    else:  # section in y-z plane
        sec = da.isel(x=index)
        sec.plot(x="y", y="depth", cmap="RdBu_r", robust=True)
        plt.gca().invert_yaxis()
        plt.title(f"Increments {var} section (x={index})")
    plt.grid(True)
    # plt.savefig(output, dpi=150); plt.close()
    plt.show()

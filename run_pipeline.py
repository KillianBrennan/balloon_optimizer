"""
Balloon Optimizer Pipeline
==========================

End-to-end pipeline that:
  1. Reads a configuration file (params.txt).
  2. Computes (or reuses) transition matrices for each ensemble member.
  3. Runs find_farthest_reachable() in parallel across members.
  4. Produces per-member and probabilistic diagnostic outputs (NetCDF + PNG).

Usage
-----
    python run_pipeline.py [params.txt]

If no path is given, the script looks for ``params.txt`` in the same
directory as this script.

Public API for notebooks / interactive inspection
-------------------------------------------------
The following functions load saved NetCDF output files and return
``matplotlib.Figure`` objects – they do **not** require re-running the
pipeline:

``plot_member_from_nc(member_dir, member, start_lat, start_lon)``
    Load ``trajectories.nc`` and ``reachable_mask.nc`` from one member
    directory and return ``(fig_map, fig_pressure_profiles)``.

``plot_member_by_target(member_dir, member, target_lat, target_lon, radius_km)``
    Return a pressure-profile figure for trajectories ending within
    *radius_km* of the given target point, or *None* if none found.

``plot_probabilistic_from_nc(run_dir, members, start_lat, start_lon)``
    Load ``probabilistic/probabilistic.nc`` (and optional landing-zone
    files) and return a dict of figures keyed by
    ``"prob_map"``, ``"histogram"``, ``"landing_zones_map"``,
    ``"landing_zones_profiles"``.

Private map / plot helpers
--------------------------
All map-drawing code is consolidated in a set of private helpers
(``_make_map_axes``, ``_set_map_extent``, ``_add_gridlines``,
``_plot_frac_contourf``, ``_make_histogram_fig``, ``_make_lz_map``,
``_make_lz_profiles``) so that the pipeline's save functions and the
notebook's plot functions share identical rendering logic.
"""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

from scipy import ndimage as ndi
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pyproj import Geod

# --------------------------------------------------------------------------- #
# Local imports — resolve relative to this file so the script is callable     #
# from any working directory.                                                  #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compute_transition as _ct
from graph_optimizer import find_farthest_reachable, build_land_mask


# =========================================================================== #
# Configuration helpers                                                         #
# =========================================================================== #

def load_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cfg.read(path)
    return cfg


def _pressure_levels(cfg: configparser.ConfigParser) -> np.ndarray:
    raw = cfg.get("pressure_levels", "levels", fallback="1000,950,900,850,800,750,700,650,600,550,500")
    return np.array([float(v.strip()) for v in raw.split(",")], dtype=float)


def _discover_members(
    cfg: configparser.ConfigParser,
    logger: logging.Logger,
) -> List[str]:
    """Discover available ensemble member identifiers from the data directory.

    For IFS: scans ``<ifs_root>/<ifs_init>/`` for subdirectories and returns
    the first *n_members* entries (sorted alphabetically).
    For ERA5: always returns ``["era5"]``.
    """
    source    = cfg.get("data", "source", fallback="era5")
    n_members = cfg.getint("data", "n_members", fallback=1)

    if source == "era5":
        logger.info("ERA5 source: using single member 'era5'.")
        return ["era5"]

    ifs_root = cfg.get("data", "ifs_root", fallback="/net/tropo/atmosdyn/eps/ncdf")
    ifs_init = cfg.get("data", "ifs_init", fallback=None)

    if not ifs_init:
        logger.error("ifs_init is required for IFS source.")
        return []

    ifs_dir = os.path.join(ifs_root, ifs_init)
    if not os.path.isdir(ifs_dir):
        logger.error("IFS directory not found: %s", ifs_dir)
        return []

    available = sorted([
        d for d in os.listdir(ifs_dir)
        if os.path.isdir(os.path.join(ifs_dir, d))
    ])

    if not available:
        logger.error("No member subdirectories found in %s", ifs_dir)
        return []

    selected = available[:n_members]
    logger.info("Discovered %d/%d IFS members in %s: %s",
                len(selected), len(available), ifs_dir, selected)
    return selected


def _members(cfg: configparser.ConfigParser) -> List[str]:
    """Legacy helper kept for compatibility – prefer _discover_members."""
    raw = cfg.get("data", "members", fallback="era5")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _surface_level_idx(plevs: np.ndarray, hpa: float) -> int:
    return int(np.argmin(np.abs(plevs - hpa)))


def _output_dir(cfg: configparser.ConfigParser) -> str:
    base = cfg.get("output", "base_dir",
                   fallback="/net/litho/atmosdyn2/kbrennan/data/balloon/solutions")
    source   = cfg.get("data", "source",   fallback="era5")
    label    = cfg.get("run",  "label",    fallback="run")
    start    = cfg.get("time", "start",    fallback="unknown")
    end      = cfg.get("time", "end",      fallback="unknown")
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder   = f"{source}_{label}_{start}_{end}_{ts}"
    return os.path.join(base, folder)


# =========================================================================== #
# Logging                                                                       #
# =========================================================================== #

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("balloon_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =========================================================================== #
# Transition computation                                                        #
# =========================================================================== #

def _patch_ct_globals(cfg: configparser.ConfigParser) -> None:
    """Patch module-level constants in compute_transition before calling main().

    This allows the pipeline to honour the domain and pressure-level settings
    from params.txt without requiring changes to compute_transition.py.
    """
    lon_min = cfg.getfloat("domain", "lon_min", fallback=-22.0)
    lon_max = cfg.getfloat("domain", "lon_max", fallback=38.0)
    lat_min = cfg.getfloat("domain", "lat_min", fallback=30.0)
    lat_max = cfg.getfloat("domain", "lat_max", fallback=75.0)

    _ct.LON_MIN  = lon_min
    _ct.LON_MAX  = lon_max
    _ct.LAT_MIN  = lat_min
    _ct.LAT_MAX  = lat_max
    _ct.DOMAIN   = (lon_min, lon_max, lat_min, lat_max)
    _ct.PRES_LEVELS_HPA = _pressure_levels(cfg)
    _ct.OUTPUT_DIR = cfg.get("output", "base_dir",
                             fallback="/net/litho/atmosdyn2/kbrennan/data/balloon/solutions")

    era5_root = cfg.get("data", "era5_root",
                        fallback="/home/kbrennan/data/era5/cdf")
    _ct.DATA_ROOT = era5_root

    ifs_root = cfg.get("data", "ifs_root",
                       fallback="/net/tropo/atmosdyn/eps/ncdf")
    _ct.IFS_ROOT = ifs_root


def compute_transition_for_member(
    cfg: configparser.ConfigParser,
    member: str,
    transition_out_dir: str,
    logger: logging.Logger,
) -> Optional[str]:
    """Compute (or locate) the transition matrix for one ensemble member.

    Returns the path to the output NetCDF, or None on failure.
    """
    source      = cfg.get("data", "source", fallback="era5")
    start       = cfg.get("time", "start")
    end         = cfg.get("time", "end")
    dt          = cfg.getfloat("time", "dt", fallback=3600.0)
    superscale  = cfg.getint("time", "superscale", fallback=3)
    ifs_init    = cfg.get("data", "ifs_init", fallback=None)

    # Expected output filename (mirrors naming inside compute_transition.main)
    if source == "era5":
        fname = f"transition_indices_{start}_{end}_dt{int(dt)}.nc"
    else:
        fname = f"transition_indices_{start}_{end}_dt{int(dt)}_ifs_{ifs_init}_{member}.nc"

    out_path = os.path.join(transition_out_dir, fname)

    # Check pre-existing transition_dir supplied in params
    precomp_dir = cfg.get("data", "transition_dir", fallback="").strip()
    if precomp_dir:
        candidate = os.path.join(precomp_dir, fname)
        if os.path.exists(candidate):
            logger.info("Using pre-existing transition file: %s", candidate)
            return candidate

    if os.path.exists(out_path):
        logger.info("Transition file already present, skipping: %s", out_path)
        return out_path

    logger.info("Computing transitions for member %s …", member)
    try:
        _patch_ct_globals(cfg)
        _ct.OUTPUT_DIR = transition_out_dir

        kwargs: Dict[str, Any] = dict(
            start=start,
            end=end,
            dt=dt,
            superscale=superscale,
            data_source=source,
        )
        if source == "ifs":
            kwargs["ifs_init"] = ifs_init
            kwargs["ifs_member"] = member

        _ct.main(**kwargs)
    except Exception:
        logger.error("Transition computation failed for member %s:\n%s",
                     member, traceback.format_exc())
        return None

    if os.path.exists(out_path):
        return out_path
    # Fallback: search for the file
    for f in os.listdir(transition_out_dir):
        if f.endswith(".nc") and start in f and end in f:
            if source == "ifs" and member in f:
                return os.path.join(transition_out_dir, f)
            elif source == "era5":
                return os.path.join(transition_out_dir, f)
    logger.error("Expected output file not found after computation: %s", out_path)
    return None


# =========================================================================== #
# Optimization worker (runs in a child process)                                #
# =========================================================================== #

def _opt_worker(args: Tuple) -> Dict:
    """Worker function for the parallel optimisation pool.

    Parameters unpacked from args tuple to allow pool.imap usage.
    """
    (
        transition_path,
        member,
        start_lat, start_lon,
        budget, k_best,
        land_only,
        surface_level_hpa,
        early_cost_penalty,
    ) = args

    result: Dict[str, Any] = {"member": member, "transition_path": transition_path}

    try:
        ds = xr.open_dataset(transition_path)
        next_i = ds["lat_idx"].values.astype(int)
        next_j = ds["lon_idx"].values.astype(int)

        lats = ds["lat"].values
        lons = ds["lon"].values
        plevs = ds["plev"].values

        i0 = int(np.argmin(np.abs(lats - start_lat)))
        j0 = int(np.argmin(np.abs(lons - start_lon)))
        l_bottom = int(np.argmax(plevs)) if surface_level_hpa is None else \
                   int(np.argmin(np.abs(plevs - surface_level_hpa)))

        land_mask = build_land_mask(lats, lons) if land_only else None

        reachable, paths, values, costs, reachable_mask = find_farthest_reachable(
            next_i=next_i,
            next_j=next_j,
            origin=(i0, j0),
            B=budget,
            allowed_levels=None,
            start_levels=[l_bottom],
            target_levels=[l_bottom],
            land_mask=land_mask,
            lats=lats,
            lons=lons,
            k_best=k_best,
            early_cost_penalty=early_cost_penalty,
            show_progress=False,  # silenced in worker; main process shows overall bar
        )

        ds.close()

        result.update(
            reachable=reachable,
            paths=paths,
            values=values,
            costs=costs,
            reachable_mask=reachable_mask,
            lats=lats,
            lons=lons,
            plevs=plevs,
            i0=i0,
            j0=j0,
            l_bottom=l_bottom,
            error=None,
        )
    except Exception:
        result.update(
            reachable=False,
            paths=None,
            values=None,
            costs=None,
            reachable_mask=None,
            error=traceback.format_exc(),
        )

    return result


# =========================================================================== #
# Per-member output                                                             #
# =========================================================================== #

def _safe_time_hours(times: np.ndarray) -> np.ndarray:
    """Convert an array of numpy datetimes to hours-since-start floats."""
    t0 = times[0]
    delta = (times - t0).astype("timedelta64[s]").astype(float)
    return delta / 3600.0


def plot_member(
    result: Dict,
    member_dir: str,
    start_lat: float,
    start_lon: float,
    logger: logging.Logger,
) -> None:
    """Produce the trajectory map and pressure-profile plot for one member."""

    if not result["reachable"] or result["paths"] is None:
        logger.warning("Member %s: no reachable paths, skipping plots.", result["member"])
        return

    paths    = result["paths"] if isinstance(result["paths"], list) else [result["paths"]]
    lats     = result["lats"]
    lons     = result["lons"]
    plevs    = result["plevs"]
    rmasked  = result["reachable_mask"]

    # ---- best path geometry -------------------------------------------------
    best_path = paths[0]
    i_best = best_path[:, 1]
    j_best = best_path[:, 2]
    lats_best = lats[i_best]
    lons_best = lons[j_best]

    # ---- 1) Map --------------------------------------------------------------
    fig, ax = _make_map_axes(
        float(lons_best.mean()), float(lats_best.mean()), figsize=(8, 7)
    )

    # Shade reachable area
    if rmasked is not None:
        ax.contourf(
            lons, lats, rmasked.astype(float),
            levels=[0.5, 1.5],
            colors=["red"],
            alpha=0.25,
            transform=ccrs.PlateCarree(),
        )

    # All trajectories (best solid, rest faint)
    for idx, p in enumerate(paths):
        lp = lats[p[:, 1]]
        lnp = lons[p[:, 2]]
        ax.plot(lnp, lp, "-", color="k",
                alpha=1.0 if idx == 0 else 0.2,
                linewidth=1.5 if idx == 0 else 0.5,
                transform=ccrs.PlateCarree())

    # Range circle around origin
    if result["values"] is not None:
        vals = result["values"] if isinstance(result["values"], np.ndarray) \
               else np.array([result["values"]])
        max_dist_m = float(vals[0])
        _plot_geodesic_circle(ax, start_lat, start_lon, max_dist_m / 1000.0)

    # Origin marker
    ax.scatter([start_lon], [start_lat], color="tab:blue", marker="o",
               s=60, zorder=8, transform=ccrs.PlateCarree(), label="Origin")

    _set_map_extent(ax, lons_best, lats_best, padding=5.0)
    _add_gridlines(ax)
    ax.set_title(f"Member {result['member']} – top {len(paths)} trajectories")
    fig.tight_layout()
    map_path = os.path.join(member_dir, "reachable_map.png")
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", map_path)

    # ---- 2) Pressure profiles ------------------------------------------------
    try:
        transition_ds = xr.open_dataset(result["transition_path"])
        times_all = transition_ds["time"].values
        transition_ds.close()
    except Exception:
        times_all = None

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for idx, p in enumerate(paths):
        k_arr = p[:, 0]
        l_arr = p[:, 3]
        plev_p = plevs[l_arr]
        if times_all is not None:
            t_hours = _safe_time_hours(times_all[k_arr])
            ax2.plot(t_hours, plev_p, "-", color="k",
                     alpha=1.0 if idx == 0 else 0.2,
                     linewidth=1.5 if idx == 0 else 0.5)
        else:
            ax2.plot(k_arr, plev_p, "-", color="k",
                     alpha=1.0 if idx == 0 else 0.2)

    ax2.set_xlabel("Time after start (h)")
    ax2.set_ylabel("Pressure (hPa)")
    ax2.invert_yaxis()
    ax2.set_title(f"Member {result['member']} – pressure profiles")
    fig2.tight_layout()
    pprof_path = os.path.join(member_dir, "pressure_profiles.png")
    fig2.savefig(pprof_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved %s", pprof_path)


def _plot_geodesic_circle(
    ax,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    n_points: int = 360,
    color: str = "black",
) -> None:
    geod = Geod(ellps="WGS84")
    angles = np.linspace(0, 360, n_points)
    lats_c, lons_c = [], []
    for ang in angles:
        lon, lat, _ = geod.fwd(center_lon, center_lat, ang, radius_km * 1000.0)
        lats_c.append(lat)
        lons_c.append(lon)
    ax.plot(lons_c, lats_c, "--", color=color, linewidth=1.0,
            alpha=0.7, transform=ccrs.PlateCarree(), zorder=6)


# =========================================================================== #
# Zone colours (used by landing-zone helpers and plots)                        #
# =========================================================================== #

_ZONE_COLORS = ["tab:orange", "tab:green", "tab:purple", "tab:red", "tab:cyan"]


# =========================================================================== #
# Private map / plot helpers                                                   #
# Shared by both the pipeline save functions and the notebook plot functions.  #
# Extracted to avoid repetition across plot_member, save_and_plot_landing_zones, #
# compute_and_save_probabilistic, plot_member_from_nc, and                     #
# plot_probabilistic_from_nc.                                                  #
# =========================================================================== #

def _make_map_axes(
    center_lon: float,
    center_lat: float,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[plt.Figure, Any]:
    """Create a Cartopy GeoAxes with RotatedPole projection centred on the data.

    The figure comes pre-loaded with coastlines, ocean, borders, and land
    background features.  Complete setup by calling::

        _set_map_extent(ax, lons, lats)   # fix the visible domain
        _add_gridlines(ax)                # add labelled gridlines
    """
    projection = ccrs.RotatedPole(
        pole_longitude=center_lon - 180.0,
        pole_latitude=90.0 - center_lat,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={"projection": projection})
    ax.coastlines(resolution="50m", alpha=0.6)
    ax.add_feature(cfeature.OCEAN,                   facecolor="lightblue",   alpha=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),                        alpha=0.2)
    ax.add_feature(cfeature.LAND,                    facecolor="lightyellow", alpha=0.3)
    return fig, ax


def _set_map_extent(
    ax,
    lons: np.ndarray,
    lats: np.ndarray,
    padding: float = 0.0,
) -> None:
    """Set the map extent from array bounds with optional padding in degrees."""
    ax.set_extent(
        [float(lons.min()) - padding, float(lons.max()) + padding,
         float(lats.min()) - padding, float(lats.max()) + padding],
        crs=ccrs.PlateCarree(),
    )


def _set_map_extent_from_nonzero(
    ax,
    lons: np.ndarray,
    lats: np.ndarray,
    frac: np.ndarray,
    padding_frac: float = 0.2,
) -> None:
    """Set extent to the bounding box of non-zero *frac* cells plus relative padding.

    Parameters
    ----------
    padding_frac:
        Fraction of the non-zero bounding-box span to add as margin on each
        side (default 0.20 → 20 %).
    """
    mask = frac > 0
    if not mask.any():
        _set_map_extent(ax, lons, lats)
        return
    # lons/lats may be 1-D coordinate vectors; build 2-D grids if needed so
    # boolean indexing with the 2-D mask always works correctly.
    if lons.ndim == 1 and lats.ndim == 1:
        lons_2d, lats_2d = np.meshgrid(lons, lats)
    else:
        lons_2d, lats_2d = lons, lats
    lons_nz = lons_2d[mask]
    lats_nz = lats_2d[mask]
    lon_min, lon_max = float(lons_nz.min()), float(lons_nz.max())
    lat_min, lat_max = float(lats_nz.min()), float(lats_nz.max())
    pad_lon = (lon_max - lon_min) * padding_frac
    pad_lat = (lat_max - lat_min) * padding_frac * np.cos(np.radians((lat_min + lat_max) / 2))
    ax.set_extent(
        [lon_min - pad_lon, lon_max + pad_lon,
         lat_min - pad_lat, lat_max + pad_lat],
        crs=ccrs.PlateCarree(),
    )


def _add_gridlines(ax) -> None:
    """Add the standard labelled gridlines to a GeoAxes."""
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False, linewidth=0.5, color="black",
                      alpha=0.5, linestyle=(0, (15, 10)))
    gl.top_labels   = False
    gl.right_labels = False


def _plot_frac_contourf(
    ax,
    fig: plt.Figure,
    lons: np.ndarray,
    lats: np.ndarray,
    frac: np.ndarray,
    alpha: float = 0.85,
    cbar_label: str = "Fraction of members\nwith grid point reachable",
    mask_zeros: bool = True,
    n_levels: int = 10,
) -> Any:
    """Draw a filled-contour overlay of *frac* and attach a colour bar.

    Parameters
    ----------
    ax, fig
        GeoAxes and parent Figure from ``_make_map_axes``.
    lons, lats
        1-D coordinate arrays matching the shape of *frac*.
    frac
        2-D fraction field, values in [0, 1].
    alpha
        Transparency of the fill; default 0.85.
    cbar_label
        Text for the colour-bar label.
    mask_zeros
        When True (default), grid points where *frac* equals exactly zero are
        masked so the background map shows through.
    n_levels : int
        Number of contour levels.  Pass the ensemble size so that each level
        corresponds to exactly one additional member being reachable
        (levels = 1/N, 2/N, …, 1).  Default 10.

    Returns
    -------
    The ``QuadContourSet`` returned by ``contourf``.
    """
    data = np.ma.masked_where(frac == 0, frac) if mask_zeros else frac
    cf = ax.contourf(lons, lats, data,
                     levels=np.linspace(0, 1, n_levels), cmap=cmc.batlowW_r,
                     transform=ccrs.PlateCarree(),
                     alpha=alpha, extend="neither")
    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.75)
    cbar.set_ticks(np.linspace(0, 1, 11))
    cbar.set_ticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)])
    cbar.set_label(cbar_label, fontsize=9)
    return cf


def _make_histogram_fig(dist_m: np.ndarray) -> Optional[plt.Figure]:
    """Return a histogram figure for per-member maximum distances (m).

    Returns *None* when there are no finite values in *dist_m*.
    """
    finite = dist_m[np.isfinite(dist_m)]
    if finite.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(finite / 1000.0, bins=min(len(finite), 20),
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.nanmedian(dist_m) / 1000.0, color="k", linestyle="--",
               linewidth=1.5, label=f"Median {np.nanmedian(dist_m)/1e3:.0f} km")
    ax.axvline(np.nanmean(dist_m) / 1000.0, color="red", linestyle=":",
               linewidth=1.5, label=f"Mean {np.nanmean(dist_m)/1e3:.0f} km")
    ax.set_xlabel("Max reachable distance (km)")
    ax.set_ylabel("Count")
    ax.set_title("Ensemble spread of maximum reachable distance")
    ax.legend()
    fig.tight_layout()
    return fig


def _make_lz_map(
    zones: List[Dict],
    lons: np.ndarray,
    lats: np.ndarray,
    frac: np.ndarray,
    start_lat: float,
    start_lon: float,
) -> plt.Figure:
    """Return the landing-zones overview map figure.

    Each element of *zones* must contain ``centroid_lat``, ``centroid_lon``,
    ``mean_frac``, and a ``trajectories`` sub-dict with list-of-array entries
    ``lat``, ``lon``, and ``member`` (as returned by
    ``identify_landing_zones``, or built from NetCDF data).
    """
    fig, ax = _make_map_axes(float(lons.mean()), float(lats.mean()))
    _plot_frac_contourf(ax, fig, lons, lats, frac, alpha=0.65,
                        cbar_label="Fraction reachable")

    for z_idx, zone in enumerate(zones):
        color = _ZONE_COLORS[z_idx % len(_ZONE_COLORS)]
        traj  = zone["trajectories"]
        for i in range(len(traj["lat"])):
            ax.plot(traj["lon"][i], traj["lat"][i], "-", color=color,
                    alpha=0.35, linewidth=0.8, transform=ccrs.PlateCarree())
        _plot_geodesic_circle(ax, zone["centroid_lat"], zone["centroid_lon"],
                              200.0, color=color)
        ax.scatter([zone["centroid_lon"]], [zone["centroid_lat"]],
                   color=color, marker="*", s=200, zorder=9,
                   transform=ccrs.PlateCarree(),
                   label=(f"Zone {z_idx}  n={len(traj['member'])}  "
                          f"frac={zone['mean_frac']:.2f}"))

    ax.scatter([start_lon], [start_lat], color="black", marker="o",
               s=80, zorder=10, transform=ccrs.PlateCarree(), label="Origin")
    _set_map_extent(ax, lons, lats)
    ax.legend(loc="lower left", fontsize=8)
    _add_gridlines(ax)
    ax.set_title("Suggested landing zones with matching trajectories", fontsize=11)
    fig.tight_layout()
    return fig


def _make_lz_profiles(zones: List[Dict]) -> plt.Figure:
    """Return the per-zone pressure-profile figure.

    *zones* must be in the same format as described for ``_make_lz_map``.  The
    ``trajectories`` sub-dict must contain ``time_h`` and ``plev`` as lists of
    1-D arrays.
    """
    n = len(zones)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for z_idx, zone in enumerate(zones):
        ax    = axes[z_idx]
        color = _ZONE_COLORS[z_idx % len(_ZONE_COLORS)]
        traj  = zone["trajectories"]
        for i in range(len(traj["time_h"])):
            t = np.asarray(traj["time_h"][i], dtype=float)
            p = np.asarray(traj["plev"][i],   dtype=float)
            ok = np.isfinite(t) & np.isfinite(p)
            ax.plot(t[ok], p[ok], "-", color=color, alpha=0.4, linewidth=0.8)
        ax.set_title(f"Zone {z_idx}\n({len(traj['member'])} traj)", fontsize=9)
        ax.set_xlabel("Time after start (h)")
        if z_idx == 0:
            ax.set_ylabel("Pressure (hPa)")
        ax.invert_yaxis()
    fig.suptitle("Pressure profiles by landing zone", fontsize=11)
    fig.tight_layout()
    return fig


def save_member_netcdf(
    result: Dict,
    member_dir: str,
    start_lat: float,
    start_lon: float,
    logger: logging.Logger,
) -> None:
    """Write reachable-mask and trajectories to NetCDF for one member."""

    rmasked = result["reachable_mask"]
    lats    = result["lats"]
    lons    = result["lons"]
    plevs   = result["plevs"]
    member  = result["member"]

    if rmasked is None:
        logger.warning("Member %s: no reachable mask, skipping NetCDF.", member)
        return

    # -- Reachable mask --------------------------------------------------------
    ds_mask = xr.Dataset(
        {"reachable": xr.DataArray(
            rmasked.astype(np.int8),
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"description": "1 = gridpoint reachable within budget, 0 = not"},
        )},
        attrs={
            "member": member,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "budget": float(result.get("costs", [0])[0]) if result.get("costs") is not None else np.nan,
        },
    )
    mask_nc = os.path.join(member_dir, "reachable_mask.nc")
    ds_mask.to_netcdf(mask_nc)
    logger.info("Saved %s", mask_nc)

    # -- Trajectories ----------------------------------------------------------
    paths = result["paths"]
    if paths is None:
        return
    if not isinstance(paths, list):
        paths = [paths]
    values = result["values"]
    costs  = result["costs"]
    if not isinstance(values, np.ndarray):
        values = np.array([values])
    if not isinstance(costs, np.ndarray):
        costs = np.array([costs])

    try:
        transition_ds = xr.open_dataset(result["transition_path"])
        times_all = transition_ds["time"].values
        transition_ds.close()
    except Exception:
        times_all = None

    # Pad trajectories to a common length with NaN
    max_len = max(len(p) for p in paths)
    n_traj  = len(paths)

    traj_lat    = np.full((n_traj, max_len), np.nan)
    traj_lon    = np.full((n_traj, max_len), np.nan)
    traj_plev   = np.full((n_traj, max_len), np.nan)
    traj_time_h = np.full((n_traj, max_len), np.nan)
    endpoint_lat = np.full(n_traj, np.nan)
    endpoint_lon = np.full(n_traj, np.nan)

    for idx, p in enumerate(paths):
        L = len(p)
        k_arr = p[:, 0]
        i_arr = p[:, 1]
        j_arr = p[:, 2]
        l_arr = p[:, 3]
        traj_lat[idx, :L]  = lats[i_arr]
        traj_lon[idx, :L]  = lons[j_arr]
        traj_plev[idx, :L] = plevs[l_arr]
        if times_all is not None:
            traj_time_h[idx, :L] = _safe_time_hours(times_all[k_arr])
        # Last valid step is the endpoint
        endpoint_lat[idx] = float(lats[i_arr[-1]])
        endpoint_lon[idx] = float(lons[j_arr[-1]])

    step_idx = np.arange(max_len)
    traj_idx = np.arange(n_traj)

    ds_traj = xr.Dataset(
        {
            "lat":    (["trajectory", "step"], traj_lat,  {"units": "degrees_north"}),
            "lon":    (["trajectory", "step"], traj_lon,  {"units": "degrees_east"}),
            "plev":   (["trajectory", "step"], traj_plev, {"units": "hPa"}),
            "time_h": (["trajectory", "step"], traj_time_h,
                       {"units": "hours",
                        "long_name": "hours since trajectory start"}),
            "distance_m": (["trajectory"], values, {"units": "m",
                "description": "Geodesic distance from origin to endpoint"}),
            "cost":   (["trajectory"], costs,  {"description": "Total vert cost used"}),
            "endpoint_lat": (["trajectory"], endpoint_lat,
                             {"units": "degrees_north",
                              "description": "Latitude of trajectory endpoint"}),
            "endpoint_lon": (["trajectory"], endpoint_lon,
                             {"units": "degrees_east",
                              "description": "Longitude of trajectory endpoint"}),
        },
        coords={"trajectory": traj_idx, "step": step_idx},
        attrs={
            "member": member,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "usage": (
                "Select trajectories near a target with: "
                "ds.where((abs(ds.endpoint_lat - target_lat) < dlat) & "
                "(abs(ds.endpoint_lon - target_lon) < dlon), drop=True)"
            ),
        },
    )
    traj_nc = os.path.join(member_dir, "trajectories.nc")
    ds_traj.to_netcdf(traj_nc)
    logger.info("Saved %s", traj_nc)


# =========================================================================== #
# Landing zone identification                                                   #
# =========================================================================== #

def identify_landing_zones(
    frac_reachable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    all_results: List[Dict],
    n_zones: int = 3,
    frac_threshold: float = 0.35,
    zone_radius_km: float = 200.0,
) -> List[Dict]:
    """Identify *n_zones* optimal landing positions from the probabilistic field.

    Algorithm
    ---------
    1. Threshold ``frac_reachable`` at *frac_threshold* and label connected
       components with ``scipy.ndimage.label``.
    2. Score each component: ``mean_fraction × area_km²``.
    3. Greedily select up to *n_zones* components whose centroids are at least
       ``1.5 × zone_radius_km`` apart (avoid near-duplicates).
    4. For each selected zone collect all trajectories from all ensemble members
       whose endpoints fall within *zone_radius_km* of the zone centroid.

    Returns a list of zone dicts (sorted by score descending), each containing
    ``centroid_lat``, ``centroid_lon``, ``area_km2``, ``mean_frac``, ``score``,
    ``mask``, and a ``trajectories`` sub-dict with per-trajectory arrays.
    """
    geod = Geod(ellps="WGS84")

    # Approximate cell area (km²) on a regular lat/lon grid
    dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 1.0
    dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 1.0
    R_earth = 6371.0
    lat_grid = np.broadcast_to(lats[:, np.newaxis], (lats.size, lons.size))
    cell_area_km2 = (
        np.deg2rad(dlat) * R_earth
        * np.deg2rad(dlon) * R_earth
        * np.cos(np.deg2rad(lat_grid))
    )

    binary = frac_reachable >= frac_threshold
    labeled, n_labels = ndi.label(binary)

    if n_labels == 0:
        return []

    # Score every component
    components: List[Dict] = []
    for label in range(1, n_labels + 1):
        mask = labeled == label
        area_km2  = float(cell_area_km2[mask].sum())
        mean_frac = float(frac_reachable[mask].mean())
        score     = area_km2 * mean_frac

        weights   = frac_reachable[mask]
        lat_idx, lon_idx = np.where(mask)
        c_lat = float(np.average(lats[lat_idx], weights=weights))
        c_lon = float(np.average(lons[lon_idx], weights=weights))

        components.append(dict(
            label=label, score=score, area_km2=area_km2,
            mean_frac=mean_frac, centroid_lat=c_lat, centroid_lon=c_lon,
            mask=mask,
        ))

    components.sort(key=lambda c: -c["score"])

    # Greedy selection with minimum-separation constraint
    selected: List[Dict] = []
    for comp in components:
        too_close = any(
            geod.inv(sel["centroid_lon"], sel["centroid_lat"],
                     comp["centroid_lon"], comp["centroid_lat"])[2] / 1000.0
            < zone_radius_km * 1.5
            for sel in selected
        )
        if not too_close:
            selected.append(comp)
        if len(selected) >= n_zones:
            break

    # Collect matching trajectories from all members
    for zone in selected:
        traj: Dict[str, List] = {k: [] for k in
            ("lat", "lon", "plev", "time_h", "distance_m",
             "member", "traj_idx", "endpoint_lat", "endpoint_lon")}

        for r in all_results:
            if r.get("paths") is None or r.get("lats") is None:
                continue
            paths  = r["paths"] if isinstance(r["paths"], list) else [r["paths"]]
            lats_r = r["lats"]
            lons_r = r["lons"]
            plevs_r = r["plevs"]
            vals_r = r["values"] if isinstance(r["values"], np.ndarray) \
                     else np.array([r["values"]])

            try:
                tds = xr.open_dataset(r["transition_path"])
                times_all = tds["time"].values
                tds.close()
            except Exception:
                times_all = None

            for p_idx, p in enumerate(paths):
                i_arr = p[:, 1]; j_arr = p[:, 2]
                l_arr = p[:, 3]; k_arr = p[:, 0]

                end_lat = float(lats_r[i_arr[-1]])
                end_lon = float(lons_r[j_arr[-1]])
                _, _, dist_m = geod.inv(
                    zone["centroid_lon"], zone["centroid_lat"],
                    end_lon, end_lat,
                )
                if dist_m / 1000.0 > zone_radius_km:
                    continue

                t_h = _safe_time_hours(times_all[k_arr]) \
                      if times_all is not None else k_arr.astype(float)

                traj["lat"].append(lats_r[i_arr])
                traj["lon"].append(lons_r[j_arr])
                traj["plev"].append(plevs_r[l_arr])
                traj["time_h"].append(t_h)
                traj["distance_m"].append(
                    float(vals_r[p_idx]) if p_idx < len(vals_r) else np.nan)
                traj["member"].append(r["member"])
                traj["traj_idx"].append(p_idx)
                traj["endpoint_lat"].append(end_lat)
                traj["endpoint_lon"].append(end_lon)

        zone["trajectories"] = traj

    return selected


def save_and_plot_landing_zones(
    zones: List[Dict],
    prob_dir: str,
    start_lat: float,
    start_lon: float,
    frac_reachable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    logger: logging.Logger,
) -> None:
    """Save landing-zone NetCDFs and diagnostic plots."""

    if not zones:
        logger.info("No landing zones identified.")
        return

    n_zones = len(zones)

    # ---- Landing zone metadata NetCDF --------------------------------------
    ds_zones = xr.Dataset(
        {
            "centroid_lat":  (["zone"], [z["centroid_lat"]  for z in zones],
                              {"units": "degrees_north"}),
            "centroid_lon":  (["zone"], [z["centroid_lon"]  for z in zones],
                              {"units": "degrees_east"}),
            "score":         (["zone"], [z["score"]         for z in zones],
                              {"description": "area_km2 × mean_fraction"}),
            "area_km2":      (["zone"], [z["area_km2"]      for z in zones],
                              {"units": "km2"}),
            "mean_fraction": (["zone"], [z["mean_frac"]     for z in zones]),
            "n_trajectories":(["zone"],
                              [len(z["trajectories"]["member"]) for z in zones],
                              {"description": "Total trajectories near this zone"}),
        },
        coords={"zone": np.arange(n_zones)},
        attrs={"start_lat": start_lat, "start_lon": start_lon, "n_zones": n_zones},
    )
    zones_nc = os.path.join(prob_dir, "landing_zones.nc")
    ds_zones.to_netcdf(zones_nc)
    logger.info("Saved %s", zones_nc)

    # ---- Per-zone trajectory NetCDFs ---------------------------------------
    for z_idx, zone in enumerate(zones):
        traj = zone["trajectories"]
        n_t  = len(traj["member"])
        if n_t == 0:
            logger.info("Zone %d: no trajectories, skipping.", z_idx)
            continue

        max_len = max(len(a) for a in traj["lat"])
        t_lat   = np.full((n_t, max_len), np.nan)
        t_lon   = np.full((n_t, max_len), np.nan)
        t_plev  = np.full((n_t, max_len), np.nan)
        t_th    = np.full((n_t, max_len), np.nan)

        for i in range(n_t):
            L = len(traj["lat"][i])
            t_lat[i, :L]  = traj["lat"][i]
            t_lon[i, :L]  = traj["lon"][i]
            t_plev[i, :L] = traj["plev"][i]
            t_th[i, :L]   = traj["time_h"][i]

        ds_z = xr.Dataset(
            {
                "lat":   (["trajectory", "step"], t_lat,  {"units": "degrees_north"}),
                "lon":   (["trajectory", "step"], t_lon,  {"units": "degrees_east"}),
                "plev":  (["trajectory", "step"], t_plev, {"units": "hPa"}),
                "time_h":(["trajectory", "step"], t_th,
                           {"units": "hours", "long_name": "hours since trajectory start"}),
                "distance_m":    (["trajectory"], np.array(traj["distance_m"]),
                                  {"units": "m"}),
                "endpoint_lat":  (["trajectory"], np.array(traj["endpoint_lat"]),
                                  {"units": "degrees_north"}),
                "endpoint_lon":  (["trajectory"], np.array(traj["endpoint_lon"]),
                                  {"units": "degrees_east"}),
                "member":        (["trajectory"], np.array(traj["member"])),
                "traj_idx_in_member": (["trajectory"], np.array(traj["traj_idx"])),
            },
            coords={"trajectory": np.arange(n_t), "step": np.arange(max_len)},
            attrs={
                "zone_index": z_idx,
                "centroid_lat": float(zone["centroid_lat"]),
                "centroid_lon": float(zone["centroid_lon"]),
                "area_km2": float(zone["area_km2"]),
                "mean_fraction": float(zone["mean_frac"]),
                "start_lat": start_lat, "start_lon": start_lon,
            },
        )
        nc_path = os.path.join(prob_dir, f"landing_zone_{z_idx}_trajectories.nc")
        ds_z.to_netcdf(nc_path)
        logger.info("Saved %s  (%d trajectories)", nc_path, n_t)

    # ---- Plot 1: landing-zones map ------------------------------------------
    fig_map = _make_lz_map(zones, lons, lats, frac_reachable, start_lat, start_lon)
    map_path = os.path.join(prob_dir, "landing_zones_map.png")
    fig_map.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig_map)
    logger.info("Saved %s", map_path)

    # ---- Plot 2: per-zone pressure profiles ----------------------------------
    fig_pprof = _make_lz_profiles(zones)
    pprof_path = os.path.join(prob_dir, "landing_zones_pressure_profiles.png")
    fig_pprof.savefig(pprof_path, dpi=150, bbox_inches="tight")
    plt.close(fig_pprof)
    logger.info("Saved %s", pprof_path)


# =========================================================================== #
# Probabilistic output                                                          #
# =========================================================================== #

def compute_and_save_probabilistic(
    all_results: List[Dict],
    prob_dir: str,
    start_lat: float,
    start_lon: float,
    logger: logging.Logger,
) -> None:
    """Aggregate across ensemble members and produce probabilistic outputs."""

    valid = [r for r in all_results if r.get("reachable_mask") is not None]
    if not valid:
        logger.warning("No valid members for probabilistic output.")
        return

    n_members = len(valid)
    lats = valid[0]["lats"]
    lons = valid[0]["lons"]

    # Stack reachable masks and compute fraction
    masks = np.stack([r["reachable_mask"].astype(float) for r in valid], axis=0)
    frac_reachable = masks.mean(axis=0)        # (Nx, Ny)
    count_reachable = masks.sum(axis=0)

    # Best (farthest) trajectory endpoint distance per member
    max_dist_per_member = []
    for r in valid:
        if r["values"] is not None:
            vals = r["values"] if isinstance(r["values"], np.ndarray) \
                   else np.array([r["values"]])
            max_dist_per_member.append(float(vals[0]))
        else:
            max_dist_per_member.append(np.nan)
    max_dist_arr = np.array(max_dist_per_member)

    # ---- Save probabilistic NetCDF ------------------------------------------
    ds_prob = xr.Dataset(
        {
            "reachable_fraction": xr.DataArray(
                frac_reachable,
                dims=["lat", "lon"],
                coords={"lat": lats, "lon": lons},
                attrs={
                    "description": (
                        "Fraction of ensemble members for which "
                        "this grid point is reachable within budget"
                    ),
                    "n_members": n_members,
                },
            ),
            "reachable_count": xr.DataArray(
                count_reachable.astype(np.int16),
                dims=["lat", "lon"],
                coords={"lat": lats, "lon": lons},
                attrs={"description": "Number of members with gridpoint reachable"},
            ),
            "max_distance_m": xr.DataArray(
                max_dist_arr,
                dims=["member"],
                coords={"member": [r["member"] for r in valid]},
                attrs={"units": "m",
                       "description": "Farthest reachable distance per member"},
            ),
        },
        attrs={
            "start_lat": start_lat,
            "start_lon": start_lon,
            "n_members": n_members,
        },
    )
    prob_nc = os.path.join(prob_dir, "probabilistic.nc")
    ds_prob.to_netcdf(prob_nc)
    logger.info("Saved %s", prob_nc)

    # ---- Probabilistic map --------------------------------------------------
    fig, ax = _make_map_axes(float(lons.mean()), float(lats.mean()))
    _plot_frac_contourf(ax, fig, lons, lats, frac_reachable, alpha=0.8,
                        mask_zeros=True, n_levels=n_members)

    # Overlay best trajectories from each member (very faint)
    for r in valid:
        if r["paths"] is None:
            continue
        ps = r["paths"] if isinstance(r["paths"], list) else [r["paths"]]
        ax.plot(r["lons"][ps[0][:, 2]], r["lats"][ps[0][:, 1]], "-",
                color="k", alpha=0.3, linewidth=0.5, transform=ccrs.PlateCarree())

    ax.scatter([start_lon], [start_lat], color="black", marker="o",
               s=80, zorder=8, transform=ccrs.PlateCarree(), label="Origin")
    _set_map_extent_from_nonzero(ax, lons, lats, frac_reachable)
    _add_gridlines(ax)
    ax.set_title(
        f"Probabilistic reachability  –  {n_members} members\n"
        f"Origin: ({start_lat:.3f}°N, {start_lon:.3f}°E)", fontsize=11,
    )
    fig.tight_layout()
    prob_map = os.path.join(prob_dir, "prob_reachable_map.png")
    fig.savefig(prob_map, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", prob_map)

    # ---- Max-range histogram ------------------------------------------------
    fig_hist = _make_histogram_fig(max_dist_arr)
    if fig_hist is not None:
        hist_path = os.path.join(prob_dir, "max_range_histogram.png")
        fig_hist.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close(fig_hist)
        logger.info("Saved %s", hist_path)

    # ---- Summary stats to log -----------------------------------------------
    logger.info("=== Probabilistic Summary ===")
    logger.info("  Members processed : %d", n_members)
    logger.info("  Median max range  : %.0f km", np.nanmedian(max_dist_arr) / 1000.0)
    logger.info("  Mean max range    : %.0f km", np.nanmean(max_dist_arr) / 1000.0)
    logger.info("  Max of maxes      : %.0f km", np.nanmax(max_dist_arr) / 1000.0)
    logger.info("  Frac(>50%% reachable gridpoints) : %.3f",
                float((frac_reachable >= 0.5).mean()))

    # ---- Landing zone identification ----------------------------------------
    logger.info("Identifying landing zones …")
    zones = identify_landing_zones(
        frac_reachable, lats, lons, all_results,
        n_zones=3, frac_threshold=0.35, zone_radius_km=200.0,
    )
    logger.info("Found %d landing zone(s).", len(zones))
    for z_idx, z in enumerate(zones):
        logger.info(
            "  Zone %d: centroid (%.2f°N, %.2f°E)  area=%.0f km²  "
            "mean_frac=%.2f  n_traj=%d",
            z_idx, z["centroid_lat"], z["centroid_lon"],
            z["area_km2"], z["mean_frac"],
            len(z["trajectories"]["member"]),
        )
    save_and_plot_landing_zones(
        zones, prob_dir, start_lat, start_lon,
        frac_reachable, lats, lons, logger,
    )


# =========================================================================== #
# Public plotting helpers — work directly from saved NetCDF files              #
# (these are the functions the inspection notebook should import)              #
# =========================================================================== #

def plot_member_from_nc(
    member_dir: str,
    member: str,
    start_lat: float,
    start_lon: float,
    padding: float = 5.0,
) -> Tuple[plt.Figure, plt.Figure]:
    """Load saved member NetCDFs and return (map_fig, pressure_profile_fig).

    Parameters
    ----------
    member_dir : str
        Path to a ``member_XX`` sub-directory produced by the pipeline.
    member : str
        Member identifier string (used in titles only).
    start_lat, start_lon : float
        Launch site coordinates.
    padding : float
        Map extent padding in degrees beyond the best trajectory's bbox.

    Returns
    -------
    fig_map, fig_pres : matplotlib.figure.Figure
    """
    ds_traj = xr.open_dataset(
        os.path.join(member_dir, "trajectories.nc"), decode_times=False
    )
    ds_mask = xr.open_dataset(os.path.join(member_dir, "reachable_mask.nc"))

    lats_grid = ds_mask["lat"].values
    lons_grid = ds_mask["lon"].values
    reachable = ds_mask["reachable"].values.astype(float)

    lats_traj = ds_traj["lat"].values       # (n_traj, steps)
    lons_traj = ds_traj["lon"].values
    plev_traj = ds_traj["plev"].values
    time_traj = ds_traj["time_h"].values
    dists_m   = ds_traj["distance_m"].values
    n_traj    = lats_traj.shape[0]

    # Best path for projection centre
    lat_best = lats_traj[0][np.isfinite(lats_traj[0])]
    lon_best = lons_traj[0][np.isfinite(lons_traj[0])]

    # ---- Map ----------------------------------------------------------------
    fig_map, ax = _make_map_axes(float(lon_best.mean()), float(lat_best.mean()),
                                 figsize=(8, 7))
    ax.contourf(lons_grid, lats_grid, reachable,
                levels=[0.5, 1.5], colors=["red"], alpha=0.25,
                transform=ccrs.PlateCarree())

    for idx in range(n_traj):
        lp  = lats_traj[idx]
        lnp = lons_traj[idx]
        ax.plot(lnp, lp, "-", color="k",
                alpha=1.0 if idx == 0 else 0.2,
                linewidth=1.5 if idx == 0 else 0.5,
                transform=ccrs.PlateCarree())

    _plot_geodesic_circle(ax, start_lat, start_lon, float(dists_m[0]) / 1000.0)
    ax.scatter([start_lon], [start_lat], color="k", marker="o",
               s=60, zorder=8, transform=ccrs.PlateCarree())

    _set_map_extent(ax, lon_best, lat_best, padding=padding)
    _add_gridlines(ax)
    ax.set_title(
        f"Member {member} – top {n_traj} trajectories  |  "
        f"best range {float(dists_m[0])/1000:.0f} km"
    )
    fig_map.tight_layout()

    # ---- Pressure profiles --------------------------------------------------
    fig_pres, ax2 = plt.subplots(figsize=(7, 4))
    for idx in range(n_traj):
        t = time_traj[idx]
        p = plev_traj[idx]
        ok = np.isfinite(t) & np.isfinite(p)
        ax2.plot(t[ok], p[ok], "-", color="k",
                 alpha=1.0 if idx == 0 else 0.2,
                 linewidth=1.5 if idx == 0 else 0.5)

    ax2.set_xlabel("Time after start (h)")
    ax2.set_ylabel("Pressure (hPa)")
    ax2.invert_yaxis()
    ax2.set_title(f"Member {member} – pressure profiles")
    fig_pres.tight_layout()

    ds_traj.close()
    ds_mask.close()
    return fig_map, fig_pres


def plot_member_by_target(
    member_dir: str,
    member: str,
    target_lat: float,
    target_lon: float,
    radius_km: float = 200.0,
) -> Optional[plt.Figure]:
    """Plot pressure profiles of trajectories ending near a target location.

    Parameters
    ----------
    member_dir : str
        Path to a ``member_XX`` sub-directory.
    target_lat, target_lon : float
        Centre of the target region (decimal degrees).
    radius_km : float
        Radius within which endpoints are selected.

    Returns
    -------
    fig or None if no trajectories match.
    """
    ds_traj = xr.open_dataset(
        os.path.join(member_dir, "trajectories.nc"), decode_times=False
    )
    end_lats = ds_traj["endpoint_lat"].values
    end_lons = ds_traj["endpoint_lon"].values
    geod = Geod(ellps="WGS84")

    selected = [
        i for i in range(len(end_lats))
        if np.isfinite(end_lats[i]) and np.isfinite(end_lons[i])
        and geod.inv(target_lon, target_lat, float(end_lons[i]), float(end_lats[i]))[2] / 1000.0
        <= radius_km
    ]

    if not selected:
        ds_traj.close()
        return None

    time_h = ds_traj["time_h"].values
    plev   = ds_traj["plev"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx in selected:
        t = time_h[idx]; p = plev[idx]
        ok = np.isfinite(t) & np.isfinite(p)
        ax.plot(t[ok], p[ok], "-", color="steelblue", alpha=0.6, linewidth=1.0)

    ax.set_xlabel("Time after start (h)")
    ax.set_ylabel("Pressure (hPa)")
    ax.invert_yaxis()
    ax.set_title(
        f"Member {member} – {len(selected)} traj ending within "
        f"{radius_km:.0f} km of ({target_lat:.2f}°N, {target_lon:.2f}°E)"
    )
    fig.tight_layout()
    ds_traj.close()
    return fig


def plot_probabilistic_from_nc(
    run_dir: str,
    members: List[str],
    start_lat: float,
    start_lon: float,
) -> Dict[str, plt.Figure]:
    """Load probabilistic.nc and produce the standard probabilistic figures.

    Parameters
    ----------
    run_dir : str
        Top-level pipeline run directory (contains ``probabilistic/``).
    members : list of str
        Member IDs in this run (to load best-trajectory overlays).
    start_lat, start_lon : float
        Launch site coordinates.

    Returns
    -------
    dict with keys ``"prob_map"``, ``"histogram"``, ``"landing_zones_map"``,
    ``"landing_zones_profiles"`` (the last two may be absent if no zones file
    exists).
    """
    prob_dir = os.path.join(run_dir, "probabilistic")
    ds_prob  = xr.open_dataset(os.path.join(prob_dir, "probabilistic.nc"))

    frac    = ds_prob["reachable_fraction"].values
    lats    = ds_prob["lat"].values
    lons    = ds_prob["lon"].values
    dist_m  = ds_prob["max_distance_m"].values
    n_mem   = int(ds_prob.attrs.get("n_members", len(dist_m)))

    figures: Dict[str, plt.Figure] = {}

    # ---- Probabilistic map --------------------------------------------------
    fig, ax = _make_map_axes(float(lons.mean()), float(lats.mean()))
    _plot_frac_contourf(ax, fig, lons, lats, frac, alpha=0.85,
                        mask_zeros=True, n_levels=n_mem)

    # Overlay best trajectory from each member
    for m in members:
        traj_nc = os.path.join(run_dir, f"member_{m}", "trajectories.nc")
        if not os.path.exists(traj_nc):
            continue
        with xr.open_dataset(traj_nc, decode_times=False) as dt:
            lp  = dt["lat"].values[0]
            lnp = dt["lon"].values[0]
        ax.plot(lnp, lp, "-", color="k", alpha=0.3, linewidth=0.5,
                transform=ccrs.PlateCarree())

    ax.scatter([start_lon], [start_lat], color="black", marker="o",
               s=80, zorder=8, transform=ccrs.PlateCarree())
    _set_map_extent_from_nonzero(ax, lons, lats, frac)
    _add_gridlines(ax)
    ax.set_title(
        f"Probabilistic reachability  –  {n_mem} members\n"
        f"Origin: ({start_lat:.3f}°N, {start_lon:.3f}°E)", fontsize=11)
    fig.tight_layout()
    figures["prob_map"] = fig

    # ---- Max-range histogram ------------------------------------------------
    fig_hist = _make_histogram_fig(dist_m)
    if fig_hist is not None:
        figures["histogram"] = fig_hist

        print("=== Probabilistic Summary ===")
        for m_id, d in zip(ds_prob["member"].values, dist_m):
            print(f"  member {m_id}: {d/1e3:.0f} km")
        print(f"  Median : {np.nanmedian(dist_m)/1e3:.0f} km")
        print(f"  Mean   : {np.nanmean(dist_m)/1e3:.0f} km")
        print(f"  Max    : {np.nanmax(dist_m)/1e3:.0f} km")
        print(f"  Frac(>=50%) reachable : {float((frac >= 0.5).mean()):.3f}")

    ds_prob.close()

    # ---- Landing zones (if computed) ----------------------------------------
    lz_nc = os.path.join(prob_dir, "landing_zones.nc")
    if os.path.exists(lz_nc):
        ds_lz = xr.open_dataset(lz_nc)
        n_zones = int(ds_lz.dims["zone"])

        # Re-read per-zone trajectory files and normalise into zones format
        zones_for_plot: List[Dict] = []
        for z_idx in range(n_zones):
            zt_nc = os.path.join(prob_dir, f"landing_zone_{z_idx}_trajectories.nc")
            if not os.path.exists(zt_nc):
                continue
            with xr.open_dataset(zt_nc, decode_times=False) as zt:
                n_t   = zt.dims["trajectory"]
                zones_for_plot.append({
                    "centroid_lat": float(ds_lz["centroid_lat"].values[z_idx]),
                    "centroid_lon": float(ds_lz["centroid_lon"].values[z_idx]),
                    "mean_frac":    float(ds_lz["mean_fraction"].values[z_idx]),
                    "trajectories": {
                        "lat":    [zt["lat"].values[i]    for i in range(n_t)],
                        "lon":    [zt["lon"].values[i]    for i in range(n_t)],
                        "plev":   [zt["plev"].values[i]   for i in range(n_t)],
                        "time_h": [zt["time_h"].values[i] for i in range(n_t)],
                        "member": list(range(n_t)),
                    },
                })

        if zones_for_plot:
            figures["landing_zones_map"]      = _make_lz_map(
                zones_for_plot, lons, lats, frac, start_lat, start_lon
            )
            figures["landing_zones_profiles"] = _make_lz_profiles(zones_for_plot)

        ds_lz.close()

    return figures


# =========================================================================== #
# Main entry point                                                              #
# =========================================================================== #

def main(params_path: str) -> None:
    # Switch to non-interactive backend for file-based output on cluster
    matplotlib.use("Agg")

    cfg = load_config(params_path)

    # ---- Create output directory -----------------------------------------------
    out_dir = _output_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    # Save a copy of the params file into the output directory
    shutil.copy(params_path, os.path.join(out_dir, "params.txt"))

    # ---- Set up logging --------------------------------------------------------
    log_path = os.path.join(out_dir, "pipeline.log")
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Balloon Optimizer Pipeline started")
    logger.info("Output directory : %s", out_dir)
    logger.info("Config file      : %s", params_path)
    logger.info("=" * 60)

    # ---- Parse parameters ------------------------------------------------------
    source      = cfg.get("data", "source", fallback="era5")
    start_lat   = cfg.getfloat("origin", "start_lat")
    start_lon   = cfg.getfloat("origin", "start_lon")
    budget      = cfg.getfloat("optimizer", "budget")
    k_best      = cfg.getint("optimizer", "k_best", fallback=30)
    land_only   = cfg.getboolean("optimizer", "land_only", fallback=True)
    early_pen   = cfg.getfloat("optimizer", "early_cost_penalty", fallback=1000.0)
    n_workers   = cfg.getint("run", "n_opt_workers", fallback=16)
    surf_hpa    = cfg.getfloat("pressure_levels", "surface_level_hpa", fallback=1000.0)
    members     = _discover_members(cfg, logger)

    logger.info("Source: %s | Members: %s", source, members)
    logger.info("Origin: (%.4f°N, %.4f°E)", start_lat, start_lon)
    logger.info("Budget: %.1f | k_best: %d | land_only: %s", budget, k_best, land_only)

    # ---- Transition computations (serial – each uses internal parallelism) ----
    transition_dir = os.path.join(out_dir, "transitions")
    os.makedirs(transition_dir, exist_ok=True)

    member_transition_files: Dict[str, Optional[str]] = {}
    for m in members:
        logger.info("--- Transition computation: member %s ---", m)
        nc_path = compute_transition_for_member(cfg, m, transition_dir, logger)
        member_transition_files[m] = nc_path
        if nc_path is None:
            logger.error("Skipping member %s (transition failed).", m)
        else:
            logger.info("Transition file ready: %s", nc_path)

    valid_members = [(m, f) for m, f in member_transition_files.items() if f]
    if not valid_members:
        logger.error("No valid transition files. Aborting.")
        return

    # ---- Parallel optimisation -----------------------------------------------
    opt_args = [
        (
            f, m, start_lat, start_lon, budget, k_best,
            land_only, surf_hpa, early_pen,
        )
        for m, f in valid_members
    ]

    logger.info("Launching optimisation across %d members with %d workers …",
                len(valid_members), min(n_workers, len(valid_members)))

    n_pool = min(n_workers, len(valid_members))
    if n_pool > 1:
        all_results: List[Dict] = []
        with Pool(processes=n_pool) as pool:
            for r in tqdm(
                pool.imap_unordered(_opt_worker, opt_args),
                total=len(opt_args),
                desc="Optimising members",
                unit="member",
            ):
                all_results.append(r)
    else:
        all_results = [
            _opt_worker(a)
            for a in tqdm(opt_args, desc="Optimising members", unit="member")
        ]

    # ---- Per-member saving ---------------------------------------------------
    for r in all_results:
        m = r["member"]
        if r.get("error"):
            logger.error("Optimisation failed for member %s:\n%s", m, r["error"])
            continue
        if not r["reachable"]:
            logger.warning("Member %s: no reachable land points within budget.", m)

        member_dir = os.path.join(out_dir, f"member_{m}")
        os.makedirs(member_dir, exist_ok=True)

        save_member_netcdf(r, member_dir, start_lat, start_lon, logger)
        plot_member(r, member_dir, start_lat, start_lon, logger)

        if r["values"] is not None:
            vals = r["values"] if isinstance(r["values"], np.ndarray) \
                   else np.array([r["values"]])
            logger.info("Member %s: best distance = %.0f km, paths returned = %d",
                        m, float(vals[0]) / 1000.0,
                        len(r["paths"]) if isinstance(r["paths"], list) else 1)

    # ---- Probabilistic output ------------------------------------------------
    prob_dir = os.path.join(out_dir, "probabilistic")
    os.makedirs(prob_dir, exist_ok=True)
    compute_and_save_probabilistic(
        all_results, prob_dir, start_lat, start_lon, logger
    )

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("Results: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balloon trajectory optimizer pipeline"
    )
    parser.add_argument(
        "params",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.txt"),
        help="Path to the configuration file (default: params.txt next to this script)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.params):
        print(f"ERROR: Configuration file not found: {args.params}", file=sys.stderr)
        sys.exit(1)

    main(args.params)

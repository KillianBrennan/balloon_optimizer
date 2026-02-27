"""
computes transition indices and writes transition matrices to NetCDF
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool

DEFAULT_DT = 3600.0  # default timestep in seconds (1 hours, is not tested for timesteps other than 1 hour and migth not work)
DEFAULT_START = "20240914_20"
DEFAULT_END = "20240917_20"
# DEFAULT_START = "20170908_19"
# DEFAULT_END = "20170911_19"
DATA_ROOT = "/home/kbrennan/data/era5/cdf"
IFS_ROOT = "/net/tropo/atmosdyn/eps/ncdf"
OUTPUT_DIR = "/home/kbrennan/data/balloon/transition_matrices"

# Horizontal domain and vertical level range as in test_transition.ipynb
# Base domain box (lon_min, lon_max, lat_min, lat_max)
DOMAIN = (-22.0, 38.0, 30.0, 75.0)
LON_MIN = DOMAIN[0]
LON_MAX = DOMAIN[1]
LAT_MIN = DOMAIN[2]
LAT_MAX = DOMAIN[3]

# Target pressure levels in hPa for the transition matrices
PRES_LEVELS_HPA = np.array(
    [1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0],
    dtype=float,
)


def _read_and_vinterp(ppath: str) -> Optional[tuple[xr.Dataset, xr.DataArray]]:
    """Read a P-file and interpolate U/V to target pressure levels.

    Parameters
    ----------
    ppath : str
        Path to a single P-file (ERA5 or IFS).

    Returns
    -------
    tuple of (ds_uv_plev, ps_da) or None
        ds_uv_plev : xr.Dataset with U, V on ``(time, plev, lat, lon)``.
        ps_da : xr.DataArray with PS on ``(time, lat, lon)`` in Pa.
    """
    if not os.path.exists(ppath):
        return None

    pfile = xr.open_dataset(ppath)

    if "U" not in pfile or "V" not in pfile:
        pfile.close()
        return None

    ds_uv = pfile[["U", "V"]].sel(
        lon=slice(LON_MIN, LON_MAX),
        lat=slice(LAT_MIN, LAT_MAX),
    )

    if "lev" not in ds_uv.dims:
        pfile.close()
        return None

    nlev_file = ds_uv.sizes["lev"]

    hyam = pfile["hyam"].values
    hybm = pfile["hybm"].values
    hyam_use = hyam[-nlev_file:]
    hybm_use = hybm[-nlev_file:]

    ps = (
        pfile["PS"]
        .sel(
            lon=slice(LON_MIN, LON_MAX),
            lat=slice(LAT_MIN, LAT_MAX),
        )
        .load()
        * 100.0  # hPa → Pa
    )

    ps_vals = ps.values
    if ps_vals.ndim == 3:
        ps_2d = ps_vals[0]
    else:
        ps_2d = ps_vals

    plevels_Pa = (
        hyam_use[:, np.newaxis, np.newaxis]
        + hybm_use[:, np.newaxis, np.newaxis] * ps_2d[np.newaxis, :, :]
    )
    plevels_hPa = plevels_Pa / 100.0

    lat_vals = ds_uv["lat"].values
    lon_vals = ds_uv["lon"].values
    time_vals = ds_uv["time"].values

    u_vals = ds_uv["U"].values.squeeze()
    v_vals = ds_uv["V"].values.squeeze()

    nlev_model, nlat, nlon = u_vals.shape
    ntarget = len(PRES_LEVELS_HPA)

    if plevels_hPa[0, 0, 0] > plevels_hPa[-1, 0, 0]:
        plevels_hPa = plevels_hPa[::-1]
        u_vals = u_vals[::-1]
        v_vals = v_vals[::-1]

    u_interp = np.empty((ntarget, nlat, nlon))
    v_interp = np.empty((ntarget, nlat, nlon))

    for j in range(nlat):
        for i in range(nlon):
            p_col = plevels_hPa[:, j, i]
            u_interp[:, j, i] = np.interp(PRES_LEVELS_HPA, p_col, u_vals[:, j, i])
            v_interp[:, j, i] = np.interp(PRES_LEVELS_HPA, p_col, v_vals[:, j, i])

    ds_uv_plev = xr.Dataset(
        {
            "U": (["time", "plev", "lat", "lon"], u_interp[np.newaxis, ...]),
            "V": (["time", "plev", "lat", "lon"], v_interp[np.newaxis, ...]),
        },
        coords={
            "time": time_vals,
            "plev": PRES_LEVELS_HPA,
            "lat": lat_vals,
            "lon": lon_vals,
        },
    )

    pfile.close()
    return ds_uv_plev, ps


def _apply_topo_mask(
    transition_ds: xr.Dataset,
    ps_pa: xr.DataArray,
) -> Optional[xr.Dataset]:
    """Redirect transitions whose destination is below the surface.

    Parameters
    ----------
    transition_ds : xr.Dataset
        Dataset returned by ``compute_transition_indices``.
    ps_pa : xr.DataArray
        Surface pressure in Pa on ``(lat, lon)`` or ``(time, lat, lon)``.
    """
    ps_on_grid = ps_pa.interp(
        lon=transition_ds["lon"],
        lat=transition_ds["lat"],
    ).squeeze()

    if ps_on_grid.ndim == 3 and "time" in ps_on_grid.dims and ps_on_grid.sizes.get(
        "time",
        1,
    ) == 1:
        ps_on_grid = ps_on_grid.isel(time=0)

    if ps_on_grid.ndim != 2:
        return None

    plev_vals_hPa = transition_ds["plev"].values
    ps_vals_pa = ps_on_grid.values * 100.0

    nlev = plev_vals_hPa.shape[0]
    nlat_ps, nlon_ps = ps_vals_pa.shape

    plev_vals_pa = plev_vals_hPa * 100.0
    below_surface_3d = (
        plev_vals_pa[:, np.newaxis, np.newaxis] > ps_vals_pa[np.newaxis, :, :]
    )

    lat_da = transition_ds["lat_idx"]
    lon_da = transition_ds["lon_idx"]

    lat_idx = lat_da.values
    lon_idx = lon_da.values

    if lat_idx.shape[-3] != nlev:
        return None
    if lat_idx.shape[-2] != nlat_ps or lat_idx.shape[-1] != nlon_ps:
        return None

    if "time" in lat_da.dims:
        lat_idx_view = lat_idx
        lon_idx_view = lon_idx
    else:
        lat_idx_view = lat_idx[np.newaxis, ...]
        lon_idx_view = lon_idx[np.newaxis, ...]

    nt_view, nlev_view, nlat, nlon = lat_idx_view.shape
    j_grid, i_grid = np.indices((nlat, nlon))

    for it in range(nt_view):
        for k in range(nlev_view):
            dest_j = lat_idx_view[it, k]
            dest_i = lon_idx_view[it, k]
            invalid = below_surface_3d[k, dest_j, dest_i]
            if not np.any(invalid):
                continue
            lat_idx_view[it, k][invalid] = j_grid[invalid]
            lon_idx_view[it, k][invalid] = i_grid[invalid]

    if "time" not in lat_da.dims:
        lat_idx = lat_idx_view[0]
        lon_idx = lon_idx_view[0]
    else:
        lat_idx = lat_idx_view
        lon_idx = lon_idx_view

    transition_ds["lat_idx"].data = lat_idx
    transition_ds["lon_idx"].data = lon_idx

    return transition_ds


def _compute_single_transition(args: tuple[str, float, int]) -> Optional[xr.Dataset]:
    """Worker: read one P-file, vertically interpolate, compute transitions."""
    ppath, dt, superscale = args
    result = _read_and_vinterp(ppath)
    if result is None:
        return None
    ds_uv_plev, ps_da = result
    ds_single = ds_uv_plev.isel(time=0)
    transition_ds = compute_transition_indices(ds_single, dt=dt, superscale=superscale)
    return _apply_topo_mask(transition_ds, ps_da)


def _compute_transition_from_uv(args) -> Optional[xr.Dataset]:
    """Worker: compute transitions from pre-interpolated U/V on pressure levels."""
    u_arr, v_arr, ps_arr, plev, lat, lon, time_val, dt, superscale = args
    ds_uv = xr.Dataset(
        {
            "U": (["plev", "lat", "lon"], u_arr),
            "V": (["plev", "lat", "lon"], v_arr),
        },
        coords={"plev": plev, "lat": lat, "lon": lon},
    )
    ps_da = xr.DataArray(
        ps_arr, dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    transition_ds = compute_transition_indices(ds_uv, dt=dt, superscale=superscale)
    result = _apply_topo_mask(transition_ds, ps_da)
    if result is not None:
        result = result.expand_dims(time=[time_val])
    return result



def main(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    dt: float = DEFAULT_DT,
    superscale: int = 3,
    data_source: str = "era5",
    ifs_init: Optional[str] = None,
    ifs_member: Optional[str] = None,
) -> None:
    """Compute transition matrices for P files in a given time range.

    Parameters
    ----------
    start : str
        Start date/time ``YYYYMMDD_HH`` (inclusive).
    end : str
        End date/time ``YYYYMMDD_HH`` (inclusive).
    dt : float
        Advection time step in seconds.  Defaults to 3600 s.
    superscale : int
        Horizontal superscaling factor.
    data_source : str
        ``"era5"`` for hourly ERA5 P-files under ``DATA_ROOT``, or
        ``"ifs"`` for 6-hourly IFS ensemble P-files under ``IFS_ROOT``.
    ifs_init : str, optional
        IFS initialisation date, e.g. ``"20260222_00"``.  Required when
        ``data_source="ifs"``.
    ifs_member : str, optional
        IFS ensemble member, e.g. ``"01"``.  Required when
        ``data_source="ifs"``.
    """

    t_start = datetime.strptime(start, "%Y%m%d_%H")
    t_end = datetime.strptime(end, "%Y%m%d_%H")

    datasets: List[xr.Dataset] = []

    if data_source == "era5":
        # ---- ERA5: hourly P-files, one transition per file ----
        nsteps = int((t_end - t_start).total_seconds() // dt) + 1
        task_args: list[tuple[str, float, int]] = []
        for istep in range(nsteps):
            current = t_start + timedelta(hours=istep)
            date_code = current.strftime("%Y%m%d_%H")
            year = current.strftime("%Y")
            month = current.strftime("%m")
            ppath = os.path.join(DATA_ROOT, year, month, f"P{date_code}")
            if os.path.exists(ppath):
                task_args.append((ppath, dt, superscale))

        if not task_args:
            print("No ERA5 P-files found in the requested time range.")
            return

        with Pool(processes=16) as pool:
            for ds in tqdm(
                pool.imap_unordered(_compute_single_transition, task_args),
                total=len(task_args),
                desc="Computing transition matrices (ERA5)",
            ):
                if ds is not None:
                    datasets.append(ds)

    elif data_source == "ifs":
        if ifs_init is None or ifs_member is None:
            raise ValueError(
                "ifs_init and ifs_member are required for data_source='ifs'"
            )

        ifs_dir = os.path.join(IFS_ROOT, ifs_init, ifs_member)

        # ---- IFS: 6-hourly P-files → temporally interpolate to hourly ----
        # 1. Collect 6h files that bracket the requested time range.
        dt_6h = timedelta(hours=6)
        t_first = datetime(
            t_start.year, t_start.month, t_start.day,
            (t_start.hour // 6) * 6,
        )
        # Round end up to next 6h boundary
        if t_end.hour % 6 != 0:
            hours_to_next = 6 - (t_end.hour % 6)
            t_last = t_end.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_next)
        else:
            t_last = t_end

        pfile_paths: list[str] = []
        t_cur = t_first
        while t_cur <= t_last:
            date_code = t_cur.strftime("%Y%m%d_%H")
            ppath = os.path.join(ifs_dir, f"P{date_code}")
            if os.path.exists(ppath):
                pfile_paths.append(ppath)
            t_cur += dt_6h

        if not pfile_paths:
            print("No IFS P-files found in the requested time range.")
            return

        # 2. Read and vertically interpolate each 6h file (parallel).
        uv_ps_pairs: list[tuple[xr.Dataset, xr.DataArray]] = []
        with Pool(processes=min(16, len(pfile_paths))) as pool:
            for result in tqdm(
                pool.imap(_read_and_vinterp, pfile_paths),
                total=len(pfile_paths),
                desc="Reading & vertical interpolation (IFS)",
            ):
                if result is not None:
                    uv_ps_pairs.append(result)

        if not uv_ps_pairs:
            print("No valid IFS P-files could be processed.")
            return

        # 3. Concatenate along time and sort.
        ds_uv_6h = xr.concat(
            [pair[0] for pair in uv_ps_pairs], dim="time",
        ).sortby("time")

        ps_das = [
            pair[1] for pair in uv_ps_pairs if "time" in pair[1].dims
        ]
        if not ps_das:
            print("No valid PS fields available.")
            return
        ps_6h = xr.concat(ps_das, dim="time").sortby("time")

        # 4. Generate hourly timestamps and interpolate U, V, PS.
        hourly_steps = int((t_end - t_start).total_seconds() // 3600) + 1
        hourly_times = np.array(
            [t_start + timedelta(hours=h) for h in range(hourly_steps)],
            dtype="datetime64[ns]",
        )

        print("Interpolating to hourly timesteps …")
        ds_uv_hourly = ds_uv_6h.interp(time=hourly_times)
        ps_hourly = ps_6h.interp(time=hourly_times)

        # 5. Build tasks for each hourly step.
        plev = PRES_LEVELS_HPA
        lat = ds_uv_hourly["lat"].values
        lon = ds_uv_hourly["lon"].values

        hourly_tasks = []
        for it in range(hourly_steps):
            u_arr = ds_uv_hourly["U"].isel(time=it).values
            v_arr = ds_uv_hourly["V"].isel(time=it).values
            ps_arr = ps_hourly.isel(time=it).values
            time_val = hourly_times[it]
            hourly_tasks.append(
                (u_arr, v_arr, ps_arr, plev, lat, lon, time_val, dt, superscale)
            )

        # 6. Compute transitions for each hourly step (parallel).
        with Pool(processes=16) as pool:
            for ds in tqdm(
                pool.imap_unordered(_compute_transition_from_uv, hourly_tasks),
                total=len(hourly_tasks),
                desc="Computing transition matrices (IFS hourly)",
            ):
                if ds is not None:
                    datasets.append(ds)

    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    if not datasets:
        print("No transition datasets were computed.")
        return

    combined = xr.concat(datasets, dim="time").sortby("time")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"_ifs_{ifs_init}_{ifs_member}" if data_source == "ifs" else ""
    outfile = os.path.join(
        OUTPUT_DIR,
        f"transition_indices_{start}_{end}_dt{int(dt)}{tag}.nc",
    )
    combined.to_netcdf(outfile)
    print(f"Wrote {outfile}")


def compute_transition_indices(
    zfile: xr.Dataset,
    dt: float = 3600.0,
    outfile: Optional[str] = None,
    superscale: int = 1,
) -> xr.Dataset:
    """Compute destination grid indices for one advection time step.

    For each horizontal grid point, compute the (lat, lon) index where a parcel
    would arrive after advecting for a time step ``dt`` using the horizontal
    wind components ``U`` (zonal, m s-1) and ``V`` (meridional, m s-1).

    The function assumes a regular lat/lon grid and uses great-circle
    approximations for metres-per-degree. Any additional dimensions (e.g.
    pressure level) are preserved; indices are computed independently at each
    vertical level.

    Parameters
    ----------
    zfile : xr.Dataset
        Dataset containing at least the variables ``U`` and ``V`` with
        coordinates ``lat`` and ``lon``.
    dt : float, optional
        Time step in seconds over which to advect the flow. Defaults to 3600 s
        (one hour).
    superscale : int, optional
        Horizontal superscaling factor. ``1`` uses the native grid, while
        values >1 linearly interpolate U and V onto a finer regular lat/lon
        grid before computing transitions.
    outfile : str, optional
        If provided, the resulting indices dataset is also written to this
        NetCDF file via ``to_netcdf``.

    Returns
    -------
    xr.Dataset
        Dataset with two integer DataArrays ``lat_idx`` and ``lon_idx`` having
        the same dimensions and coordinates as ``U`` / ``V``.
    """

    # Extract wind components
    if "U" not in zfile or "V" not in zfile:
        raise KeyError("Dataset must contain 'U' and 'V' variables")

    u_da: xr.DataArray = zfile["U"]
    v_da: xr.DataArray = zfile["V"]

    # Ensure latitude and longitude coordinates are present
    if "lat" not in u_da.coords or "lon" not in u_da.coords:
        raise KeyError("Wind fields must have 'lat' and 'lon' coordinates")

    if superscale is None:
        superscale = 1
    if superscale < 1:
        raise ValueError("superscale must be an integer >= 1")

    # Optionally superscale the horizontal grid by linearly interpolating
    # U and V onto a finer regular lat/lon grid.
    if superscale > 1:
        lat_orig = u_da["lat"].values
        lon_orig = u_da["lon"].values

        if lat_orig.size < 2 or lon_orig.size < 2:
            raise ValueError(
                "Latitude and longitude coordinates must have length >= 2 to superscale",
            )

        dlat = (lat_orig[1] - lat_orig[0]) / float(superscale)
        dlon = (lon_orig[1] - lon_orig[0]) / float(superscale)

        nlat_new = (lat_orig.size - 1) * superscale + 1
        nlon_new = (lon_orig.size - 1) * superscale + 1

        new_lat = lat_orig[0] + dlat * np.arange(nlat_new)
        new_lon = lon_orig[0] + dlon * np.arange(nlon_new)

        u_da = u_da.interp(lat=new_lat, lon=new_lon)
        v_da = v_da.interp(lat=new_lat, lon=new_lon)

    # Use (possibly superscaled) coordinates from u_da
    lat = u_da["lat"].values
    lon = u_da["lon"].values

    if lat.size < 2 or lon.size < 2:
        raise ValueError("Latitude and longitude coordinates must have length >= 2")

    # Reorder so that horizontal dimensions are the last two: (..., lat, lon)
    other_dims = [d for d in u_da.dims if d not in ("lat", "lon")]
    dims = tuple(other_dims + ["lat", "lon"])
    u = u_da.transpose(*dims).values
    v = v_da.transpose(*dims).values

    # Record original shape to reshape results later
    orig_shape = u.shape
    nlat, nlon = orig_shape[-2], orig_shape[-1]

    # Flatten any leading dimensions for simpler broadcasting
    if u.ndim > 2:
        lead = int(np.prod(orig_shape[:-2]))
        u_flat = u.reshape(lead, nlat, nlon)
        v_flat = v.reshape(lead, nlat, nlon)
    else:
        lead = 1
        u_flat = u.reshape(1, nlat, nlon)
        v_flat = v.reshape(1, nlat, nlon)

    # Grid spacing in radians (keep sign to handle descending latitude grids)
    dlat_rad = np.deg2rad(lat[1] - lat[0])
    dlon_rad = np.deg2rad(lon[1] - lon[0])

    # Earth radius in metres
    R_earth = 6_371_000.0

    # Meridional distance per latitude index (signed for grid orientation)
    dy_per_index = R_earth * dlat_rad  # metres per step in latitude index

    # Zonal distance per longitude index depends on latitude
    lat_rad = np.deg2rad(lat)
    dx_per_index_lat = R_earth * np.cos(lat_rad) * dlon_rad  # (nlat,)

    # Broadcast dx_per_index_lat to full grid shape (lead, nlat, nlon)
    dx_per_index = dx_per_index_lat[np.newaxis, :, np.newaxis]

    # Displacement in metres over time step dt
    disp_x = u_flat * dt  # east-west
    disp_y = v_flat * dt  # north-south

    # Convert physical displacement to index displacement
    with np.errstate(divide="ignore", invalid="ignore"):
        dlat_idx = disp_y / dy_per_index
        dlon_idx = disp_x / dx_per_index

    # Starting indices
    j_idx = np.arange(nlat)[np.newaxis, :, np.newaxis]
    i_idx = np.arange(nlon)[np.newaxis, np.newaxis, :]

    j_new = j_idx + dlat_idx
    i_new = i_idx + dlon_idx

    # Round to nearest grid point and clip to domain
    j_new_rounded = np.rint(j_new).astype(int)
    i_new_rounded = np.rint(i_new).astype(int)

    j_new_rounded = np.clip(j_new_rounded, 0, nlat - 1)
    i_new_rounded = np.clip(i_new_rounded, 0, nlon - 1)

    # Enforce self-pointers for any origin cell that lies on the
    # outermost grid (true boundary nodes), so trajectories
    # cannot leave the domain or slide along its edge.
    if u.ndim > 2:
        lead = int(np.prod(orig_shape[:-2]))
    else:
        lead = 1

    j_orig = np.broadcast_to(
        np.arange(nlat)[np.newaxis, :, np.newaxis],
        (lead, nlat, nlon),
    )
    i_orig = np.broadcast_to(
        np.arange(nlon)[np.newaxis, np.newaxis, :],
        (lead, nlat, nlon),
    )

    boundary_mask = (
        (j_orig == 0)
        | (j_orig == nlat - 1)
        | (i_orig == 0)
        | (i_orig == nlon - 1)
    )

    j_new_rounded[boundary_mask] = j_orig[boundary_mask]
    i_new_rounded[boundary_mask] = i_orig[boundary_mask]

    # Reshape back to original dimensions
    lat_indices = j_new_rounded.reshape(orig_shape)
    lon_indices = i_new_rounded.reshape(orig_shape)

    # Build DataArrays with original dimensions and coordinates
    coords = {d: u_da[d].values for d in dims if d in u_da.coords}

    lat_da = xr.DataArray(
        lat_indices,
        dims=dims,
        coords=coords,
        name="lat_idx",
    )
    lon_da = xr.DataArray(
        lon_indices,
        dims=dims,
        coords=coords,
        name="lon_idx",
    )

    ds = xr.Dataset({"lat_idx": lat_da, "lon_idx": lon_da})
    ds.attrs["dt_seconds"] = float(dt)
    ds.attrs["superscale"] = int(superscale)

    if outfile is not None:
        ds.to_netcdf(outfile)

    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute transition matrices")
    parser.add_argument("--source", default="era5", choices=["era5", "ifs"],
                        help="Data source (default: era5)")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="Start date YYYYMMDD_HH")
    parser.add_argument("--end", default=DEFAULT_END,
                        help="End date YYYYMMDD_HH")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT,
                        help="Advection timestep in seconds (default: 3600)")
    parser.add_argument("--superscale", type=int, default=3,
                        help="Horizontal superscaling factor (default: 3)")
    parser.add_argument("--ifs-init", default=None,
                        help="IFS init date, e.g. 20260222_00")
    parser.add_argument("--ifs-member", default=None,
                        help="IFS ensemble member, e.g. 01")
    args = parser.parse_args()
    main(
        start=args.start,
        end=args.end,
        dt=args.dt,
        superscale=args.superscale,
        data_source=args.source,
        ifs_init=args.ifs_init,
        ifs_member=args.ifs_member,
    )

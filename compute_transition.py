"""
computes transition indices and writes transition matrices to NetCDF
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm

DEFAULT_DT = 3600.0  # default timestep in seconds (1 hours, is not tested for timesteps other than 1 hour and migth not work)
DEFAULT_START = "20240914_20"
DEFAULT_END = "20240917_20"
DATA_ROOT = "/home/kbrennan/data/era5/cdf"
OUTPUT_DIR = "/home/kbrennan/data/balloon/transition_matrices"

# Horizontal domain and vertical level range as in test_transition.ipynb
# Base domain box (lon_min, lon_max, lat_min, lat_max)
DOMAIN = (-25.0, 35.0, 33.0, 72.0)
LON_MIN = DOMAIN[0]
LON_MAX = DOMAIN[1]
LAT_MIN = DOMAIN[2]
LAT_MAX = DOMAIN[3]

# Pressure level limits in Pa (500–900 hPa)
PLEV_MIN = 50_000.0
PLEV_MAX = 90_000.0


def main(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    dt: float = DEFAULT_DT,
    superscale: int = 1,
) -> None:
    """Compute transition matrices for all Z files in a given time range.

    Parameters
    ----------
    start : str, optional
        Start date/time in the format ``YYYYMMDD_HH`` (inclusive).
    end : str, optional
        End date/time in the format ``YYYYMMDD_HH`` (inclusive).
    dt : float, optional
        Advection time step in seconds. Defaults to 3600 s.
    superscale : int, optional
        Horizontal superscaling factor. ``1`` uses the native grid,
        while e.g. ``2`` linearly interpolates U and V onto a grid
        with roughly twice as many points in each horizontal
        direction before computing transitions.
    """

    t_start = datetime.strptime(start, "%Y%m%d_%H")
    t_end = datetime.strptime(end, "%Y%m%d_%H")

    datasets: List[xr.Dataset] = []

    # Number of hourly steps between start and end (inclusive)
    nsteps = int((t_end - t_start).total_seconds() // dt) + 1

    for istep in tqdm(range(nsteps), desc="Computing transition matrices"):
        current = t_start + timedelta(hours=istep)

        date_code = current.strftime("%Y%m%d_%H")
        year = current.strftime("%Y")
        month = current.strftime("%m")
        zpath = os.path.join(DATA_ROOT, year, month, f"Z{date_code}")
        ppath = os.path.join(DATA_ROOT, year, month, f"P{date_code}")

        if not os.path.exists(zpath):
            # Skip missing files rather than aborting the whole range
            continue

        zfile = xr.open_dataset(zpath)
        # Only keep U and V for the transition computation and
        # restrict to the same domain and pressure levels used in
        # test_transition.ipynb
        zfile_uv = zfile[["U", "V"]].sel(
            lon=slice(LON_MIN, LON_MAX),
            lat=slice(LAT_MIN, LAT_MAX),
            plev=slice(PLEV_MIN, PLEV_MAX),
        )

        transition_ds = compute_transition_indices(zfile_uv, dt=dt, superscale=superscale)

        # If a corresponding P-file with PS exists, use it to adjust
        # transitions so that no trajectory can enter grid points
        # below the surface (where p > PS). PS is stored in hPa,
        # while plev is in Pa, so convert PS to Pa before comparing.
        if os.path.exists(ppath):
            pfile = xr.open_dataset(ppath)
            try:
                ps = pfile["PS"].sel(
                    lon=slice(LON_MIN, LON_MAX),
                    lat=slice(LAT_MIN, LAT_MAX),
                )

                # Interpolate PS to the (possibly superscaled) grid
                # used by the transition dataset, then squeeze any
                # singleton dimensions.
                ps = ps.interp(
                    lon=transition_ds["lon"],
                    lat=transition_ds["lat"],
                ).squeeze()

                # Ensure ps is 2D (lat, lon)
                if ps.ndim == 3 and "time" in ps.dims and ps.sizes.get("time", 1) == 1:
                    ps = ps.isel(time=0)

                if ps.ndim != 2:
                    raise ValueError("Unexpected PS dimensions; expected 2D lat/lon")

                plev_vals = transition_ds["plev"].values  # (Nz,)
                # PS is in hPa in the P-files; convert to Pa
                ps_vals_pa = ps.values * 100.0  # (nlat, nlon), on same grid as transitions

                nlev = plev_vals.shape[0]
                nlat_ps, nlon_ps = ps_vals_pa.shape

                # Adjust transition indices so that any transition whose
                # destination would land below topography instead points
                # back to the origin cell (self-pointer). This way we do
                # not need to store an additional validity mask.

                lat_da = transition_ds["lat_idx"]
                lon_da = transition_ds["lon_idx"]

                lat_idx = lat_da.values
                lon_idx = lon_da.values

                # Ensure vertical and horizontal sizes match PS grid
                if lat_idx.shape[-3] != nlev:
                    raise ValueError("Mismatch between plev dimension and PS levels")
                if lat_idx.shape[-2] != nlat_ps or lat_idx.shape[-1] != nlon_ps:
                    raise ValueError("Mismatch between horizontal grid and PS grid")

                # Mask of grid points below the surface for each level
                # Shape: (Nz, nlat, nlon)
                below_surface_3d = (
                    plev_vals[:, np.newaxis, np.newaxis] > ps_vals_pa[np.newaxis, :, :]
                )

                # Work with a unified view where the first dimension is
                # time-like (can be 1 if there is no explicit time dim).
                if "time" in lat_da.dims:
                    lat_idx_view = lat_idx
                    lon_idx_view = lon_idx
                else:
                    lat_idx_view = lat_idx[np.newaxis, ...]
                    lon_idx_view = lon_idx[np.newaxis, ...]

                nt_view, nlev_view, nlat, nlon = lat_idx_view.shape

                # Origin indices for self-pointers
                j_grid, i_grid = np.indices((nlat, nlon))

                # Redirect any transition whose destination is below the
                # surface to point back to its origin cell.
                for it in range(nt_view):
                    for k in range(nlev_view):
                        dest_j = lat_idx_view[it, k]
                        dest_i = lon_idx_view[it, k]

                        # Boolean mask of destinations that are below surface
                        invalid = below_surface_3d[k, dest_j, dest_i]

                        if not np.any(invalid):
                            continue

                        lat_idx_view[it, k][invalid] = j_grid[invalid]
                        lon_idx_view[it, k][invalid] = i_grid[invalid]

                # If there was no explicit time dimension, drop the extra axis
                if "time" not in lat_da.dims:
                    lat_idx = lat_idx_view[0]
                    lon_idx = lon_idx_view[0]
                else:
                    lat_idx = lat_idx_view
                    lon_idx = lon_idx_view

                # Write adjusted indices back into the Dataset
                transition_ds["lat_idx"].data = lat_idx
                transition_ds["lon_idx"].data = lon_idx
            finally:
                pfile.close()

        datasets.append(transition_ds)

    if not datasets:
        return

    # Concatenate along time dimension. This assumes each input file has a
    # single time step and consistent grid/levels.
    combined = xr.concat(datasets, dim="time")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(
        OUTPUT_DIR,
        f"transition_indices_{start}_{end}_dt{int(dt)}.nc",
    )
    combined.to_netcdf(outfile)


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
    main()

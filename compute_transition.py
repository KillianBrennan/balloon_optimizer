'''
computes transition indices and writes transition matrices to NetCDF
'''

import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm

DEFAULT_DT = 2 * 3600.0  # default timestep in seconds (2 hours)
DEFAULT_START = "20240914_20"
DEFAULT_END = "20240917_20"
DATA_ROOT = "/home/kbrennan/data/era5/cdf"
OUTPUT_DIR = "/home/kbrennan/data/balloon/transition_matrices"


def main(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    dt: float = DEFAULT_DT,
) -> None:
    """Compute transition matrices for all Z files in a given time range.

    Parameters
    ----------
    start : str, optional
        Start date/time in the format ``YYYYMMDD_HH`` (inclusive).
    end : str, optional
        End date/time in the format ``YYYYMMDD_HH`` (inclusive).
    dt : float, optional
        Advection time step in seconds. Defaults to 2 hours.
    """

    t_start = datetime.strptime(start, "%Y%m%d_%H")
    t_end = datetime.strptime(end, "%Y%m%d_%H")

    datasets: List[xr.Dataset] = []

    # Number of hourly steps between start and end (inclusive)
    nsteps = int((t_end - t_start).total_seconds() // 3600) + 1

    for istep in tqdm(range(nsteps), desc="Computing transition matrices"):
        current = t_start + timedelta(hours=istep)

        date_code = current.strftime("%Y%m%d_%H")
        year = current.strftime("%Y")
        month = current.strftime("%m")
        zpath = os.path.join(DATA_ROOT, year, month, f"Z{date_code}")

        if not os.path.exists(zpath):
            # Skip missing files rather than aborting the whole range
            continue

        zfile = xr.open_dataset(zpath)
        # only keep U and V for the transition computation
        zfile_uv = zfile[["U", "V"]]

        transition_ds = compute_transition_indices(zfile_uv, dt=dt)
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

    if outfile is not None:
        ds.to_netcdf(outfile)

    return ds

if __name__ == '__main__':
    main()
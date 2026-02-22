'''
computes transition indices
'''

import numpy as np
import xarray as xr

def main():
    '''
    '''
    return

def compute_transition_indices(zfile: xr.Dataset, dt: float = 3600.0) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Integer array of shape ``(2, *U.shape)`` where the first entry holds
        the destination latitude indices and the second the destination
        longitude indices for each grid point.
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
    u = u_da.transpose(*other_dims, "lat", "lon").values
    v = v_da.transpose(*other_dims, "lat", "lon").values

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

    # Stack into a single array: (2, *orig_shape)
    transition_indices = np.stack((lat_indices, lon_indices), axis=0)

    return transition_indices

if __name__ == '__main__':
    main()
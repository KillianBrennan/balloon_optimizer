# Balloon Optimizer

This package implements a simple graph-based optimizer for stratopheric balloon trajectories on top of precomputed ERA5 transition matrices.

The workflow is split into two stages:

1. **Transition matrix generation** from ERA5 wind fields (`compute_transition.py`).
2. **Graph-based optimization and diagnostics** on those transitions (`graph_optimizer.py` and the notebooks).

---

## 1. Transition Matrices

### Source data

The transitions are built from ERA5 Z and P files (U, V, and surface pressure):

- Z files: 3D wind fields `U`, `V` on pressure levels `plev`, with latitude `lat` and longitude `lon`.
- P files: surface pressure `PS` on the same horizontal grid.

Only a subset of the horizontal domain and pressure range is used, as configured in `compute_transition.py`.

### `compute_transition.py`

This script reads ERA5 files over a time range and produces a netCDF file containing two integer fields:

- `lat_idx(time, plev, lat, lon)`  – destination latitude index for each origin cell and time step.
- `lon_idx(time, plev, lat, lon)`  – destination longitude index.

For each time step and level, the script:

- Crops ERA5 to a fixed domain and pressure range (e.g. 500–900 hPa).
- Optionally **superscales** the horizontal grid by interpolating `U` and `V`.
- Converts winds to displacements over a fixed `dt` and maps them to integer index jumps.
- Enforces **topography masking** using surface pressure `PS`:
  - Cells where level pressure exceeds `PS` are turned into **self-pointers** (stay in place).
- Enforces **boundary self-pointers**:
  - All origin cells on the outermost grid rows/columns are forced to point to themselves.

The resulting transition file is what the graph optimizer uses.

---

## 2. Graph Optimizer

The core optimization logic lives in `graph_optimizer.py`.

### Land mask

`build_land_mask(lats, lons) -> np.ndarray[bool]`

- Uses Natural Earth land polygons via cartopy + shapely.
- Returns a boolean mask `land_mask[i, j]` (True over land) on the model grid.
- Shows a progress bar while computing the mask.

### Point-to-point optimization

`optimize_point_to_point(next_i, next_j, origin, target, B, ...)`

Dynamic-programming optimizer on the grid defined by the transition indices.

- **State** at time `k`: `(i, j, l)` (lat index, lon index, level).
- **Transitions**:
  - Horizontal moves: `next_i[k, l_new, i, j]`, `next_j[k, l_new, i, j]`.
  - Vertical moves: restricted to `|l_new - l| ≤ 1` with a configurable vertical cost.
- **Budget** `B`: total consumable cost (e.g. sum of vertical level changes).
- **Objective**: reward = reduction of (grid) distance to the target, accumulated over time.

Key features:

- `allowed_levels` – restricts levels participating in the DP.
- `start_levels` / `target_levels` – enforce that trajectories start and end on specified levels
  (typically the **bottom** level = highest pressure).
- `earliest_arrival` / `fixed_arrival_time` – control when arrival is allowed.
- `k_best` – return the top-k paths ranked by (value desc, cost asc).

Return values (for `k_best == 1`):

- `reachable` (bool)
- `path` – array of shape `(L, 4)` with rows `[k, i, j, l]`.
- `best_value` – objective value.
- `best_cost` – used budget.

### Farthest-reachable optimization

`find_farthest_reachable(next_i, next_j, origin, B, ...)`

Dynamic program to find trajectories that **maximize distance from the origin** under budget.

Differences from the point-to-point case:

- There is **no fixed spatial target**; any grid cell can be an endpoint.
- The objective is the **increase in distance from origin** at each step.
- Endpoints can be restricted by:
  - `land_mask` – only land cells are considered as valid endpoints.
  - `target_levels` – only states on these levels are counted as endpoints
    (typically `[l_bottom]` to enforce arrival at the lowest level).

Distance metric:

- If `lats` and `lons` (1D arrays for the grid) are provided:
  - Uses a `pyproj.Geod(ellps="WGS84")` to compute **geodesic distance** from the origin.
  - The objective is then the final great-circle distance in meters (or km).
- If `lats`/`lons` are omitted, falls back to Euclidean distance in index space.

In addition to paths and values, this function returns a **reachable mask**:

- `reachable_mask[i, j]` is True if there exists at least one state `(k, i, j, l)`
  on any of the `target_levels` with finite value and cost ≤ `B`.
- This is used for plotting “all bottom-level grid points that are reachable within budget”.

Return values (for `k_best == 1`):

- `reachable` (bool)
- `path` – farthest path `[k, i, j, l]`.
- `best_value` – farthest distance from origin (meters if geodesic).
- `best_cost` – used budget.
- `reachable_mask` – 2D boolean array over `(i, j)`.

---

## 3. Full Pipeline (`run_pipeline.py`)

`run_pipeline.py` is the end-to-end orchestrator.  It reads `params.txt`,
computes transitions for every ensemble member, runs the optimizer in parallel,
and writes all outputs to a time-stamped directory.

### Running the pipeline

```bash
python run_pipeline.py [params.txt]
```

If no path is given, the script expects `params.txt` next to itself.

### Output directory structure

```
<base_dir>/<source>_<label>_<start>_<end>_<timestamp>/
│
├── params.txt                   # copy of the config used
├── pipeline.log                 # full run log
│
├── transitions/
│   └── transition_indices_*.nc  # one file per member
│
├── member_<id>/                 # one sub-directory per ensemble member
│   ├── trajectories.nc          # top-k trajectories (lat, lon, plev, time_h, …)
│   ├── reachable_mask.nc        # 2-D boolean reachability mask
│   ├── reachable_map.png        # map plot saved by the pipeline
│   └── pressure_profiles.png
│
└── probabilistic/
    ├── probabilistic.nc             # ensemble-fraction field + per-member ranges
    ├── prob_reachable_map.png       # ensemble fraction map
    ├── max_range_histogram.png
    ├── landing_zones.nc             # zone metadata (centroid, score, …)
    ├── landing_zones_map.png
    ├── landing_zones_pressure_profiles.png
    └── landing_zone_<n>_trajectories.nc   # per-zone trajectory set
```

### Key `params.txt` settings

| Section | Key | Description |
|---------|-----|-------------|
| `[data]` | `source` | `era5` or `ifs` |
| `[data]` | `n_members` | Number of IFS members to process |
| `[origin]` | `start_lat`, `start_lon` | Launch site (°N, °E) |
| `[time]` | `start`, `end` | ISO-8601 range consumed by transition computation |
| `[optimizer]` | `budget` | Vertical-cost budget |
| `[optimizer]` | `k_best` | Top-k trajectories to store per member |
| `[optimizer]` | `land_only` | Restrict endpoints to land cells |
| `[output]` | `base_dir` | Root directory for run output |

### Public plotting API

When working in a notebook you can import `run_pipeline` and call any of the
following functions against an **existing** output directory without re-running
the pipeline:

| Function | Returns | Description |
|----------|---------|-------------|
| `plot_member_from_nc(member_dir, member, start_lat, start_lon)` | `(fig_map, fig_pres)` | Map + pressure profiles for a single member |
| `plot_member_by_target(member_dir, member, target_lat, target_lon, radius_km)` | `fig` or `None` | Profiles of trajectories ending near a target |
| `plot_probabilistic_from_nc(run_dir, members, start_lat, start_lon)` | `dict[str, Figure]` | Fraction map, histogram, landing-zone plots |

See `inspect_pipeline_output.ipynb` for ready-to-run examples.

### Private map/plot helpers

All map rendering logic is centralised in private helpers shared by both the
pipeline save functions and the notebook plot functions:

- `_make_map_axes(center_lon, center_lat, figsize)` – creates a RotatedPole
  figure with coastlines, ocean, land and borders.
- `_set_map_extent(ax, lons, lats, padding)` – sets the visible domain.
- `_add_gridlines(ax)` – adds labelled gridlines with standard style.
- `_plot_frac_contourf(ax, fig, lons, lats, frac, ...)` – draws the filled
  fraction contour with colour bar; masks zero-fraction cells by default.
- `_make_histogram_fig(dist_m)` – returns a max-range histogram figure.
- `_make_lz_map(zones, lons, lats, frac, start_lat, start_lon)` – landing-zones
  overview map figure.
- `_make_lz_profiles(zones)` – per-zone pressure-profile figure.

---

## 4. Notebooks

Two main notebooks demonstrate and debug the workflow:

### `test_transition.ipynb`

- Recomputes a single-step transition matrix from a Z file.
- Provides diagnostics:
  - Histogram of transition distances in grid cells.
  - Index-space arrow plot of transitions with square aspect ratio.
  - Map of self-pointer nodes (e.g. below-surface cells, boundaries).
  - Geographic arrow plot of transition vectors on a PlateCarree map with coastlines.

Useful for verifying:

- Domain and pressure-level cropping.
- Topography masking via surface pressure.
- Boundary self-pointer behavior.
- Basic plausibility of the advective displacements.

### `test_graph_optimizer.ipynb`

- Loads a precomputed transition file.
- Maps chosen start/target lat/lon to grid indices and selects the bottom level.
- Runs `optimize_point_to_point` to obtain top-k trajectories between origin and target:
  - Plots trajectories on a rotated map and their pressure profiles over time.
- Runs `find_farthest_reachable` with:
  - `start_levels=[l_bottom]` and `target_levels=[l_bottom]` (start and end at lowest level).
  - `land_mask` to restrict endpoints to land.
  - `lats`/`lons` so the objective and plots use geodesic distance.
- Visualizes:
  - The farthest bottom-level land trajectory.
  - A shaded mask of all bottom-level grid points reachable within budget.
  - A geodesic circle centered at the launch site with radius equal to the
    farthest endpoint distance.
  - Pressure profile of the farthest trajectory.

### `inspect_pipeline_output.ipynb`

Interactive inspection notebook for a completed pipeline run.  Import
`run_pipeline as rp` and point `RUN_DIR` at an output directory to:

- Print the xarray dataset structure of `probabilistic.nc`.
- Display per-member trajectory maps and pressure profiles.
- Plot trajectories ending near a custom target point.
- Reproduce the ensemble-fraction map, max-range histogram, and
  landing-zone figures using `rp.plot_probabilistic_from_nc`.

All plots are produced from the saved NetCDF files with no re-computation.

---

## 5. Dependencies

Key Python packages:

- `numpy`
- `xarray`
- `matplotlib`
- `cartopy`
- `shapely`
- `tqdm`
- `pyproj`

You also need access to ERA5 Z/P files and appropriate paths configured in
`compute_transition.py`.

A minimal (non-exhaustive) installation example:

```bash
pip install numpy xarray matplotlib cartopy shapely tqdm pyproj
```

---

## 6. Typical workflow

1. **Generate transition matrices**

   - Configure `params.txt` (data source, domain, time range, origin).
   - Either let the pipeline compute them automatically (step 3) or run
     `compute_transition.py` directly.

2. **Inspect transitions (optional)**

   - Open `test_transition.ipynb` and run all cells.

3. **Run the pipeline**

   ```bash
   python run_pipeline.py params.txt
   ```

   This creates a time-stamped directory under `output.base_dir` containing
   all NetCDF files and PNG diagnostics.

4. **Inspect results in the notebook**

   - Open `inspect_pipeline_output.ipynb`, set `RUN_DIR` to the output
     directory, and run all cells to browse every diagnostic figure.

---

## 7. Notes

- The vertical cost and allowed levels are configurable; current defaults
  use `abs(l_new - l_prev)` as cost and allow adjacent-level moves only.
- Boundary and topography masking are enforced in the transition matrices,
  not in the optimizer itself.
- The farthest-reachable search is budget-limited; increasing `B` generally
  increases the reachable region and the maximum distance.

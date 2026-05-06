# ODC-1.2 — Grid-to-points colocation (along-track / drifters)

**Source survey item:** [ocean-data-challenges-survey.md §1.2](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `mod_interp.py` from `2024_DC_SSH_mapping_SWOT_OSE` (and equivalents in the 2023 / 2024c repos).

---

## 1. Motivation

Every notebook in the three ocean-data-challenges repos performs the same
operation: take a gridded reconstruction `ds_maps(time, lat, lon)` and
sample its variables along an along-track satellite product or a list of
drifter trajectories. The interpolated values are then differenced
against the observed values to compute RMSE, PSD, and effective-resolution
scores.

This is the single most-reused function across the three repos and is
the natural input for ODC-1.3 (segmented PSD scores) and ODC-1.5 (drifter
deviation skill). It is also the canonical "grid → points" primitive
listed as a placeholder in
[`xr_toolz.interpolate._src.grid_to_points`](../../src/xr_toolz/interpolate/_src/grid_to_points.py)
("Future home for `SampleAtPoints` / `AlongTrack` primitives.").

The upstream implementation depends on `pyinterp` for the interpolation
engine and reimplements a memory-management chunking loop (`TimeSeries`,
`periods`). Both are unnecessary in xarray: `Dataset.interp` with
advanced (pointwise) indexing performs the colocation natively, and dask
handles chunking automatically.

## 2. User stories

### 2.1 Colocate a gridded SSH map onto SWOT along-track (primary)

> *I have a DUACS L4 reconstruction `ds_maps(time, lat, lon)` and a SWOT
> along-track product `ds_track(num_lines)` carrying `longitude`,
> `latitude`, `time`, `ssha`. I want the reconstruction sampled at every
> along-track point, joined back to the track Dataset, so I can score
> the residual.*

```python
import xarray as xr
from xr_toolz.geo import grid_to_along_track

ds_maps  = xr.open_dataset("duacs_l4.nc")            # (time, lat, lon)
ds_track = xr.open_dataset("swot_along_track.nc")    # (num_lines,)

ds_colocated = grid_to_along_track(
    ds_maps, ds_track,
    method="linear",
    suffix="_interp",
)
# ds_colocated has the original along-track variables plus
# "ssh_interp", "ugos_interp", ... aligned on (num_lines,).
```

### 2.2 Colocate gridded velocities onto drifter trajectories

> *I have a list of Copernicus surface drifter Datasets and want u/v from
> my reconstruction sampled at every drifter fix.*

```python
from xr_toolz.geo import grid_to_drifters

ds_drifters_uv = grid_to_drifters(
    ds_maps, ds_drifters,    # ds_drifters: (time,) with longitude/latitude vars
    method="linear",
    suffix="_interp",
)
```

### 2.3 Generic point sampling (any coord set)

> *I have a multi-var gridded dataset and an arbitrary list of points
> (lon, lat, depth, time). I want every variable sampled at every point.*

```python
from xr_toolz.interpolate import sample_at_points

points = xr.Dataset(
    coords={
        "longitude": ("obs", lon),
        "latitude":  ("obs", lat),
        "depth":     ("obs", z),
        "time":      ("obs", t),
    }
)
ds_pointwise = sample_at_points(ds_maps, points, method="linear")
```

### 2.4 As a Layer-1 Operator inside a Sequential

> *I want colocation to be a configurable, serializable Operator I can
> drop into a `Sequential`.*

```python
from xr_toolz.interpolate import SampleAtPoints
from xr_toolz.core import Sequential

pipeline = Sequential([
    BandpassWavelength(...),         # ODC-1.1
    SampleAtPoints(points=ds_track, suffix="_interp"),
    SegmentedPSDScore(...),          # ODC-1.3 (future)
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| Placeholder module | [`interpolate/_src/grid_to_points.py`](../../src/xr_toolz/interpolate/_src/grid_to_points.py) | Implement |
| Points → grid (sklearn-NN bin) | [`points_to_grid.py`](../../src/xr_toolz/interpolate/_src/points_to_grid.py) | Untouched |
| Grid → grid (`regrid_like`) | [`grid_to_grid.py`](../../src/xr_toolz/interpolate/_src/grid_to_grid.py) | Untouched |
| Pointwise interp engine | — | Use `xr.Dataset.interp` w/ advanced indexing |
| Convenience: along-track | — | **Add** `grid_to_along_track` |
| Convenience: drifters | — | **Add** `grid_to_drifters` |
| Generic primitive | — | **Add** `sample_at_points` |
| `Operator` wrapper | — | **Add** `SampleAtPoints` |

## 4. Design

### 4.1 Why xarray, not pyinterp

The upstream uses `pyinterp` for two reasons:

1. **Interpolation engine** — `xarray.Dataset.interp(...)` with advanced
   (pointwise) indexing is the native equivalent. When all interp
   coordinates share the same output dim, xarray performs pointwise
   (not orthogonal) interpolation, calling
   `scipy.interpolate.RegularGridInterpolator` under the hood. Methods
   available: `nearest`, `linear`, `cubic`, `quadratic`, `slinear`,
   `splinef2d`, `pchip`, `akima`.
2. **Memory chunking via `periods`** — `pyinterp` materializes grids,
   forcing a manual sliding window. xarray's `.interp` is lazy when
   maps are dask-backed, so chunking happens automatically along the
   gridded dimensions.

We therefore drop `pyinterp` from the proposal. It remains a viable
*optional* backend (faster on huge datasets, true IDW / bicubic) but
duck-typed: a future `backend="pyinterp"` kwarg can dispatch to a
pyinterp-aware path without making it a hard dependency.

### 4.2 Tier B — generic primitive

```python
# src/xr_toolz/interpolate/_src/grid_to_points.py
def sample_at_points(
    ds: xr.Dataset,
    points: xr.Dataset | xr.DataArray | pd.DataFrame | Mapping[str, ArrayLike],
    *,
    coords: Mapping[str, str] | None = None,
    method: str = "linear",
    point_dim: str = "obs",
    suffix: str | None = None,
    keep_coords: bool = True,
) -> xr.Dataset:
    """Sample every variable in ``ds`` at the given pointwise coordinates.

    Parameters
    ----------
    ds
        Source Dataset on a regular grid. Variables that depend on a
        coord listed in ``coords`` (default: any coord shared between
        ``ds`` and ``points``) are sampled; others pass through.
    points
        Pointwise coordinate locations. Coerced to a Dataset with all
        coordinate variables aligned on ``point_dim``. Accepted forms:
        ``xr.Dataset`` (uses every variable named in ``coords``),
        ``xr.DataArray`` (must carry coord variables along its single
        dim), ``pd.DataFrame`` (uses columns), ``Mapping`` (each entry
        is a 1-D array; all must share length).
    coords
        Map-side coord → point-side variable/column. ``None`` (default)
        auto-detects names matching between ``ds.coords`` and
        ``points``.
    method
        Forwarded to ``xr.Dataset.interp``.
    point_dim
        Name of the output point dimension.
    suffix
        If set, sampled variables get this suffix (e.g.
        ``"ssh"`` → ``"ssh_interp"``). ``None`` keeps original names.
    keep_coords
        If True, copy the point coords through to the output Dataset.
    """
```

Implementation core — once `points` is normalized to a Dataset with
1-D variables on `point_dim`:

```python
interp_kwargs = {
    grid_coord: xr.DataArray(points[track_coord].values, dims=point_dim)
    for grid_coord, track_coord in coords.items()
}
sampled = ds.interp(**interp_kwargs, method=method)
if suffix:
    sampled = sampled.rename({v: f"{v}{suffix}" for v in sampled.data_vars})
if keep_coords:
    sampled = sampled.assign_coords({c: points[c] for c in coords.values()})
return sampled
```

That's the entire engine. The work is in the input-coercion helper.

### 4.3 Domain wrappers

```python
# src/xr_toolz/geo/_src/along_track.py — same module as ODC-1.1 utilities
def grid_to_along_track(
    ds_maps: xr.Dataset,
    ds_track: xr.Dataset,
    *,
    method: str = "linear",
    lon: str = "longitude",
    lat: str = "latitude",
    time: str = "time",
    suffix: str = "_interp",
) -> xr.Dataset:
    """Colocate gridded maps onto an along-track Dataset.

    Returns ``ds_track`` with sampled variables joined as new data_vars.
    """

def grid_to_drifters(
    ds_maps: xr.Dataset,
    ds_drifters: xr.Dataset | Sequence[xr.Dataset],
    *,
    method: str = "linear",
    lon: str = "longitude",
    lat: str = "latitude",
    time: str = "time",
    suffix: str = "_interp",
    drifter_dim: str = "trajectory",
) -> xr.Dataset:
    """Colocate gridded maps onto drifter fixes.

    Accepts a single Dataset or a list (concatenated on the point dim)."""
```

Both delegate to `sample_at_points` with `coords={lon: lon, lat: lat,
time: time}`. The drifter wrapper additionally supports a list-of-Datasets
input.

The choice to put these in `geo/_src/along_track.py` (alongside the
ODC-1.1 helpers `median_dx_km` and `bandpass_wavelength`) keeps all
along-track-specific conveniences in one place.

### 4.4 Layer-1 Operator

```python
# src/xr_toolz/interpolate/operators.py
class SampleAtPoints(Operator):
    """Sample gridded variables at a fixed set of point locations."""

    def __init__(
        self, *,
        points: xr.Dataset | xr.DataArray | pd.DataFrame | Mapping,
        coords: Mapping[str, str] | None = None,
        method: str = "linear",
        point_dim: str = "obs",
        suffix: str | None = None,
        keep_coords: bool = True,
    ): ...

    def __call__(self, ds: xr.Dataset) -> xr.Dataset: ...
    def get_config(self) -> dict: ...
    def __repr__(self) -> str: ...
```

The `points` Dataset is captured at construction time (matches the
upstream `TimeSeries(ds_maps)` initialization pattern but inverted —
points are the fixed object, the gridded source flows through). For
serialization, `get_config()` emits a NetCDF-encodable representation
of the points Dataset (or a path, if loaded from disk).

## 5. Library leverage

| Need | Library |
|---|---|
| Pointwise multi-D interpolation | `xarray.Dataset.interp` with advanced indexing (built-in) |
| Underlying interp engine | `scipy.interpolate.RegularGridInterpolator` (transitive via xarray) |
| DataFrame coercion | `pandas.DataFrame.to_xarray` |
| Lazy chunking | dask (already a transitive dep via xarray) |
| Geodesic distance for QC checks | `pyproj.Geod` (already a dep) |

No new dependencies. No `pyinterp`.

## 6. API summary (public surface)

```python
# Generic primitive
xr_toolz.interpolate.sample_at_points(ds, points, *, coords, method,
                                      point_dim, suffix, keep_coords)

# Domain wrappers
xr_toolz.geo.grid_to_along_track(ds_maps, ds_track, *, method, lon, lat, time, suffix)
xr_toolz.geo.grid_to_drifters(ds_maps, ds_drifters, *, method, lon, lat, time,
                              suffix, drifter_dim)

# Operator
xr_toolz.interpolate.SampleAtPoints(...)
```

## 7. Tests

| Test | Asserts |
|---|---|
| Pointwise interp on analytic field `f(x,y,t) = x + y + t` | Sampled values match analytic within tol |
| `method="nearest"` at grid-centre points | Identity recovery |
| Out-of-domain points | NaN, no exception |
| DataFrame / dict / DataArray / Dataset inputs | All produce identical sampled Datasets |
| Multi-var Dataset | Every data_var present in output |
| `suffix` renaming | `"ssh"` → `"ssh_interp"`; non-suffix vars untouched |
| Dask-backed maps stay lazy | Output is dask-backed; `.data.compute()` matches eager |
| Coord auto-detection | Common-name coords detected without explicit `coords=` |
| `coords` override | Explicit map applies even when names differ |
| `grid_to_along_track` end-to-end | Sampled values added to track Dataset, point coords preserved |
| `grid_to_drifters` list input | List of drifters concatenated on `drifter_dim` |
| `SampleAtPoints` round-trip | `get_config` → reconstructed Operator produces identical output |

Target: ~12 new test cases.

## 8. Out of scope

- **Optional `pyinterp` backend** — duck-typed dispatch can be added
  later if we need true IDW / bicubic on huge grids. Not blocking.
- **Memory chunking iterator** — relies on dask + xarray laziness; no
  hand-rolled `periods`/`TimeSeries` machinery.
- **Vendor-specific drifter reformatting** (`reformat_drifter_dataset`)
  — that's a Copernicus-specific I/O concern, not a library primitive.
  Lives in user code or `xr_toolz.data`.
- **Vector-field colocation helper** — `interpolate_current` (u, v
  together) is just `sample_at_points` over a 2-var Dataset; no special
  API needed.

## 9. Effort

≈80 LOC implementation + ≈100 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `sample_at_points` + input coercion | 50 |
| `grid_to_along_track`, `grid_to_drifters` | 25 |
| `SampleAtPoints` operator | 20 |
| Tests | ~100 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **`xr.Dataset.interp` time handling.** `datetime64` coords are
   supported but require `points`'s time variable to be the same
   numeric/dt encoding. Document and add a coercion step
   (`np.asarray(t).astype(maps_time.dtype)`).
2. **Method dispatch on extrapolation.** `xr.interp` extrapolates
   `linear` by default but returns NaN for `cubic` outside the bounds.
   We document the inconsistency and provide `bounds_error=False` /
   `fill_value=np.nan` as kwargs forwarded through.
3. **Operator config for `points`.** Capturing a Dataset in
   `get_config()` is awkward — Datasets aren't JSON-serializable. We'll
   emit a NetCDF representation by default and accept a `points_path`
   alternative for from-disk reconstruction. Mirrors how
   `xr_toolz.data` operators serialize.
4. **Naming overlap with existing `ds_maps` variables.** If
   `ds_track` already has a `ssh` variable and `ds_maps` does too, a
   merge collides. Default `suffix="_interp"` avoids this; document the
   default.
5. **Choice of placement for wrappers** — `geo/_src/along_track.py`
   already hosts ODC-1.1's `median_dx_km` / `bandpass_wavelength`.
   Putting `grid_to_along_track` and `grid_to_drifters` there keeps
   along-track-specific helpers in one place. The generic
   `sample_at_points` stays in `interpolate.grid_to_points` since it's
   domain-agnostic.

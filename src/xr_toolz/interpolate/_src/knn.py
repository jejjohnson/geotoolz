"""k-nearest-neighbour inverse-distance interpolation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from xr_toolz.interpolate._src.binning import Grid
from xr_toolz.interpolate._src.spatial import _build_tree, _to_metric_xy
from xr_toolz.utils._src.finite import _finite_filter, _finite_mask
from xr_toolz.utils._src.validation import _validate_idw_args


Metric = Literal["euclidean", "haversine"]


def _idw_kernel(
    tree: Any,
    src_values: np.ndarray,
    queries: np.ndarray,
    k: int,
    power: float,
    max_distance: float | None,
    eps: float,
) -> np.ndarray:
    """Evaluate the inverse-distance weighted mean at query coordinates."""
    if queries.size == 0:
        return np.empty(0, dtype=float)

    k_eff = min(k, len(src_values))
    distances, indices = tree.query(queries, k=k_eff)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    valid = (
        distances <= max_distance
        if max_distance is not None
        else np.ones_like(distances, dtype=bool)
    )
    values = src_values[indices]
    exact = (distances == 0.0) & valid

    safe_distances = np.where(valid, distances + eps, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(valid, 1.0 / safe_distances**power, 0.0)
        numerator = np.sum(weights * values, axis=1)
        denominator = np.sum(weights, axis=1)
        out = np.where(denominator > 0.0, numerator / denominator, np.nan)

    if exact.any():
        first_exact = exact.argmax(axis=1)
        rows = np.arange(values.shape[0])
        out = np.where(exact.any(axis=1), values[rows, first_exact], out)

    return out


def _prepare_sources(
    lons: ArrayLike,
    lats: ArrayLike,
    values: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten, size-check, and finite-filter scattered source arrays."""
    lon_arr = np.ravel(np.asarray(lons, dtype=float))
    lat_arr = np.ravel(np.asarray(lats, dtype=float))
    value_arr = np.ravel(np.asarray(values, dtype=float))
    if not (lon_arr.size == lat_arr.size == value_arr.size):
        raise ValueError("lons, lats, and values must have the same size")

    lon_arr, lat_arr, value_arr = _finite_filter(lon_arr, lat_arr, value_arr)
    if lon_arr.size == 0:
        raise ValueError("IDW interpolation requires at least one finite source")

    return lon_arr, lat_arr, value_arr


def idw_to_points(
    src_lons: ArrayLike,
    src_lats: ArrayLike,
    src_values: ArrayLike,
    dst_lons: ArrayLike,
    dst_lats: ArrayLike,
    *,
    k: int = 8,
    power: float = 2.0,
    metric: Metric = "euclidean",
    max_distance: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Inverse-distance interpolate scattered samples onto target points.

    Args:
        src_lons: Source longitudes.
        src_lats: Source latitudes.
        src_values: Source values.
        dst_lons: Target longitudes.
        dst_lats: Target latitudes.
        k: Number of nearest finite source neighbours to use.
        power: Inverse-distance exponent. ``0`` gives an unweighted kNN mean.
        metric: ``"euclidean"`` for degree-space distances or ``"haversine"``
            for great-circle distances.
        max_distance: Optional neighbour cutoff. For ``metric="haversine"``,
            this is interpreted in radians (approximately ``km / 6371``).
        eps: Small non-negative offset for non-exact distance weights.

    Returns:
        NumPy array matching the broadcast shape of ``dst_lons`` and
        ``dst_lats``.
    """
    _validate_idw_args(k, power, metric, max_distance, eps)
    src_lon, src_lat, src_value = _prepare_sources(src_lons, src_lats, src_values)
    dst_lon, dst_lat = np.broadcast_arrays(
        np.asarray(dst_lons, dtype=float),
        np.asarray(dst_lats, dtype=float),
    )

    out = np.full(dst_lon.shape, np.nan, dtype=float)
    finite_queries = _finite_mask(dst_lon, dst_lat)
    if not finite_queries.any():
        return out

    src_xy = _to_metric_xy(src_lon, src_lat, metric)
    query_xy = _to_metric_xy(dst_lon[finite_queries], dst_lat[finite_queries], metric)
    tree = _build_tree(src_xy, metric)
    out[finite_queries] = _idw_kernel(
        tree,
        src_value,
        query_xy,
        k,
        power,
        max_distance,
        eps,
    )
    return out


def idw_to_grid(
    lons: ArrayLike,
    lats: ArrayLike,
    values: ArrayLike,
    grid: Grid,
    *,
    k: int = 8,
    power: float = 2.0,
    metric: Metric = "euclidean",
    max_distance: float | None = None,
    eps: float = 1e-12,
) -> xr.DataArray:
    """Inverse-distance interpolate scattered points onto ``grid``.

    For ``metric="haversine"``, ``max_distance`` is interpreted in radians
    (for example, ``50 / 6371`` for approximately 50 km).

    Args:
        lons: Source longitudes.
        lats: Source latitudes.
        values: Source values.
        grid: Target regular lon/lat grid.
        k: Number of nearest finite source neighbours to use.
        power: Inverse-distance exponent. ``0`` gives an unweighted kNN mean.
        metric: ``"euclidean"`` for degree-space distances or ``"haversine"``
            for great-circle distances.
        max_distance: Optional neighbour cutoff. For ``metric="haversine"``,
            this is interpreted in radians.
        eps: Small non-negative offset for non-exact distance weights.

    Returns:
        DataArray with ``("lat", "lon")`` dimensions on ``grid``.
    """
    lon_grid, lat_grid = np.meshgrid(grid.lon, grid.lat, indexing="xy")
    out = idw_to_points(
        lons,
        lats,
        values,
        lon_grid,
        lat_grid,
        k=k,
        power=power,
        metric=metric,
        max_distance=max_distance,
        eps=eps,
    )
    return xr.DataArray(
        out,
        dims=("lat", "lon"),
        coords={"lat": grid.lat, "lon": grid.lon},
    )


def fillnan_idw(
    da: xr.DataArray,
    *,
    lon: str = "lon",
    lat: str = "lat",
    k: int = 8,
    power: float = 2.0,
    metric: Metric = "euclidean",
    max_distance: float | None = None,
    eps: float = 1e-12,
) -> xr.DataArray:
    """Fill NaNs in a 2-D lon/lat field by IDW from finite neighbours.

    Operates slice-by-slice along any leading dimensions. For
    ``metric="haversine"``, coordinate values are interpreted as lon/lat
    degrees and converted to radians internally; ``max_distance`` is in
    radians. For ``metric="euclidean"``, distances are measured in array-index
    space (column/row positions), not coordinate-value space.
    This differs from :func:`idw_to_points` and :func:`idw_to_grid`, whose
    Euclidean mode uses source and target lon/lat coordinate values directly.

    Args:
        da: Input DataArray with ``lat`` and ``lon`` dimensions.
        lon: Name of the longitude coordinate and dimension.
        lat: Name of the latitude coordinate and dimension.
        k: Number of nearest finite neighbours to use.
        power: Inverse-distance exponent. ``0`` gives an unweighted kNN mean.
        metric: ``"euclidean"`` for array-index distances or ``"haversine"``
            for great-circle distances from coordinate values.
        max_distance: Optional neighbour cutoff. For ``metric="haversine"``,
            this is interpreted in radians; for ``"euclidean"``, in index
            units.
        eps: Small non-negative offset for non-exact distance weights.

    Returns:
        Same-shaped DataArray with NaNs filled where finite neighbours are
        available.
    """
    _validate_idw_args(k, power, metric, max_distance, eps)
    lon_coord = np.asarray(da.coords[lon].values, dtype=float)
    lat_coord = np.asarray(da.coords[lat].values, dtype=float)

    if metric == "haversine":
        lon_grid, lat_grid = np.meshgrid(lon_coord, lat_coord, indexing="xy")
    else:
        rows, cols = np.indices((lat_coord.size, lon_coord.size))

    def _fill_slice(arr: np.ndarray) -> np.ndarray:
        finite = _finite_mask(arr)
        if finite.all() or not finite.any():
            return arr

        if metric == "haversine":
            src_xy = _to_metric_xy(lon_grid[finite], lat_grid[finite], "haversine")
            query_xy = _to_metric_xy(
                lon_grid[~finite],
                lat_grid[~finite],
                "haversine",
            )
            tree = _build_tree(src_xy, "haversine")
        else:
            src_xy = np.column_stack([cols[finite], rows[finite]])
            query_xy = np.column_stack([cols[~finite], rows[~finite]])
            tree = _build_tree(src_xy, "euclidean")

        filled = _idw_kernel(
            tree,
            arr[finite].astype(float, copy=False),
            query_xy,
            k,
            power,
            max_distance,
            eps,
        )
        out = arr.copy()
        out[~finite] = filled
        return out

    return xr.apply_ufunc(
        _fill_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )


__all__ = ["fillnan_idw", "idw_to_grid", "idw_to_points"]

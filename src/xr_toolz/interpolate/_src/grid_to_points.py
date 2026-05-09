"""Grid-to-points value resampling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


Method = Literal["linear", "nearest", "slinear", "cubic", "quintic"]


def _as_numeric_axis(values: np.ndarray, *, name: str) -> np.ndarray:
    """Convert source coordinate values to a finite numeric interpolation axis."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        if np.any(np.isnat(arr)):
            raise ValueError(f"coord {name!r} contains non-finite values")
        return arr.astype("datetime64[ns]").astype(np.int64)
    try:
        numeric = arr.astype(np.float64)
    except (TypeError, ValueError) as err:
        raise ValueError(f"coord {name!r} must be numeric or datetime-like") from err
    if np.any(~np.isfinite(numeric)):
        raise ValueError(f"coord {name!r} contains non-finite values")
    return numeric


def _as_numeric_points(
    values: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert target point values to numeric values plus a finite-value mask."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        valid = ~np.isnat(arr)
        numeric = arr.astype("datetime64[ns]").astype(np.int64)
        return numeric, valid
    try:
        numeric = arr.astype(np.float64)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"points variable {name!r} must be numeric or datetime-like"
        ) from err
    return numeric, np.isfinite(numeric)


def _normalize_axis(da: xr.DataArray, name: str) -> tuple[xr.DataArray, np.ndarray]:
    """Validate a 1-D coordinate axis and return it in ascending order."""
    if name not in da.dims:
        raise ValueError(f"da is missing required coord dim {name!r}")
    if name not in da.coords:
        raise ValueError(f"da is missing required coord {name!r}")
    if da[name].dims != (name,):
        raise ValueError(f"coord {name!r} must be 1-D on its matching dimension")

    axis = _as_numeric_axis(da[name].values, name=name)
    if axis.size < 2:
        raise ValueError(f"coord {name!r} must have length >= 2")

    diffs = np.diff(axis)
    if np.all(diffs > 0):
        return da, axis
    if np.all(diffs < 0):
        reversed_da = da.isel({name: slice(None, None, -1)})
        return reversed_da, axis[::-1]
    raise ValueError(f"coord {name!r} must be strictly monotone")


def _normalize_points(
    points: xr.Dataset | Mapping[str, object],
    *,
    coords: tuple[str, ...],
    point_dim: str,
) -> xr.Dataset:
    """Normalize point mappings or datasets to a validated point dataset."""
    if isinstance(points, Mapping):
        arrays = {name: np.asarray(points[name]) for name in coords if name in points}
        lengths = {arr.shape[0] for arr in arrays.values() if arr.ndim == 1}
        if len(arrays) != len(coords):
            missing = [name for name in coords if name not in points]
            raise ValueError(f"points is missing required variables: {missing}")
        if any(arr.ndim != 1 for arr in arrays.values()):
            raise ValueError("points mapping values must be 1-D arrays")
        if len(lengths) != 1:
            raise ValueError("points mapping values must all have the same length")
        return xr.Dataset(
            {name: ((point_dim,), values) for name, values in arrays.items()}
        )

    if not isinstance(points, xr.Dataset):
        raise TypeError("points must be an xarray Dataset or a mapping")

    for name in coords:
        if name not in points:
            raise ValueError(f"points is missing variable {name!r}")
        if points[name].dims != (point_dim,):
            raise ValueError(f"points variable {name!r} must be 1-D on {point_dim!r}")
    return points


def _collect_point_coordinates(
    points: xr.Dataset,
    coords: tuple[str, ...],
    point_dim: str,
    *,
    retained_dims: tuple[str, ...] = (),
) -> dict[str, tuple[str, np.ndarray]]:
    """Collect output coordinates that are defined along the point dimension.

    Coordinates whose names collide with retained source dimensions
    (``retained_dims``) are skipped to avoid attaching an alternate
    1-D coord on ``point_dim`` that conflicts with an existing
    output dimension coord.
    """
    skip = set(retained_dims)
    assigned: dict[str, tuple[str, np.ndarray]] = {}
    for name in coords:
        if name in skip:
            continue
        assigned[name] = (point_dim, points[name].values)
    for name, coord in points.coords.items():
        if (
            isinstance(name, str)
            and coord.dims == (point_dim,)
            and name not in assigned
            and name not in skip
        ):
            assigned[name] = (point_dim, coord.values)
    return assigned


def _check_dask_chunks(da: xr.DataArray, coords: tuple[str, ...]) -> None:
    """Raise if dask chunks split any interpolation dimension."""
    if da.chunks is None:
        return
    chunked = [
        dim for dim in coords if dim in da.chunksizes and len(da.chunksizes[dim]) > 1
    ]
    if chunked:
        raise ValueError(
            "sample_at_points requires interpolation dimensions to be contained "
            f"in a single chunk; rechunk dimensions {chunked!r} before calling."
        )


def sample_at_points(
    da: xr.DataArray,
    points: xr.Dataset | Mapping[str, object],
    *,
    coords: tuple[str, ...] = ("lat", "lon"),
    method: Method = "linear",
    fill_value: float | None = np.nan,
    bounds_error: bool = False,
    point_dim: str = "points",
) -> xr.DataArray:
    """Sample a gridded field at scattered target points using RGI.

    Args:
        da: Source data with 1-D, strictly monotone coordinates for every
            interpolation dimension in ``coords``.
        points: Target locations as a dataset or mapping. Each coordinate in
            ``coords`` must be a 1-D variable on ``point_dim``.
        coords: Source/target coordinate names to interpolate over.
        method: Interpolation method supported by
            :class:`scipy.interpolate.RegularGridInterpolator`.
        fill_value: Value used for finite out-of-bounds points when
            ``bounds_error`` is false. ``None`` enables scipy extrapolation.
        bounds_error: Whether finite out-of-bounds points raise an error.
        point_dim: Name of the target-point dimension.

    Returns:
        Values sampled at ``points`` with all non-interpolated source
        dimensions retained and ``point_dim`` appended.
    """
    if not coords:
        raise ValueError("coords must contain at least one interpolation dimension")

    coords = tuple(coords)
    points_ds = _normalize_points(points, coords=coords, point_dim=point_dim)
    _check_dask_chunks(da, coords)

    grid_axes: list[np.ndarray] = []
    normalized = da
    for name in coords:
        normalized, axis = _normalize_axis(normalized, name)
        grid_axes.append(axis)

    point_columns: list[np.ndarray] = []
    valid = np.ones(points_ds.sizes[point_dim], dtype=bool)
    for name in coords:
        numeric, finite = _as_numeric_points(points_ds[name].values, name=name)
        point_columns.append(numeric)
        valid &= finite
    interp_pts = np.column_stack(point_columns)

    if np.issubdtype(normalized.dtype, np.floating) or np.issubdtype(
        normalized.dtype,
        np.complexfloating,
    ):
        result_dtype = normalized.dtype
    else:
        result_dtype = np.dtype(np.float64)

    def _interp_block(arr: np.ndarray) -> np.ndarray:
        out = np.full(interp_pts.shape[0], np.nan, dtype=result_dtype)
        if not np.any(valid):
            return out
        rgi = RegularGridInterpolator(
            tuple(grid_axes),
            arr,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
        out[valid] = rgi(interp_pts[valid])
        return out

    out = xr.apply_ufunc(
        _interp_block,
        normalized,
        input_core_dims=[list(coords)],
        output_core_dims=[[point_dim]],
        exclude_dims=set(coords),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[result_dtype],
        dask_gufunc_kwargs={
            "output_sizes": {point_dim: interp_pts.shape[0]},
            "allow_rechunk": False,
        },
    )
    # `out.dims` keeps the source dims that are NOT being interpolated over;
    # skip those names when collecting per-point coords so we don't try to
    # attach a 1-D point coord on top of an existing output dim coord.
    retained = tuple(str(d) for d in out.dims if d != point_dim)
    return out.assign_coords(
        _collect_point_coordinates(
            points_ds,
            coords,
            point_dim,
            retained_dims=retained,
        )
    )


def along_track(
    da: xr.DataArray,
    track: xr.Dataset,
    *,
    coords: tuple[str, ...] = ("time", "lat", "lon"),
    method: Method = "linear",
    fill_value: float | None = np.nan,
    point_dim: str = "points",
) -> xr.DataArray:
    """Collocate a gridded field onto an observation track.

    This is a convenience wrapper around :func:`sample_at_points` for
    time-collocated tracks such as satellite altimetry, drifters, or buoys.
    It validates that datetime-like coordinates are datetime-like in both the
    source and target before delegating to the generic sampler.

    Args:
        da: Source gridded field with 1-D coordinates for every name in
            ``coords``.
        track: Track dataset containing each coordinate in ``coords`` as a
            1-D variable on ``point_dim``.
        coords: Coordinate names to interpolate over. Defaults to
            ``("time", "lat", "lon")``.
        method: Interpolation method supported by
            :class:`scipy.interpolate.RegularGridInterpolator`.
        fill_value: Value used for finite out-of-bounds track points.
        point_dim: Name of the target-track dimension.

    Returns:
        Values from ``da`` collocated onto ``track`` along ``point_dim``.
    """
    for name in coords:
        # Validate presence first so the user gets a clear ValueError instead
        # of a bare KeyError from da[name] / track[name] indexing.
        if name not in da.coords and name not in da.dims:
            raise ValueError(f"da is missing coord {name!r}")
        if name not in track:
            raise ValueError(f"track is missing variable {name!r}")
        source_is_time = np.issubdtype(da[name].values.dtype, np.datetime64)
        target_is_time = np.issubdtype(
            track[name].values.dtype,
            np.datetime64,
        )
        if source_is_time != target_is_time:
            raise ValueError(
                f"coord {name!r} must be datetime-like in both da and track, "
                "or numeric in both"
            )
    return sample_at_points(
        da,
        track,
        coords=coords,
        method=method,
        fill_value=fill_value,
        bounds_error=False,
        point_dim=point_dim,
    )


__all__ = ["along_track", "sample_at_points"]

"""Spatial, temporal, and variable subsetting."""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr


def subset_bbox(
    ds: xr.Dataset,
    lon_bnds: tuple[float, float],
    lat_bnds: tuple[float, float],
    lon: str = "lon",
    lat: str = "lat",
) -> xr.Dataset:
    """Keep points whose ``(lon, lat)`` fall inside a bounding box.

    Works with both 1-D rectilinear grids (where ``sel`` with slices would
    also work) and 2-D lon/lat coordinate arrays, because it masks via
    ``where(..., drop=True)``.

    Args:
        ds: Input dataset.
        lon_bnds: ``(lon_min, lon_max)``.
        lat_bnds: ``(lat_min, lat_max)``.
        lon: Name of the longitude coordinate.
        lat: Name of the latitude coordinate.

    Returns:
        Dataset restricted to the bounding box.
    """
    lon_min, lon_max = lon_bnds
    lat_min, lat_max = lat_bnds
    mask = (
        (ds[lon] >= lon_min)
        & (ds[lon] <= lon_max)
        & (ds[lat] >= lat_min)
        & (ds[lat] <= lat_max)
    )
    return ds.where(mask, drop=True)


def subset_where(
    ds: xr.Dataset,
    variable: str,
    min_val: float,
    max_val: float,
    drop: bool = True,
) -> xr.Dataset:
    """Keep points where ``variable`` is in ``[min_val, max_val]``.

    Args:
        ds: Input dataset.
        variable: Name of the variable used to build the mask.
        min_val: Lower bound (inclusive).
        max_val: Upper bound (inclusive).
        drop: Whether to drop cells outside the range (default ``True``);
            if ``False``, masked cells are replaced with NaN.

    Returns:
        Dataset with cells outside the range masked or dropped.
    """
    mask = (ds[variable] >= float(min_val)) & (ds[variable] <= float(max_val))
    return ds.where(mask, drop=drop)


def subset_time(
    ds: xr.Dataset,
    time_min,
    time_max,
    time: str = "time",
) -> xr.Dataset:
    """Restrict a dataset to the time range ``[time_min, time_max]``.

    Args:
        ds: Input dataset.
        time_min: Lower bound, any type xarray accepts in a ``slice``
            (``str``, ``numpy.datetime64``, ``pandas.Timestamp``, ...).
        time_max: Upper bound, same types as ``time_min``.
        time: Name of the time coordinate.

    Returns:
        Dataset sliced along the time dimension.
    """
    return ds.sel({time: slice(time_min, time_max)})


def select_variables(
    ds: xr.Dataset,
    variables: str | Sequence[str],
) -> xr.Dataset:
    """Return a dataset with only the requested variables.

    Args:
        ds: Input dataset.
        variables: A single variable name or a sequence of names.

    Returns:
        Dataset restricted to ``variables``.
    """
    names = [variables] if isinstance(variables, str) else list(variables)
    return ds.drop_vars([name for name in ds.data_vars if name not in names])

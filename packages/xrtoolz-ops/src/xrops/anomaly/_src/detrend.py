"""Climatology and anomaly primitives.

Uses the split-object pattern: ``calculate_climatology`` returns a
climatology as a pure data object, and ``remove_climatology`` /
``add_climatology`` apply it. This keeps every step
``Dataset → Dataset`` so pipelines stay uniform — state is computed
explicitly upstream, not hidden inside a fit/transform duality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from diagnostix.climate import Climatology, smooth_climatology

from xrops._src.calendar import DAYS_OF_YEAR, day_of_year


CLIMATOLOGY_DIMS: dict[str, str] = {
    "day": "dayofyear",
    "month": "month",
    "year": "year",
}

SEASONS: dict[int, str] = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def calculate_climatology(
    data: xr.DataArray | xr.Dataset,
    freq: str = "day",
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Compute the climatology by grouping ``time`` at a given frequency.

    Args:
        data: Input DataArray or Dataset with a ``time`` coordinate.
        freq: One of ``"day"``, ``"month"``, ``"year"``. A day is a date on a
            leap-year calendar, so 29 February is 60 and 1 March is 61 in
            every year (``diagnostix.climate.calendar_day``); a calendar
            without leap days has its own day of the year.
        time: Name of the time coordinate (default ``"time"``).

    Returns:
        Climatology, same container type as ``data``, with ``time``
        replaced by the grouping dimension
        (``dayofyear`` / ``month`` / ``year``).
    """
    return _grouped(data, _climatology_dim(freq), time).mean(time)


def calculate_climatology_smoothed(
    data: xr.DataArray | xr.Dataset,
    window: int = 61,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Daily climatology smoothed with a running mean around the year.

    The window is centred on its day and wraps at the end of the year, and a
    day counts as often as it was sampled, so 29 February counts for a quarter
    of its neighbours (``diagnostix.climate.smooth_climatology``).

    Args:
        data: Input DataArray or Dataset with a ``time`` coordinate.
        window: Length of the running window in days, odd, so that it has a
            middle.
        time: Name of the time coordinate.

    Returns:
        Smoothed climatology on every day of the year, NaN where the whole
        window has no data.

    Raises:
        ValueError: If ``window`` is not a positive odd integer of at most the
            year.
    """
    if window <= 0 or window % 2 != 1:
        raise ValueError(f"window must be a positive odd integer, got {window}.")
    if isinstance(data, xr.Dataset):
        return data.map(
            lambda da: (
                calculate_climatology_smoothed(da, window=window, time=time)
                if time in da.dims
                else da
            )
        )
    days = DAYS_OF_YEAR.get(data[time].dt.calendar, 366)
    if window > days:
        raise ValueError(f"window must be at most the year of {days}, got {window}.")
    groups = _grouped(data, "dayofyear", time)
    year = {"dayofyear": np.arange(1, days + 1)}
    mean = groups.mean(time).reindex(year)
    count = groups.count(time).reindex(year, fill_value=0)

    def smooth(mean_: Any, count_: Any) -> Any:
        return smooth_climatology(Climatology(mean_, count_, days), window).mean

    out = xr.apply_ufunc(
        smooth,
        mean,
        count,
        input_core_dims=[["dayofyear"]] * 2,
        output_core_dims=[["dayofyear"]],
        dask="parallelized",
        output_dtypes=[mean.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    return out.transpose(*mean.dims).rename(data.name)


def calculate_climatology_season(
    data: xr.DataArray | xr.Dataset,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Climatology grouped by meteorological season (DJF/MAM/JJA/SON).

    Args:
        data: Input DataArray or Dataset with a ``time`` coordinate.
        time: Name of the time coordinate.

    Returns:
        Climatology with ``season`` as the grouping dimension.
    """
    season = data[time].dt.month.copy(
        data=[SEASONS[int(m)] for m in data[time].dt.month.values]
    )
    with_season = data.assign_coords(season=(time, season.values))
    return with_season.groupby("season").mean(time)


def remove_climatology(
    data: xr.DataArray | xr.Dataset,
    climatology: xr.DataArray | xr.Dataset,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Subtract a precomputed climatology from ``data``.

    The grouping frequency is inferred from ``climatology``: it must
    carry exactly one of the dimensions ``dayofyear``, ``month``, or
    ``year``.

    Args:
        data: Input with a ``time`` coordinate.
        climatology: Output of :func:`calculate_climatology` or friends.
        time: Name of the time coordinate.

    Returns:
        Anomalies ``data - climatology``.
    """
    dim = _infer_climatology_dim(climatology)
    return _grouped(data, dim, time) - climatology


def add_climatology(
    data: xr.DataArray | xr.Dataset,
    climatology: xr.DataArray | xr.Dataset,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Add a climatology back onto ``data`` (inverse of
    :func:`remove_climatology`).

    Args:
        data: Anomaly field with a ``time`` coordinate.
        climatology: Climatology computed via :func:`calculate_climatology`.
        time: Name of the time coordinate.

    Returns:
        Reconstructed field ``data + climatology``.
    """
    dim = _infer_climatology_dim(climatology)
    return _grouped(data, dim, time) + climatology


def calculate_anomaly(
    data: xr.DataArray | xr.Dataset,
    freq: str = "day",
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Convenience: compute climatology at ``freq`` and subtract it."""
    clim = calculate_climatology(data, freq=freq, time=time)
    return remove_climatology(data, clim, time=time)


def calculate_anomaly_smoothed(
    data: xr.DataArray | xr.Dataset,
    window: int = 61,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Convenience: smoothed daily climatology then subtract it."""
    clim = calculate_climatology_smoothed(data, window=window, time=time)
    return remove_climatology(data, clim, time=time)


def remove_mean(
    data: xr.DataArray | xr.Dataset,
    dims: str | tuple[str, ...],
) -> xr.DataArray | xr.Dataset:
    """Subtract the mean over ``dims`` (per-variable, NaN-aware).

    Useful for cheap anomaly fields when no climatology is available
    (e.g. spatial mean removal per timestep, or zonal-mean removal).
    Distinct from :func:`calculate_anomaly`, which subtracts a
    time-grouped climatology.
    """
    dim_list = [dims] if isinstance(dims, str) else list(dims)
    return data - data.mean(dim=dim_list)


def _grouped(data: xr.DataArray | xr.Dataset, dim: str, time: str) -> Any:
    """``data`` grouped along ``time`` by month, year, or calendar date."""
    if dim == "dayofyear":
        return data.groupby(day_of_year(data[time]))
    return data.groupby(f"{time}.{dim}")


def _climatology_dim(freq: str) -> str:
    if freq not in CLIMATOLOGY_DIMS:
        raise ValueError(
            f"freq must be one of {sorted(CLIMATOLOGY_DIMS)}, got {freq!r}."
        )
    return CLIMATOLOGY_DIMS[freq]


def _infer_climatology_dim(climatology: Any) -> str:
    candidates = set(CLIMATOLOGY_DIMS.values())
    present = list(candidates & set(climatology.dims))
    if len(present) != 1:
        raise ValueError(
            "climatology must have exactly one of the grouping dimensions "
            f"{sorted(candidates)}, got dims={list(climatology.dims)}."
        )
    return present[0]

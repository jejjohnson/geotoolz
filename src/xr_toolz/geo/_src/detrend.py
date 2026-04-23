"""Climatology and anomaly primitives.

Uses the split-object pattern: ``calculate_climatology`` returns a
climatology as a pure data object, and ``remove_climatology`` /
``add_climatology`` apply it. This keeps every step
``Dataset → Dataset`` so pipelines stay uniform — state is computed
explicitly upstream, not hidden inside a fit/transform duality.
"""

from __future__ import annotations

from typing import Any

import xarray as xr


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
        freq: One of ``"day"``, ``"month"``, ``"year"``. Maps to
            xarray's ``time.dayofyear``, ``time.month``, ``time.year``.
        time: Name of the time coordinate (default ``"time"``).

    Returns:
        Climatology, same container type as ``data``, with ``time``
        replaced by the grouping dimension
        (``dayofyear`` / ``month`` / ``year``).
    """
    dim = _climatology_dim(freq)
    return data.groupby(f"{time}.{dim}").mean(time)


def calculate_climatology_smoothed(
    data: xr.DataArray | xr.Dataset,
    window: int = 60,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Daily climatology smoothed with a circular rolling window.

    Pads the climatology using wrap mode so the smoothing doesn't bias
    the January / December edges, then applies a centered rolling mean.

    Args:
        data: Input DataArray or Dataset with a ``time`` coordinate.
        window: Length of the rolling window in days (must be even so
            that the half-window ``window // 2`` padding is symmetric).
        time: Name of the time coordinate.

    Returns:
        Smoothed day-of-year climatology.
    """
    if window <= 0 or window % 2 != 0:
        raise ValueError(f"window must be a positive even integer, got {window}.")

    pad = window // 2
    clim = calculate_climatology(data, freq="day", time=time)
    clim = clim.pad(dayofyear=(pad, pad), mode="wrap")
    clim = clim.rolling(dayofyear=window, center=True, min_periods=1).mean()
    return clim.isel(dayofyear=slice(pad, -pad))


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
    return data.groupby(f"{time}.{dim}") - climatology


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
    return data.groupby(f"{time}.{dim}") + climatology


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
    window: int = 60,
    time: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Convenience: smoothed daily climatology then subtract it."""
    clim = calculate_climatology_smoothed(data, window=window, time=time)
    return remove_climatology(data, clim, time=time)


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

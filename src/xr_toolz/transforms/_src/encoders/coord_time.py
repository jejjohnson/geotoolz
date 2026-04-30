"""Coordinate-time encoders — rescaling and periodic time-component encodings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from xr_toolz.transforms._src.encoders.basis import cyclical_encode


def time_rescale(
    ds: xr.Dataset,
    freq_dt: float = 1.0,
    freq_unit: str = "s",
    t0: str | np.datetime64 | None = None,
    time: str = "time",
) -> xr.Dataset:
    """Rescale a ``datetime64`` time axis to a float offset.

    ``t' = (t - t_0) / freq_dt``, where ``freq_dt`` is expressed in
    ``freq_unit``. Stores ``t0``, ``freq``, and ``units`` as attrs so
    that :func:`time_unrescale` can round-trip back to ``datetime64``.

    Args:
        ds: Input dataset with a ``time`` coordinate.
        freq_dt: Size of the time step in ``freq_unit`` units.
        freq_unit: Pandas timedelta unit (``"s"``, ``"m"``, ``"h"``,
            ``"D"``, ...).
        t0: Reference time. Defaults to ``ds[time].min()``.
        time: Name of the time coordinate.

    Returns:
        Copy of ``ds`` with ``time`` replaced by a ``float32`` offset.
    """
    ds = ds.copy()
    delta = pd.Timedelta(freq_dt, unit=freq_unit)
    delta_ns = np.int64(delta.asm8.astype("timedelta64[ns]").astype(np.int64))

    if t0 is None:
        t0_val = np.datetime64(ds[time].min().values, "ns")
    else:
        t0_val = np.datetime64(t0, "ns")

    td_ns = (
        (ds[time].values.astype("datetime64[ns]") - t0_val)
        .astype("timedelta64[ns]")
        .astype(np.int64)
    )
    rescaled = (td_ns.astype(np.float64) / float(delta_ns)).astype(np.float32)
    ds = ds.assign_coords({time: rescaled})
    ds[time].attrs.update(
        units=freq_unit,
        freq=float(freq_dt),
        t0=str(t0_val),
    )
    return ds


def time_unrescale(ds: xr.Dataset, time: str = "time") -> xr.Dataset:
    """Inverse of :func:`time_rescale` using its stored attrs."""
    ds = ds.copy()
    attrs = ds[time].attrs
    if "t0" not in attrs or "freq" not in attrs or "units" not in attrs:
        raise ValueError(
            "time coord is missing t0/freq/units attrs — rescale with "
            "time_rescale first."
        )
    delta = pd.Timedelta(attrs["freq"], unit=attrs["units"])
    t0 = np.datetime64(attrs["t0"], "ns")
    delta_ns = np.int64(delta.asm8.astype("timedelta64[ns]").astype(np.int64))
    offsets = (ds[time].values.astype(np.float64) * delta_ns).astype("timedelta64[ns]")
    ds = ds.assign_coords({time: t0 + offsets})
    ds[time].attrs = {}
    return ds


def encode_time_cyclical(
    ds: xr.Dataset,
    components: Sequence[str] = ("dayofyear", "hour"),
    time: str = "time",
) -> xr.Dataset:
    """Attach sin/cos encodings of datetime components as new variables.

    For each ``component`` in ``components`` (any name accepted by
    xarray's ``.dt`` accessor), two coordinates are added:
    ``{component}_sin`` and ``{component}_cos``.

    Args:
        ds: Input dataset with a ``time`` coordinate.
        components: Iterable of datetime attributes to encode.
        time: Name of the time coordinate.

    Returns:
        Dataset with the requested encodings attached.
    """
    periods = {
        "dayofyear": 366.0,
        "day": 31.0,
        "month": 12.0,
        "hour": 24.0,
        "minute": 60.0,
        "second": 60.0,
        "weekday": 7.0,
    }
    ds = ds.copy()
    for name in components:
        if name not in periods:
            raise ValueError(
                f"Unknown time component {name!r}; known: {sorted(periods)}."
            )
        values = getattr(ds[time].dt, name).values.astype(float)
        sin, cos = cyclical_encode(values, period=periods[name])
        ds = ds.assign_coords({f"{name}_sin": (time, sin), f"{name}_cos": (time, cos)})
    return ds


def encode_time_ordinal(
    ds: xr.Dataset,
    reference_date: str | np.datetime64 | None = None,
    time: str = "time",
    unit: str = "D",
) -> xr.Dataset:
    """Attach an ordinal float-day encoding of the time coordinate.

    Adds a ``{time}_ordinal`` coord to ``ds``.
    """
    ref = (
        np.datetime64(ds[time].min().values)
        if reference_date is None
        else np.datetime64(reference_date)
    )
    delta = pd.Timedelta(1, unit=unit)
    ordinal = ((ds[time].values - ref) / delta).astype(np.float64)
    return ds.assign_coords({f"{time}_ordinal": (time, ordinal)})


__all__ = [
    "encode_time_cyclical",
    "encode_time_ordinal",
    "time_rescale",
    "time_unrescale",
]

"""Coordinate-time encoders — rescaling and periodic time-component encodings.

Per the PR γ primitive-flip (``docs/design/xarray-native-primitives.md``),
the Layer-0 primitives in this module take a single positional
``DataArray`` (the time coordinate / variable) and return either a
``DataArray`` (``time_rescale``, ``time_unrescale``,
``encode_time_ordinal``) or a ``Dataset`` of derived variables
(``encode_time_cyclical``). The Dataset assignment / merge lives in the
Layer-1 ``TimeRescale``, ``TimeUnrescale``, ``EncodeTimeCyclical``,
``EncodeTimeOrdinal`` operators in :mod:`xrtoolz.transforms.operators`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from xrtoolz.transforms._src.encoders.basis import cyclical_encode


def time_rescale(
    time: xr.DataArray,
    *,
    freq_dt: float = 1.0,
    freq_unit: str = "s",
    t0: str | np.datetime64 | None = None,
) -> xr.DataArray:
    """Rescale a ``datetime64`` time axis to a float offset.

    ``t' = (t - t_0) / freq_dt``, where ``freq_dt`` is expressed in
    ``freq_unit``. Stores ``t0``, ``freq``, and ``units`` as attrs so
    that :func:`time_unrescale` can round-trip back to ``datetime64``.

    Args:
        time: Time coordinate / variable to rescale.
        freq_dt: Size of the time step in ``freq_unit`` units.
        freq_unit: Pandas timedelta unit (``"s"``, ``"m"``, ``"h"``,
            ``"D"``, ...).
        t0: Reference time. Defaults to ``time.min()``.

    Returns:
        DataArray of ``float32`` offsets, with ``units`` / ``freq`` /
        ``t0`` attrs attached for :func:`time_unrescale`.
    """
    delta = pd.Timedelta(freq_dt, unit=freq_unit)
    delta_ns = np.int64(delta.asm8.astype("timedelta64[ns]").astype(np.int64))

    if t0 is None:
        t0_val = np.datetime64(time.min().values, "ns")
    else:
        t0_val = np.datetime64(t0, "ns")

    td_ns = (
        (time.values.astype("datetime64[ns]") - t0_val)
        .astype("timedelta64[ns]")
        .astype(np.int64)
    )
    rescaled = (td_ns.astype(np.float64) / float(delta_ns)).astype(np.float32)
    out = xr.DataArray(
        rescaled,
        dims=time.dims,
        coords={cname: c for cname, c in time.coords.items() if cname != time.name},
        name=time.name,
        attrs={
            **dict(time.attrs),
            "units": freq_unit,
            "freq": float(freq_dt),
            "t0": str(t0_val),
        },
    )
    return out


def time_unrescale(time: xr.DataArray) -> xr.DataArray:
    """Inverse of :func:`time_rescale` using its stored attrs."""
    attrs = time.attrs
    if "t0" not in attrs or "freq" not in attrs or "units" not in attrs:
        raise ValueError(
            "time coord is missing t0/freq/units attrs — rescale with "
            "time_rescale first."
        )
    delta = pd.Timedelta(attrs["freq"], unit=attrs["units"])
    t0 = np.datetime64(attrs["t0"], "ns")
    delta_ns = np.int64(delta.asm8.astype("timedelta64[ns]").astype(np.int64))
    offsets = (time.values.astype(np.float64) * delta_ns).astype("timedelta64[ns]")
    restored = t0 + offsets
    return xr.DataArray(
        restored,
        dims=time.dims,
        coords={cname: c for cname, c in time.coords.items() if cname != time.name},
        name=time.name,
        attrs={},
    )


def encode_time_cyclical(
    time: xr.DataArray,
    *,
    components: Sequence[str] = ("dayofyear", "hour"),
) -> xr.Dataset:
    """Sin/cos encodings of datetime components as a Dataset of new variables.

    For each ``component`` in ``components`` (any name accepted by
    xarray's ``.dt`` accessor), two variables are produced:
    ``{component}_sin`` and ``{component}_cos``. The Layer-1
    ``EncodeTimeCyclical`` operator merges these into the input Dataset
    as new coordinates.

    Args:
        time: Time coordinate / variable.
        components: Iterable of datetime attributes to encode.

    Returns:
        Dataset whose data variables are the requested sin/cos pairs,
        each carrying the same dims as ``time``.
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
    out_vars: dict[str, xr.DataArray] = {}
    for name in components:
        if name not in periods:
            raise ValueError(
                f"Unknown time component {name!r}; known: {sorted(periods)}."
            )
        values = getattr(time.dt, name).values.astype(float)
        sin, cos = cyclical_encode(values, period=periods[name])
        out_vars[f"{name}_sin"] = xr.DataArray(sin, dims=time.dims, coords=time.coords)
        out_vars[f"{name}_cos"] = xr.DataArray(cos, dims=time.dims, coords=time.coords)
    # Carry the source time coords on the output Dataset so direct
    # callers can still do label-based selection / alignment.
    return xr.Dataset(out_vars, coords=time.coords)


def encode_time_ordinal(
    time: xr.DataArray,
    *,
    reference_date: str | np.datetime64 | None = None,
    unit: str = "D",
) -> xr.DataArray:
    """Ordinal float encoding of a time coordinate.

    Args:
        time: Time coordinate / variable.
        reference_date: Reference time. Defaults to ``time.min()``.
        unit: Pandas timedelta unit used to scale the offset.

    Returns:
        DataArray of ordinal floats with the same dims as ``time``.
        The Layer-1 ``EncodeTimeOrdinal`` operator attaches this as a
        ``{time.name}_ordinal`` coordinate on the input Dataset.
    """
    ref = (
        np.datetime64(time.min().values)
        if reference_date is None
        else np.datetime64(reference_date)
    )
    delta = pd.Timedelta(1, unit=unit)
    ordinal = ((time.values - ref) / delta).astype(np.float64)
    # Keep every input coord (including the source timestamp coord at
    # ``time.name``) so direct callers can still ``out.sel(time=...)``.
    return xr.DataArray(
        ordinal,
        dims=time.dims,
        coords=time.coords,
        name=f"{time.name}_ordinal" if time.name is not None else "time_ordinal",
    )


__all__ = [
    "encode_time_cyclical",
    "encode_time_ordinal",
    "time_rescale",
    "time_unrescale",
]

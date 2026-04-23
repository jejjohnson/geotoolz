"""Coordinate and positional encoders.

Includes coordinate-range transforms for longitude and latitude, time
rescaling to/from float offsets, cyclical and Fourier feature encoders,
and a positional-encoding helper in the style used by NeRF / ViT.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray


def lon_360_to_180(coord: ArrayLike) -> NDArray:
    """Wrap longitudes from ``[0, 360)`` into ``[-180, 180)``.

    Args:
        coord: Array of longitude values in the ``[0, 360)`` convention.

    Returns:
        Array of longitude values in the ``[-180, 180)`` convention, with
        the same shape as the input.
    """
    return (np.asarray(coord) + 180.0) % 360.0 - 180.0


def lon_180_to_360(coord: ArrayLike) -> NDArray:
    """Wrap longitudes from ``[-180, 180)`` into ``[0, 360)``.

    Args:
        coord: Array of longitude values in the ``[-180, 180)`` convention.

    Returns:
        Array of longitude values in the ``[0, 360)`` convention, with the
        same shape as the input.
    """
    return np.asarray(coord) % 360.0


def lat_180_to_90(coord: ArrayLike) -> NDArray:
    """Wrap latitudes from ``[0, 180)`` into ``[-90, 90)``.

    Useful when a dataset stores latitude as a 0-based index rather than
    the standard geographic convention.

    Args:
        coord: Array of latitude values in the ``[0, 180)`` convention.

    Returns:
        Array of latitude values in the ``[-90, 90)`` convention.
    """
    return (np.asarray(coord) + 90.0) % 180.0 - 90.0


def lat_90_to_180(coord: ArrayLike) -> NDArray:
    """Wrap latitudes from ``[-90, 90)`` into ``[0, 180)``.

    Args:
        coord: Array of latitude values in the ``[-90, 90)`` convention.

    Returns:
        Array of latitude values in the ``[0, 180)`` convention.
    """
    return np.asarray(coord) % 180.0


# ---------- time -----------------------------------------------------------


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


# ---------- cyclical / Fourier --------------------------------------------


def cyclical_encode(
    values: ArrayLike,
    period: float,
) -> tuple[NDArray, NDArray]:
    """Sin/cos embedding of a periodic variable.

    Args:
        values: Input values with period ``period``.
        period: Period length (e.g. 365.25 for day-of-year, 24 for hour
            of day, ``2 * np.pi`` for radians).

    Returns:
        ``(sin_component, cos_component)`` pair, each with the same
        shape as ``values``.
    """
    x = 2.0 * np.pi * np.asarray(values) / period
    return np.sin(x), np.cos(x)


def fourier_features(
    values: ArrayLike,
    num_freqs: int,
    scale: float = 1.0,
) -> NDArray:
    """Deterministic Fourier-feature encoding.

    Returns an array of shape ``(..., 2 * num_freqs)`` whose columns
    alternate ``sin(2^k * scale * x)`` and ``cos(2^k * scale * x)``.

    Args:
        values: Input array.
        num_freqs: Number of frequency octaves to use.
        scale: Base frequency.
    """
    values = np.asarray(values)
    freqs = (2.0 ** np.arange(num_freqs)) * scale
    angles = values[..., None] * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def random_fourier_features(
    values: ArrayLike,
    num_features: int,
    sigma: float = 1.0,
    seed: int | None = None,
) -> NDArray:
    """Random Fourier features in the style of Rahimi & Recht, 2007.

    Approximates the RBF kernel feature map. Returns an array of shape
    ``(..., num_features)``.

    Args:
        values: Input array with the feature dim as the last axis, or a
            1-D scalar coordinate.
        num_features: Output feature dimension (must be even).
        sigma: Bandwidth of the underlying RBF kernel.
        seed: Seed for the RNG used to draw the random frequencies.
    """
    if num_features % 2 != 0:
        raise ValueError("num_features must be even.")
    values = np.asarray(values)
    if values.ndim == 0:
        values = values[None]
    d = values.shape[-1] if values.ndim > 1 else 1
    x = values if values.ndim > 1 else values[..., None]

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((d, num_features // 2)) / sigma
    projection = x @ omega  # (..., num_features // 2)
    return np.concatenate([np.sin(projection), np.cos(projection)], axis=-1)


def positional_encoding(
    values: ArrayLike,
    num_freqs: int,
    include_input: bool = True,
) -> NDArray:
    """NeRF-style positional encoding.

    Output shape is ``(..., (2 * num_freqs) + include_input)``.

    Args:
        values: Input array.
        num_freqs: Number of octave frequencies.
        include_input: If ``True``, concatenate the raw input as an
            additional column.
    """
    encoded = fourier_features(values, num_freqs=num_freqs, scale=np.pi)
    if include_input:
        values = np.asarray(values)[..., None]
        return np.concatenate([values, encoded], axis=-1)
    return encoded


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


__all__: list[str] = [
    "cyclical_encode",
    "encode_time_cyclical",
    "encode_time_ordinal",
    "fourier_features",
    "lat_90_to_180",
    "lat_180_to_90",
    "lon_180_to_360",
    "lon_360_to_180",
    "positional_encoding",
    "random_fourier_features",
    "time_rescale",
    "time_unrescale",
]


# Silence "unused import" lints for types only touched in annotations.
_ANNOTATION_ONLY: tuple[Any, ...] = (ArrayLike, NDArray)

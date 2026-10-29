"""Time-axis resampling primitives."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


_ALLOWED_RESAMPLE_METHODS = frozenset(
    {"mean", "sum", "max", "min", "median", "std", "var", "first", "last", "count"}
)


def resample_time(
    da: xr.DataArray,
    freq: str = "1D",
    method: str = "mean",
    time: str = "time",
    *,
    interp_method: Literal["linear", "nearest", "cubic"] = "linear",
) -> xr.DataArray:
    """Resample a DataArray along the time axis via xarray's built-in resampler.

    Per the PR β primitive-flip, this primitive is DataArray-only.
    The Layer-1 ``ResampleTime`` operator handles ``ds.map``-style
    dispatch when a Dataset is supplied.

    Args:
        da: Input DataArray.
        freq: Pandas-style frequency string (e.g. ``"1D"``, ``"6H"``,
            ``"1M"``).
        method: Zero-argument aggregation method; one of ``"mean"``,
            ``"sum"``, ``"max"``, ``"min"``, ``"median"``, ``"std"``,
            ``"var"``, ``"first"``, ``"last"``, ``"count"``. Methods
            that require additional arguments (e.g. ``reduce``, ``quantile``)
            are intentionally not exposed here. Use ``"interpolate"`` for
            upsampling via xarray's resampler interpolation.
        time: Name of the time dimension.
        interp_method: Interpolation method forwarded to xarray when
            ``method="interpolate"``.

    Returns:
        Resampled DataArray.
    """
    if method not in _ALLOWED_RESAMPLE_METHODS | {"interpolate"}:
        raise ValueError(
            f"Unknown resample method {method!r}; expected one of "
            f"{sorted(_ALLOWED_RESAMPLE_METHODS | {'interpolate'})}."
        )

    if method == "interpolate":
        _ensure_datetime_like_time(da, time=time)
        if _target_is_coarser_than_source(da, freq=freq, time=time):
            raise ValueError(
                "resample_time(method='interpolate') only supports upsampling; "
                f"got target frequency {freq!r} coarser than the input."
            )
        return da.resample({time: freq}).interpolate(interp_method)

    resampler = da.resample({time: freq})
    return getattr(resampler, method)()


def _ensure_datetime_like_time(da: xr.DataArray, *, time: str) -> None:
    try:
        _ = da[time].dt.year
    except AttributeError as exc:
        raise ValueError(
            f"{time!r} coordinate must be datetime-like for "
            "resample_time(method='interpolate')."
        ) from exc


def _target_is_coarser_than_source(da: xr.DataArray, *, freq: str, time: str) -> bool:
    coord = da[time]
    if coord.size < 2:
        return False

    try:
        values_ns = np.asarray(coord.values, dtype="datetime64[ns]").astype("int64")
    except (TypeError, ValueError):
        # Non-numpy datetime indexes (for example CFTimeIndex) may still be
        # accepted by xarray; skip the coarse-frequency guard in that case.
        return False

    deltas = np.diff(values_ns)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        return False
    source_delta_ns = int(np.median(positive_deltas))
    start = pd.Timestamp(coord.values[0])
    if not isinstance(start, pd.Timestamp) or pd.isna(start):
        return False
    target_delta_ns = _target_delta_ns(freq, start)

    return target_delta_ns > source_delta_ns


def _target_delta_ns(freq: str, start: pd.Timestamp) -> int:
    offset = pd.tseries.frequencies.to_offset(freq)
    try:
        return int(offset.nanos)
    except ValueError:
        # Variable-length offsets such as "MS" do not expose fixed nanos;
        # estimate their first step from the input coordinate anchor.
        try:
            return int((start + offset - start).value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not infer target frequency duration for {freq!r}; "
                'ensure freq is a valid pandas frequency string such as "1h", '
                '"12h", or "1D".'
            ) from exc

"""Time-axis resampling primitives."""

from __future__ import annotations

import xarray as xr


_ALLOWED_RESAMPLE_METHODS = frozenset(
    {"mean", "sum", "max", "min", "median", "std", "var", "first", "last", "count"}
)


def resample_time(
    ds: xr.Dataset | xr.DataArray,
    freq: str = "1D",
    method: str = "mean",
    time: str = "time",
) -> xr.Dataset | xr.DataArray:
    """Resample along the time axis via xarray's built-in resampler.

    Args:
        ds: Input.
        freq: Pandas-style frequency string (e.g. ``"1D"``, ``"6H"``,
            ``"1M"``).
        method: Zero-argument aggregation method; one of ``"mean"``,
            ``"sum"``, ``"max"``, ``"min"``, ``"median"``, ``"std"``,
            ``"var"``, ``"first"``, ``"last"``, ``"count"``. Methods
            that require additional arguments (e.g. ``reduce``,
            ``quantile``) are intentionally not exposed here.
        time: Name of the time dimension.

    Returns:
        Resampled container.
    """
    if method not in _ALLOWED_RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown resample method {method!r}; expected one of "
            f"{sorted(_ALLOWED_RESAMPLE_METHODS)}."
        )
    resampler = ds.resample({time: freq})
    return getattr(resampler, method)()

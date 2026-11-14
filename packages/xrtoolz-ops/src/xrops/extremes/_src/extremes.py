"""Extreme-value analysis primitives.

Covers the standard trio:
- block maxima / minima (Generalised Extreme Value framework)
- peaks over threshold (Generalised Pareto framework)
- point-process statistics over time blocks
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr


def block_maxima(
    da: xr.DataArray,
    block_size: int = 365,
    time: str = "time",
    boundary: str = "trim",
    side: str = "left",
) -> xr.DataArray:
    """Coarsen along ``time`` and take the maximum per block.

    Args:
        da: Input DataArray with a ``time`` dimension.
        block_size: Number of time steps per block.
        time: Name of the time dimension.
        boundary: Passed to ``.coarsen`` (``"trim"``, ``"pad"``).
        side: Passed to ``.coarsen`` (``"left"`` or ``"right"``).

    Returns:
        DataArray of block maxima.
    """
    return da.coarsen({time: block_size}, boundary=boundary, side=side).max()


def block_minima(
    da: xr.DataArray,
    block_size: int = 365,
    time: str = "time",
    boundary: str = "trim",
    side: str = "left",
) -> xr.DataArray:
    """Coarsen along ``time`` and take the minimum per block."""
    return da.coarsen({time: block_size}, boundary=boundary, side=side).min()


def pot_threshold(
    da: xr.DataArray,
    quantile: float = 0.98,
    time: str = "time",
) -> xr.DataArray:
    """Threshold for peaks-over-threshold: the ``quantile`` of ``da``."""
    return da.quantile(quantile, dim=time)


def pot_exceedances(
    da: xr.DataArray,
    quantile: float = 0.98,
    decluster_freq: int | None = None,
    time: str = "time",
) -> xr.DataArray:
    """Return values above the ``quantile`` threshold, optionally declustered.

    When ``decluster_freq`` is set, values are declustered by taking the
    block maximum over that many steps; this converts the peaks series
    into independent exceedances (standard POT practice).

    Args:
        da: Input DataArray.
        quantile: Quantile (in [0, 1]) used to compute the threshold.
        decluster_freq: If given, apply a block-maximum declustering
            filter with this block size and drop resulting NaNs.
        time: Name of the time dimension.

    Returns:
        DataArray of exceedances. If ``decluster_freq is None`` the
        returned array has the same shape as ``da`` with sub-threshold
        points replaced by NaN.
    """
    threshold = pot_threshold(da, quantile=quantile, time=time)
    above = da.where(da >= threshold)
    if decluster_freq is None:
        return above
    return block_maxima(above, block_size=decluster_freq, time=time).dropna(time)


def pp_counts(
    da: xr.DataArray,
    quantile: float = 0.98,
    block_size: int = 5,
    time: str = "time",
    boundary: str = "trim",
    side: str = "left",
) -> xr.DataArray:
    """Count exceedances of a high quantile within each time block.

    Args:
        da: Input DataArray.
        quantile: Quantile used to build the threshold.
        block_size: Size of the time block in which to count.
        time: Name of the time dimension.
        boundary, side: Passed to :func:`xarray.DataArray.coarsen`.

    Returns:
        DataArray with one count per block along the (coarsened) time
        dimension.
    """
    return _pp_apply(
        da,
        quantile=quantile,
        block_size=block_size,
        time=time,
        boundary=boundary,
        side=side,
        reducer=lambda vals, thr: int(np.sum(vals > thr)),
    )


def pp_stats(
    da: xr.DataArray,
    quantile: float = 0.98,
    block_size: int = 5,
    statistic: Callable[[np.ndarray], float] = np.mean,
    time: str = "time",
    boundary: str = "trim",
    side: str = "left",
) -> xr.DataArray:
    """Summarize exceedances within each time block.

    Applies ``statistic`` to the subset of values in each block that
    exceed the ``quantile`` threshold.

    Args:
        da: Input DataArray.
        quantile: Quantile used to build the threshold.
        block_size: Size of the time block.
        statistic: Reduction to apply to the exceedances (default
            ``numpy.mean``).
        time: Name of the time dimension.
        boundary, side: Passed to :func:`xarray.DataArray.coarsen`.

    Returns:
        DataArray of per-block summary statistics.
    """
    return _pp_apply(
        da,
        quantile=quantile,
        block_size=block_size,
        time=time,
        boundary=boundary,
        side=side,
        reducer=lambda vals, thr: (
            float(statistic(vals[vals > thr])) if np.any(vals > thr) else np.nan
        ),
    )


def _pp_apply(
    da: xr.DataArray,
    quantile: float,
    block_size: int,
    time: str,
    boundary: str,
    side: str,
    reducer: Callable[[np.ndarray, float], float],
) -> xr.DataArray:
    # For gridded inputs, `pot_threshold` returns a DataArray over the
    # non-time dims (e.g. lat/lon). Keep it as a DataArray and let
    # xarray broadcast it against `blocks` via apply_ufunc instead of
    # scalarizing with `.item()`, which would fail on size > 1.
    threshold = pot_threshold(da, quantile=quantile, time=time)
    blocks = da.coarsen({time: block_size}, boundary=boundary, side=side).construct(
        {time: (time, "_block")}
    )
    return xr.apply_ufunc(
        reducer,
        blocks,
        threshold,
        input_core_dims=[["_block"], []],
        output_core_dims=[[]],
        vectorize=True,
    )

"""Shared percentile clip-and-stretch core.

The single NaN-aware percentile stretch behind
``radiometry.percentile_clip`` / ``radiometry.PercentileClip``,
``normalize.HistogramStretch`` and ``viz.StretchToUint8`` (all spelled
``lower`` / ``upper`` / ``reduce_axes``). It owns the ``upper > lower``
validation, so the wrappers delegate without re-checking. Both functions
are written on the array namespace, so they run on NumPy and JAX arrays
(under ``jax.jit`` too) and on the NumPy blocks of a dask cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float, Shaped
from pipekit_array import array_namespace


__all__ = ["check_percentiles", "percentile_stretch", "stretch_between"]


def check_percentiles(lower: float, upper: float) -> None:
    """Raise ``ValueError`` unless ``upper > lower``.

    Example:
        >>> check_percentiles(2.0, 98.0)
    """
    if upper <= lower:
        raise ValueError(
            f"percentile_stretch requires upper > lower; got {lower=}, {upper=}"
        )


def stretch_between(
    arr: Shaped[np.ndarray, "*dims"], lo: Any, hi: Any
) -> Float[np.ndarray, "*dims"]:
    r"""Clip ``arr`` to ``[lo, hi]`` and map that range onto ``[0, 1]``.

    .. math::

        y \;=\; \operatorname{clip}\!\left(\frac{x - lo}{hi - lo}, 0, 1\right)

    A constant range (``hi == lo``) maps to ``0`` rather than dividing by
    zero; ``NaN`` propagates.

    Args:
        arr: Input array of any shape.
        lo: Lower bound, broadcasting against ``arr``.
        hi: Upper bound, broadcasting against ``arr``.

    Returns:
        Float array of ``arr``'s shape with values in ``[0, 1]``.

    Example:
        >>> stretch_between(np.array([0.0, 1.0, 3.0]), 1.0, 2.0)
        array([0., 0., 1.])
    """
    xp = array_namespace(arr)
    denom = xp.where(hi > lo, hi - lo, 1.0)
    return xp.clip((arr - lo) / denom, 0.0, 1.0)


def percentile_stretch(
    arr: Shaped[np.ndarray, "*dims"],
    lower: float,
    upper: float,
    *,
    reduce_axes: int | tuple[int, ...] | None = (-2, -1),
) -> Float[np.ndarray, "*dims"]:
    """Clip to percentile bounds and stretch the result into ``[0, 1]``.

    NaN-aware: percentiles are computed with ``nanpercentile`` so
    fill/NaN pixels don't poison the thresholds. A constant slice
    (``hi == lo``) maps to ``0`` rather than dividing by zero.

    Args:
        arr: Input array of any shape.
        lower: Lower percentile in ``[0, 100]``.
        upper: Upper percentile in ``[0, 100]``; must exceed ``lower``.
        reduce_axes: Axis (or axes) to compute percentiles over. The default
            ``(-2, -1)`` stretches each leading band / time slice
            independently; ``None`` uses one global pair of thresholds.

    Returns:
        Float array of the same shape with values in ``[0, 1]`` (NaNs
        propagate through untouched).

    Raises:
        ValueError: If ``upper <= lower``.
    """
    check_percentiles(lower, upper)
    xp = array_namespace(arr)
    # `keepdims=True` so lo/hi broadcast back over the reduced axes.
    lo = xp.nanpercentile(arr, lower, axis=reduce_axes, keepdims=True)
    hi = xp.nanpercentile(arr, upper, axis=reduce_axes, keepdims=True)
    return stretch_between(arr, lo, hi)

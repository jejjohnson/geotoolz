"""Tier A — array kernels for finite-difference calculus.

Pure-array entry points for the canonical central-difference derivatives
on uniform Cartesian grids. Tier B (xarray, ``dim=``) wrappers in
:mod:`xr_toolz.calc._src.cartesian` add coord/attr handling and route to
``finitediffx`` for higher-order accuracies; this module is the
numpy-only computational core for the default 2nd-order central-difference
case (``accuracy=1, method="central"``), built on :func:`numpy.gradient`.

Pilot scope: ``partial`` and ``gradient`` only. ``divergence`` /
``curl`` / ``laplacian`` are deferred to follow-up issues.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def partial(
    x: ArrayLike,
    *,
    axis: int,
    spacing: float = 1.0,
) -> NDArray[np.floating]:
    """Partial derivative ``∂x/∂<axis>`` via 2nd-order central differences.

    Args:
        x: Input array.
        axis: Axis to differentiate along.
        spacing: Uniform sample spacing along ``axis``.

    Returns:
        Array with the same shape as ``x``.
    """
    arr = np.asarray(x)
    return np.gradient(arr, spacing, axis=axis)


def gradient(
    x: ArrayLike,
    *,
    axes: tuple[int, ...] | None = None,
    spacing: float | tuple[float, ...] = 1.0,
) -> tuple[NDArray[np.floating], ...]:
    """Gradient ``∇x`` over ``axes`` via 2nd-order central differences.

    Args:
        x: Input array.
        axes: Axes to differentiate against. Defaults to every axis of
            ``x``, in order.
        spacing: Scalar sample spacing applied to every axis, or a
            per-axis tuple matching the length of ``axes``.

    Returns:
        Tuple of arrays — one partial derivative per axis in ``axes``,
        each with the same shape as ``x``.
    """
    arr = np.asarray(x)
    target = tuple(range(arr.ndim)) if axes is None else tuple(axes)
    if isinstance(spacing, (int, float)):
        steps: tuple[float, ...] = (float(spacing),) * len(target)
    else:
        steps = tuple(float(s) for s in spacing)
        if len(steps) != len(target):
            raise ValueError(
                f"spacing tuple length ({len(steps)}) does not match number "
                f"of axes ({len(target)})."
            )
    result = np.gradient(arr, *steps, axis=target)
    if isinstance(result, tuple):
        return result
    return (result,)


__all__ = [
    "gradient",
    "partial",
]

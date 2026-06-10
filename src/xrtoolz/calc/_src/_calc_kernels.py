"""Private numpy kernels for finite-difference calculus.

Implementation detail — no stability guarantees. Pure-array entry points
for the canonical central-difference derivatives on uniform Cartesian
grids. The Layer 0 (xarray, ``dim=``) wrappers in
:mod:`xrtoolz.calc._src.cartesian` add coord/attr handling and route to
``finitediffx`` for higher-order accuracies; this module is the
numpy-only computational core for the default 2nd-order central-difference
case (``accuracy=1, method="central"``), built on :func:`numpy.gradient`.

Array shapes are annotated with :mod:`jaxtyping` (``Float[np.ndarray,
"*shape"]``): the named ``*shape`` axes document that each derivative has
the same shape as its input. See
``docs/design/conventions/array-typing.md``.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float


def partial(
    x: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    spacing: float = 1.0,
) -> Float[np.ndarray, "*shape"]:
    """Partial derivative ``∂x/∂<axis>`` via 2nd-order central differences.

    Args:
        x: Real-valued field of arbitrary shape ``(*shape)``.
        axis: Axis to differentiate along.
        spacing: Uniform sample spacing along ``axis``.

    Returns:
        The derivative ``∂x/∂<axis>``, same shape ``(*shape)`` as ``x``.
    """
    arr = np.asarray(x)
    return np.gradient(arr, spacing, axis=axis)


def gradient(
    x: Float[np.ndarray, "*shape"],
    *,
    axes: tuple[int, ...] | None = None,
    spacing: float | tuple[float, ...] = 1.0,
) -> tuple[Float[np.ndarray, "*shape"], ...]:
    """Gradient ``∇x`` over ``axes`` via 2nd-order central differences.

    Args:
        x: Real-valued field of arbitrary shape ``(*shape)``.
        axes: Axes to differentiate against. Defaults to every axis of
            ``x``, in order.
        spacing: Scalar sample spacing applied to every axis, or a
            per-axis tuple matching the length of ``axes``.

    Returns:
        One partial derivative per axis in ``axes``, each of shape
        ``(*shape)`` matching ``x``.
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

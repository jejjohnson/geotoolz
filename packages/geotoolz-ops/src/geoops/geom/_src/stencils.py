"""Finite-difference stencils: offsets and coefficients, from integers alone.

The lowest layer of the gradient stack. Nothing here touches an array: a
stencil depends only on the derivative order, the accuracy order and the
method, so it is computed once on the host (and cached) and the array
primitives in `geoops.geom._src.array` apply it by slicing on any backend.

The rules are finitediffx's, so the two give the same numbers:

- a *forward* stencil of accuracy :math:`p` for derivative :math:`n` uses
  offsets :math:`0, 1, \\dots, n + p - 1`;
- a *backward* stencil mirrors it, :math:`-(n + p - 1), \\dots, 0`;
- a *central* stencil uses accuracy :math:`p + 1` (so ``accuracy=1`` is the
  second-order three-point stencil, as in ``numpy.gradient``):
  :math:`-\\lfloor (n + p) / 2 \\rfloor, \\dots, \\lfloor (n + p) / 2 \\rfloor`.

The coefficients :math:`c_k` of offsets :math:`s_k` solve the Taylor
conditions :math:`\\sum_k c_k s_k^m = n!\\,\\delta_{mn}` for
:math:`m = 0, \\dots, K - 1`.

Example:
    >>> stencil_offsets(derivative=1, accuracy=1, method="central")
    (-1, 0, 1)
    >>> stencil_coeffs((-1, 0, 1), derivative=1)
    (-0.5, 0.0, 0.5)
"""

from __future__ import annotations

import functools
import math
from typing import Literal

import numpy as np


__all__ = ["METHODS", "Method", "stencil_coeffs", "stencil_offsets"]

Method = Literal["central", "forward", "backward"]

#: The stencil methods, widest support first.
METHODS: tuple[Method, ...] = ("central", "forward", "backward")


@functools.lru_cache(maxsize=256)
def stencil_offsets(
    *, derivative: int = 1, accuracy: int = 1, method: Method = "central"
) -> tuple[int, ...]:
    """The offsets of one stencil, finitediffx's rules.

    Args:
        derivative: The derivative order, at least 1.
        accuracy: The accuracy order, at least 1. A central stencil is built
            at ``accuracy + 1``, the order its symmetry gives for free.
        method: ``"central"``, ``"forward"`` or ``"backward"``.

    Returns:
        The integer offsets, ascending.

    Raises:
        ValueError: An order below 1, or an unknown method.

    Examples:
        >>> stencil_offsets(derivative=1, accuracy=2, method="forward")
        (0, 1, 2)
        >>> stencil_offsets(derivative=2, accuracy=1, method="central")
        (-1, 0, 1)
    """
    if derivative < 1 or accuracy < 1:
        raise ValueError(
            f"derivative and accuracy must be at least 1; got {derivative}, {accuracy}"
        )
    if method == "forward":
        return tuple(range(derivative + accuracy))
    if method == "backward":
        return tuple(range(-(derivative + accuracy - 1), 1))
    if method == "central":
        half = (derivative + accuracy) // 2
        return tuple(range(-half, half + 1))
    raise ValueError(f"method must be one of {METHODS}; got {method!r}")


@functools.lru_cache(maxsize=256)
def stencil_coeffs(offsets: tuple[int, ...], derivative: int = 1) -> tuple[float, ...]:
    r"""The coefficients of a stencil on ``offsets``, from the Taylor conditions.

    Solves :math:`\sum_k c_k s_k^m = n!\,\delta_{mn}` for
    :math:`m = 0, \dots, K - 1` in float64.

    Args:
        offsets: The stencil's integer offsets, distinct.
        derivative: The derivative order; fewer than ``derivative + 1``
            offsets cannot resolve it.

    Returns:
        One coefficient per offset; :math:`f^{(n)}(x) \approx h^{-n} \sum_k
        c_k f(x + s_k h)`.

    Raises:
        ValueError: Too few offsets for the derivative.

    Examples:
        >>> stencil_coeffs((-1, 0, 1), derivative=2)
        (1.0, -2.0, 1.0)
        >>> [round(c, 6) for c in stencil_coeffs((0, 1, 2), derivative=1)]
        [-1.5, 2.0, -0.5]
    """
    n = len(offsets)
    if derivative >= n:
        raise ValueError(
            f"{n} offsets cannot resolve derivative {derivative}; need at least "
            f"{derivative + 1}"
        )
    powers = np.vander(np.asarray(offsets, dtype=np.float64), n, increasing=True).T
    rhs = np.zeros(n)
    rhs[derivative] = math.factorial(derivative)
    coeffs = np.linalg.solve(powers, rhs)
    # round off solver noise so exact stencils stay exact (0.5, not 0.4999…)
    return tuple(float(c) for c in np.round(coeffs, 12) + 0.0)

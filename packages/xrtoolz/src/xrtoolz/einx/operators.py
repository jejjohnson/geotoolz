"""Deprecated alias — forwards to :mod:`xreinx.operators`.

Kept so historical ``from xrtoolz.einx.operators import Einsum`` imports
keep resolving through 0.x; the package-level ``DeprecationWarning``
fires via :mod:`xrtoolz.einx`.
"""

from __future__ import annotations

from xreinx.operators import (
    BatchMatmul,
    Einsum,
    Matmul,
    Outer,
    Rearrange,
    Reduce,
    Repeat,
)


__all__ = [
    "BatchMatmul",
    "Einsum",
    "Matmul",
    "Outer",
    "Rearrange",
    "Reduce",
    "Repeat",
]

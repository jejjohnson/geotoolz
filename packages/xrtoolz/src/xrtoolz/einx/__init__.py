"""Deprecated alias — ``xrtoolz.einx`` moved to the ``xrtoolz-einx``
distribution (import name :mod:`xreinx`).

Every public name is re-exported here unchanged, so existing
``xrtoolz.einx`` imports keep working through the 0.x series. The shim
is removed at 1.0 — switch to ``import xreinx``.

The bridge's laziness is preserved: importing this module (or
:mod:`xreinx`) does not import the ``einx`` backend — that happens on
the first pattern call.
"""

from __future__ import annotations

import warnings

from xreinx import (
    BatchMatmul,
    CoordMismatch,
    Einsum,
    EinxBridgeError,
    Matmul,
    Outer,
    PatternError,
    Rearrange,
    Reduce,
    Repeat,
    batch_matmul,
    einsum,
    matmul,
    outer,
    pack_dataset,
    rearrange,
    reduce,
    repeat,
    unpack_dataset,
)


warnings.warn(
    "xrtoolz.einx has moved to the xrtoolz-einx distribution; "
    "use `import xreinx` instead. This alias is removed at xrtoolz 1.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BatchMatmul",
    "CoordMismatch",
    "Einsum",
    "EinxBridgeError",
    "Matmul",
    "Outer",
    "PatternError",
    "Rearrange",
    "Reduce",
    "Repeat",
    "batch_matmul",
    "einsum",
    "matmul",
    "outer",
    "pack_dataset",
    "rearrange",
    "reduce",
    "repeat",
    "unpack_dataset",
]

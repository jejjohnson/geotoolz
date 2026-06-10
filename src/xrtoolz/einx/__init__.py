"""``xrtoolz.einx`` — labeled named-tensor algebra bridging xarray + einx.

Pattern axis tokens are DataArray *dim names*; the bridge transposes to
match before dispatching to `einx <https://github.com/fferflo/einx>`_ and
rewraps the result as a labeled DataArray with coords forwarded from the
inputs. See ``docs/design/bridges/einx/`` for the full design.

Two surfaces:

- **Functions** (Layer 0): :func:`einsum`, :func:`rearrange`,
  :func:`reduce`, :func:`repeat`, plus :func:`matmul` / :func:`outer` /
  :func:`batch_matmul` conveniences and :func:`pack_dataset` /
  :func:`unpack_dataset`.
- **Operators** (Layer 1): :class:`Einsum`, :class:`Rearrange`,
  :class:`Reduce`, :class:`Repeat`, :class:`Matmul`, :class:`Outer`,
  :class:`BatchMatmul`.

Example:
    ```pycon
    >>> import xrtoolz.einx as xnx
    >>> total = xnx.einsum("time lat lon, lat lon -> time", field, mask)
    ```
"""

from __future__ import annotations

from xrtoolz.einx._src.core import einsum, rearrange, reduce, repeat
from xrtoolz.einx._src.errors import (
    CoordMismatch,
    EinxBridgeError,
    PatternError,
)
from xrtoolz.einx._src.linalg import batch_matmul, matmul, outer
from xrtoolz.einx.dataset import pack_dataset, unpack_dataset
from xrtoolz.einx.operators import (
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

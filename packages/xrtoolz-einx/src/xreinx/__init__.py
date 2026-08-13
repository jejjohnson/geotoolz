"""xreinx — labeled named-tensor algebra bridging xarray + einx.

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
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xreinx as xnx
    >>> field = xr.DataArray(np.ones((2, 3, 4)), dims=("time", "lat", "lon"))
    >>> mask = xr.DataArray(np.ones((3, 4)), dims=("lat", "lon"))
    >>> xnx.einsum("time lat lon, lat lon -> time", field, mask).shape
    (2,)

    ```
"""

from __future__ import annotations

from xreinx._src.core import einsum, rearrange, reduce, repeat
from xreinx._src.errors import (
    CoordMismatch,
    EinxBridgeError,
    PatternError,
)
from xreinx._src.linalg import batch_matmul, matmul, outer
from xreinx.dataset import pack_dataset, unpack_dataset
from xreinx.operators import (
    BatchMatmul,
    Einsum,
    Matmul,
    Outer,
    Rearrange,
    Reduce,
    Repeat,
)


__version__ = "0.0.0"  # x-release-please-version

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
    "__version__",
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

"""`geopatcher.run` — running a patcher at scale.

`SpatialPatcher.split` / `merge` are the plain loop; this module holds the
ways to run it faster, in parallel, or on another engine:

- `parallel_map` — apply an operator over every patch on a thread or
  process pool (batched ``select_many`` reads where the field supports
  them, prefetch, backpressure); `prefetch_iterable` — read ahead of any
  patch stream.
- `to_delayed` / `to_dask_bag` — one Dask task per patch (``[dask]``).
- `batch_split` / `unbatch` / `BatchedPatch` — fixed-size batches of
  patches as JAX pytrees (``[jax]``); `stack_patches` — stack patch data
  along a new leading axis.
- `PatchCache` — on-disk cache of read patches keyed by the field's
  identity; `IndexedPatchView` — random access (``len`` / ``[i]``) over a
  patcher's patches, for ML data loaders.
"""

from __future__ import annotations

from geopatcher._src.cache import PatchCache, UnstableIdentityError
from geopatcher._src.dask import to_dask_bag, to_delayed
from geopatcher._src.indexed import IndexedPatchView
from geopatcher._src.jax import BatchedPatch, batch_split, unbatch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.runners import Backend, ErrorPolicy, parallel_map
from geopatcher._src.stacking import stack_patches


__all__ = [
    "Backend",
    "BatchedPatch",
    "ErrorPolicy",
    "IndexedPatchView",
    "PatchCache",
    "UnstableIdentityError",
    "batch_split",
    "parallel_map",
    "prefetch_iterable",
    "stack_patches",
    "to_dask_bag",
    "to_delayed",
    "unbatch",
]

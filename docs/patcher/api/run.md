# Running at scale — `geopatcher.run`

`split` / `merge` are the plain loop. These run it faster, in parallel,
on another engine, or with random access for ML loaders.

```python
from geopatcher.run import PatchCache, parallel_map
```

## Parallel and prefetched runs

::: geopatcher.run.parallel_map
::: geopatcher.run.prefetch_iterable

## Dask (`[dask]`)

::: geopatcher.run.to_delayed
::: geopatcher.run.to_dask_bag

## JAX batching (`[jax]`)

Stack patch payloads on a leading axis for jitted / vmapped models, then
unpack model outputs back into patches.

::: geopatcher.run.BatchedPatch
::: geopatcher.run.batch_split
::: geopatcher.run.unbatch
::: geopatcher.run.stack_patches

## Caching and random access

::: geopatcher.run.PatchCache
::: geopatcher.run.UnstableIdentityError
::: geopatcher.run.IndexedPatchView

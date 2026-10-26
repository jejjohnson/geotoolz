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

## Batches of patches

`batch_split` groups a raster field adapter's patches into fixed-size
`PatchBatch` tuples: the patches as one `GeoArray` with a `patch` batch
dim (one transform row per patch) and a `(patch,)` mask of the real ones
(ADR-013). The batch is NumPy; move it to JAX once with
`batch.patches.to_jax()` (`[jax]`), map a model over it with
`geoarray.map_over`, and `unbatch` the output back into patches that
`patcher.merge` places from their transform rows.

```python
for patches, valid in batch_split(patcher, field, batch_size=64):
    scores = geoarray.map_over(patches.to_jax(), model, "patch")  # (64, y, x)
```

::: geopatcher.run.PatchBatch
::: geopatcher.run.batch_split
::: geopatcher.run.unbatch
::: geopatcher.run.BatchedPatch
::: geopatcher.run.stack_patches

## Caching and random access

::: geopatcher.run.PatchCache
::: geopatcher.run.UnstableIdentityError
::: geopatcher.run.IndexedPatchView

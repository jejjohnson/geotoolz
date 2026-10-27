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
`geoarray.map_over`, and hand the scored batches straight to
`patcher.merge` (or `merge_to_field`), which places each row from its
transform with the patcher's window weights and drops the padding.
`unbatch` does the placement alone, for code that wants `Patch` objects.

```python
scored = [
    PatchBatch(geoarray.map_over(patches.to_jax(), model, "patch").to_numpy(), valid)
    for patches, valid in batch_split(patcher, field, batch_size=64)
]
mosaic = patcher.merge_to_field(scored, field)  # GeoArray on the field's grid
```

A row whose pixel size or rotation differs from the target grid's raises
`ValueError` naming it: merging never resamples.

::: geopatcher.run.PatchBatch
::: geopatcher.run.batch_split
::: geopatcher.run.unbatch
::: geopatcher.run.BatchedPatch
::: geopatcher.run.stack_patches

## Caching and random access

::: geopatcher.run.PatchCache
::: geopatcher.run.UnstableIdentityError
::: geopatcher.run.IndexedPatchView

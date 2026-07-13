# Integrations API

Thin bridges from the patcher core into ML and operator-graph
frameworks. Each lives at the package top level (outside the private
`_src` core) and is gated behind the matching extra.

## JAX batching (`geopatcher.jax`, `[jax]` extra)

Stack patch payloads on a leading axis for jitted / vmapped models,
then unpack model outputs back into patches:

```python
from geopatcher.jax import BatchedPatch, batch_split, unbatch
```

::: geopatcher.jax.BatchedPatch
::: geopatcher.jax.batch_split
::: geopatcher.jax.unbatch

## pipekit operator bridge (`geopatcher.integrations.pipekit`, `[pipekit]` extra)

Operator wrappers that plug a `SpatialPatcher` into a `pipekit`
`Sequential` / `Graph` pipeline:

```python
from geopatcher.integrations.pipekit import GridSampler, ApplyToChips, Stitch
```

::: geopatcher.integrations.pipekit.GridSampler
::: geopatcher.integrations.pipekit.ApplyToChips
::: geopatcher.integrations.pipekit.Stitch

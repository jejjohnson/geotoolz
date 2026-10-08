# Integrations API

Thin bridges from the patcher core into operator-graph frameworks, each
gated behind the matching extra. (JAX batching and the Dask bridge are
runners: see [Running at scale](run.md).)

## pipekit operator bridge (`geopatcher.integrations.pipekit`, `[pipekit]` extra)

Operator wrappers that plug a `SpatialPatcher` into a `pipekit`
`Sequential` / `Graph` pipeline:

```python
from geopatcher.integrations.pipekit import GridSampler, ApplyToChips, MergePatches
```

::: geopatcher.integrations.pipekit.GridSampler
::: geopatcher.integrations.pipekit.ApplyToChips
::: geopatcher.integrations.pipekit.MergePatches

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

## pipekit-train dataset (`geopatcher.integrations.train`, `[train]` extra)

`XarrayWindowDataset`: a pipekit-train `TrainingDataset` that samples
`(input, target)` windows along a dimension from lazy xarray / Zarr stores,
using the temporal stencils in `geopatcher.temporal.stencils`. It rides the
Equinox adapter's Grain path (shuffle, checkpoint and resume), and with
`block_reader=True` reads coalesced blocks with window shuffling and
multi-host sharding. It moved here from `pipekit_train.xarray_window`, so
pipekit does not depend on geopatcher.

```python
from geopatcher.integrations.train import XarrayWindowDataset
```

::: geopatcher.integrations.train.XarrayWindowDataset

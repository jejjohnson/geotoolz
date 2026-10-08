"""`geopatcher` — split a geospatial field into patches, process them, stitch them back.

The root holds what every patching job touches:

- **Patchers** — `SpatialPatcher` (and `AsyncSpatialPatcher`),
  `TemporalPatcher`, `SpatioTemporalPatcher`: ``split`` a field into
  patches, ``merge`` processed patches back into one field.
- **Carriers** — `Patch`, `TemporalPatch`, `SpatioTemporalPatch`.
- **Fields** — the `Field` / `AsyncField` / `Domain` protocols and
  `RasterField`, the adapter for any georeader reader (a GeoTIFF, a COG
  through rasterio, an in-memory `GeoTensor`).

Everything else lives in one sub-namespace per task:

- `geopatcher.spatial` / `geopatcher.temporal` — the four axes of a patcher
  (``geometry``, ``sampler``, ``window``, ``aggregation``), plus temporal
  ``stencils``.
- `geopatcher.fields` — the other Field adapters (xarray, dask, vector,
  points, cloud COGs) and the domain types.
- `geopatcher.matched` — patching co-registered sources together.
- `geopatcher.run` — parallel and batched runners, Dask / JAX bridges,
  the on-disk `PatchCache` and random-access views.
- `geopatcher.observe` — hooks, the resumable journal, error records and
  strict mode.
- `geopatcher.config` — ``get_config`` round-trips (`axis_envelope`,
  `from_config`).
- `geopatcher.integrations.pipekit` — pipekit operator wrappers
  (``[pipekit]`` extra).

```python
from geopatcher import RasterField, SpatialPatcher, spatial

patcher = SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(256, 256)),
    sampler=spatial.sampler.RegularStride(step=192),
    window=spatial.window.Hann(),
    aggregation=spatial.aggregation.OverlapAdd(),
)
field = RasterField(reader)
outputs = [p.with_data(model(p.data)) for p in patcher.split(field)]
merged = patcher.merge(outputs, field.domain)
```
"""

from __future__ import annotations

from geopatcher import (
    config,
    fields,
    matched,
    observe,
    run,
    spatial,
    temporal,
)
from geopatcher._src.fields import RasterField
from geopatcher._src.patch import Patch, SpatioTemporalPatch, TemporalPatch
from geopatcher._src.protocols import AsyncField, Domain, Field
from geopatcher._src.spatial.patcher import AsyncSpatialPatcher, SpatialPatcher
from geopatcher._src.spatial_time import SpatioTemporalPatcher
from geopatcher._src.temporal.patcher import TemporalPatcher


__version__ = "0.10.0"  # x-release-please-version

__all__ = [
    "AsyncField",
    "AsyncSpatialPatcher",
    "Domain",
    "Field",
    "Patch",
    "RasterField",
    "SpatialPatcher",
    "SpatioTemporalPatch",
    "SpatioTemporalPatcher",
    "TemporalPatch",
    "TemporalPatcher",
    "__version__",
    "config",
    "fields",
    "matched",
    "observe",
    "run",
    "spatial",
    "temporal",
]

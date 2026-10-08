# `geopatcher` — API reference

One page per namespace; each public name has exactly one home. For the
conceptual walkthrough see [Patching](../patching.md).

| Namespace | Holds | Page |
|---|---|---|
| `geopatcher` | the patchers, the patch carriers, `Field` / `Domain`, `RasterField` | [Core](core.md) |
| `geopatcher.spatial` | `geometry` · `sampler` · `window` · `aggregation` | [Spatial axes](spatial.md) |
| `geopatcher.temporal` | the same four axes along time, plus `stencils` | [Temporal axes](temporal.md) |
| `geopatcher.fields` | the other Field adapters and the domain types | [Fields](fields.md) |
| `geopatcher.matched` | patching co-registered sources together | [Matched](matched.md) |
| `geopatcher.run` | parallel / batched runners, Dask & JAX, `PatchCache`, random access | [Running at scale](run.md) |
| `geopatcher.observe` | hooks, the journal, error records, strict mode | [Observing](observe.md) |
| `geopatcher.config` | `get_config` round-trips | [Config](config.md) |
| `geopatcher.integrations.pipekit` | pipekit operator wrappers | [Integrations](integrations.md) |

Cloud-Optimized GeoTIFF reads and the shared object-store pool live in
[geotoolz-cloud](../../cloud/index.md) (`geocloud`); `geopatcher.fields.CogField`
is their `Field`.

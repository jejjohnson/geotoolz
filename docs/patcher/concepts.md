# Concepts

`geopatcher` answers one question: *what slice of the data does my
operator see at once, and how do local outputs become a global field?*
It answers it with four independent axes over a `Field`. This page is the
mental model; exact edge, fill and determinism rules live in the
[Behaviour reference](patching.md).

## The four-axis abstraction

Each axis is a small strategy object you pick on its own. Swapping one
never forces a change in the other three.

![SpatialPatcher: a Field is split along the Geometry, Sampler, Window and Aggregation axes, your operator runs on each Patch, and the aggregation merges the patches into a stitched field](../assets/diagrams/patcher-axes.png)

### Geometry — the shape of a patch

The geometry turns an anchor into the cells a patch covers.

| `spatial.geometry.` | Patch shape | Domain |
|---|---|---|
| `Rectangular(size=(h, w))` | an `h × w` window; takes `boundary=` for the edges | raster, grid |
| `SphericalCap(radius_km)` | a cap of great-circle radius on the sphere | grid |
| `KNNGraph(k)` | the `k` nearest neighbours | points |
| `RadiusGraph(radius)` | every neighbour within a metric radius | points |
| `PolygonIntersection(polygons)` | an arbitrary polygon | raster, vector |

### Sampler — where patches go

The sampler places anchors. Overlap is not a parameter: it follows from
the stride relative to the geometry size (`overlap = size − step`).

| `spatial.sampler.` | Anchors |
|---|---|
| `RegularStride(step)` | a regular grid |
| `JitteredStride(step, jitter, seed)` | a grid plus bounded noise |
| `Random(n_samples, seed)` | uniform random |
| `PoissonDisk(min_dist, seed)` | random with a minimum spacing |
| `Explicit(anchors_)` | pixel anchors you supply |
| `ExplicitCoords(coords, crs)` | map coordinates you supply, centred |
| `AlongTrack(track, spacing, crs)` | evenly spaced along a track |

### Window — how each cell is weighted

The window weights each cell of a patch. Overlap-add divides by the
summed weights, so a taper feathers the seams.

| `spatial.window.` | Weights |
|---|---|
| `Boxcar()` | flat 1.0 — exact for non-overlapping tiles |
| `Hann()` | periodic cosine taper |
| `Tukey(alpha)` | flat top with cosine flanks |
| `Gaussian(sigma)` | radial Gaussian, never zero |
| `Custom(fn)` | any callable of the geometry |

### Aggregation — how local outputs merge

The aggregation folds patches into a result on the domain's grid.

| `spatial.aggregation.` | Result |
|---|---|
| `OverlapAdd` | weighted sum ÷ summed weights; `streaming=True` writes zarr or a COG |
| `Sum`, `Mean`, `Max`, `Min`, `WeightedSum`, `Variance`, `InvVarWeightedMean` | per-cell statistics |
| `HardVote`, `SoftVote` | per-cell class votes |
| `Median`, `Mode`, `Learned` | per-cell, but need every patch in memory |
| `MeanStd`, `MinMax`, `ApproxQuantile`, `ApproxMode`, … | one summary for the whole field |
| `ByIndex` | the `[(anchor, data), …]` pairs, for ragged geometries |

## Patch lifecycle

A run reads each anchor once, hands you a `Patch`, and merges what you
hand back. `split` is an iterator, so only the patches you hold are in
memory.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 256, 256), dtype=np.float32),                   # (1, 256, 256) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(48, 48)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)

patch: gp.Patch = next(patcher.split(field))
print(patch.anchor, patch.indices)                  # (0, 0) Window(col_off=0, row_off=0, width=64, height=64)
weights: np.ndarray = patch.weights                 # (64, 64) float64

doubled: list[gp.Patch] = [
    p.with_data(np.asarray(p.data) * 2.0)           # (1, 64, 64) float32 → (1, 64, 64) float32
    for p in patcher.split(field)
]
merged: np.ndarray = patcher.merge(doubled, field.domain)          # (1, 256, 256) float64
stitched: GeoTensor = patcher.merge_to_field(doubled, field)       # (1, 256, 256) float32, georeferenced
```

| `Patch` attribute | Set by | Holds |
|---|---|---|
| `data` | the field's `select` | the payload — a `GeoTensor`, `DataArray`, `GeoDataFrame`, … |
| `anchor` | the sampler | the patch origin (`(row, col)` on a raster) |
| `indices` | the geometry | what was read: a rasterio `Window`, neighbour ids, a polygon |
| `weights` | the window | per-cell weights for the merge |

`patch.with_data(new)` keeps the anchor, indices and weights, so your
operator only touches the payload. `merge` returns the aggregation's raw
output; `merge_to_field` rebuilds the field's carrier around it.

## Fields and domains

A `Field` is anything with three members: `domain` (I/O-free metadata
such as shape, CRS and transform), `select(indexer)` (read one patch) and
`with_data(array)` (rebuild the carrier). The raster path wraps a
georeader `GeoTensor` or lazy reader in `geopatcher.RasterField`; the
other adapters live in `geopatcher.fields` behind extras. The
[Fields and domains table](patching.md#fields-and-domains) lists each
adapter and its domain.

Samplers and geometries only read the domain. Patches read pixels, one
window at a time, so a lazy reader never loads the whole scene.

## Streaming and eager

Streaming is the default: `split` yields one patch at a time and a
streaming-safe aggregation folds it in and drops it. Materialise with
`list(patcher.split(field))` when the field is small. For outputs bigger
than RAM, `OverlapAdd(streaming=True)` keeps its accumulators on disk —
see [Stream to disk](recipes/streaming-overlap-add.md); which
aggregations can stream is in
[Streaming aggregations](patching.md#streaming-aggregations).

## Patcher families

| Patcher | Splits | Typical use |
|---|---|---|
| `SpatialPatcher` | space — raster, grid, points, polygons | sliding-window inference, tiling, training chips |
| `AsyncSpatialPatcher` | space, over an `AsyncField` | high-latency cloud reads — [Async and prefetch](patching/async-prefetch.md) |
| `TemporalPatcher` | a time axis | lookback / horizon windows, forecasts |
| `SpatioTemporalPatcher` | space × time | dense cubes, event-triggered patches |
| `geopatcher.matched.MatchedSpatialPatcher` (and temporal siblings) | several co-registered sources | multi-sensor matchups — [API](api/matched.md) |

The temporal side mirrors the four axes and adds coordinate-aware
`TimeStencil` windows: see [Temporal patching](recipes/temporal-patching.md)
and [Temporal stencils](recipes/temporal-stencils.md).

## Random access

`split` is the canonical, lazy path. ML loaders that need `dataset[i]`
wrap the patcher and field in `geopatcher.run.IndexedPatchView`, a
picklable `Sequence[Patch]` — see
[xarray N-D patching](recipes/xarray-nd-patching.md).

## Where the framework draws the line

- **Unstructured meshes** (`uxarray`) have no `Field` adapter yet.
- **Hierarchical patching** (patches of patches) is a recipe, not a class:
  see [Stream to disk](recipes/streaming-overlap-add.md#patcher-of-patchers).
- **Global context** (normalising by the scene's mean) runs as two passes
  through `patcher.reduce` and `patcher.two_pass`: see
  [Global statistics](recipes/global-statistics.md).

The [design decisions](decisions.md) record why each of these choices
was made.

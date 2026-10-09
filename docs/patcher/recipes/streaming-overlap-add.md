# Stream to disk

Stitch an output bigger than RAM: `spatial.aggregation.OverlapAdd(streaming=True)`
keeps its two accumulators in a zarr store on disk, so peak memory is one
patch plus one store block. It needs the `[streaming]` extra (zarr ≥ 3);
see [Install](../index.md#install).

## Stream patches into a zarr store

Feed the merge a generator, so only one patch is alive at a time.

```python
import numpy as np
import rasterio
import zarr
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.outer(np.linspace(0, 1, 256), np.linspace(0, 1, 256)).astype(np.float32)[None],  # (1, 256, 256) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(32, 32)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
on_disk: gp.spatial.aggregation.OverlapAdd = gp.spatial.aggregation.OverlapAdd(
    streaming=True,
    target_path="out/scene.zarr",
    chunks=(64, 64),                                    # match the patch size
)

result: zarr.Array = on_disk.merge(
    (p.with_data(np.asarray(p.data) * 2.0) for p in patcher.split(field)),  # (1, 64, 64) float32 each
    field.domain,
)                                                       # (1, 256, 256) float32, on disk
values: np.ndarray = result[:]                          # (1, 256, 256) float32, read back
```

- **What lands on disk.** The result is `<target_path>/rec.zarr`; the
  summed weights stay in `wsum.zarr`. Open it later with `zarr.open` or
  `xarray.open_zarr`.
- **Normalisation.** The final `Σ w·x / Σ w` runs one block at a time,
  with `fill_value` (NaN by default) where `Σ w = 0`.

## Write a Cloud-Optimized GeoTIFF

`writer="cog"` streams through a temporary zarr store beside the output,
then converts it block by block into a tiled COG with overviews and
`nodata = fill_value`.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.ones((1, 512, 512), dtype=np.float32),       # (1, 512, 512) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(96, 96)),
    window=gp.spatial.window.Hann(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
to_cog: gp.spatial.aggregation.OverlapAdd = gp.spatial.aggregation.OverlapAdd(
    streaming=True,
    target_path="scene.tif",
    writer="cog",
    cog={"blocksize": 256, "compress": "DEFLATE"},
)
path: str = to_cog.merge(patcher.split(field), field.domain)    # "scene.tif", (1, 512, 512) float32
```

The parent directory must exist. `chunks` defaults to the COG block size. Other `cog` keys pass through as
GDAL COG creation options, such as `overview_resampling`.

## Patcher of patchers

Run a large scene super-tile by super-tile: an outer patcher cuts
1024-pixel tiles, an inner patcher runs the model on each, and the outer
merge writes tiles to disk. This is a recipe, not a class.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.random.default_rng(0).random((1, 2048, 2048), dtype=np.float32),  # (1, 2048, 2048) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
outer: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(1024, 1024)),
    sampler=gp.spatial.sampler.RegularStride(step=(1024, 1024)),
    window=gp.spatial.window.Boxcar(),                  # super-tiles do not overlap
    aggregation=gp.spatial.aggregation.OverlapAdd(
        streaming=True, target_path="out/hier.zarr", chunks=(1024, 1024)
    ),
)
inner: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(256, 256)),
    sampler=gp.spatial.sampler.RegularStride(step=(192, 192)),
    window=gp.spatial.window.Gaussian(),                # never zero: no NaN ring per tile
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)


def model(chip: np.ndarray) -> np.ndarray:
    """Stand-in for a per-chip model."""
    return chip * 2.0                                   # (1, 256, 256) float32 → (1, 256, 256) float32


def run_inner(tile: GeoTensor) -> np.ndarray:
    """Patch one super-tile with the inner patcher and stitch it in RAM."""
    sub: gp.RasterField = gp.RasterField(tile)          # (1, 1024, 1024) float32
    chips = (p.with_data(model(np.asarray(p.data))) for p in inner.split(sub))
    return inner.merge(chips, sub.domain)               # (1, 1024, 1024) float64


tiles = (p.with_data(run_inner(p.data)) for p in outer.split(field))
mosaic = outer.aggregation.merge(tiles, field.domain)  # zarr.Array (1, 2048, 2048) float32
```

Peak memory is one super-tile plus one chip, which suits a Dask or
Kubernetes worker. Pick the inner stride so it tiles the super-tile
exactly (`4 × 192 + 256 = 1024` here), or set `boundary="pad"`.

## Bound the iterator

`split(max_in_flight=...)` caps the number of live patches, and
`max_in_flight_bytes=...` caps their total payload, so the reader never
runs ahead of the operator.

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.zeros((1, 512, 512), dtype=np.float32),      # (1, 512, 512) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(128, 128)),
    sampler=gp.spatial.sampler.RegularStride(step=(128, 128)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)

total: float = 0.0
for patch in patcher.split(field, prefetch=2, max_in_flight_bytes=4 * 128 * 128 * 4):
    with patch:                                         # releases the slot on exit
        total += float(np.asarray(patch.data).sum())   # (1, 128, 128) float32
```

A patch's size is its payload's `.nbytes`, so a lazy dask chip is budgeted
without computing it. With `prefetch=`, leaving the loop early is safe:
close or drop the iterator and the background reader stops.

## Pitfalls

- **`chunks` is required** for `writer="zarr"`. Match the patch size so
  each patch writes one block; leading dims missing from `chunks` get
  their full extent.
- **dtype.** The store defaults to `dtype="float32"`; the in-RAM path is
  float64. Pass `dtype="float64"` to match bit for bit.
- **Re-runs.** Merging onto an existing store raises `FileExistsError`;
  pass `overwrite=True` to replace it.
- **Not every aggregation streams.** `Median`, `Mode` and `Learned` need
  every patch in RAM — see [Streaming aggregations](../patching.md#streaming-aggregations).

# Cache patch reads across runs

Skip the source reads when you rerun a patcher on the same data — while
iterating on an operator, say. `geopatcher.run.PatchCache` is a
content-addressed on-disk cache: the second process reads only the
field's `domain` metadata, never the pixels.

## Cache a split

```python
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp
from geopatcher.run import IndexedPatchView, PatchCache

field: gp.RasterField = gp.RasterField(
    GeoTensor(
        np.random.default_rng(0).random((1, 256, 256), dtype=np.float32),  # (1, 256, 256) float32
        transform=rasterio.Affine(10, 0, 500_000, 0, -10, 4_300_000),
        crs="EPSG:32611",
    )
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(64, 64)),
    sampler=gp.spatial.sampler.RegularStride(step=(64, 64)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
cache: PatchCache = PatchCache(
    "./.geopatcher_cache",
    max_bytes=20 * 2**30,                               # 20 GiB, least recently used first out
    field_id="synthetic-scene",                         # in-memory fields need a stable name
)

run1: list[gp.Patch] = list(patcher.split(field, cache=cache))  # 16 reads, 16 entries written
run2: list[gp.Patch] = list(patcher.split(field, cache=cache))  # 16 hits, zero source reads
print(cache.stats())                                  # {'hits': 16, 'misses': 16, 'bytes': …, 'entries': 16}
view: IndexedPatchView = IndexedPatchView(patcher, field, cache=cache)  # random access, same cache
```

It composes with `journal=` and `prefetch=`, and with
`patcher.patch_at(field, anchor, cache=cache)`.

## What the key covers

Each entry is keyed by `sha256(field_id ‖ geometry and window config ‖ anchor)`.
`field_id` covers everything the field reads:

- **The source.** A reader's file paths (real path, mtime and size of
  every file), a `url`, or the `encoding["source"]` file of an
  `xr.open_dataset` / `rioxarray.open_rasterio` array.
- **The domain.** CRS, transform, shape and dtype, or a digest of the
  grid coordinates for `XarrayField`.
- **The reader's** band selection and boundless fill.
- **The adapter's own `cache_id()`.** `CogField` adds its store, path and
  IFD; `ReprojectingRasterField` its `dst_crs`, `resolution` and
  `resampling`.

A custom `Field` can define `cache_id() -> str`. It is trusted, not
checked: it must change whenever the patches could. Raise
`geopatcher.run.UnstableIdentityError` rather than return an identity
that can change.

## Pitfalls

- **In-memory fields** (`GeoTensor`- or `DataArray`-backed) have no stable
  identity: pass `field_id=`, or the split raises `ValueError`. Do the
  same for a file-backed `DataArray` you changed in memory.
- **Objects overwritten in place.** A plain `url` carries no version.
  `CogField` sends one `HEAD` per split (ETag, else size and
  last-modified); for other URL fields, put a version in `cache_id()` or
  call `cache.clear()`.
- **Some carriers cannot be cached.** Hits are bit-identical: a
  `GeoTensor` comes back with its transform, CRS, fill and attrs, a
  `DataArray` with its dims, coords, attrs and encoding. A carrier that
  cannot round-trip (a `GeoDataFrame`, an object array, a `datetime`
  attribute) raises `TypeError`; drop `cache=` for that field.
- **Only real reads are stored.** `on_error="mask"` placeholders are not
  written. A damaged entry is a miss and is rewritten; writes are atomic.
- **The size cap is per instance.** With several processes writing one
  directory, `max_bytes` applies per writer. An entry larger than
  `max_bytes` is not stored, with a `RuntimeWarning`.
- **Format changes orphan entries.** Keys include a cache format version;
  after an upgrade that bumps it, old entries are never hit again. Call
  `cache.clear()` to reclaim the space.

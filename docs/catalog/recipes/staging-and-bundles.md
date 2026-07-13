# Recipe: Staging and bundles

Two adjacent concepts that often get conflated:

- **`stage()`** — copy remote URIs into a *local file cache* so
  loaders can read them without going back to S3 / Azure / HTTPS.
- **`CatalogBundle.ingest()`** — pull rows from a `Source` (STAC,
  EarthAccess, CMR) into a persistent on-disk *bundle* with full
  provenance.

They compose well together (`bundle.ingest(...)` discovers, then
`stage(bundle.catalog, ...)` materialises), but you can use either
without the other.

## When to use `stage()`

Use `stage()` when:

- Your catalog rows point at `s3://` / `https://` / `az://` URIs and
  your loader / model trainer does many reads per file.
- You're hitting per-request rate limits (PC blob storage, S3
  unsigned reads).
- You want byte-reproducible inputs for a benchmark run.

Skip `stage()` when:

- The catalog rows are already local.
- You're doing one read per file (the cache miss is the only read
  — staging just adds an extra disk hop).
- You're working with very large files where the cache root would
  fill the disk faster than you can use the data.

### The basic flow

```python
import geocatalog as gc
from geocatalog.staging import LocalCache, stage

remote_catalog = gc.from_stac_search(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    collections=["sentinel-2-l2a"],
    bbox=(-120.25, 38.85, -119.85, 39.30),
    datetime="2024-06-01/2024-09-30",
)

cache = LocalCache(root="/scratch/geocatalog-cache", ttl_days=30)
local_catalog = stage(
    remote_catalog,
    cache=cache,
    parallel=8,            # threads — fsspec releases the GIL on I/O
    retries=3,
)

# local_catalog.gdf['filepath'] now points to local paths.
tensor = gc.load_raster(local_catalog, aoi, band_indexes=[1])
```

`stage()` returns a **new** catalog with rewritten `filepath`s
(originals preserved under `extras["_staged_from"]`). The input is
not mutated. Cache hits are detected by sha256(uri); a re-run is a
no-op if the cache is warm.

### `LocalCache` tuning

```python
cache = LocalCache(
    root="/scratch/geocatalog-cache",   # default: $GEOCATALOG_CACHE or ~/.cache/geocatalog
    ttl_days=30,                         # None = forever
)
```

The cache layout is `{root}/{sha256(uri)[:2]}/{sha256(uri)}{ext}` —
the two-letter prefix keeps any one directory under a few thousand
entries on a large catalog.

### Asset selection

When the row has a JSON-encoded asset map (typical for STAC bundle
rows), pass `assets=[...]` to fetch only the bands you need:

```python
local = stage(remote_catalog, cache=cache, assets=["B04", "B08"])
```

### Error handling

```python
local = stage(remote, cache=cache, on_error="skip")
```

- `on_error="raise"` (default) — first failure aborts the stage.
- `on_error="skip"` — keep the original URI in the asset map and
  continue. Useful when 1-2% of assets are flaky.

## When to use `CatalogBundle.ingest()`

Use a bundle when:

- You need provenance — *which query produced which rows?*
- You'll re-run the ingest periodically.
- You're combining multiple `Source`s into one catalog.
- You want a self-contained directory you can ship to a collaborator.

Skip bundles when:

- The catalog is short-lived (a notebook scratch space).
- The discovery is a one-liner (use `from_stac_search` instead).

### The basic flow

```python
import pandas as pd
from geocatalog.bundle import CatalogBundle
from geocatalog.sources import STACSource, EarthAccessSource

bundle = CatalogBundle.empty(target_crs="EPSG:4326")

# Discover Sentinel-2 from STAC...
bundle.ingest(
    STACSource.planetary_computer(),
    bounds=(-120.25, 38.85, -119.85, 39.30),
    interval=pd.Interval(
        pd.Timestamp("2024-06-01", tz="UTC"),
        pd.Timestamp("2024-09-30", tz="UTC"),
        closed="both",
    ),
    collection="sentinel-2-l2a",
    filters={"eo:cloud_cover": {"lt": 20}},
)

# ...and HLS L30 from NASA EarthAccess.
bundle.ingest(
    EarthAccessSource(),
    bounds=(-120.25, 38.85, -119.85, 39.30),
    interval=pd.Interval(
        pd.Timestamp("2024-06-01", tz="UTC"),
        pd.Timestamp("2024-09-30", tz="UTC"),
        closed="both",
    ),
    collection="HLSL30.v2.0",
)

print(f"items: {len(bundle.catalog)}")
print(f"queries: {len(bundle.queries)}   # 2 — one per ingest call")

bundle.to_directory("./tahoe_summer_2024_bundle/")
```

The directory layout:

```
tahoe_summer_2024_bundle/
  items.parquet       # the catalog rows (GeoParquet 1.1)
  queries.parquet     # one row per ingest() call
  matchups.parquet    # downstream matchups (empty here)
  manifest.json       # schema version + target CRS
```

Reload anywhere:

```python
bundle = CatalogBundle.from_directory("./tahoe_summer_2024_bundle/")
```

## Composing: bundle, then stage

The two patterns compose cleanly. Discover with a bundle (for
provenance), then stage the catalog locally for fast loader access:

```python
bundle = CatalogBundle.empty(target_crs="EPSG:4326")
bundle.ingest(STACSource.planetary_computer(), bounds=..., interval=...)

local_catalog = stage(
    bundle.catalog,
    cache=LocalCache(root="/scratch/cache"),
    assets=["B04", "B08"],   # only what we need
)

# Save the bundle for provenance; train against the staged catalog.
bundle.to_directory("./bundle/")
```

## See also

- [Recipes: STAC ingestion](from-stac.md) — `from_stac_search` vs `STACSource`
- [Recipes: large archives](large-archives.md) — when the catalog itself is too big to stage everything
- [API: `stage`](../api/reference.md)
- [API: `CatalogBundle`](../api/reference.md)
- [Catalog backends walkthrough ↗](https://github.com/jejjohnson/research_notebook/blob/main/projects/geostack/notebooks/catalog/02_backends.ipynb) — raster (Sentinel-2) + xarray (Copernicus DEM) + vector (Natural Earth) backends end-to-end.

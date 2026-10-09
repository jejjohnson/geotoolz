# Stage remote files

`geocatalog.staging.stage` copies the remote files a catalog points at into
a local cache, and returns the same catalog pointing at the local copies.
Remote URIs need the `[cloud]` extra.

## When to stage

Stage when you read each file many times, when the archive rate-limits
you, or when a benchmark needs the same bytes on every run. Skip it when
the rows are already local, or when you read each file once: the download
would be the only read.

## Stage a catalog

```python
import tempfile

import pandas as pd
import planetary_computer
import pystac_client
from georeader.geotensor import GeoTensor

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog
from geocatalog.staging import LocalCache, stage

client: pystac_client.Client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,                        # sign every asset href
)
remote: InMemoryGeoCatalog = gc.sources.from_stac_search(
    client,
    collections=["sentinel-2-l2a"],
    bounds=(-120.10, 39.05, -120.05, 39.10),                         # Lake Tahoe west shore
    datetime="2024-06-01/2024-06-30",
    asset_key="SCL",                                                 # scene classification, 20 m, small
    limit=2,
)                                                                    # 2 rows of https:// URLs

cache: LocalCache = LocalCache(root=tempfile.mkdtemp(), ttl_days=30)
local: InMemoryGeoCatalog = stage(remote, cache=cache, parallel=8)  # 2 rows of local paths

aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(-120.10, 39.05, -120.05, 39.10),
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(0.0002, 0.0002),                                     # ~20 m
    crs="EPSG:4326",
)
scl: GeoTensor = gc.load.load_raster(local, aoi)                     # (1, 250, 250) uint8, read from disk
```

What `stage` does:

- **Returns a new catalog.** Each `filepath` follows the row's primary
  asset, and a JSON `staged_from` column keeps the original URIs. The
  input is not changed.
- **Leaves local paths in place.** Staging a local catalog needs no extra.
- **Downloads once per URI** through
  [`geocloud.files`](../../cloud/index.md#moving-files-geocloudfiles), into
  a temp file renamed into place when complete. Credentials come from
  [`geocloud.credentials`](../../cloud/index.md#credentials-geocloudcredentials).
- **Skips cache hits,** so a second run with a warm cache downloads
  nothing.

## Tune the cache

`LocalCache(root=, ttl_days=, timeout=)` sets the directory, the lifetime
and the per-request timeout. The defaults are `$GEOCATALOG_CACHE` or
`~/.cache/geocatalog`, no expiry, and 60 s per 16 MiB range.

Files land at `{root}/{key[:2]}/{key}{ext}`. The `key` is the sha256 of
the URI without its expiring signature (Azure SAS, `X-Amz-*`, `X-Goog-*`,
CloudFront), so a re-signed URL hits the same file. `cache.prune()`
deletes expired files and abandoned `*.part` downloads.

## Pick assets and handle errors

| Argument | Effect |
| --- | --- |
| `assets=["B04", "B08"]` | fetch only these keys of a row's JSON `assets` map (bundle rows); an unknown key raises `ValueError` |
| `on_error="raise"` (default) | the first failure stops the stage and cancels pending downloads |
| `on_error="skip"` | keep the original URI for a failed asset and continue |
| `retries=3` | retry transient network errors; `FileNotFoundError` is not retried |

A [bundle](stac-ingestion.md#record-provenance-with-a-bundle) and staging
compose: ingest for provenance, then `stage(bundle.catalog, assets=[...])`
for fast reads. To hand a staged catalog to a patcher, see
[Catalog → patcher](catalog-to-patcher.md).

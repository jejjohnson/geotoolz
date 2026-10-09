# Recipe: STAC ingestion

There are two paths from a STAC API to a `GeoCatalog`. They look
similar but have different sweet spots — pick by *what you need to
keep around afterwards*.

## TL;DR

| | `from_stac_search` | `STACSource` + `CatalogBundle` |
| --- | --- | --- |
| One-liner? | Yes | No (3-4 lines) |
| Keeps the query that produced the catalog? | No | Yes (in `bundle.queries`) |
| Per-row provenance? | Minimal | Full (`source`, `query_id`, `fetched_at`) |
| Persist as portable bundle? | `to_geoparquet` only | `bundle.to_directory(...)` |
| Reuse across many queries? | One catalog per call | One bundle accumulates multiple queries |
| Best for | Notebooks, quick scripts | Production pipelines, repeatable builds |

## Path A — `from_stac_search` (the one-liner)

The simplest path. Hands a STAC URL + collection + bbox, returns a
catalog. Equivalent to calling `pystac-client` yourself and feeding
the items into `from_stac_items`.

```python
import geocatalog as gc

catalog = gc.sources.from_stac_search(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    collections=["sentinel-2-l2a"],
    bounds=(-120.25, 38.85, -119.85, 39.30),    # Lake Tahoe
    datetime="2024-06-01/2024-09-30",
    asset_key="B04",                           # red band
    limit=100,
    extra_properties=("eo:cloud_cover", "platform"),
)

print(f"len(catalog): {len(catalog)}")
print(catalog.gdf[["filepath", "eo:cloud_cover"]].head())
```

Notes:

- `asset_key="*"` emits one row per asset on each item — handy when
  you want every band as a separate row.
- `engine="duckdb"` + `out_path="..."` streams items straight into a
  GeoParquet artifact (no in-RAM accumulation).
- `extra_properties` preserves arbitrary STAC properties as catalog
  columns — `eo:cloud_cover` is the canonical use.
- Both paths read an item the same way: the footprint is the item's
  `geometry` (its `bbox` only when the geometry is missing), split at
  the antimeridian; the `crs` column is the asset's native CRS from the
  projection extension (`proj:code`, `proj:epsg`, `proj:wkt2` or
  `proj:projjson`); times are UTC; relative asset hrefs are resolved
  against the item's self link.

### Signing Planetary Computer URLs

`from_stac_search` does **not** sign URLs by default. For Planetary
Computer assets behind blob-storage tokens, sign explicitly:

```python
import planetary_computer as pc
from pystac_client import Client
import geocatalog as gc

client = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,    # signs every Item and Asset
)
catalog = gc.sources.from_stac_search(
    client,
    collections=["sentinel-2-l2a"],
    bounds=(-120.25, 38.85, -119.85, 39.30),
    datetime="2024-06-01/2024-09-30",
)
```

`STACSource.planetary_computer()` (Path B) handles signing
automatically — that's its main convenience.

Signed hrefs carry a SAS token that expires after about an hour, so a
catalog or bundle saved with them stops opening once the token lapses.
Rows whose `filepath` is signed have `href_signed=True`; re-sign those
hrefs (for example with `planetary_computer.sign(href)`) before reading
a persisted catalog.

## Path B — `STACSource` + `CatalogBundle.ingest`

The provenance-preserving path. Use this when you'll re-run the
ingest periodically, or when downstream consumers need to know
*exactly* which query produced the catalog.

```python
import pandas as pd
import geocatalog as gc
from geocatalog.storage import CatalogBundle
from geocatalog.sources import STACSource

bundle = CatalogBundle.empty(crs="EPSG:4326")

bundle.ingest(
    STACSource.planetary_computer(),         # signs URLs automatically
    bounds=(-120.25, 38.85, -119.85, 39.30),
    interval=pd.Interval(
        pd.Timestamp("2024-06-01", tz="UTC"),
        pd.Timestamp("2024-09-30", tz="UTC"),
        closed="both",
    ),
    collection="sentinel-2-l2a",
    filters={"eo:cloud_cover": {"lt": 20}},
    limit=100,
)

print(f"len(bundle.catalog): {len(bundle.catalog)}")
print(f"len(bundle.queries): {len(bundle.queries)}")  # 1 — one ingest call
```

Multiple `ingest` calls accumulate into the same bundle. Each call
appends a `QueryRecord` to `bundle.queries` and merges new items
into `bundle.catalog`.

`(source, collection, id)` is the items table's primary key (STAC item
ids are only unique within a collection). Re-ingesting an item that is
already in the bundle raises by default; pass `on_duplicate="skip"` to
make a repeated ingest a no-op, or `on_duplicate="replace"` to refresh
the stored rows.

### Persisting the bundle

```python
bundle.to_directory("./tahoe_s2_summer_2024/")
# Layout:
#   tahoe_s2_summer_2024/
#     items.parquet         # the catalog rows
#     queries.parquet       # one row per ingest() call
#     matchups.parquet      # (empty here)
#     manifest.json
```

A bundle directory round-trips cleanly:

```python
bundle = CatalogBundle.from_directory("./tahoe_s2_summer_2024/")
```

Pass `bundle.catalog` anywhere a `GeoCatalog` is expected.

## Which to use when

- **You're prototyping or exploring** → Path A. Less code, no extra
  files.
- **You need to repeat this query weekly** → Path B. The bundle
  records the query, so the next run is a diff.
- **You're ingesting from multiple sources** (STAC + EarthAccess +
  CMR) into one catalog → Path B. The bundle is the merge point.
- **You need a single GeoParquet artifact** → Path A with
  `engine="duckdb"` + `out_path=...`.

## Performance tips

- `limit=None` plus wide `bounds` can return tens of thousands of
  items — STAC paging is the bottleneck, not the catalog build. Use
  `limit=...` defensively while exploring. `limit=0` is a dry run on
  every source (no request, no rows); `limit=None` is unbounded.
- `STACSource` caches the underlying `pystac_client.Client` across
  calls; reuse the same `STACSource` instance for multiple ingests
  to skip the root-catalog fetch.
- For 10⁵+ STAC items, switch to `engine="duckdb"` so the catalog
  streams to disk instead of accumulating in RAM.

## See also

- [Quickstart](../quickstart.md) — full Lake Tahoe walkthrough
- [Recipes: staging & bundles](staging-and-bundles.md) — what
  `bundle.ingest` actually does
- [Recipes: large archives](large-archives.md) — when the STAC query
  itself returns 10⁶+ items
- [API: `STACSource`](../api/sources.md)
- [Catalog intro walkthrough ↗](https://github.com/jejjohnson/research_notebook/blob/main/projects/geostack/notebooks/catalog/01_intro.ipynb) — build → query → load against a real eight-scene Sentinel-2 archive on MPC.

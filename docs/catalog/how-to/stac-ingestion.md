# Ingest from STAC

Turn a STAC API search into a catalog. Use `from_stac_search` for a
one-off catalog, and `STACSource` with a `CatalogBundle` when you must
keep the queries that produced the rows. Both need the `[stac]` extra.

| | `from_stac_search` | `STACSource` + `CatalogBundle.ingest` |
| --- | --- | --- |
| Code | one call | three lines |
| Keeps the query | no | yes, in `bundle.queries` |
| Several queries or sources in one catalog | no | yes |
| Saved as | GeoParquet (`to_geoparquet`) | a bundle directory |
| Streams to disk for 10⁵+ items | yes (`engine="duckdb"`) | no |

## Build a catalog with `from_stac_search`

Open the client with Planetary Computer's signer, then name the asset each
row should point at:

```python
import planetary_computer
import pystac_client

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog

client: pystac_client.Client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,                        # sign every asset href
)
catalog: InMemoryGeoCatalog = gc.sources.from_stac_search(
    client,
    collections=["sentinel-2-l2a"],
    bounds=(-120.25, 38.85, -119.85, 39.30),                         # Lake Tahoe, lon/lat
    datetime="2024-06-01/2024-09-30",
    asset_key="B04",                                                 # one row per item → its red band
    limit=50,
    extra_properties=["eo:cloud_cover", "platform"],                 # kept as columns
)                                                                    # ≤ 50 rows
```

The options that matter:

- **`asset_key`** defaults to `"data"`, which Sentinel-2 items do not have.
  Name a band, or pass `"*"` for one row per asset.
- **`extra_properties`** copies STAC properties into columns, so you can
  filter with `catalog.where("`eo:cloud_cover` < 10")`.
- **`engine="duckdb"` with `out_path=`** streams items into GeoParquet
  instead of RAM.
- **`crs=`** reprojects the lon/lat footprints. The `crs` column always
  holds each asset's native CRS from the projection extension.

Each footprint is the item's `geometry`, split at the antimeridian. Times
are UTC, and relative hrefs are resolved against the item's self link.
`from_stac_items` builds the same catalog from items you already have.

## Signed URLs

A signed href carries a token that expires after about an hour. Rows with
a signed `filepath` have `href_signed=True`. Before reading a saved
catalog, re-sign those hrefs (`planetary_computer.sign(href)`), or
[stage](staging.md) the files while the tokens are fresh.

## Record provenance with a bundle

`CatalogBundle.ingest` runs a `Source.query`, adds the rows and records the
call as a `QueryRecord`. Repeated calls accumulate into one catalog:

```python
import tempfile
from pathlib import Path

import pandas as pd

from geocatalog.backends import InMemoryGeoCatalog
from geocatalog.sources import STACSource
from geocatalog.storage import CatalogBundle

summer: pd.Interval = pd.Interval(
    pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-06-30", tz="UTC"), closed="both"
)
tahoe: tuple[float, float, float, float] = (-120.25, 38.85, -119.85, 39.30)
source: STACSource = STACSource.planetary_computer()                 # signs hrefs for you

bundle: CatalogBundle = CatalogBundle.empty(crs="EPSG:4326")
bundle.ingest(source, bounds=tahoe, interval=summer, collection="sentinel-2-l2a",
              filters={"eo:cloud_cover": {"lt": 20}}, limit=10, tag="tahoe")
bundle.ingest(source, bounds=tahoe, interval=summer, collection="landsat-c2-l2",
              limit=10, tag="tahoe")

catalog: InMemoryGeoCatalog = bundle.catalog                         # ≤ 20 rows, both collections
out: Path = Path(tempfile.mkdtemp()) / "tahoe_bundle"
bundle.to_directory(out)
reloaded: CatalogBundle = CatalogBundle.from_directory(out)
```

`bundle.queries` now holds two records. Bundle rows keep every asset in a
JSON `assets` map; `filepath` points at `primary_asset`, or the first
asset when you do not name one.

The saved directory:

```text
tahoe_bundle/
  items.parquet      # the catalog rows (GeoParquet 1.1)
  queries.parquet    # one QueryRecord per ingest() call
  matchups.parquet   # only after bundle.write_matchups(...)
  _meta.json         # bundle schema version, CRS ("target_crs"), kind ("backend"), timestamps
```

`(source, collection, id)` is the items' primary key. Re-ingesting a row
already in the bundle raises; pass `on_duplicate="skip"` to ignore it, or
`on_duplicate="replace"` to refresh it.

## Pitfalls

- **Unbounded searches.** `limit=None` over a wide box can return tens of
  thousands of items, and STAC paging dominates the time. `limit=0` is a
  dry run that sends no request.
- **Client reuse.** A `STACSource` caches its `pystac_client.Client`;
  reuse one instance across ingests.
- **Other archives.** `EarthAccessSource` and `CMRSource` plug into
  `ingest` the same way; see the [Sources API](../api/sources.md).

Next, [stage](staging.md) the assets for repeated reads, or
[match](matchup.md) the two collections.

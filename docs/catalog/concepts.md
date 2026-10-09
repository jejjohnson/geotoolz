# Concepts

A catalog is an index of files, not of pixels. Each row holds one file's
footprint, time interval, CRS and path. A query reads only that index, and
a loader opens only the files that overlap.

![geocatalog: sources are built into an InMemory or DuckDB catalog, queried with a GeoSlice, and only the hits are loaded as a GeoTensor or a geopatcher RasterField](../assets/diagrams/catalog-flow.png)

Work moves through three layers:

1. **Discover and build.** A `Source` adapter or a local builder turns
   files into rows.
2. **Index.** A `GeoCatalog` holds the rows and answers queries.
3. **Load.** A loader reads the hits onto a `GeoSlice` grid.

## GeoSlice

A `geocatalog.GeoSlice` is one request for data: `bounds`, `interval`,
`resolution` and `crs`. Catalogs answer it, loaders read onto its grid, and
patchers produce it.

```python
import dataclasses

import pandas as pd

import geocatalog as gc

aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(500_000, 4_300_000, 505_000, 4_303_000),                 # xmin, ymin, xmax, ymax in metres
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(10.0, 10.0),                                         # x, y pixel size
    crs="EPSG:32611",
)
shape: tuple[int, int] = aoi.shape                                   # (300, 500) = (H, W)
wider: gc.GeoSlice = dataclasses.replace(aoi, bounds=(499_000, 4_300_000, 505_000, 4_303_000))
```

The rules of the contract:

- **Frozen.** Change a slice with `dataclasses.replace`. Slices hash, so
  they work as cache and dict keys.
- **Equal by meaning.** CRSs that PROJ considers equivalent compare equal
  even when their WKT differs.
- **Naive UTC time.** A tz-aware interval is stored as naive UTC, like
  catalog rows. Convert with `geocatalog.utils.to_naive_utc`.
- **Rounded shape.** `shape` rounds `extent / resolution` half up. For an
  exact pixel grid, see [Grid alignment](how-to/grid-alignment.md).

## Row schema

Every backend stores the same columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `geometry` | shapely geometry | The file's footprint, in the catalog CRS |
| `start_time`, `end_time` | naive UTC timestamp | Time interval, also the `IntervalIndex` (`closed="both"`) |
| `filepath` | str | Local path or URI |
| `crs` | str | The file's own CRS, which may differ from the catalog CRS |

Builders add their own columns:

- **xarray** adds `n_timesteps` and `time_var`.
- **vector** adds `layer`.
- **STAC** adds `asset_key`, `stac_item_id`, `stac_collection`,
  `href_signed`, plus any `extra_properties`.
- **Bundles** add `id`, `source`, `collection`, `assets` (JSON) and
  `provenance`.

A query is two index lookups intersected: an R-tree or bbox column for
space, the `IntervalIndex` for time. Files without a date get a sentinel
interval (`geocatalog.utils.TIME_INVARIANT_START` to `TIME_INVARIANT_END`)
so they match any time. Saved catalogs carry a schema version; see
[Schema versions](schema-versions.md).

## Backends

Two classes satisfy the `GeoCatalog` protocol, so code that accepts a
`GeoCatalog` takes either one:

| | `InMemoryGeoCatalog` | `DuckDBGeoCatalog` |
| --- | --- | --- |
| Install | base | `[duckdb]` |
| Storage | `GeoDataFrame` in RAM | GeoParquet 1.1 on disk, `s3://`, `hf://` |
| Spatial index | R-tree | per-row `bbox` column, row-group pruning |
| Scale | up to ~10⁵ rows | 10⁶+ rows |
| Build | eager | streamed in bounded memory (`engine="duckdb"`) |
| Extras off the protocol | `where(pandas_query)`, `intersect(join=…)` | `sql(where=…)`, `to_geoparquet`, `materialize()` |

Start with InMemory; move to DuckDB when the row count or a remote
artifact calls for it. `geocatalog.open_catalog` picks DuckDB when it is
installed, or takes `engine="duckdb"` / `engine="memory"`. Building and
querying archives at that scale is the
[Large archives](how-to/large-archives.md) how-to.

## Set algebra

`query`, `intersect` and `union` each return a new catalog and leave their
inputs untouched:

| Call | Returns |
| --- | --- |
| `gc.query(cat, slice_)` or `cat.query(slice_)` | rows overlapping the slice in space and time |
| `gc.intersect(left, right)` | one row per overlapping pair, clipped to the shared footprint and interval |
| `gc.intersect(left, right, spatial_only=True)` | the same, ignoring time: imagery against static labels |
| `gc.union(left, right)` | every row of both, in `left`'s CRS, without de-duplication |

```python
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog


def tile(path: Path, x0: float) -> Path:
    """A 3 km × 3 km single-band tile at 10 m, UTM 11N."""
    with rasterio.open(
        path, "w", driver="GTiff", width=300, height=300, count=1, dtype="uint8",
        crs="EPSG:32611", transform=from_origin(x0, 4_303_000, 10, 10),
    ) as dst:
        dst.write(np.ones((1, 300, 300), dtype=np.uint8))           # (1, 300, 300) uint8
    return path


root: Path = Path(tempfile.mkdtemp())
imagery: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    [tile(root / "img_20240605.tif", 500_000), tile(root / "img_20240606.tif", 503_000)],
    filename_regex=r"_(?P<date>\d{8})\.tif", crs="EPSG:32611",
)                                                                    # 2 dated rows
labels: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    [tile(root / "labels.tif", 501_500)], crs="EPSG:32611"
)                                                                    # 1 undated row

pairs: InMemoryGeoCatalog = gc.intersect(imagery, labels, spatial_only=True)  # 2 rows, clipped footprints
everything: InMemoryGeoCatalog = gc.union(imagery, labels)                    # 3 rows
```

## Provenance

A `geocatalog.sources.Source` (`STACSource`, `EarthAccessSource`,
`CMRSource`) yields `SourceRow`s: rows that still know the query that
produced them. `geocatalog.storage.CatalogBundle` collects those rows, the
queries and any matchups, and saves them as one directory.
`bundle.catalog` is an ordinary `InMemoryGeoCatalog`.

Use a bundle when you must answer *"which query produced this row?"*, or
when several sources feed one catalog. The recipe is
[Record provenance with a bundle](how-to/stac-ingestion.md#record-provenance-with-a-bundle);
the [design record](design/query-matchup.md) explains the choices.

## See also

- [Quickstart](quickstart.md): the model on local files and STAC.
- [Catalog → patcher](how-to/catalog-to-patcher.md): `field_for` and
  `CatalogDomain`.
- [Logging](logging.md): `logger.enable("geocatalog")` shows what the
  builders skip.

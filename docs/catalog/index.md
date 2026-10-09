# geotoolz-catalog

> **A spatiotemporal index over geospatial files.** Ask *"what overlaps
> this AOI between these dates?"* and get an answer in milliseconds,
> without opening a single file.

## Where it comes from

You have thousands of GeoTIFFs, NetCDFs, Zarrs or shapefiles on local disk,
in a bucket or behind a STAC API. You want the few that touch one area in
one season, and opening every file to find out takes hours.

`geocatalog` indexes each file once: its footprint, time interval, CRS and
path. Queries then touch only that index, and a loader opens only the
files that overlap. The import name is `geocatalog`; how it fits with the
other packages is on [How the packages interlock](../geostack.md).

![geocatalog: sources are built into an InMemory or DuckDB catalog, queried with a GeoSlice, and only the hits are loaded as a GeoTensor or a geopatcher RasterField](../assets/diagrams/catalog-flow.png)

## Install

```bash
pip install geotoolz-catalog
pip install 'geotoolz-catalog[duckdb,stac]'   # add extras as needed
```

| Extra | Pulls in | Needed for |
| --- | --- | --- |
| *(base)* | InMemory backend, raster + vector builders and loaders, GeoParquet round-trip | Local files, under 10⁵ rows |
| `[duckdb]` | `DuckDBGeoCatalog`, streaming builds (`engine="duckdb"`) | 10⁶+ rows, remote artifacts |
| `[streaming]` | Same as `[duckdb]` (the streaming writer itself is pyarrow) | Streaming builds |
| `[xarray-raster]` | `build_xarray_catalog`, `load_xarray` | NetCDF / Zarr |
| `[stac]` | `STACSource`, `from_stac_search`, `from_stac_items`, `to_stac_collection` | STAC API ingestion |
| `[earthaccess]` / `[gee]` | `EarthAccessSource` / `GEESource` | NASA Earthdata / Earth Engine discovery |
| `[sources-all]` | `[earthaccess]` + `[stac]` + `[gee]` | Every source adapter |
| `[cloud]` | geotoolz-cloud with its `[fsspec]` extra | `s3://` `gs://` `az://` `https://` `hf://` reads in builders and loaders (`geocloud.fs`) and `stage()` of remote URIs (`geocloud.files`), on one client pool and credential registry |
| `[patch]` | geotoolz-patcher | `geocatalog.patch.field_for` |
| `[full]` | All of the above | One-shot install |

## Quickstart

Index three scenes, ask which ones touch a 5 km box in June, and load only
those onto the box's grid:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog

# Three 4-band scenes, 6 km × 6 km at 10 m in UTM 11N, dated by file name.
root: Path = Path(tempfile.mkdtemp())
for date in ("20240605", "20240612", "20240801"):
    with rasterio.open(
        root / f"s2_{date}.tif", "w", driver="GTiff", width=600, height=600, count=4,
        dtype="uint16", crs="EPSG:32611", transform=from_origin(501_000, 4_308_000, 10, 10),
    ) as dst:
        dst.write(np.full((4, 600, 600), 1_000, dtype=np.uint16))     # (4, 600, 600) uint16

catalog: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    sorted(root.glob("*.tif")),
    filename_regex=r"s2_(?P<date>\d{8})\.tif",                        # time from the file name
    crs="EPSG:32611",
)                                                                     # 3 rows · footprints, times, paths

aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(502_000, 4_302_000, 507_000, 4_307_000),                  # 5 km × 5 km, UTM 11N metres
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(10.0, 10.0),                                          # → grid (H, W) = (500, 500)
    crs="EPSG:32611",
)

hits: InMemoryGeoCatalog = catalog.query(aoi)                         # 2 rows — no file opened
tensor: GeoTensor = gc.load.load_raster(hits, aoi, band_indexes=[1, 2, 3])  # (3, 500, 500) uint16
```

`hits` is itself a catalog, so queries chain and set-combine. Only the
loader opens files, and only the ones that overlap. The
[Quickstart](quickstart.md) adds STAC discovery and saving the catalog.

## What's inside

The root holds `GeoCatalog`, `GeoSlice`, `open_catalog` and
`query` / `intersect` / `union`. Every other name has one home, named for
the step it serves:

| Step | Namespace | Main names | Guide |
|---|---|---|---|
| discover | `geocatalog.sources` | `STACSource`, `CMRSource`, `EarthAccessSource`, `GEESource`, `from_stac_search` | [STAC ingestion](how-to/stac-ingestion.md) |
| index | `geocatalog.build` | `build_raster_catalog`, `build_xarray_catalog`, `build_vector_catalog`, `append_files` | [Large archives](how-to/large-archives.md) |
| hold | `geocatalog.backends` | `InMemoryGeoCatalog`, `DuckDBGeoCatalog`, `CatalogRow`, the errors | [Backends](concepts.md#backends) |
| join | `geocatalog.matchup` | `matchup`, `MatchupRow`, the spatial and temporal strategies | [Match sources](how-to/matchup.md) |
| read | `geocatalog.load` | `load_raster`, `aload_raster`, `load_raster_timeseries`, `load_xarray`, `load_vector` | [Load API](api/load.md) |
| save / share | `geocatalog.storage` | `to_geoparquet`, `from_geoparquet`, `StreamingParquetWriter`, `to_stac_collection`, `CatalogBundle` | [Provenance bundles](how-to/stac-ingestion.md#record-provenance-with-a-bundle) |
| stage | `geocatalog.staging` | `stage` (into a `geocloud.cache.LocalCache`) | [Staging](how-to/staging.md) |
| patch | `geocatalog.patch` | `field_for`, `CatalogDomain` | [Catalog → patcher](how-to/catalog-to-patcher.md) |
| grids | `geocatalog.grid` | `slice_to_window`, `is_grid_aligned`, `count_steps` | [Grid alignment](how-to/grid-alignment.md) |
| helpers | `geocatalog.utils` | `parse_uri`, `retry_transient_io`, UTC time helpers | [Utils API](api/utils.md) |

## Next steps

- [Concepts](concepts.md): `GeoSlice`, the row schema, the two backends,
  set algebra and provenance.
- [Quickstart](quickstart.md): local files, then Sentinel-2 from STAC.
- How-tos: [STAC ingestion](how-to/stac-ingestion.md),
  [staging](how-to/staging.md), [large archives](how-to/large-archives.md),
  [matching sources](how-to/matchup.md),
  [catalog → patcher](how-to/catalog-to-patcher.md),
  [grid alignment](how-to/grid-alignment.md).
- [Lake Tahoe tutorial](notebooks/end_to_end_lake_tahoe.ipynb): discover
  Sentinel-2 on Planetary Computer, filter, load, then an NDVI pipeline
  with geotoolz.
- [API reference](api/reference.md), [CLI](cli.md) and the
  [parameter vocabulary](vocabulary.md).

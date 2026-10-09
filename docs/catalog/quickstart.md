# Quickstart

Build a catalog, query it, load the hits and save the index for next time.
Part 1 runs offline on generated files; part 2 does the same on Sentinel-2
from Microsoft Planetary Computer.

## 1. Local files

Build an index of four GeoTIFFs, combine two queries, load a mosaic and
reopen the catalog from GeoParquet:

```python
# Shapes: H × W = 300 × 400 pixels on the query grid.
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog

# Two adjacent 3 km tiles on two dates: 4 files, 4 bands, 10 m, UTM 11N.
root: Path = Path(tempfile.mkdtemp())
for date in ("20240605", "20240720"):
    for i, x0 in enumerate((500_000, 503_000)):
        with rasterio.open(
            root / f"tile{i}_{date}.tif", "w", driver="GTiff", width=300, height=300, count=4,
            dtype="uint16", crs="EPSG:32611", transform=from_origin(x0, 4_303_000, 10, 10),
        ) as dst:
            dst.write(np.full((4, 300, 300), 1_000, dtype=np.uint16))   # (4, 300, 300) uint16

catalog: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    sorted(root.glob("*.tif")), filename_regex=r"_(?P<date>\d{8})\.tif", crs="EPSG:32611"
)                                                                    # 4 rows

june: gc.GeoSlice = gc.GeoSlice(
    bounds=(501_000, 4_300_000, 505_000, 4_303_000),                 # 4 km × 3 km, across both tiles
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(10.0, 10.0),
    crs="EPSG:32611",
)
june_hits: InMemoryGeoCatalog = catalog.query(june)                  # 2 rows: both June tiles
east: InMemoryGeoCatalog = gc.query(catalog, bounds=(504_000, 4_300_000, 506_000, 4_303_000))
both: InMemoryGeoCatalog = gc.union(june_hits, east)                 # 4 rows (no de-duplication)

mosaic: GeoTensor = gc.load.load_raster(june_hits, june)             # (4, 300, 400) uint16

gc.storage.to_geoparquet(catalog, root / "catalog.parquet")
reopened: gc.GeoCatalog = gc.open_catalog(root / "catalog.parquet")  # DuckDB if installed, else InMemory
```

What happened:

- `build_raster_catalog` opened each file once for its bounds and CRS, and
  read the date from the file name.
- `query` and `union` returned new catalogs and opened no file.
- `load_raster` opened the two June tiles and mosaicked them onto the
  slice grid.
- `to_geoparquet` wrote a GeoParquet 1.1 file that DuckDB, geopandas and
  GDAL can read; `open_catalog` reads it back.

## 2. Sentinel-2 from STAC

The same flow on real data needs the `[stac]` extra and network access.
`from_stac_search` builds one catalog row per item, pointing at one asset:

```python
import pandas as pd
import planetary_computer
import pystac_client
from georeader.geotensor import GeoTensor

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
    datetime="2024-06-01/2024-06-30",
    asset_key="B04",                                                 # red band; the default "data" is absent on S2
    limit=20,
    extra_properties=["eo:cloud_cover"],                             # keep as a column
)                                                                    # ≤ 20 rows, EPSG:4326 footprints

clearest: InMemoryGeoCatalog = catalog.where("`eo:cloud_cover` < 1")
aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(-120.10, 39.05, -120.05, 39.10),                         # ~4 km box on the west shore
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(0.0001, 0.0001),                                     # ~10 m at this latitude
    crs="EPSG:4326",
)
red: GeoTensor = gc.load.load_raster(clearest, aoi)                  # (1, 500, 500) uint16, warped to lon/lat
```

`load_raster` reads only the windows it needs over HTTP and warps each
asset from its UTM zone onto the slice grid. Signed URLs expire after
about an hour, so re-run the search for a fresh catalog.

## Next steps

- [Concepts](concepts.md): why queries open no file, and when to pick
  DuckDB.
- [STAC ingestion](how-to/stac-ingestion.md): every band per item,
  provenance bundles, signing.
- [Staging](how-to/staging.md): cache remote assets on local disk before
  many reads.
- [Lake Tahoe tutorial](notebooks/end_to_end_lake_tahoe.ipynb): the
  Sentinel-2 flow continued into a geotoolz NDVI pipeline.

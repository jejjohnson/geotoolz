# Catalogs

`geotoolz.catalog` is a **queryable spatiotemporal index over geospatial
files**. Each row records a file's footprint (bbox), time interval, CRS,
and path. Given a query like *"files overlapping AOI X between dates Y
and Z,"* the catalog returns the matching rows fast without opening any
file.

The submodule is the Phase 1 of the geodatabase design — see
`research_journal_v2/notes/geotoolz/plans/geodatabase/geocatalog.md`.
Phase 2 (`DuckDBGeoCatalog`) ships in a follow-up release; both
backends honour the same `GeoCatalog` Protocol.

## Mental model

```
┌──────────────────────────┐
│  GeoCatalog (Protocol)   │
│  query / intersect /     │
│  union / iter_slices /   │
│  to_geoparquet           │
└──────────────────────────┘
              ▲
              │  v0.1
              │
┌──────────────────────────┐
│   InMemoryGeoCatalog     │
│                          │
│   gpd.GeoDataFrame       │
│   + IntervalIndex (time) │
│   + R-tree (space)       │
│   ~10⁵ rows in RAM       │
└──────────────────────────┘
```

The shared row schema is the same across all three backends:

| Column | Type | Meaning |
| --- | --- | --- |
| `geometry` | Shapely Polygon | The file's footprint in the catalog CRS |
| `start_time` / `end_time` *(promoted to `IntervalIndex`)* | Timestamp | Time interval, `closed='both'` |
| `filepath` | str | Path or URI to the source file |

Extra columns depend on the backend (xarray adds `n_timesteps` and
`time_var`; vector adds `layer`).

## `GeoSlice` — the unit of work

A `GeoSlice` is the **inter-layer contract** that catalogs produce and
loaders consume. It carries everything a loader needs to fetch a chip
without consulting the catalog again:

```python
from geotoolz import GeoSlice
import pandas as pd

aoi = GeoSlice(
    bounds=(500_000, 4_000_000, 540_000, 4_040_000),
    interval=pd.Interval(
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-06-30"),
        closed="both",
    ),
    resolution=(10.0, 10.0),
    crs="EPSG:32629",
)
```

The dataclass is `frozen=True` so slices can be cached / hashed /
shipped across function boundaries. Use `dataclasses.replace(...)` to
"change" a slice — explicit by design.

## Three backends

### Raster

Always available. Bounds come from each file's `rasterio` metadata,
optionally reprojected via a lazy `WarpedVRT` if `target_crs` is
specified.

```python
import geotoolz as gz

catalog = gz.build_raster_catalog(
    filepaths=glob("/data/s2/T29SND_*.tif"),
    filename_regex=r"T29SND_(?P<date>\d{8}).*\.tif",
    target_crs="EPSG:32629",
)
tensor = gz.load_raster(catalog, aoi, band_indexes=[2, 3, 4, 8])
```

### Xarray *(extras: `[xarray-raster]`)*

For NetCDF / Zarr / HDF stores. Each row's footprint is derived from
the dataset's coordinate min/max.

```python
catalog = gz.build_xarray_catalog(
    filepaths=glob("/data/modis/MOD13A2_*.nc"),
    target_crs="EPSG:4326",
    data_vars=["NDVI"],
    time_var="time",
)
ds = gz.load_xarray(catalog, aoi, data_vars=["NDVI"])
```

### Vector *(base install — `shapely` + `geopandas`)*

Each row's footprint is the file's `total_bounds`. Loaders rasterise
into a label `GeoTensor` for `semantic_segmentation` or
`instance_segmentation`.

```python
catalog = gz.build_vector_catalog(
    filepaths=glob("/labels/*.gpkg"),
    target_crs="EPSG:32629",
)
labels = gz.load_vector(
    catalog, aoi,
    task="semantic_segmentation",
    label_field="class_id",
)
```

## Set algebra

Catalogs support `query`, `intersect`, and `union` — same shape across
backends. Returned catalogs are new instances; the originals are
untouched.

```python
imagery = gz.build_raster_catalog(...)
labels  = gz.build_vector_catalog(...)
paired  = gz.intersect(imagery, labels)  # rows that overlap in space AND time
combined = gz.union(catalog_2023, catalog_2024)
```

`intersect(left, right, spatial_only=True)` is the right tool for
pairing imagery with static labels.

`query` is the daily-bread operation — *"give me rows that overlap this
slice."* The recommended path passes a `GeoSlice` directly; bounds in
a non-catalog CRS are reprojected internally (the cross-CRS query
footgun described in §10.1 of the design plan).

## Portable artifact: GeoParquet

```python
gz.to_geoparquet(catalog, "cat.parquet")
# ... share the file ...
catalog = gz.from_geoparquet("cat.parquet")
```

The artifact is GeoParquet 1.1 compatible — readable by DuckDB,
geopandas, GDAL, pandas. The Phase 2 DuckDB backend reads the *same*
file.

## Bridging to `geotoolz.patch`

`CatalogDomain` adapts a catalog into a `geotoolz.patch.Domain` so the
`SpatialPatcher` can tile across a multi-file archive. Each catalog
row contributes one sub-domain; the patcher iterates them.

```python
import geotoolz as gz

catalog = gz.build_raster_catalog(...)
domain = gz.CatalogDomain(catalog=catalog, resolution=(10.0, 10.0))
for slice_ in domain.slices():
    chip = gz.load_raster(catalog, slice_, band_indexes=[2, 3, 4, 8])
    yield model(chip.values)
```

## When to use which

| Scale | Backend | Why |
| --- | --- | --- |
| Hundreds–thousands of files | `InMemoryGeoCatalog` | GeoDataFrame fits in RAM, sub-millisecond queries |
| 10⁶+ files | `DuckDBGeoCatalog` *(v0.2)* | SQL pushdown, on-disk artifact |
| Anywhere | Both | They share the `GeoCatalog` Protocol |

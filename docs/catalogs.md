# Catalogs

`geotoolz.catalog` is a **queryable spatiotemporal index over geospatial
files**. Each row records a file's footprint (bbox), time interval, CRS,
and path. Given a query like *"files overlapping AOI X between dates Y
and Z,"* the catalog returns the matching rows fast without opening any
file.

Two backends honour the same `GeoCatalog` Protocol: `InMemoryGeoCatalog`
(Phase 1 — a `GeoDataFrame` with an R-tree + `IntervalIndex`, good up
to ~10⁵ rows) and `DuckDBGeoCatalog` (Phase 2 — a lazy SQL relation
over a GeoParquet file, good to 10⁶+ rows and queryable from a remote
URI). See the design plans in
`research_journal_v2/notes/geotoolz/plans/geodatabase/`.

## Mental model

```
                ┌──────────────────────────┐
                │  GeoCatalog (Protocol)   │
                │  query / intersect /     │
                │  union / iter_rows /     │
                │  iter_slices /           │
                │  to_geoparquet           │
                └──────────────────────────┘
                       ▲          ▲
                       │          │
       ┌───────────────┘          └───────────────┐
       │                                          │
┌──────────────────────┐         ┌────────────────────────────┐
│ InMemoryGeoCatalog   │         │   DuckDBGeoCatalog         │
│ (v0.1, base install) │         │   (v0.2, [duckdb] extra)   │
│                      │         │                            │
│ gpd.GeoDataFrame     │         │ DuckDB + GeoParquet 1.1    │
│ + IntervalIndex      │         │ + spatial extension        │
│ + R-tree             │         │ + bbox-column pushdown     │
│ ~10⁵ rows in RAM     │         │ 10⁶+ rows, lazy on disk    │
└──────────────────────┘         └────────────────────────────┘
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
| 10⁶+ files | `DuckDBGeoCatalog` | SQL pushdown via GeoParquet 1.1 bbox column, on-disk / remote artifact |
| Anywhere | Both | They share the `GeoCatalog` Protocol |

## The DuckDB backend

`DuckDBGeoCatalog` (v0.2; `pip install 'geotoolz[duckdb]'`) wraps a
GeoParquet file as a lazy SQL relation. The Protocol surface is
unchanged — `query`, `intersect`, `union`, `iter_slices` all work — but
the implementation routes through DuckDB's `spatial` extension. The
two wins:

- **Predicate pushdown.** GeoParquet 1.1 carries a per-row `bbox`
  covering struct that DuckDB uses to skip row-groups without touching
  any WKB. A small-AOI query against a 10⁶-row catalog reads ~10⁵ rows
  of bbox metadata, not 10⁶ geometries.
- **Portable artifact.** The catalog *is* the GeoParquet file. Share
  it via S3 / GCS / HuggingFace (DuckDB's `httpfs` extension reads
  remote Parquet lazily, only fetching the row groups your query
  touches).

`open_catalog(path)` is the factory:

```python
import geotoolz as gz

# Prefers DuckDB when the [duckdb] extra is installed,
# falls back to the in-memory backend otherwise.
catalog = gz.open_catalog("cat.parquet")
print(catalog)
# DuckDBGeoCatalog(backend='raster', crs='EPSG:32629')

# All Protocol ops work — and return new lazy relations:
clean = catalog.query(bounds=(-3.8, 40.3, -3.6, 40.5), crs="EPSG:4326")
joint = clean.intersect(label_catalog)
joint.to_geoparquet("paired.parquet")

# `iter_rows` is the streaming surface — useful for large catalogs.
for row in catalog.iter_rows():
    print(row.filepath, row.geometry.bounds)

# `materialize()` pulls the relation into an InMemoryGeoCatalog when
# the rest of your pipeline expects a GeoDataFrame.
mem = catalog.materialize()
```

`CatalogRow` is the backend-neutral row view yielded by `iter_rows`;
loaders consume it without caring whether the underlying store was a
gdf or a SQL relation.

### Streaming build (`backend="duckdb"`)

The default builders (`build_raster_catalog` / `build_vector_catalog` /
`build_xarray_catalog`) collect every row in RAM before returning an
`InMemoryGeoCatalog`. That's fine up to ~10⁵ files — beyond that, the
build step itself becomes the bottleneck.

Pass `backend="duckdb"` to stream rows directly into a GeoParquet artifact
in bounded memory (peak ≈ `batch_size × row_size`, not `O(n_rows)`):

```python
catalog = build_raster_catalog(
    filepaths,                       # 10⁶ Sentinel-2 scenes
    filename_regex=r"S2_T\w+_(?P<date>\d{8}).*\.tif",
    backend="duckdb",
    out_path="s2_archive.parquet",   # required for backend="duckdb"
    n_workers=8,                     # parallel rasterio.open across files
    sort_by=("start_time", "geometry_hilbert"),  # row-group pruning
)
# DuckDBGeoCatalog opened on s2_archive.parquet
```

What this turns on:

- **Streaming Arrow batches** to `pyarrow.parquet.ParquetWriter` — peak
  RAM is `batch_size × row_size` (default ~10 MB at `batch_size=10_000`).
- **Process-pool extraction** when `n_workers > 1` — rasterio / fiona /
  xarray release the GIL during I/O, so a spawn-based pool feeds the
  single writer thread with no contention.
- **EPSG:4326 canonicalization** when `target_crs=None` — the design's
  prescribed wire format for shared GeoParquet artifacts.
- **Hilbert-sorted output** via a DuckDB post-write rewrite —
  `(start_time, ST_Hilbert(ST_Centroid(geometry)))` ordering enables
  row-group pruning at query time. Pass `sort_by=None` to skip.
- **GeoParquet 1.1 covering bbox** — per-row `bbox` struct column
  (`xmin`/`ymin`/`xmax`/`ymax`), used by DuckDB and geopandas readers
  for predicate pushdown.

The result is a `DuckDBGeoCatalog` opened on `out_path`. The artifact
also round-trips through `geopandas.read_parquet` and any other
GeoParquet-aware tool.

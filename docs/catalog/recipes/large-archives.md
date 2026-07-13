# Recipe: Large archives

When you're indexing 10⁵+ files, the eager `build_*_catalog` calls
that return an `InMemoryGeoCatalog` stop scaling — the GeoDataFrame
itself becomes the bottleneck. This recipe walks the three patterns
that keep things sane up to 10⁷+ rows.

## Pattern 1 — Streaming build into GeoParquet

Use `backend="duckdb"` on any builder. Rows are extracted in parallel
and streamed straight into the Parquet writer; peak RAM is
`batch_size × row_size`, not `O(n_rows)`.

```python
import geocatalog as gc

catalog = gc.build_raster_catalog(
    filepaths=glob("/data/s2/**/*.tif"),     # 1M Sentinel-2 scenes
    filename_regex=r"S2_T\w+_(?P<date>\d{8}).*\.tif",
    backend="duckdb",
    out_path="/data/s2_archive.parquet",     # required
    target_crs="EPSG:4326",                  # canonical for shared artifacts
    n_workers=16,                             # parallel rasterio.open
    sort_by=("start_time", "geometry_hilbert"),
)
# DuckDBGeoCatalog opened on the new artifact.
```

The output is GeoParquet 1.1 with a per-row covering `bbox` struct
and Hilbert-sorted geometry. Both features enable row-group pruning
at query time — small-AOI queries read kilobytes of metadata instead
of gigabytes of geometry.

### Why `target_crs="EPSG:4326"`?

The design's prescribed wire format for shared GeoParquet artifacts.
The streaming build canonicalises footprints to EPSG:4326 unless you
override. (Exception: `build_xarray_catalog` requires you to pass the
native CRS explicitly — it has no `WarpedVRT` analogue to reproject
on the fly.)

## Pattern 2 — Hive-partitioned + incremental append

For archives that grow over time, write to a directory of
Hive-partitioned shards. Each new batch of files writes only the new
rows, into the right partitions, without rewriting existing shards.

```python
from geocatalog import append_files

# Initial build — write to a directory, partition by year/month.
catalog = append_files(
    archive="/data/s2_archive/",            # directory, not a .parquet file
    filepaths=initial_paths,
    extract_fn=extract_raster_row,           # picklable function
    crs="EPSG:4326",
    backend="raster",
    partition_by=("year", "month"),          # derived from start_time
    n_workers=8,
)

# Later: add a new acquisition day. Only the new shard is written.
catalog = append_files(
    archive="/data/s2_archive/",
    filepaths=todays_new_paths,
    extract_fn=extract_raster_row,
    crs="EPSG:4326",
    backend="raster",
    partition_by=("year", "month"),
)
```

`partition_by` is validated against the archive's existing layout —
mismatched layouts raise `ValueError` rather than silently producing
a mixed-layout archive that downstream readers can't reconstruct.

### Reading a partitioned archive

`open_catalog` accepts a directory or a glob and lets DuckDB treat it
as one virtual table:

```python
catalog = gc.open_catalog("/data/s2_archive/")
hits = catalog.query(aoi)   # DuckDB skips partitions that don't overlap
```

## Pattern 3 — Remote artifacts on S3 / GCS / HF

The whole point of a GeoParquet archive is that you can write it once,
upload it, and let readers query it lazily over HTTP-range. DuckDB's
`httpfs` extension fetches only the row-groups your query touches:

```python
catalog = gc.open_catalog("s3://my-bucket/s2_archive.parquet")

# Small-AOI query — DuckDB pushes the bbox predicate down and reads
# only the bbox column + matching geometry row-groups (~MB, not GB).
hits = catalog.query(gc.GeoSlice(
    bounds=(-120.25, 38.85, -119.85, 39.30),
    interval=pd.Interval(
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-06-30"),
        closed="both",
    ),
    resolution=(0.0001, 0.0001),
    crs="EPSG:4326",
))
```

For directory-of-shards archives on S3:

```python
catalog = gc.open_catalog("s3://my-bucket/s2_archive/")
```

DuckDB recurses through the Hive partitions and prunes whole
directory subtrees when the partition column is in a query predicate.

## Performance knobs

| Knob | When to bump | Trade-off |
| --- | --- | --- |
| `batch_size` | Lots of small rows; underutilised writer | Higher peak RAM |
| `n_workers` | I/O-bound extraction (S3 reads, rasterio opens) | Disk/network saturation |
| `sort_by=("start_time", "geometry_hilbert")` | Workloads dominated by AOI+time queries | Slightly slower build |
| `ordered=True` | You need byte-reproducible output for CI | Slower / less parallel — earliest input stalls all subsequent yields |
| Per-partition row count | Bigger partitions = fewer files, but more I/O per query | Aim for ~100k–1M rows per shard |

## When *not* to bother

If you have under ~10⁵ files, the default eager `InMemoryGeoCatalog`
is faster end-to-end (no Parquet round-trip, no DuckDB warmup). The
GeoDataFrame fits in RAM and the R-tree + `IntervalIndex` give you
sub-millisecond queries.

The Protocol is the same in both directions — you can start with
InMemory and swap to DuckDB when the row count grows.

## See also

- [Concepts: persistence](../concepts.md#persistence) — schema and bbox-column details
- [Recipes: STAC ingestion](from-stac.md) — populating an archive from STAC
- [Recipes: staging & bundles](staging-and-bundles.md) — when to materialise to local disk
- [API: `append_files`](../api/reference.md) — full signature
- [DuckDB scale-out walkthrough ↗](https://github.com/jejjohnson/research_notebook/blob/main/projects/geostack/notebooks/catalog/04_duckdb.ipynb) — `DuckDBGeoCatalog` against a real S2 GeoParquet artifact, plus the Overture buildings recipe (billions of rows).

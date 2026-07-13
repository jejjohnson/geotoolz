# geocatalog

> A queryable spatiotemporal index over geospatial files. Two backends,
> one Protocol, one `GeoSlice` contract.

## What is a Catalog?

A **catalog** is an *index over geospatial files*. Each row holds a
single file's footprint (bbox), time interval, CRS, and path/URI.
Given a query like *"files overlapping AOI X between dates Y and Z,"*
the catalog answers in milliseconds — **without opening any file.**

That's the whole idea. Everything else (DuckDB pushdown, GeoParquet
artifacts, staging caches, STAC ingestion, patcher bridges) is
plumbing in service of that one operation.

The unit of work is `GeoSlice` — a `(bounds, interval, crs,
resolution)` 4-tuple. Catalogs **produce** slices; loaders **consume**
them. Anywhere a slice flows through your pipeline, it carries
everything the next stage needs to fetch a chip without consulting the
catalog again.

## Is this the right tool for me?

```mermaid
flowchart TD
    Q[I have a bunch of<br/>geospatial files] --> A{Do I need to<br/>query by bbox<br/>and time?}
    A -- No, I just<br/>read one file --> NO1[Use rasterio /<br/>xarray / fiona<br/>directly]
    A -- Yes --> B{How many files?}
    B -- &lt; 10^5 --> IM[InMemoryGeoCatalog<br/><sub>base install</sub>]
    B -- 10^5 to 10^6+ --> C{Remote storage<br/>S3 / GCS / HF?}
    C -- No, local --> DD[DuckDBGeoCatalog<br/><sub>[duckdb] extra</sub>]
    C -- Yes --> DD2[DuckDBGeoCatalog<br/>over remote<br/>GeoParquet<br/><sub>[duckdb] + [fsspec]</sub>]
    B -- I don't know yet --> BUILD[Start with<br/>InMemory, swap to<br/>DuckDB later<br/><sub>same Protocol</sub>]
    style IM fill:#fbe8e9,stroke:#C44E52
    style DD fill:#e8f0fb,stroke:#4C72B0
    style DD2 fill:#e8f0fb,stroke:#4C72B0
    style BUILD fill:#fbf6e3,stroke:#CCB974
    style NO1 fill:#eee,stroke:#888
```

## Mental model — one paragraph

You build a `GeoCatalog` once (from a directory of files, a STAC
search, an EarthAccess CMR call, or a hand-rolled list). You hand it
`GeoSlice`s to query. The catalog hands you back smaller catalogs
(the matching rows). When you're ready to materialise pixels you pass
the slice + matching catalog to a loader (`load_raster`, `load_vector`,
`load_xarray`). The loader returns a `GeoTensor` (raster, vector) or
`xr.Dataset` (xarray). At every step the geometry, time, and CRS are
explicit — there is no hidden state.

## Quick links

- **[Concepts](concepts.md)** — architecture, schema, backends, set algebra, persistence
- **[Quickstart](quickstart.md)** — 15-minute Lake Tahoe Sentinel-2 walkthrough
- **Recipes**
    - [Large archives](recipes/large-archives.md) — partitioned Parquet + S3
    - [STAC ingestion](recipes/from-stac.md) — `STACSource` vs `from_stac_search`
    - [Staging & bundles](recipes/staging-and-bundles.md) — when to `stage()`, when to use `CatalogBundle.ingest()`
- **[End-to-end notebook](notebooks/end_to_end_lake_tahoe.ipynb)** — discover → query → load → patch → stitch (cross-repo with `geotoolz` and `geopatcher`); this PR's canonical worked example.
- **Extended examples ↗** — the deep dives on build/query/load, raster + xarray + vector backends, query/intersect/union set algebra, the DuckDB scale-out backend, and the catalog↔patcher bridge live in [`research_notebook/projects/geostack/notebooks/catalog`](https://github.com/jejjohnson/research_notebook/tree/main/projects/geostack/notebooks/catalog). They execute against a real Sentinel-2 archive on MPC + Natural Earth admin-1 polygons; this repo's docs reference them by name.
- **[API Reference](api/reference.md)** — full mkdocstrings reference
- **[CLI](cli.md)**, **[Logging](logging.md)**, **[Schema versions](schema-versions.md)**

## Install

```bash
pip install geocatalog                  # base: InMemory + raster + vector
pip install 'geocatalog[duckdb]'        # DuckDB backend
pip install 'geocatalog[xarray-raster]' # xarray (NetCDF / Zarr) backend
pip install 'geocatalog[stac]'          # STAC ingestion
pip install 'geocatalog[full]'          # everything
```

Or with `uv`:

```bash
uv add geocatalog
```

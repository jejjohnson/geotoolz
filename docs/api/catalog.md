# `geotoolz.catalog` — Catalog API

Curated mkdocstrings reference. For the conceptual walkthrough see
[Catalogs](../catalogs.md).

## Cross-cutting type

::: geotoolz.types._src.geoslice.GeoSlice
::: geotoolz.types._src.geoslice.slice_to_window
::: geotoolz.types._src.geoslice.window_to_slice

## Protocol

::: geotoolz.catalog._src.base.GeoCatalog
::: geotoolz.catalog._src.base.CatalogRow

## Backends

::: geotoolz.catalog._src.memory.InMemoryGeoCatalog

### DuckDB *(extras: `[duckdb]`)*

::: geotoolz.catalog._src.duckdb_backend.DuckDBGeoCatalog

## Factory

::: geotoolz.catalog.open_catalog

## Raster

::: geotoolz.catalog._src.raster.build_raster_catalog
::: geotoolz.catalog._src.raster.load_raster
::: geotoolz.catalog._src.raster.load_raster_timeseries

## Xarray *(extras: `[xarray-raster]`)*

::: geotoolz.catalog._src.xarray_backend.build_xarray_catalog
::: geotoolz.catalog._src.xarray_backend.load_xarray

## Vector

::: geotoolz.catalog._src.vector.build_vector_catalog
::: geotoolz.catalog._src.vector.load_vector

## Set algebra

::: geotoolz.catalog._src.ops.query
::: geotoolz.catalog._src.ops.intersect
::: geotoolz.catalog._src.ops.union

## GeoParquet roundtrip

::: geotoolz.catalog._src.parquet.to_geoparquet
::: geotoolz.catalog._src.parquet.from_geoparquet

## Bridge to `geotoolz.patch`

::: geotoolz.catalog._src.domain.CatalogDomain

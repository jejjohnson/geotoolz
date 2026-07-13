# `geocatalog` — API Reference

Curated mkdocstrings reference. For the conceptual walkthrough see
[Concepts](../concepts.md); for a worked example see the
[Quickstart](../quickstart.md).

## Cross-cutting type

::: geocatalog._src.geoslice.GeoSlice
::: geocatalog._src.geoslice.slice_to_window
::: geocatalog._src.geoslice.window_to_slice

## Protocol

::: geocatalog._src.base.GeoCatalog
::: geocatalog._src.base.CatalogRow

## Backends

::: geocatalog._src.memory.InMemoryGeoCatalog

### DuckDB *(extras: `[duckdb]`)*

::: geocatalog._src.duckdb_backend.DuckDBGeoCatalog

## Factory

::: geocatalog.open_catalog

## Raster

::: geocatalog._src.raster.build_raster_catalog
::: geocatalog._src.raster.load_raster
::: geocatalog._src.raster.load_raster_timeseries

## Xarray *(extras: `[xarray-raster]`)*

::: geocatalog._src.xarray_backend.build_xarray_catalog
::: geocatalog._src.xarray_backend.load_xarray

## Vector

::: geocatalog._src.vector.build_vector_catalog
::: geocatalog._src.vector.load_vector

## Set algebra

::: geocatalog._src.ops.query
::: geocatalog._src.ops.intersect
::: geocatalog._src.ops.union

## GeoParquet roundtrip

::: geocatalog._src.parquet.to_geoparquet
::: geocatalog._src.parquet.from_geoparquet

## Schema migration

::: geocatalog._src.parquet.SCHEMA_VERSION_CURRENT
::: geocatalog._src.parquet.migrate_geoparquet
::: geocatalog._src.base.CatalogSchemaError

## Discovery sources

The `Source` Protocol and its adapters live under
`geocatalog.sources`. Adapters are extras-gated and imported lazily.

::: geocatalog.sources.Source
::: geocatalog.sources.SourceRow
::: geocatalog.sources.AuthStatus

### STAC *(extras: `[stac]`)*

::: geocatalog.sources.STACSource

### earthaccess *(extras: `[earthaccess]`)*

::: geocatalog.sources.EarthAccessSource

### CMR

::: geocatalog.sources.CMRSource

## STAC conversion *(extras: `[stac]`)*

::: geocatalog._src.stac.from_stac_items
::: geocatalog._src.stac.from_stac_search
::: geocatalog._src.stac.to_stac_collection

## Matchup

::: geocatalog.matchup.matchup
::: geocatalog.matchup.MatchupRow

### Spatial strategies

::: geocatalog.matchup.SpatialStrategy
::: geocatalog.matchup.Intersects
::: geocatalog.matchup.IouAtLeast
::: geocatalog.matchup.CentroidWithin
::: geocatalog.matchup.Contains

### Temporal strategies

::: geocatalog.matchup.TemporalStrategy
::: geocatalog.matchup.NearestInTime
::: geocatalog.matchup.WithinWindow
::: geocatalog.matchup.Synchronous

## Bundle

::: geocatalog.bundle.CatalogBundle
::: geocatalog.bundle.QueryRecord
::: geocatalog.bundle.source_row_to_gdf_row

## Staging

::: geocatalog.staging.stage
::: geocatalog.staging.LocalCache
::: geocatalog.staging.field_for

## Grid alignment

::: geocatalog.types.Align
::: geocatalog.types.divide_evenly
::: geocatalog.types.is_grid_aligned
::: geocatalog.types.GridAlignmentWarning

## Object-store pool

Shared `obstore` client pool used by the raster catalog builders for
remote URIs. Internal-but-stable knobs for tuning long-running
processes.

::: geocatalog._src.objstore.get_obstore
::: geocatalog._src.objstore.clear_obstore_pool
::: geocatalog._src.objstore.set_obstore_pool_maxsize

## Bridge to a patcher

::: geocatalog._src.domain.CatalogDomain

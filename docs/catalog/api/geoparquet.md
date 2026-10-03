# GeoParquet interchange

The persisted, schema-versioned catalog format: round-trip,
migration, and streaming writes.

## GeoParquet roundtrip

::: geocatalog.catalog.to_geoparquet
::: geocatalog.catalog.from_geoparquet

## Schema migration

::: geocatalog.catalog.SCHEMA_VERSION_CURRENT
::: geocatalog.catalog.migrate_geoparquet
::: geocatalog.catalog.CatalogSchemaError
::: geocatalog.catalog.CatalogMetadataError

## Streaming writes

::: geocatalog.catalog.StreamingParquetWriter
::: geocatalog.catalog.append_files
::: geocatalog.catalog.sort_geoparquet

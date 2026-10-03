# GeoParquet interchange

The persisted, schema-versioned catalog format: round-trip,
migration, and streaming writes.

## GeoParquet roundtrip

::: geocatalog.catalog.to_geoparquet
::: geocatalog.catalog.from_geoparquet

## Schema migration

::: geocatalog.catalog.SCHEMA_VERSION_CURRENT
::: geocatalog.catalog.migrate_geoparquet

## Errors

Every error `geocatalog` raises on purpose derives from
`GeoCatalogError` and from the builtin callers caught before
(`ValueError` / `RuntimeError`); see the
[parameter vocabulary](../design/vocabulary.md#errors).

::: geocatalog.catalog.GeoCatalogError
::: geocatalog.catalog.CatalogSchemaError
::: geocatalog.catalog.CatalogMetadataError
::: geocatalog.catalog.CatalogClosedError

## Streaming writes

::: geocatalog.catalog.StreamingParquetWriter
::: geocatalog.catalog.append_files
::: geocatalog.catalog.sort_geoparquet

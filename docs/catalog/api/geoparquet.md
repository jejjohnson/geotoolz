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

Errors about a catalog's state or artifacts derive from
`GeoCatalogError` and from the builtin callers caught before
(`ValueError` / `RuntimeError`). Invalid arguments raise the builtin
`ValueError` / `TypeError`; see the
[parameter vocabulary](../design/vocabulary.md#errors).

::: geocatalog.catalog.GeoCatalogError
::: geocatalog.catalog.CatalogSchemaError
::: geocatalog.catalog.CatalogMetadataError
::: geocatalog.catalog.CatalogClosedError

## Streaming writes

::: geocatalog.catalog.StreamingParquetWriter
::: geocatalog.catalog.append_files
::: geocatalog.catalog.sort_geoparquet

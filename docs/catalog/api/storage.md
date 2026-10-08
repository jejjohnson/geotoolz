# Storage — `geocatalog.storage`

Save, share and reopen catalogs: the schema-versioned GeoParquet format,
streaming writes, STAC export, and the provenance-recording
`CatalogBundle`.

## GeoParquet round-trip

::: geocatalog.storage.to_geoparquet
::: geocatalog.storage.from_geoparquet

## Schema migration

::: geocatalog.storage.SCHEMA_VERSION_CURRENT
::: geocatalog.storage.migrate_geoparquet

## Streaming writes

::: geocatalog.storage.StreamingParquetWriter
::: geocatalog.storage.sort_geoparquet

## STAC export *(extras: `[stac]`)*

::: geocatalog.storage.to_stac_collection

## Provenance bundles

::: geocatalog.storage.CatalogBundle
::: geocatalog.storage.QueryRecord

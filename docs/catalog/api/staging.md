# Staging & the patcher bridge

Resolve remote URIs into a local cache and hand staged rows to
`geopatcher` as Fields.

## Staging

::: geocatalog.staging.stage
::: geocatalog.staging.LocalCache
::: geocatalog.staging.field_for

## Object-store pool

geocatalog does not use the pooled `obstore` client: staging downloads
through fsspec and the builders read through rasterio / GDAL. To read the
staged or catalogued objects with range requests, install
`geotoolz-cloud` and take the client from
[`geocloud.store`](../../cloud/api.md) — one pool per process for the
whole stack.

## Bridge to a patcher

::: geocatalog.catalog.CatalogDomain

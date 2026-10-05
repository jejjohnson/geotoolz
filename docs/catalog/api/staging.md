# Staging & the patcher bridge

Resolve remote URIs into a local cache and hand staged rows to
`geopatcher` as Fields.

## Staging

::: geocatalog.staging.stage
::: geocatalog.staging.LocalCache
::: geocatalog.staging.field_for

## Object-store pool

The pooled `obstore` client lives in geopatcher — one pool per process
for the whole stack. `stage` and the catalog builders do not use it
(staging downloads through fsspec; the builders read through
rasterio / GDAL). `pip install 'geotoolz-catalog[obstore]'` installs it;
see [`geopatcher.objstore`](../../patcher/api/core.md#object-store-pool).

## Bridge to a patcher

::: geocatalog.catalog.CatalogDomain

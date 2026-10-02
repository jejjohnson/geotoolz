# Staging & the patcher bridge

Resolve remote URIs into a local cache, share pooled object-store
clients, and hand staged rows to `geopatcher` as Fields.

## Staging

::: geocatalog.staging.stage
::: geocatalog.staging.LocalCache
::: geocatalog.staging.field_for

## Object-store pool

The pooled `obstore` client lives in geopatcher — one pool per process
for the whole stack. `pip install 'geotoolz-catalog[obstore]'` installs it;
see [`geopatcher.objstore`](../../patcher/api/core.md#object-store-pool).

## Bridge to a patcher

::: geocatalog._src.domain.CatalogDomain

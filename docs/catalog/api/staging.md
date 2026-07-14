# Staging & the patcher bridge

Resolve remote URIs into a local cache, share pooled object-store
clients, and hand staged rows to `geopatcher` as Fields.

## Staging

::: geocatalog.staging.stage
::: geocatalog.staging.LocalCache
::: geocatalog.staging.field_for

## Object-store pool

Shared `obstore` client pool used by the raster catalog builders for
remote URIs. Internal-but-stable knobs for tuning long-running
processes.

::: geocatalog._src.objstore.get_obstore
::: geocatalog._src.objstore.clear_obstore_pool
::: geocatalog._src.objstore.set_obstore_pool_maxsize

## Bridge to a patcher

::: geocatalog._src.domain.CatalogDomain

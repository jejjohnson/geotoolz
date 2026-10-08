# Staging — `geocatalog.staging`

Copy the remote assets a catalog points at into a local cache, and get
back the same catalog with its paths rewritten to the local copies.
Hand the result to the [loaders](load.md) or to
[`field_for`](patch.md).

## Staging

::: geocatalog.staging.stage
::: geocatalog.staging.LocalCache

## Object-store pool

geocatalog does not use the pooled `obstore` client: staging downloads
through fsspec and the builders read through rasterio / GDAL. To read the
staged or catalogued objects with range requests, install
`geotoolz-cloud` and take the client from
[`geocloud.store`](../../cloud/api.md) — one pool per process for the
whole stack.

# Core framework

The framework spine: patch carriers, the `Field` / `Domain`
protocols, concrete domains, field adapters, the top-level patchers,
and strictness / error types.

## Carriers

::: geopatcher._src.patch.Patch
::: geopatcher._src.patch.TemporalPatch
::: geopatcher._src.patch.SpatioTemporalPatch

## Protocols

::: geopatcher._src.hooks.PatcherHook
::: geopatcher._src.protocols.Field
::: geopatcher._src.protocols.AsyncField
::: geopatcher._src.protocols.Domain

## Domains

::: geopatcher._src.domains.GridDomain
::: geopatcher._src.domains.VectorDomain
::: geopatcher._src.domains.PointDomain

`RasterDomain` is the existing `GeoDataBase` protocol re-exported from
[`georeader`](https://github.com/IPL-UV/georeader) — import it as
`from geopatcher import RasterDomain`; see georeader's docs for the
protocol members.

## Field adapters

::: geopatcher._src.fields.raster.RasterField
::: geopatcher._src.fields.raster.AsyncRasterField

The remaining adapters are extras-gated; import via the public
submodule path:

```python
from geopatcher.fields import XarrayField, GeoPandasField, XvecField
from geopatcher.fields import RioXarrayField, DaskField, ObstoreCogField
```

::: geopatcher._src.fields.rio_xarray.RioXarrayField
::: geopatcher._src.fields.dask.DaskField
::: geopatcher._src.fields.obstore_cog.ObstoreCogField

## Object-store pool

`geopatcher.objstore` is the process-wide `obstore` client pool for the
whole stack: `ObstoreCogField`, geotoolz's sensor readers and geocatalog
all go through it, so one bucket / Azure container gets one client and
one HTTP/2 connection pool. Install with
`pip install 'geotoolz-patcher[obstore]'`.

| URI | Pooled store | `object_key` |
| --- | --- | --- |
| `s3://bucket/key`, `gs://bucket/key` | `S3Store` / `GCSStore(bucket)` | `key` |
| `az://account/container/key` | `AzureStore(container, account)` | `key` |
| `abfs[s]://container@account.dfs.core.windows.net/key` | `AzureStore(container, account)` | `key` |
| `https://account.blob.core.windows.net/container/key` | `AzureStore(container, account)` | `key` |
| `http[s]://host/path?query` (pre-signed / SAS) | `HTTPStore(origin?query)` | `path` |

Pooled stores never carry a prefix, `storage_options` are part of the
pool key, and an `http(s)` query string is kept on the store so signed
URLs are requested signed.

::: geopatcher.objstore.get_obstore
::: geopatcher.objstore.object_key
::: geopatcher.objstore.get_range_bytes
::: geopatcher.objstore.clear_obstore_pool
::: geopatcher.objstore.set_obstore_pool_maxsize

## Top-level patchers

::: geopatcher._src.spatial.patcher.SpatialPatcher
    options:
      inherited_members: true
::: geopatcher._src.spatial.patcher.AsyncSpatialPatcher
    options:
      inherited_members: true
::: geopatcher._src.time.patcher.TemporalPatcher
::: geopatcher._src.spatial_time.SpatioTemporalPatcher

## Config round-trip

Every axis, stencil and patcher exposes `get_config()`. Nested components
serialise as `{"class": ..., "config": ...}` envelopes, and
`from_config(axis_envelope(obj))` rebuilds `obj` — unless its type is
`forbid_in_yaml` (closures, polygons, backend-native anchors), whose
config is a debug summary that `from_config` refuses.

::: geopatcher._src._serialize.axis_envelope
::: geopatcher._src._serialize.from_config
::: geopatcher._src._serialize.config_from_fields

## Strictness and errors

::: geopatcher._src.config.get_strict
::: geopatcher._src.config.set_strict
::: geopatcher._src.exceptions.IncompleteScanConfiguration
::: geopatcher._src.walk.PatchErrorRecord

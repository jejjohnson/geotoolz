# Core framework

The framework spine: patch carriers, the `Field` / `Domain`
protocols, concrete domains, field adapters, the top-level patchers,
and strictness / error types.

## Carriers

::: geopatcher.Patch
::: geopatcher.TemporalPatch
::: geopatcher.SpatioTemporalPatch

## Protocols

::: geopatcher.PatcherHook
::: geopatcher.hooks.UNKNOWN_TOTAL
::: geopatcher.Field
::: geopatcher.AsyncField
::: geopatcher.Domain

## Domains

::: geopatcher.GridDomain
::: geopatcher.VectorDomain
::: geopatcher.PointDomain

`RasterDomain` is the existing `GeoDataBase` protocol re-exported from
[`georeader`](https://github.com/IPL-UV/georeader) — import it as
`from geopatcher import RasterDomain`; see georeader's docs for the
protocol members.

## Field adapters

::: geopatcher.RasterField
::: geopatcher.AsyncRasterField
::: geopatcher.ReprojectingRasterField

The remaining adapters are extras-gated; import via the public
submodule path:

```python
from geopatcher.fields import XarrayField, GeoPandasField, XvecField
from geopatcher.fields import RioXarrayField, DaskField, ObstoreCogField
```

::: geopatcher.fields.XarrayField
::: geopatcher.fields.GeoPandasField
::: geopatcher.fields.XvecField
::: geopatcher.fields.RioXarrayField
::: geopatcher.fields.DaskField
::: geopatcher.fields.ObstoreCogField

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

::: geopatcher.SpatialPatcher
    options:
      inherited_members: true
::: geopatcher.AsyncSpatialPatcher
    options:
      inherited_members: true
::: geopatcher.TemporalPatcher
::: geopatcher.SpatioTemporalPatcher

## Config round-trip

Every axis, stencil and patcher exposes `get_config()`. Nested components
serialise as `{"class": ..., "config": ...}` envelopes, and
`from_config(axis_envelope(obj))` rebuilds `obj` — unless its type is
`forbid_in_yaml` (closures, polygons, backend-native anchors), whose
config is a debug summary that `from_config` refuses.

::: geopatcher.axis_envelope
::: geopatcher.from_config
::: geopatcher.config_from_fields

## Strictness and errors

::: geopatcher.get_strict
::: geopatcher.set_strict
::: geopatcher.IncompleteScanConfiguration
::: geopatcher.PatchErrorRecord

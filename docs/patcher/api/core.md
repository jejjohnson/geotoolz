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

## Top-level patchers

::: geopatcher._src.spatial.patcher.SpatialPatcher
::: geopatcher._src.spatial.patcher.AsyncSpatialPatcher
::: geopatcher._src.time.patcher.TemporalPatcher
::: geopatcher._src.spatial_time.SpatioTemporalPatcher

## Strictness and errors

::: geopatcher._src.config.get_strict
::: geopatcher._src.config.set_strict
::: geopatcher._src.exceptions.IncompleteScanConfiguration
::: geopatcher._src.spatial.patcher.PatchErrorRecord

# Core — `geopatcher`

What every patching job touches: the patchers, the patch carriers, the
`Field` / `Domain` protocols and `RasterField`, the adapter for any
`geoarray.RasterSource` (a `GeoArray` or a lazy reader).

```python
from geopatcher import RasterField, SpatialPatcher, spatial
```

## Patchers

::: geopatcher.SpatialPatcher
    options:
      inherited_members: true
::: geopatcher.AsyncSpatialPatcher
    options:
      inherited_members: true
::: geopatcher.TemporalPatcher
::: geopatcher.SpatioTemporalPatcher

## Carriers

::: geopatcher.Patch
::: geopatcher.TemporalPatch
::: geopatcher.SpatioTemporalPatch

## Protocols

::: geopatcher.Field
::: geopatcher.AsyncField
::: geopatcher.Domain

## The raster adapter

::: geopatcher.RasterField

# Fields — `geopatcher.fields`

Field adapters for every substrate other than a georeader reader
(`RasterField`, at the root), and the domain types samplers plan over.
The extras-gated adapters resolve lazily, so importing the module never
needs an optional dependency.

```python
from geopatcher.fields import CogField, RioXarrayField, XarrayField
```

## Raster readers

::: geopatcher.fields.AsyncRasterField
::: geopatcher.fields.ReprojectingRasterField

## Cloud-Optimized GeoTIFFs (`[cog]`)

`CogField` is geotoolz-cloud's [`CogSource`](../../cloud/api.md) as a
`Field`: tiles a batch of windows touches are fetched once, in concurrent
groups, and `parallel_map` / `AsyncSpatialPatcher` use the batched and
async paths automatically.

::: geopatcher.fields.CogField
    options:
      inherited_members: false

## Gridded N-D data (`[grid]`, `[xarray-raster]`, `[dask]`)

::: geopatcher.fields.XarrayField
::: geopatcher.fields.RioXarrayField
::: geopatcher.fields.DaskField

## Vector features and points (`[vector]`, `[point]`)

::: geopatcher.fields.GeoPandasField
::: geopatcher.fields.XvecField

## Domains

`RasterDomain` is georeader's `GeoDataBase` protocol, re-exported; see
[georeader](https://github.com/IPL-UV/georeader) for its members.

::: geopatcher.fields.GridDomain
::: geopatcher.fields.VectorDomain
::: geopatcher.fields.PointDomain

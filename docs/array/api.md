# API reference

## `GeoArray`

The grid carrier: a NumPy or JAX array with its stored dims, coordinates,
transform, CRS, fill value and attrs. The coordinates are on
[their own page](api/coords.md), the readable types on
[theirs](api/types.md).

::: geoarray.GeoArray
::: geoarray.SPATIAL_DIMS
::: geoarray.DIMS

## `GeoSwath` and `GeoPoints`

Carriers whose georeference is not an affine grid: per-pixel lat/lon on
`row col`, and scattered samples on `obs`.

::: geoarray.GeoSwath
::: geoarray.GeoPoints
::: geoarray.pad_points

## Dim-level functions

pipekit-array's functions, re-exported: `geoarray.reduce is
pipekit_array.reduce`. They are documented in pipekit-array; geoarray's
adapters are registered with pipekit at import, and give carriers these
fast paths:

| Function | On a geoarray carrier |
| --- | --- |
| `isel(x, **dims)`, `sel(x, **labels)` | slice the data, coordinates and geometry; an index that breaks the grid gives a `DimArray` |
| `transpose(x, order)` | a carrier while the spatial dims stay last; else a `DimArray` |
| `reduce(x, fn, dim)`, `argreduce(x, fn, dim)`, `take_along(x, index, dim)` | a carrier while the spatial (and batch) dims survive; else a `DimArray` |
| `map_over(x, fn, dim)` | one `jax.vmap` over the data and the matching leaves on JAX; a loop on NumPy |
| `concat(xs, dim)` | joins coordinates, per-band attrs and, along a batch dim, transforms |
| `combine(fn, *xs)` | one exact grid, one fill; mixed NumPy and JAX inputs warn (0.3) and raise `BackendError` from 0.4 |

## The named-array adapters

::: geoarray.GeoArrayAdapter
::: geoarray.GeoSwathAdapter
::: geoarray.GeoPointsAdapter
::: geoarray.to_dimarray

## Sampling at points

::: geoarray.to_pixel
::: geoarray.interp_at

## `Grid`

A pixel grid without data.

::: geoarray.Grid

## `PixelWindow`

A rectangle of pixels.

::: geoarray.PixelWindow

## `FrozenAttrs`

Hashable metadata.

::: geoarray.FrozenAttrs
::: geoarray.PER_BAND_KEYS

## Geometry

Re-index or re-describe pixels without computing new values.

::: geoarray.crop
::: geoarray.read_window
::: geoarray.write_window
::: geoarray.pad
::: geoarray.pixel_coords
::: geoarray.footprint

## Joining carriers

Carriers on one grid: the same CRS, spatial shape and transform
(`combine` and `concat` are above).

::: geoarray.stack
::: geoarray.concatenate

## Lazy readers

What a reader provides so code can read windows of it on demand; a
`GeoArray` is one too.

::: geoarray.RasterSource
::: geoarray.AsyncRasterSource

## Backends and interop

`to_jax()` / `to_numpy()` cross between the backends (see
[When to jit](how-to/when-to-jit.md)). Nothing crosses silently: mixed
NumPy and JAX inputs to the joins warn in 0.3 and raise `BackendError` from
0.4, so call `to_jax()` first. `np.asarray(g)` copies JAX data to the host
and raises `GeoArrayTraceError` under a trace.

::: geoarray.set_options
::: geoarray.get_options
::: geoarray.HostTransferWarning
::: geoarray.interop.georeader.from_georeader
::: geoarray.interop.georeader.to_georeader
::: geoarray.interop.georeader.source_from_georeader
::: geoarray.interop.georeader.source_to_georeader

## xarray

Lossless converters with a CF `spatial_ref` and GDAL `GeoTransform`
(`geotoolz-array[xarray]`); see [Cross to xarray](how-to/xarray.md).

::: geoarray.interop.xarray.to_xarray
::: geoarray.interop.xarray.from_xarray
::: geoarray.interop.xarray.to_dataset
::: geoarray.interop.xarray.from_dataset
::: geoarray.interop.xarray.to_datatree
::: geoarray.interop.xarray.from_datatree

## Coercions

Turn transforms and CRSs from other libraries into geoarray's.

::: geoarray.as_transform
::: geoarray.as_crs

## Errors

Every error geoarray raises on its own account subclasses `GeoArrayError`.

::: geoarray.GeoArrayError
::: geoarray.X64Error
::: geoarray.GeoArrayTraceError
::: geoarray.GridMismatchError
::: geoarray.LabelError
::: geoarray.types.DimsError

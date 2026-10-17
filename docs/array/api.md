# API reference

## `GeoArray`

The carrier: a NumPy or JAX array with its transform, CRS, fill value and attrs.

::: geoarray.GeoArray
::: geoarray.DIMS

## `Grid`

A pixel grid without data.

::: geoarray.Grid

## `PixelWindow`

A rectangle of pixels.

::: geoarray.PixelWindow

## `FrozenAttrs`

Hashable metadata.

::: geoarray.FrozenAttrs

## Coercions

Turn transforms and CRSs from other libraries into geoarray's.

::: geoarray.as_transform
::: geoarray.as_crs

## Errors

Every error geoarray raises on its own account subclasses `GeoArrayError`.

::: geoarray.GeoArrayError
::: geoarray.X64Error
::: geoarray.GeoArrayTraceError

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

Carriers on one grid: the same CRS, spatial shape and transform.

::: geoarray.combine
::: geoarray.stack
::: geoarray.concatenate

## Lazy readers

What a reader provides so code can read windows of it on demand; a
`GeoArray` is one too.

::: geoarray.RasterSource
::: geoarray.AsyncRasterSource

## Backends and interop

NumPy and JAX carriers mix eagerly; `np.asarray(g)` copies JAX data to the
host and raises `GeoArrayTraceError` under a trace.

::: geoarray.set_options
::: geoarray.get_options
::: geoarray.HostTransferWarning
::: geoarray.interop.georeader.from_georeader
::: geoarray.interop.georeader.to_georeader
::: geoarray.interop.georeader.source_from_georeader
::: geoarray.interop.georeader.source_to_georeader

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

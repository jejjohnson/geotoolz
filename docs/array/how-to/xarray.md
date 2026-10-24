# Cross to xarray

Goal: hand a carrier to xarray for analysis by value (climatologies,
`sel(time=slice(...))`, `interp`), or write it to NetCDF, and come back to a
carrier losslessly at the `jit` boundary. Needs `geotoolz-array[xarray]`.

```python
import numpy as np
import xarray as xr

from geoarray import GeoArray
from geoarray.coords import SpectralAxis

scene: GeoArray = GeoArray(
    np.random.default_rng(0).random((2, 4, 5)).astype(np.float32),  # (C, H, W) float32
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
    "EPSG:32613",
    fill_value=np.nan,
    dims=("band", "y", "x"),
    coords=(SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0)),),
)
da: xr.DataArray = scene.to_xarray()                 # (band, y, x) float32
swir2: xr.DataArray = da.sel(wavelength=2190.0)      # (y, x): wavelength is indexed
back: GeoArray = GeoArray.from_xarray(da)            # identical to scene
```

## What is written

| Carrier | xarray |
| --- | --- |
| `GeoArray` | a `DataArray` with the same dims; `x` / `y` centre coordinates when axis-aligned |
| `GeoSwath` | a `DataArray` with 2-D `lat` / `lon` auxiliary coordinates |
| `GeoPoints` | a CF Discrete Sampling Geometry `Dataset` on `obs`, `featureType` from the layout |
| CRS and transform | a scalar `spatial_ref`: `crs_wkt`, `CRS.to_cf()` attributes, GDAL `GeoTransform` |
| `fill_value` | the `_FillValue` encoding and `attrs["fill_value"]` |
| `SpectralAxis` | `band` names; indexed `wavelength` and `fwhm` (nm) |
| `VariableAxis`, `VerticalAxis`, `TimeAxis` | `variable` + `variable_units`; `level` with CF `units` / `positive`; `time` |

This is the convention rioxarray, GDAL's netCDF driver, geocloud and
geopatcher read; geoarray writes it without importing rioxarray.

## Reading back

A grid is read from, in order of trust:

1. the `GeoTransform` on the variable's `grid_mapping` (`spatial_ref`);
2. else the centre coordinates, which must be regular: the transform's
   origin is the first pixel's *edge*, half a pixel from its centre;
3. else georeader's `attrs["crs"]`, with a `DeprecationWarning`.

When a `GeoTransform` and centres both exist, the centres must sit at
T(j + ½, i + ½) to 10⁻⁹ of a pixel, or `GridMismatchError` is raised: edge
coordinates under a centre transform are caught, never silently shifted. A
one-pixel axis has no spacing, so it needs the `GeoTransform`.

A `DataArray` on `lat lon` (or with CF `axis` `Y` / `X` coordinates) reads
as a grid, its horizontal dims named `y x` on the `GeoArray`. Lazy (dask)
data is computed: a carrier holds NumPy or JAX.

## Files

Open what you wrote with `decode_coords="all"`, so `spatial_ref` comes
back as a coordinate rather than a data variable:

```python
from pathlib import Path

import numpy as np
import xarray as xr

from geoarray import GeoArray

g: GeoArray = GeoArray(
    np.zeros((4, 5), dtype=np.float32),              # (H, W) float32
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
    "EPSG:32613",
    dims=("y", "x"),
)
path = Path("scene.nc")
g.to_xarray().to_dataset(name="radiance").to_netcdf(path)
with xr.open_dataset(path, decode_coords="all") as ds:
    again: GeoArray = GeoArray.from_xarray(ds["radiance"])  # identical to g
```

## Several carriers

- `geoarray.to_dataset({"radiance": rad, "mask": mask})` — carriers on one
  grid, one `Dataset`; mismatched grids raise.
- `geoarray.to_datatree({"r10m": g10, "r20m": g20, "r60m": g60})` — several
  grids, one node each; `geoarray.from_datatree` gives them back by node.

## Readable types on xarray

The converters register a kind reader for `DataArray`, so
`geoarray.types.matches(cube, GeoMapSeries)` is true for a lazy cube on
`time lat lon` without computing or renaming it, and a curvilinear model
grid (2-D `lat` / `lon`) reads as a swath.

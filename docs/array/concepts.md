# Concepts

geoarray has three carriers, one rule for what compiles, and one set of
dim-level functions shared with every named array of the stack. This page
is the mental model; the [reference](api.md) has every signature.

## A new carrier only when the georeference changes

A carrier holds an array (NumPy or JAX) and says where its samples are.
The *where* decides the carrier class; everything else is a named dim.

| Carrier | Georeference | Spatial dims (last) | Typical data |
| --- | --- | --- | --- |
| `GeoArray` | an affine transform `(…, 6)` | `y x` | a Sentinel-2 scene, an SLA analysis, a geopatcher batch |
| `GeoSwath` | per-pixel `lat`, `lon` `(…, row, col)` | `row col` | EMIT or TROPOMI in sensor geometry, an ORCA model grid |
| `GeoPoints` | `xy (obs, 2)`; optional `z`, `time`, `valid (obs,)` | `obs` | Argo profiles, weather stations, along-track altimetry |

Every axis has a name, stored on the carrier (`dims`), and a non-spatial
dim may have a coordinate from `geoarray.coords` saying what its entries
are:

| Coordinate | Dim | Holds |
| --- | --- | --- |
| `SpectralAxis` | `band` | band names, centre wavelengths and FWHM (nm) |
| `VariableAxis` | `variable` | names and units of physical quantities |
| `VerticalAxis` + `Vertical` | `level` | shared levels: kind, units, CF `positive`, datum |
| `TimeAxis` | `time` | shared `datetime64[ns]` stamps |

`band` and `variable` are separate on purpose: spectral channels share units
and have wavelengths, so a normalized difference makes sense over them;
station variables carry different units, and an operator that mixes them
must refuse.

```python
import numpy as np

from geoarray import GeoArray
from geoarray.coords import SpectralAxis, TimeAxis

# Shapes: T = 3 dates, C = 2 bands, H × W = 4 × 5 pixels of 20 m
dates: TimeAxis = TimeAxis(["2026-06-01", "2026-06-11", "2026-06-21"])
swir: SpectralAxis = SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0))
stack: GeoArray = GeoArray(
    np.zeros((3, 2, 4, 5), dtype=np.float32),       # (T, C, H, W) float32
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),   # (a, b, c, d, e, f)
    "EPSG:32613",
    fill_value=np.nan,
    dims=("time", "band", "y", "x"),
    coords=(dates, swir),
)
b12: GeoArray = stack.sel(band="B12")                # (T, H, W) float32: dims time y x
first: GeoArray = stack.isel(time=0)                 # (C, H, W) float32: dims band y x
```

`sel` takes exact labels only (a range raises `LabelError` and names
`to_xarray()`); selection by value belongs to xarray.

## Static or leaf

A carrier is a JAX pytree. What fixes a compiled program is **static**;
what changes from call to call is a **leaf**, so a new scene, patch or day
of observations never recompiles.

| Field | Role | Changing it |
| --- | --- | --- |
| `data` | leaf | free (same shape and dtype) |
| `transform`, swath `lat` / `lon`, point `xy` / `z` / `time` / `valid` | leaf | free; differentiable |
| `TimeAxis` stamps (`int64` ns) | leaf | free: new dates of the same length reuse the program |
| `dims`, `coords` (minus stamps), `crs`, `fill_value`, `attrs` | static | recompiles |

A JAX transform is float64 (`jax_enable_x64`), because a float32 northing
in UTM is only good to 0.25 m. `to_jax()` and `to_numpy()` are the only
crossings between the two backends.

## One set of dim-level functions

`isel`, `sel`, `transpose`, `take_along`, `map_over`, `concat` and
`combine` are pipekit-array's functions (`geoarray.isel is
pipekit_array.isel`). The named reductions, `reduce` and `argreduce`, are
pipekit-array's too, but they compute, so geoarray does not re-export them:
`geoops.compositing.reduce` / `argreduce` leave nodata out first (GA-B6). They reach a carrier through
the named-array adapter geoarray registers with pipekit at import, so the
same call runs on a `GeoArray`, a `pipekit_array.DimArray` and an
`xr.DataArray`, and so does a pipekit-array `KernelOp`.

The adapter keeps the georeference while it still describes the result:

- a reduction over `time` or `band` is still a `GeoArray` on the same grid;
- a reduction over `y` or `x` (per-column statistics) is a `DimArray`,
  because no grid describes it any more;
- joins check the grid exactly and never align: under `jit`, at run time.

```python
import numpy as np
import pipekit_array as pa
from pipekit_array import DimArray

from geoarray import GeoArray

from geoops.compositing import reduce

scene: GeoArray = GeoArray(
    np.random.default_rng(0).random((3, 4, 5)),     # (T, H, W) float64
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
    "EPSG:32613",
    dims=("time", "y", "x"),
)
mean: GeoArray = reduce(scene, "mean", "time")          # (H, W) float64, on the grid
profile: DimArray = pa.reduce(scene, "mean", "x")        # (T, H) float64, no grid
```

## Readable types

A readable type is a name, a dims pattern and a geometry kind, not a
subclass. Slicing a `GeoImageSeries` at one date simply is a `GeoImage`.

| Type | Kind and dims |
| --- | --- |
| `GeoMap` | grid `y x` |
| `GeoImage` | grid `band y x` |
| `GeoMapSeries` | grid `time y x` |
| `GeoImageSeries` | grid `time band y x` |
| `GeoVolume` | grid `level y x` |
| `GeoPatches` | grid `patch band y x` |
| `GeoSwathImage` | swath `band row col` |
| `GeoSwathProfile` | swath `level row col` |
| `GeoTimeSeries` | points `time variable obs` |
| `GeoTracks` | points `variable obs`, with a per-point time |
| `GeoProfiles` | points `variable obs`, with a z |

A time series is not a video: an image series is frames on a grid; a time
series is values at fixed places. Any carrier whose adapter and kind fit
matches a readable type, so a lazy `DataArray` cube on `time lat lon`
(with `geotoolz-array[xarray]`) is a `GeoMapSeries` too.
`geoarray.types.check_dims` checks annotated arguments and returns at the
call:

```python
import numpy as np

from geoarray import GeoArray
from geoarray.types import GeoImage, GeoMap, check_dims


@check_dims
def first_band(scene: GeoImage) -> GeoMap:
    return scene.isel(band=0)


chip: GeoArray = GeoArray(
    np.zeros((2, 4, 4)),                             # (C, H, W) float64
    (1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
    None,
    dims="band y x",
)
band: GeoArray = first_band(chip)                    # (H, W) float64
```

Calling `first_band` on a `(time, band, y, x)` stack raises `DimsError`
and names the fix: `geoarray.map_over(x, first_band, "time")`.

## The membership test

geoarray and geoops split the work in one place:

![What geoarray does and what geoops does: geoarray re-indexes (isel, sel, crop, read_window, pad), joins (stack, concatenate, concat), carries your function (combine, map_over, map), re-describes (dims, coords, grid, pixel_coords, footprint) and locates pixels (to_pixel); its two documented exceptions, valid_mask and astype, compute; geoops computes new values on the same grid, new grids, values at points (interp_at, InterpAt) and nodata-aware reductions (reduce, argreduce)](../assets/diagrams/geoarray-verbs.png)

```
geoarray: re-indexes and re-describes; computes no values of its own
  re-index     isel, sel, crop, read_window, write_window, pad
  join         stack, concatenate, concat        one grid
  carry        combine, map_over, GeoArray.map    one grid; your fn computes
  re-describe  dims, coords, pixel_coords, footprint
  locate       to_pixel                           coordinates → pixels
  exceptions   valid_mask, astype                 documented below

geoops: computes
  new values   indices, radiometry, mask, normalize, …   same grid
  new grid     reproject, resample, warp, mosaic, rasterize
  at points    geom.interp_at, InterpAt                  H(x)
  reductions   compositing.reduce, argreduce             nodata left out
```

The test: if the output holds only input values, fills and coordinates,
it is geoarray. Otherwise it is geoops, unless it is one of the two
documented exceptions below. A new function that computes values goes to
geoops, or joins this list with its reason; never neither.

Two geoarray functions compute, each for a stated reason:

| Function | What it computes | Why it stays in geoarray |
|---|---|---|
| `GeoArray.valid_mask` | a bool per pixel: every band finite and not the fill (a reduction over bands) | it defines the nodata model: `fill_value` is a field of the carrier, and packages that may not import geoops (geoproducts, geopatcher) need it |
| `GeoArray.astype` | a cast to another dtype; float to int truncates | it changes the representation only and carries the fill; scaling and rounding are geoops' (`radiometry`, `normalize`) |

Two more used to live here and moved to geoops (geostack GA-B6): H(x),
which blends the four pixels around each point, is `geoops.geom.interp_at`
(and the `InterpAt` operator); the named reductions are
`geoops.compositing.reduce` / `argreduce`, which leave nodata out
(`pipekit_array.reduce` / `argreduce` reduce raw values).

`combine`, `map_over` and `GeoArray.map` align carriers on one grid and
run your function on their data; geoarray computes nothing of its own.
`to_pixel`, `pixel_coords` and `footprint(crs=…)` compute coordinates, not
values.

`sample` and `member` are dim names, not functions: a `sample` is one
item of an ML batch, a `member` one draw of an ensemble, and `obs` one
point of a `GeoPoints`.

## Moving from 0.2

0.3.0 stores the dims. Every 0.2 form still works for one release, with a
`DeprecationWarning` that points at your code:

| 0.2 | 0.3 |
| --- | --- |
| `GeoArray(data, T, crs)`, dims from the rank | `GeoArray(data, T, crs, dims="band y x")` |
| `attrs={"band_names": [...]}` | `coords=(SpectralAxis(names, wavelength_nm),)` (the attrs are read into one) |
| `stack(gs)`, `stack(gs, axis=1)` | `stack(gs, "time")` |
| `concatenate(gs, axis=-3)` | `concat(gs, "band")` |
| `GeoArray[np.ndarray]` | `GeoArray["band y x"]` |

The geoops kernels still read the band axis as −3 until they take `dim=`,
so positionally (`g[:, k]`) a `(time, band, y, x)` carrier never loses its
band axis alone; by name (`g.isel(band=k)`) it may.

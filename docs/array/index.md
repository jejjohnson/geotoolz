# geotoolz-array

> **One georeferenced array for NumPy and JAX.** A `GeoArray` holds the
> array, its affine transform, CRS, fill value and metadata, and knows which
> pixel sits where — imported as `geoarray`.

## Where it comes from

A raster is an array plus where its pixels are. Keep the two apart and
every function has to carry both; keep them together by subclassing the
array type, and the first operation from another array library loses the
georeferencing. JAX arrays cannot be subclassed at all.

`GeoArray` *holds* its array instead, so one carrier works on NumPy and on
JAX. It is a JAX pytree: the data and a float64 transform are leaves, which
`jit`, `grad` and `vmap` trace, and the CRS, fill value and attrs are static.
A new transform (another scene, another patch) reuses compiled code, and
gradients reach the georeferencing.

![A GeoArray: the data (..., H, W) and the (6,) float64 transform are pytree leaves, traced by jit, grad and vmap; the CRS, fill value and attrs are static and key the compile cache; g.grid derives the Grid (transform, CRS and (H, W) without pixels), and a PixelWindow is a rectangle of pixels on it](../assets/diagrams/geoarray-carrier.png)

## Install

```bash
pip install geotoolz-array              # NumPy carriers, Grid, PixelWindow
pip install 'geotoolz-array[jax]'       # + JAX carriers and the pytree
pip install 'geotoolz-array[vector]'    # + footprint as a shapely polygon
pip install 'geotoolz-array[georeader]' # + GeoTensor <-> GeoArray
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | numpy, pyproj, affine | `GeoArray` on NumPy data, `Grid`, `PixelWindow`, `FrozenAttrs` |
| `[jax]` | jax | JAX data, `GeoArray.to_jax`, `jit` / `grad` / `vmap` over carriers |
| `[vector]` | shapely | `footprint` |
| `[georeader]` | georeader-spaceml | `geoarray.interop.georeader` (`from_georeader`, `to_georeader`) |

A JAX carrier keeps its transform in float64, so it needs
`jax.config.update("jax_enable_x64", True)`. The data keeps its own dtype.

## Quickstart

Wrap a Sentinel-2 chip in UTM 13N, read its grid, then move it to JAX and
jit a function of it.

```python
import jax
import numpy as np

from geoarray import GeoArray, Grid

jax.config.update("jax_enable_x64", True)

# Shapes: C = 2 bands (B11, B12), H × W = 600 × 600 pixels of 20 m
chip: GeoArray = GeoArray(
    np.zeros((2, 600, 600), dtype=np.float32),  # (C, H, W) float32
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),  # (a, b, c, d, e, f)
    "EPSG:32613",
    fill_value=np.nan,
    attrs={"band_names": ["B11", "B12"]},
)
grid: Grid = chip.grid  # transform, CRS, (600, 600)
grid.bounds  # (600000.0, 3538000.0, 612000.0, 3550000.0)

on_device: GeoArray = chip.to_jax()  # (C, H, W) float32 · transform float64
total = jax.jit(lambda g: g.data.sum())(on_device)  # () float32
```

Slice it, cut a window by map bounds, and crop patches inside `vmap`:
each result carries the transform of the pixels it holds.

```python
import jax
import jax.numpy as jnp
import numpy as np

from geoarray import GeoArray, PixelWindow, crop, read_window

jax.config.update("jax_enable_x64", True)

# Shapes: C = 2 bands, H × W = 600 × 600 pixels of 20 m
scene: GeoArray = GeoArray(
    np.zeros((2, 600, 600), dtype=np.float32),  # (C, H, W) float32
    (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),  # (a, b, c, d, e, f)
    "EPSG:32613",
    fill_value=np.nan,
).to_jax()

sub: GeoArray = scene[..., 100:356, 200:456]  # (C, 256, 256): the origin moves
window: PixelWindow = scene.grid.window((604000.0, 3542880.0, 609120.0, 3548000.0))
same: GeoArray = read_window(scene, window)  # the same pixels as sub

offsets = jnp.array([[0, 0], [256, 256], [344, 344]])  # (N, 2) int
patches: GeoArray = jax.jit(jax.vmap(lambda o: crop(scene, o, (256, 256))))(offsets)
patches.batch_shape  # (3,): (N, C, 256, 256) data, one (6,) transform per patch
valid = scene.valid_mask()  # (H, W) bool: every band finite and not the fill
```

## Backends and georeader

NumPy and JAX carriers mix eagerly, as NumPy and `jnp` arrays do. Inside
`jit`, `grad` or `vmap` the data are tracers, and NumPy code on them raises
`GeoArrayTraceError` instead of JAX's generic error.

| Call | NumPy carrier | JAX carrier |
|---|---|---|
| `np.asarray(g)` | the data, no copy | a host copy; raises under a trace; warns with `set_options(warn_on_transfer=True)` |
| `combine`, `stack`, `concatenate`, `write_window` with mixed inputs | — | NumPy inputs move to JAX (needs x64) |
| `from_georeader(gt)` / `to_georeader(g)` | `GeoTensor` ↔ `GeoArray`, sharing memory | `to_georeader` needs `g.to_numpy()` first |

```python
import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor

from geoarray import GeoArray
from geoarray.interop.georeader import from_georeader, to_georeader

gt: GeoTensor = GeoTensor(
    np.zeros((2, 600, 600), dtype=np.float32),  # (C, H, W) float32
    Affine(20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
    "EPSG:32613",
    fill_value_default=np.nan,
)
g: GeoArray = from_georeader(gt)  # (C, H, W) float32, the same memory
back: GeoTensor = to_georeader(g[..., :256, :256])  # (C, 256, 256), origin kept
```

## What's inside

| Name | Use it to… | Guide |
|---|---|---|
| `GeoArray` | carry data with its transform, CRS, fill value and attrs; `grid`, `with_data`, `to_jax`, `to_numpy` | [Reference](api.md#geoarray) |
| `Grid` | describe a pixel grid without data: `bounds`, `res`, `shift`, `crop`, `from_bounds`, `isclose`, `get_config` | [Reference](api.md#grid) |
| `PixelWindow` | name a rectangle of pixels, rasterio's `Window` without rasterio; `Grid.window` / `Grid.bounds_of` convert to and from map bounds | [Reference](api.md#pixelwindow) |
| `g[...]`, `isel` | slice with the transform following; select bands with their per-band attrs | [Reference](api.md#geoarray) |
| `map`, `shift_origin`, `squeeze`, `expand_dims` | change the data on the same grid, move the georeferencing, add or drop leading axes | [Reference](api.md#geoarray) |
| `crop`, `read_window`, `write_window`, `pad` | cut, read past the edges, write and pad pixels; `crop` inside `jit` / `vmap` | [Reference](api.md#geometry) |
| `valid_mask`, `pixel_coords`, `footprint` | find the pixels that hold data, their map coordinates, the outline (`[vector]`) | [Reference](api.md#geometry) |
| `combine`, `stack`, `concatenate` | join carriers that share a grid | [Reference](api.md#joining-carriers) |
| `FrozenAttrs` | keep metadata hashable, so it can key a compiled function | [Reference](api.md#frozenattrs) |
| `RasterSource`, `AsyncRasterSource` | describe a lazy reader: its grid, and reads that return a `GeoArray` (`read_window`, `load`); a `GeoArray` is one | [Reference](api.md#lazy-readers) |
| `set_options`, `get_options`, `HostTransferWarning` | find the NumPy steps that copy JAX data to the host | [Reference](api.md#backends-and-interop) |
| `from_georeader`, `to_georeader` | bridge to georeader's `GeoTensor`, zero-copy, and (`source_from_georeader` / `source_to_georeader`) its lazy readers (`[georeader]`) | [Reference](api.md#backends-and-interop) |
| `as_transform`, `as_crs` | coerce an `affine.Affine`, six numbers, an EPSG code or WKT | [Reference](api.md#coercions) |

## Next steps

- [API reference](api.md).
- The design, its decisions and its roadmap: [geoarray: one NumPy/JAX
  carrier for geotoolz](https://github.com/jejjohnson/geostack/blob/main/docs/design/01_geoarray.md).

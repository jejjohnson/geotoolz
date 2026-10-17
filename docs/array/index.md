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
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | numpy, pyproj, affine | `GeoArray` on NumPy data, `Grid`, `PixelWindow`, `FrozenAttrs` |
| `[jax]` | jax | JAX data, `GeoArray.to_jax`, `jit` / `grad` / `vmap` over carriers |

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

## What's inside

| Name | Use it to… | Guide |
|---|---|---|
| `GeoArray` | carry data with its transform, CRS, fill value and attrs; `grid`, `with_data`, `to_jax`, `to_numpy` | [Reference](api.md#geoarray) |
| `Grid` | describe a pixel grid without data: `bounds`, `res`, `shift`, `isclose`, `get_config` | [Reference](api.md#grid) |
| `PixelWindow` | name a rectangle of pixels, rasterio's `Window` without rasterio | [Reference](api.md#pixelwindow) |
| `FrozenAttrs` | keep metadata hashable, so it can key a compiled function | [Reference](api.md#frozenattrs) |
| `as_transform`, `as_crs` | coerce an `affine.Affine`, six numbers, an EPSG code or WKT | [Reference](api.md#coercions) |

## Next steps

- [API reference](api.md).
- The design, its decisions and its roadmap: [geoarray: one NumPy/JAX
  carrier for geotoolz](https://github.com/jejjohnson/geostack/blob/main/docs/design/01_geoarray.md).

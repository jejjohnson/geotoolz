# Concepts

Every product is read the same way. A reader is a georeader `GeoData`,
`stack` puts readers on one grid, and recipes and presets turn bands into
pictures and masks. The sensor pages build on these three ideas.

## The ProductReader contract

A reader is a `geoproducts.ProductReader`, which is a georeader `GeoData`.
It answers metadata without reading pixels, and decodes only the window
you ask for.

| Part | What it gives you |
|---|---|
| `crs`, `transform`, `shape`, `dtype` | the grid, known at construction |
| `bands`, `fill_value_default` | band names (also in every read's `attrs["band_names"]`) and the nodata value |
| `read_from_window`, `read_from_bounds`, `load` | a `GeoTensor`; windows past the grid are padded with the fill |
| `track` | `"A"` for a clean affine grid; `"B"` for irregular per-pixel geolocation |
| `set_obstore_client` | byte-range reads through the shared [`geocloud.store`](../cloud/concepts.md#the-pool) pool (`[obstore]`) |

Every reader shipped today is Track A: GOES and Himawari recover the
geostationary fixed grid, Carbon Mapper rasters are GeoTIFFs. Because a
reader is a `GeoData`, georeader's `read_*` helpers, `geopatcher.RasterField`
and geotoolz's `io` operators accept it as it is.

```python
import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor

from geoproducts import ProductReader, toy_sensor

reader: ProductReader = toy_sensor.Reader(
    "scene",
    data=np.random.default_rng(0).random((4, 256, 256), dtype=np.float32),
    transform=Affine(0.001, 0.0, 10.0, 0.0, -0.001, 45.0),
)                                                        # (4, 256, 256) float32 · EPSG:4326
track: str = reader.track                                # 'A'
box: GeoTensor = reader.read_from_bounds((10.0, 44.9, 10.1, 45.0))  # (4, 100, 100) float32
```

How to write one — the layout, the toolkit and the tests — is in
[Add a product reader](product-readers.md).

## One grid: `stack`

`geoproducts.stack(readers)` puts several readers' bands on the grid of
one reference reader (the first, or `like=`). It reads every other reader
only where it overlaps that grid and warps it there with georeader.

- **Finer float grids** are averaged; **coarser** ones interpolated
  bilinearly.
- **Integer readers** (masks, classes) are resampled with `mode` /
  `nearest` and stay integer when every reader is integer.
- **Mixed** integer and float stacks decode integer fills to `NaN`.

```python
import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor

import geoproducts
from geoproducts import toy_sensor

rng: np.random.Generator = np.random.default_rng(0)
fine: toy_sensor.Reader = toy_sensor.Reader(
    "fine", data=rng.random((4, 256, 256), dtype=np.float32),
    transform=Affine(0.001, 0.0, 10.0, 0.0, -0.001, 45.0),
)                                                        # (4, 256, 256) float32
coarse: toy_sensor.Reader = toy_sensor.Reader(
    "coarse", data=rng.random((4, 128, 128), dtype=np.float32),
    transform=Affine(0.002, 0.0, 10.0, 0.0, -0.002, 45.0),
)                                                        # (4, 128, 128) float32

scene: GeoTensor = geoproducts.stack([fine, coarse])     # (4 + 4, 256, 256) float32 on fine's grid
```

Pass `bounds=` (with `crs_bounds=`) to read only an area of the
reference grid. The result names every band, in reader order.

## Recipes and presets

A **recipe** is data: three band expressions, each stretched between
fixed bounds and gamma-corrected. Recipes live in `<sensor>.recipes`,
follow the published quick guides, and need no extra.

A **preset** is a geotoolz operator bound to a sensor's band names:
`goes.TrueColor()` is a `gz.viz.RGBRecipe` built from
`goes.recipes.TRUE_COLOR`, and `goes.NDVI()` is `gz.NDVI(red="C02",
nir="C03")`. Presets need the `[operators]` extra; geotoolz is imported
only when you call one.

```python
import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor
from pipekit import Operator

from geoproducts import goes

spec: goes.recipes.Recipe = goes.recipes.TRUE_COLOR      # red C02 · synthetic green · blue C01
needs: tuple[str, ...] = spec.channels                   # ('C01', 'C02', 'C03')

scene: GeoTensor = GeoTensor(
    np.random.default_rng(0).random((3, 64, 64), dtype=np.float32),
    transform=Affine(2004.0, 0.0, 0.0, 0.0, -2004.0, 0.0), crs="EPSG:4326",
    fill_value_default=np.nan, attrs={"band_names": ("C01", "C02", "C03")},
)                                                        # (3, 64, 64) float32 reflectance
true_color: Operator = goes.TrueColor()                  # gz.viz.RGBRecipe
rgb: GeoTensor = true_color(scene)                       # (3, 64, 64) float32 → (3, 64, 64) float32 in [0, 1]
```

Presets resolve bands by name. They work on any stack with the right
`band_names`: L1b reads, L2 imagery, or a `stack` of both. An operator that is genuinely new belongs in a geotoolz
family; a preset only binds names.

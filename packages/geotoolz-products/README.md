# geoproducts
> Part of the [geotoolz monorepo](https://github.com/jejjohnson/geotoolz) — ships as the `geotoolz-products` distribution; the import name is `geoproducts`.

[![Tests](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Every Earth-observation product, read as a georeader `GeoTensor`.**
> One reader per mission or provider; what comes out is what every geotoolz operator and geopatcher field takes.

<p align="center"><img src="../../docs/assets/diagrams/products-architecture.png" alt="geoproducts: one namespace per product (Reader, bucket or API helpers, BANDS, recipes, presets), built on a shared toolkit and the ProductReader contract, produces GeoTensors for geotoolz and geopatcher" width="100%"></p>

## 30-second pitch

Every mission ships its own format, band naming, calibration tables and,
for provider APIs, its own authentication and catalogue quirks. That
churn belongs in one place, not scattered across analysis code.
`geoproducts` gives each product a reader that hides it: `ProductReader`
is a georeader `GeoData` with a lazy windowed read, named bands and an
optional pooled cloud byte-range path, and each sensor ships as a small
namespace (`Reader`, `BANDS`, `CONSTANTS`, `presets`). Mission readers
such as **GOES-R ABI** and **Himawari AHI** recover the native grid and calibrate from the
file's own coefficients; provider clients such as **Carbon Mapper** turn
REST and STAC responses into typed records and lazy rasters. The package depends on georeader only — never on
geotoolz — so readers release on their own schedule while their output
drops straight into the operators, the patcher and the catalog loaders.

## Install

```bash
pip install geotoolz-products                    # readers (georeader only)
pip install 'geotoolz-products[obstore]'         # pooled cloud byte-range reads + NOAA bucket helpers
pip install 'geotoolz-products[operators]'       # sensor presets (geotoolz operators)
pip install 'geotoolz-products[goes]'            # GOES-R ABI reader (h5py) + goes.aws (geotoolz-cloud)
pip install 'geotoolz-products[himawari]'        # Himawari L2 products (h5py) + himawari.aws (geotoolz-cloud)
pip install 'geotoolz-products[carbonmapper]'    # Carbon Mapper plume catalogue + STAC
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | georeader, numpy, rasterio | `ProductReader`, `toy_sensor` |
| `[obstore]` | `geotoolz-cloud` | `ProductReader._read_bytes` over `s3://` / `gs://` / `az://` through the shared `geocloud.store` pool; the `aws` bucket helpers |
| `[operators]` | `geotoolz` | per-sensor `presets` (e.g. `toy_sensor.presets.NDVI`) |
| `[goes]` | h5py, geotoolz-cloud | `geoproducts.goes.Reader`, the `goes.aws` bucket helpers |
| `[himawari]` | h5py, geotoolz-cloud | `geoproducts.himawari.L2Reader`, the `himawari.aws` bucket helpers (the HSD `Reader` needs no extra) |
| `[carbonmapper]` | requests, pydantic, shapely, geopandas, pandas | `geoproducts.carbonmapper` |

## Quickstart — a sensor reader

`toy_sensor` is the in-memory reference reader that exercises the full
`ProductReader` contract:

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geoproducts import ProductReader, toy_sensor

data: np.ndarray = np.random.default_rng(0).random((4, 256, 256), dtype=np.float32)  # (4, 256, 256)
reader: ProductReader = toy_sensor.Reader("scene", data=data)   # shape (4, 256, 256) · crs EPSG:4326

chip: GeoTensor = reader.read_from_window(Window(0, 0, 64, 64)).load()  # (4, 64, 64) float32
chip.attrs["band_names"]                                         # ('blue', 'green', 'red', 'nir')
```

Because `chip` names its bands, geotoolz operators resolve them by name
(`gz.NDVI(nir="nir", red="red")(chip)` → `(64, 64)`), and because
`reader` is a `GeoData`, `geopatcher.RasterField(reader)` tiles it.

## GOES-R ABI

GOES-16 … GOES-19, L1b radiances and every L2 product, straight from NOAA's
public, anonymous AWS buckets. Every file sits on the `+proj=geos` fixed
grid; windows decompress only the chunks they touch:

```python
from datetime import datetime
from pathlib import Path

from georeader.geotensor import GeoTensor

from geoproducts import goes, stack
from geoproducts.goes import aws

files: list[aws.ABIFile] = aws.list_files(
    satellite="G19", product="ABI-L1b-RadC",                  # GOES-East, CONUS sector
    start=datetime(2026, 10, 7, 18), end=datetime(2026, 10, 7, 18, 10),
    channel=13,                                               # 10.3 µm clean IR window
)                                                             # 2 scans · ~4 MB each
reader: goes.Reader = goes.Reader(
    aws.download(files[0], "data/goes"), calibration="brightness_temperature"
)                                                             # (1, 1500, 2500) · 2 km
bt: GeoTensor = reader.read_from_bounds(
    (-100.0, 30.0, -95.0, 35.0), crs_bounds="EPSG:4326"
)                                                             # (1, 219, 266) float32 K

cmi: goes.L2Reader = goes.L2Reader(mcmip_path)                # (16, 500, 500) C01 … C16, calibrated
acm: goes.L2Reader = goes.L2Reader(acm_path)                  # (2, 500, 500) uint8 BCM, ACM
acha: goes.L2Reader = goes.L2Reader(acha_path)                # (1, 250, 250) cloud-top height, 4 km
scene: GeoTensor = stack([cmi, acm, acha])               # (19, 500, 500) on one grid
rgb: GeoTensor = goes.DayCloudPhase()(scene)                  # (3, 500, 500) float32 in [0, 1]
```

<p align="center"><img src="../../docs/assets/figures/goes-recipes.jpg" alt="True colour, natural colour, day cloud phase and fire temperature RGBs of one GOES-19 mesoscale scan" width="100%"></p>

Calibration uses the coefficients in each file. The `[operators]` presets
add `goes.TrueColor()`, `NaturalColor()`, `DayCloudPhase()`,
`FireTemperature()` (on the generic `gz.viz.RGBRecipe`), `MaskClouds()`,
`NDVI()`, `SyntheticGreen()` and `ParallaxCorrect()`.

## Himawari AHI

Himawari-8 / -9 at 140.7°E. JMA's binary HSD segments are decoded natively
(numpy + standard library), straight from NOAA's public buckets (the
`[himawari]` extra brings the bucket helpers); the
segments of a band assemble onto the `+proj=geos` grid, and a window
decodes only the segments it overlaps:

```python
from geoproducts import himawari
from geoproducts.himawari import aws

segments: list[aws.Segment] = aws.list_segments(
    start=datetime(2026, 10, 7, 3), band=13, segments=[5, 6],  # 10°N – 10°S strips
)                                                             # 2 files · ~3 MB each
reader: himawari.Reader = himawari.Reader(
    [aws.download(s, "data/ahi") for s in segments], calibration="brightness_temperature"
)                                                             # (1, 5500, 5500) · 2 km
bt: GeoTensor = reader.read_from_bounds(
    (110.0, -5.0, 120.0, 5.0), crs_bounds="EPSG:4326"
)                                                             # (1, 546, 477) float32 K
mask: himawari.L2Reader = himawari.L2Reader(cmsk_path)        # (2, 5500, 5500) int8 NOAA cloud mask
```

<p align="center"><img src="../../docs/assets/figures/himawari-japan.jpg" alt="Himawari-9 Japan area: true colour, day cloud phase and the NOAA L2 cloud mask on one grid" width="100%"></p>

The same presets as GOES — `himawari.TrueColor()` (with a hybrid green),
`NaturalColor()`, `DayCloudPhase()`, `FireTemperature()`, `MaskClouds()`,
`NDVI()`, `HybridGreen()` and `ParallaxCorrect()` — bind to AHI band names.

## Carbon Mapper

```python
from georeader.geotensor import GeoTensor

from geoproducts import carbonmapper as cm

config: cm.CarbonMapperConfig = cm.CarbonMapperConfig.load()     # ~/.geoproducts/auth_carbonmapper.json or env
token: str = config.get_token() or config.refresh_access_token()

plumes: list[cm.CMRawPlume] = cm.list_plumes(token, bbox=(-104.5, 31.5, -103.5, 32.5), limit=50)
image: cm.CMPlumeImage = cm.CMPlumeImage.from_plume_id(plumes[0].plume_id, token=token)
cmf: GeoTensor = image.tile_cmf(pad_px=64)                       # (1, h, w) — L2B retrieval around the plume

sources: list[cm.CMSource] = cm.list_sources(token, bbox=(-104.5, 31.5, -103.5, 32.5), sectors=["1B2"])
labels: GeoTensor = cm.rasterize_sources_like(sources, cmf, buffer_m=150.0)  # (h, w) uint8 on cmf's grid
```

Credentials resolve from `~/.geoproducts/auth_carbonmapper.json`, or from
`CARBONMAPPER_TOKEN` / `CARBONMAPPER_EMAIL` + `CARBONMAPPER_PASSWORD`;
the file is written with owner-only permissions, and the token is only
ever sent to Carbon Mapper's API host. Without the `[carbonmapper]`
extra, `import geoproducts.carbonmapper` raises an `ImportError` naming
it.

## Next steps

- **Products docs:** [home](https://jejjohnson.github.io/geotoolz/products/) ·
  [adding a product reader](https://jejjohnson.github.io/geotoolz/products/product-readers/) ·
  [GOES-R ABI](https://jejjohnson.github.io/geotoolz/products/goes/) ·
  [Himawari AHI](https://jejjohnson.github.io/geotoolz/products/himawari/) ·
  [Carbon Mapper](https://jejjohnson.github.io/geotoolz/products/carbonmapper/) ·
  [API reference](https://jejjohnson.github.io/geotoolz/products/api/).
- **The whole stack:** the root [README](https://github.com/jejjohnson/geotoolz#readme)
  walks a Carbon Mapper + Sentinel-2 methane example through all four packages.

## License

MIT — see [LICENSE](../../LICENSE).

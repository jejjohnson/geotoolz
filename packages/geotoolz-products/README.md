# geoproducts
> Part of the [geotoolz monorepo](https://github.com/jejjohnson/geotoolz) — ships as the `geotoolz-products` distribution; the import name is `geoproducts`.

[![Tests](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geotoolz/actions/workflows/typecheck.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Every Earth-observation product, read as a georeader `GeoTensor`.**
> One reader per mission or provider; what comes out is what every geotoolz operator and geopatcher field takes.

<p align="center"><img src="../../docs/assets/diagrams/products-architecture.png" alt="geoproducts: ProductReader readers such as toy_sensor and the carbonmapper provider client both produce GeoTensors for geotoolz and geopatcher" width="100%"></p>

## 30-second pitch

Every mission ships its own format, band naming, calibration tables and,
for provider APIs, its own authentication and catalogue quirks. That
churn belongs in one place, not scattered across analysis code.
`geoproducts` gives each product a reader that hides it: `ProductReader`
is a georeader `GeoData` with a lazy windowed read, named bands and an
optional pooled cloud byte-range path, and each sensor ships as a small
namespace (`Reader`, `BANDS`, `CONSTANTS`, `presets`). Provider clients
such as **Carbon Mapper** turn REST and STAC responses into typed records
and lazy rasters. The package depends on georeader only — never on
geotoolz — so readers release on their own schedule while their output
drops straight into the operators, the patcher and the catalog loaders.

## Install

```bash
pip install geotoolz-products                    # readers (georeader only)
pip install 'geotoolz-products[obstore]'         # pooled cloud byte-range reads
pip install 'geotoolz-products[operators]'       # sensor presets (geotoolz operators)
pip install 'geotoolz-products[carbonmapper]'    # Carbon Mapper plume catalogue + STAC
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(base)* | georeader, numpy, rasterio | `ProductReader`, `toy_sensor` |
| `[obstore]` | `geotoolz-patcher[obstore]` | `ProductReader._read_bytes` over `s3://` / `gs://` / `az://` through the shared `geopatcher.objstore` pool |
| `[operators]` | `geotoolz` | per-sensor `presets` (e.g. `toy_sensor.presets.NDVI`) |
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
  [Carbon Mapper](https://jejjohnson.github.io/geotoolz/products/carbonmapper/) ·
  [API reference](https://jejjohnson.github.io/geotoolz/products/api/).
- **The whole stack:** the root [README](https://github.com/jejjohnson/geotoolz#readme)
  walks a Carbon Mapper + Sentinel-2 methane example through all four packages.

## License

MIT — see [LICENSE](../../LICENSE).

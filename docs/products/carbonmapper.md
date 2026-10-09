# Carbon Mapper

`geoproducts.carbonmapper` reads the [Carbon Mapper](https://carbonmapper.org)
methane and CO₂ plume catalogue and its STAC API. Plume and source
records come back as typed objects; per-plume (L3A) and per-scene (L2B)
rasters come back as lazy georeader readers or `GeoTensor`s.

```bash
pip install 'geotoolz-products[carbonmapper]'   # requests, pydantic, shapely, geopandas, pandas
```

Without the extra, `import geoproducts.carbonmapper` raises an
`ImportError` naming it; the rest of geoproducts is unaffected. Every
fence on this page calls the live API, so it needs an account.

## What a product is

| Product | What it is | Returned as |
|---|---|---|
| Plume (`CMRawPlume`) | one detection: location, gas, emission rate, instrument, scene | pydantic record |
| Source (`CMSource`) | a cluster of plumes at one facility, with its sector | frozen dataclass |
| L3A plume bundle (`CMPlumeImage`) | mask, concentrations, RGB and outline cropped around one plume | lazy rasters + shapely outline |
| L2B scene (`CMImageRaster`) | the full matched-filter retrieval of one flight line or satellite scene | four lazy rasters |

`products` names every asset (`CMProduct`, `DEFAULT_PLUME_PRODUCTS`, …),
so a bundle downloads only what you select.

## Credentials

`CarbonMapperConfig.load()` resolves credentials in this order:

1. an explicit path, else `~/.geoproducts/auth_carbonmapper.json` (then
   the other `CONFIG_SEARCH_PATHS`);
2. environment variables, which override the file: `CARBONMAPPER_TOKEN`,
   or `CARBONMAPPER_EMAIL` + `CARBONMAPPER_PASSWORD`.

With nothing configured, `load()` writes a placeholder file to edit
(`create_placeholder=False` keeps the filesystem untouched).
`get_token()` renews an expired token from the stored refresh token or
the email / password pair. `save()` writes the email and tokens
owner-only, never the password, and the token is only ever sent to
Carbon Mapper's API host.

## Find plumes and sources

```python
from datetime import datetime

from geoproducts import carbonmapper as cm

config: cm.CarbonMapperConfig = cm.CarbonMapperConfig.load()
token: str = config.get_token() or config.refresh_access_token()
permian: tuple[float, float, float, float] = (-104.5, 31.5, -103.5, 32.5)   # lon/lat

plumes: list[cm.CMRawPlume] = cm.list_plumes(
    token, bbox=permian, datetime_min=datetime(2025, 1, 1), gas=cm.Gas.CH4, limit=50,
)                                                     # ≤ 50 typed records, never raw dicts
sources: list[cm.CMSource] = cm.list_sources(token, bbox=permian, sectors=["1B2"])  # oil & gas
```

`get_plume`, `get_source`, `list_tiles` and the `*_for_plume` /
`*_for_source` / `*_for_tile` helpers walk between plumes, sources and
scenes; `get_plume_context` fetches all three at once.

## Read the rasters

```python
from georeader.geotensor import GeoTensor

from geoproducts import carbonmapper as cm

token: str = cm.CarbonMapperConfig.load().get_token() or ""
plume: cm.CMRawPlume = cm.list_plumes(token, bbox=(-104.5, 31.5, -103.5, 32.5), limit=1)[0]

image: cm.CMPlumeImage = cm.CMPlumeImage.from_plume_id(plume.plume_id, token=token)
cmf: GeoTensor = image.tile_cmf(pad_px=64)            # (1, h, w) float32 · L2B retrieval, full resolution
```

`tile_cmf` crops the analysis-grade L2B retrieval to the plume outline
plus a margin. The pre-cropped L3A thumbnails (`plume-concentrations.tif`)
are only tens of pixels across.

## Label sources on a grid

`rasterize_sources_like` burns sources onto any `GeoData` grid — here the
plume's retrieval — to build training labels or screening masks:

```python
from georeader.geotensor import GeoTensor

from geoproducts import carbonmapper as cm

token: str = cm.CarbonMapperConfig.load().get_token() or ""
permian: tuple[float, float, float, float] = (-104.5, 31.5, -103.5, 32.5)
plume: cm.CMRawPlume = cm.list_plumes(token, bbox=permian, limit=1)[0]
cmf: GeoTensor = cm.CMPlumeImage.from_plume_id(plume.plume_id, token=token).tile_cmf(pad_px=64)  # (1, h, w) float32

sources: list[cm.CMSource] = cm.list_sources(token, bbox=permian, sectors=["1B2"])
labels: GeoTensor = cm.rasterize_sources_like(sources, cmf, buffer_m=150.0)  # (h, w) uint8 · 1 = within 150 m
```

`CMSourceRaster` is the lazy version, and `rasterize_sources` takes an
explicit grid.

## Module layout

| Module | Contents |
|---|---|
| `config` | `CarbonMapperConfig` — token persistence (file / env / in-memory) |
| `download` | raw HTTP / JSON primitives (`obtain_token`, `stac_search`, `download_asset`, …) |
| `api_queries` | typed wrappers (`get_plume`, `list_plumes`, `list_tiles`, `get_source`, …) and the `CMAPIError` hierarchy |
| `plume` / `source` | `CMRawPlume` (pydantic) and `CMSource` records |
| `products` | the product vocabulary (`CMProduct`, `CMCollectionSpec`, product sets) |
| `image` / `rasters` | `CMPlumeImage` (L3A per-plume bundle) and `CMImageRaster` (L2B scene) |
| `sources_raster` | sources onto a grid (`rasterize_sources`, `rasterize_sources_like`, `CMSourceRaster`) |

The live-API tests are marked `live` and always deselected in CI; run
them with a token configured:

```console
cd packages/geotoolz-products && uv run pytest -m live tests/carbonmapper/test_live.py
```

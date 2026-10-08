# Carbon Mapper

`geoproducts.carbonmapper` reads the [Carbon Mapper](https://carbonmapper.org)
plume catalogue and STAC API: plume records and sources come back as typed
objects, and the per-plume (L3A) and per-scene (L2B) rasters come back as lazy
georeader readers / `GeoTensor`s.

```bash
pip install 'geotoolz-products[carbonmapper]'   # requests, pydantic, shapely, geopandas
```

Without the extra, `import geoproducts.carbonmapper` raises an `ImportError`
naming it; the rest of `geoproducts` is unaffected.

## Credentials

`CarbonMapperConfig.load()` resolves credentials in this order:

1. an explicit path, else the first file found at
   `~/.geoproducts/auth_carbonmapper.json` (then the other
   `CONFIG_SEARCH_PATHS`);
2. environment variables, which override the file:
   `CARBONMAPPER_TOKEN`, or `CARBONMAPPER_EMAIL` + `CARBONMAPPER_PASSWORD`.

When nothing is configured, `load()` writes a placeholder file at
`~/.geoproducts/auth_carbonmapper.json` to edit (pass
`create_placeholder=False` to keep the filesystem untouched). `get_token()`
renews an expired access token from the stored refresh token or the
email / password pair. `save()` writes the email and tokens (owner-only)
but never the password: log in once with `refresh_access_token()`, save,
and later sessions renew from the refresh token.

## Usage

```python
from datetime import datetime

from georeader.geotensor import GeoTensor

from geoproducts import carbonmapper as cm

config: cm.CarbonMapperConfig = cm.CarbonMapperConfig.load()
token: str = config.get_token() or config.refresh_access_token()

# Typed catalogue queries: plume records, never raw dicts.
plumes: list[cm.CMRawPlume] = cm.list_plumes(
    token,
    bbox=(-104.5, 31.5, -103.5, 32.5),
    datetime_min=datetime(2025, 1, 1),
    gas=cm.Gas.CH4,
    limit=50,
)

# One plume's product bundle: mask, concentrations, RGB, outline, …
image: cm.CMPlumeImage = cm.CMPlumeImage.from_plume_id(plumes[0].plume_id, token=token)

# The L2B retrieval cropped to the plume outline, at full resolution.
cmf: GeoTensor = image.tile_cmf(pad_px=64)
```

The module layout:

| Module | Contents |
|---|---|
| `config` | `CarbonMapperConfig` — token persistence (file / env / in-memory) |
| `download` | Raw HTTP / JSON primitives (`obtain_token`, `stac_search`, `download_asset`, …) |
| `api_queries` | Typed wrappers (`get_plume`, `list_plumes`, `list_tiles`, `get_source`, …) and the `CMAPIError` hierarchy |
| `plume` / `source` | `CMRawPlume` (pydantic) and `CMSource` records |
| `products` | The product vocabulary (`CMProduct`, `CMCollectionSpec`, product sets) |
| `image` / `rasters` | `CMPlumeImage` (L3A per-plume bundle) and `CMImageRaster` (L2B scene) |
| `sources_raster` | Rasterize sources onto a grid (`rasterize_sources_like`) |

The live-API tests are marked `live` and always deselected in CI; run them with
a token configured:

```bash
cd packages/geotoolz-products && uv run pytest -m live tests/carbonmapper/test_live.py
```

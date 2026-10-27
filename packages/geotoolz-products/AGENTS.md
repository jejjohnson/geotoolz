# geotoolz-products (`geoproducts`) — agent rules

Readers for Earth-observation products, each turning a mission's or
provider's files into a lazy `geoarray.RasterSource` whose reads return a
`geoarray.GeoArray`. The root [`AGENTS.md`](../../AGENTS.md) applies too;
this file adds what is specific to the readers. geotoolz is an optional
extra (`[operators]`, for presets) and never a hard dependency. georeader
is a base dependency used only internally, as an engine (map-coordinate
windows, reprojection, rasterize, `RasterioReader`): every public API takes
and returns `GeoArray` / `RasterSource`, never `GeoTensor` / `GeoData`.
Satpy is never a dependency — it may be read for reference, not imported.

## The reader contract

Every reader subclasses `ProductReader` (`_src/base.py`), which is a
`geoarray.RasterSource` (no georeader base class). Implement
`_read_window(window: PixelWindow) -> np.ndarray` (usually through
`_read_boundless`) and the metadata properties `_crs`, `_transform`,
`_shape`, `_dtype`, `_bands`, `_fill_value` (set explicitly), `_track`
(Track A: a clean affine grid; Track B: irregular geolocation). The base
class supplies the `RasterSource` surface — `grid`, `shape`, `dtype`,
`fill_value`, `attrs` (`band_names` + `_band_attrs()`),
`read_window(window, *, boundless=True)`, `load()` — plus `crs`,
`transform` (an `Affine`), `res`, `bounds`, `bands`, `dims`, `track` and
`read_from_bounds` / `read_from_center_coords` (georeader's window maths,
through `geoarray.interop.georeader.source_to_georeader`), all returning
`GeoArray`.

- Call georeader only at the engine boundary:
  `from_georeader(read.read_from_bounds(source_to_georeader(src), ...))`.
- A georeader `GeoTensor` / `GeoData` passed in raises `TypeError` naming
  `from_georeader` / `source_from_georeader`: `_src/carrier.py`
  (`reject_georeader`, `require_source`).
- Lazy georeader readers a module opens (Carbon Mapper's `RasterioReader`)
  are handed out wrapped with `source_from_georeader`.

File formats (`geocloud.hdf`) and object storage come from geotoolz-cloud,
through each sensor's extra.

## A sensor subpackage

Copy the shape of `goes/` and `himawari/` (the most complete) or
`toy_sensor/` (the in-memory reference):

```
geoproducts/<sensor>/
├── __init__.py     # public surface + a docstring listing it
├── reader.py / l1b.py / l2.py   # Reader classes (ProductReader subclasses)
├── aws.py          # list / download from public buckets (via _src/s3.py)
├── constants.py    # BANDS table, flag codes, platform constants
├── data/           # packaged calibration tables (loaded via _src/constants.py)
├── recipes.py      # RGB recipes as data (via _src/presets.rgb_recipe)
├── presets.py      # geotoolz operators bound to band names ([operators])
└── _src/           # format parsing private to this sensor
```

Heavy format dependencies sit behind a per-sensor extra (`[goes]` → h5py)
and are imported lazily.

## The shared reader toolkit — use it, extend it

Before writing any plumbing in a sensor package, use `geoproducts/_src/`;
if something is missing, add it there so the next reader gets it:

| Module | Use it for |
|---|---|
| `net.py` | HTTP API clients: retries, `Retry-After`, backoff, atomic writes of response chunks (`retrying`, `retry_after_seconds`, `stream_to_file`, `bearer_headers_for`); plain URL / object downloads go through `geocloud.files` |
| `s3.py` | public buckets without credentials (`list_objects`, `download_object`, `object_url`), unsigned through `geocloud.files` |
| `files.py` | atomic writes (`atomic_path`, `atomic_write_text`, `write_private_json`) |
| `credentials.py` | `~/.geoproducts/auth_<provider>.json` + env vars (`auth_path`, `read_json_config`, `jwt_expiry`) |
| `query.py` | lon/lat bbox validation and UTC time windows for API queries |
| `hdf.py` | `PackedGridReader` (NetCDF-4 / HDF5 bands on one grid, local or remote); the CF decoders are `geocloud.hdf`'s, reached lazily as `hdf.attr(...)` so readers import on a base install |
| `geostationary.py` | the geostationary fixed grid (`FixedGrid`, `geos_crs`, `scan_angle_transform`, `on_earth`) |
| `stack.py` | several readers on one reference grid (`geoproducts.stack`) |
| `presets.py` | `Recipe`, `rgb_recipe`, `geotoolz_module` (the lazy geotoolz import), `parallax_correct` |
| `constants.py` | lazy loaders for packaged CSV / JSON tables; `band_values` (one number per band from a `BANDS` table) |
| `extras.py` | `require` / `missing_extra` / `install_hint` for optional dependencies |
| `base.py` | `ProductReader`, `resolve_fill_value` |
| `carrier.py` | `reject_georeader`, `require_source` (refuse georeader carriers at a public boundary) |

Object-store reads go through geotoolz-cloud (`geocloud.hdf.open_hdf5`,
`geocloud.files`, `geocloud.cog`), never a client of your own.

## Tests

- Run from this directory: `uv run pytest tests/goes -v`.
- Readers are tested on small synthetic files built in the tests, checked
  against an independent reference where one exists (rasterio / GDAL, or
  published values). georeader may be an oracle in tests (through
  `to_georeader` / `from_georeader`).
- Every reader is asserted to be a `geoarray.RasterSource`; inputs that
  accept a raster are tested to refuse a `GeoTensor`.
- `live` marks tests against real APIs needing credentials (always
  deselected); `integration` marks tests that download real public data
  (run manually via the "Extended Tests" workflow); `slow` for long builds.
- Coverage gate: 80 %.
- Each reader gets a docs page under `docs/products/` with a typed,
  shape-annotated example.

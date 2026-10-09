# Add a product reader

A new sensor or provider gets its own subpackage, `geoproducts.<sensor>`,
with the same small surface as the others. This page is the one home of
that layout and of the shared toolkit every reader builds on. What a
reader must return is the [ProductReader contract](concepts.md#the-productreader-contract).

## The subpackage layout

Copy the shape of `goes/` or `himawari/` (the most complete) or
`toy_sensor/` (the in-memory reference):

```text
geoproducts/<sensor>/
├── __init__.py          # public surface; its docstring lists every name
├── reader.py            # Reader(ProductReader) — or l1b.py / l2.py per product level
├── aws.py               # list / download from public buckets (via _src/s3.py) — or an API client
├── constants.py         # BANDS table, flag codes, platform constants
├── data/                # packaged calibration tables (loaded via _src/constants.py)
├── recipes.py           # RGB recipes as data (_src/presets.Recipe)
├── presets.py           # geotoolz operators bound to band names ([operators])
└── _src/                # format parsing private to this sensor
```

| Sensor | Reader modules | Finding files | Private `_src/` |
|---|---|---|---|
| `goes` | `l1b.py` (`Reader`, `QualityReader`), `l2.py` (`L2Reader`) | `aws.py` | `base.py` (ABI grid mapping), `metadata.py` |
| `himawari` | `reader.py` (`Reader`), `l2.py` (`L2Reader`) | `aws.py` | `hsd.py` (the HSD decoder) |
| `carbonmapper` | `image.py`, `rasters.py`, `sources_raster.py` | `api_queries.py`, `download.py` | — |
| `toy_sensor` | `reader.py` | — | — |

## Write the reader

Subclass `geoproducts.ProductReader` and implement:

- the metadata properties `_crs`, `_transform`, `_shape`, `_dtype`,
  `_bands`, `_fill_value` and `_track` (`"A"` for an affine grid, `"B"`
  for irregular geolocation);
- `_read_window(window)` — usually one line over
  `self._read_boundless(window, read)`, which pads windows past the grid
  with the fill so `read(rows, cols)` only sees in-grid slices;
- `_band_attrs()` (optional) to add per-band `attrs` (`units`,
  `wavelengths` in nm) to every read.

Calibrate with the file's own coefficients and document the units and
equation in the docstring. Register the subpackage in
`geoproducts/__init__.py`.

## Data, extras and presets

- **Calibration tables** go under `data/`, small, loaded through
  `_src.constants.load_csv` / `load_json`. The loaders are cached, so
  importing a sensor reads no file.
- **Heavy dependencies** go in the sensor's extra in
  `packages/geotoolz-products/pyproject.toml`, imported at use time with
  `_src.extras.require(module, feature, extra)`. The `ImportError` then
  names the extra. Anything that runs on the standard library (bucket
  helpers, HSD decoding) needs no extra.
- **Presets** bind generic geotoolz operators to band names, behind
  `[operators]`, importing geotoolz inside the function
  (`_src.presets.geotoolz_module`). An operator that is genuinely new
  belongs in a geotoolz family.

## The shared toolkit (`geoproducts._src`)

Check here before writing a helper: these are the pieces every reader so
far has needed, written once and tested on their own. They are private
to the package; if a new sensor needs something the next one will too,
add it here first.

| Module | Use it for | Used by |
|---|---|---|
| `_src.base` | `ProductReader`, `Track`, `_read_boundless`, `_band_attrs`, `resolve_fill_value` | every reader |
| `_src.hdf` | `PackedGridReader` — bands that are CF-packed 2-D variables of one HDF5 / NetCDF-4 file (integer masks, CF flag tables, chunked window reads), from a local path or any URI with ranged reads. The decoding (`PackedVariable`, `attr`, `scalar`, `time_attr`, `unpacked`, `grid_variables`, `open_hdf5`) is geotoolz-cloud's [`geocloud.hdf`](../cloud/how-to/hdf.md); call it as `hdf.attr(...)` so a reader still imports on a base install | `goes`, `himawari.l2` |
| `_src.geostationary` | `FixedGrid` — the affine `+proj=geos` grid from a CF grid mapping and scan angles; `geos_crs`, `scan_angle_transform`, `on_earth` | `goes`, `himawari` (MTG, SEVIRI next) |
| `_src.stack` | `geoproducts.stack` — any readers onto one reference grid | public |
| `_src.presets` | `Recipe`, `rgb_recipe`, `geotoolz_module` (the lazy geotoolz import), `parallax_correct` | `goes`, `himawari` recipes and presets |
| `_src.net` | `retrying` (one retry loop for raising *and* response-returning clients), `retry_after_seconds` (`Retry-After` or backoff, clamped), `stream_to_file` (atomic `.part` writes), `bearer_headers_for` (a token only ever sent to its own host) | `carbonmapper`, `_src.s3` |
| `_src.s3` | anonymous public buckets, unsigned through `geocloud.files`: `list_objects`, `download_object`, `object_url`, `s3_wait` (S3's spurious `NoSuchBucket`) | `goes.aws`, `himawari.aws` |
| `_src.files` | `atomic_path` / `atomic_write_text` / `write_private_json` — complete-or-absent writes; `private=True` is owner-only (`0600`) from the first byte | `himawari.aws`, `_src.credentials`, `_src.net` |
| `_src.credentials` | `auth_path(provider)` (`~/.geoproducts/auth_<provider>.json`), `read_json_config`, `write_private_json`, `jwt_expiry` | `carbonmapper` |
| `_src.query` | `validate_lonlat_bbox`, `as_utc`, `rfc3339_utc`, `time_interval` (STAC / OGC ranges) | `carbonmapper`, `goes.aws`, `himawari.aws` |
| `_src.constants` | `load_csv` / `load_json` for packaged tables | `toy_sensor`, `goes`, `himawari` |
| `_src.extras` | `require`, `missing_extra`, `install_hint` for optional dependencies | every extra-gated feature |

Object-store byte reads go through `geocloud.store` (the `[obstore]`
extra), never a client of your own. A provider client keeps only what is
specific to it: its URLs and payloads, which statuses it retries, and its
credential fields and renewal flow.

## Tests and docs

- Build small synthetic files in a fixture module under
  `packages/geotoolz-products/tests/<sensor>/` (like
  `tests/goes/_goes_abi.py`); no downloads in the fast tier.
- Check geolocation and calibrated values against an independent
  reference (GDAL / rasterio, the provider's tool, published values).
- Mark real downloads `integration` (public data) or `live` (needs
  credentials). The coverage gate is 80 %.
- Add a sensor page `docs/products/<sensor>.md` following the template of
  [GOES-R ABI](goes.md): what a file is → find and download → read and
  calibrate → L2 → one grid → recipes and presets → module layout. Add it
  to the sensor table on the [landing page](index.md) and to the nav.

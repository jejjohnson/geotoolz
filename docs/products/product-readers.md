# Adding a new product reader

Sensor and product integrations live under `geoproducts.<sensor>` and keep
the same small surface so each one can be audited in isolation.

```text
geoproducts/<sensor>/
  __init__.py      # Reader, BANDS, CONSTANTS, presets
  reader.py        # ProductReader subclass
  constants.py     # lazy calibration table accessors
  presets.py       # zero-argument geotoolz operators bound to the band names
  data/            # packaged calibration files
```

A reader should subclass `geoproducts.ProductReader`, implement the
metadata properties (`_crs`, `_transform`, `_shape`, `_dtype`, `_bands`,
`_fill_value`, `_track`), and provide `_read_window(window)` — usually a
one-liner over `self._read_boundless(window, read)`, which pads windows
past the grid with the fill so `read(rows, cols)` only ever sees in-grid
slices. Override `_band_attrs()` to add per-band `attrs` (`units`,
`wavelengths`, …) to every read. Track `"A"` means a clean affine grid;
track `"B"` is reserved for sensors with irregular geolocation.

Calibration data should be packaged below `data/`, kept small, and loaded via
`geoproducts._src.constants.load_csv()` or `load_json()`. These loaders are
cached, so importing a sensor module does not read calibration files and later
accesses reuse the parsed table.

Format-specific dependencies belong in the sensor's optional extra in
`packages/geotoolz-products/pyproject.toml`. Import them at use time with
`geoproducts._src.extras.require(module, feature, extra)`, which raises an
`ImportError` naming the extra (`pip install 'geotoolz-products[<sensor>]'`)
instead of surfacing a library-internal error.

`geoproducts` never depends on `geotoolz`: readers produce `GeoTensor`s that
the operators consume. `presets.py` bridges the two with zero-argument
wrappers over generic operators — for example `toy_sensor.NDVI()` returning
`geotoolz.indices.NDVI(red="red", nir="nir")` — importing geotoolz inside the
function, behind the `[operators]` extra. An operator that is genuinely new
belongs in a geotoolz family, not in the reader package.

## The shared toolkit (`geoproducts._src`)

Before writing a helper, check here: these are the pieces every reader so
far has needed, written once and tested on their own. They are private to
the package (readers import them; users don't).

| Module | Use it for | Used by |
|---|---|---|
| `_src.base` | `ProductReader`, `_read_boundless`, `_band_attrs`, `resolve_fill_value` | every reader |
| `_src.hdf` | `PackedGridReader` — bands that are CF-packed 2-D variables of one HDF5 / NetCDF-4 file (`_Unsigned`, `_FillValue`, `scale_factor` / `add_offset`, masks kept integer, CF flag tables, windowed chunk reads); `PackedVariable`, `attr`, `scalar`, `unpacked` | `goes` |
| `_src.geostationary` | `FixedGrid.from_cf(f, grid_mapping)` — the affine `+proj=geos` grid from a CF `geostationary` grid mapping and scan angles; `geos_crs`, `scan_angle_transform` | `goes` (MTG, Himawari next) |
| `_src.stack` | `geoproducts.stack` — any readers onto one reference grid | public |
| `_src.net` | `retrying` (one retry loop for raising *and* response-returning clients), `retry_after_seconds` (API rate limits: `Retry-After` or backoff, clamped), `stream_to_file` (atomic `.part` writes), `bearer_headers_for` (a token only ever sent to its own API host) | `carbonmapper`, `_src.s3` |
| `_src.s3` | Anonymous public buckets, unsigned through `geocloud.files`: `list_objects`, `download_object`, `object_url`, `s3_wait` (S3's spurious `NoSuchBucket`) | `goes.aws`, `himawari.aws` |
| `_src.files` | `atomic_path` / `atomic_write_text` / `write_private_json` — complete-or-absent writes; `private=True` is owner-only (`0600`) from the first byte | downloads, credentials |
| `_src.credentials` | `auth_path(provider)` (`~/.geoproducts/auth_<provider>.json`), `read_json_config`, `write_private_json`, `jwt_expiry` | `carbonmapper` |
| `_src.query` | `validate_lonlat_bbox`, `as_utc`, `rfc3339_utc`, `time_interval` (STAC / OGC ranges) | `carbonmapper`, `goes.aws` |
| `_src.extras` | `require(module, feature, extra)` for optional dependencies | every extra-gated feature |
| `_src.constants` | `load_csv` / `load_json` for packaged calibration tables | `toy_sensor`, `goes` |

A provider client keeps only what is specific to it: its URLs and payloads,
which statuses its API retries (Carbon Mapper retries 429 with its
`Retry-After`; anonymous S3 also retries a spurious `NoSuchBucket`), and its
credential fields and renewal flow.

## Reference implementation: `toy_sensor`

`geoproducts.toy_sensor` is the worked example of the contract. It is an
in-memory reader (no external file format) used to exercise the framework in
tests and to demonstrate the layout for real sensors.

`geoproducts.goes` is the first file-format reader built on the toolkit,
and a template for a mission with several product levels: `goes/_src/`
holds only what is ABI-specific (the `goes_imager_projection` grid
mapping, the file's global metadata) on top of `PackedGridReader` and
`FixedGrid`; thin public modules per level (`l1b`, `l2`) sit over it, with
the `aws` bucket helpers (file-name parsing over `_src.s3`), recipe data in
`recipes.py` and presets over `geotoolz` operators, all behind the `[goes]`
extra. The remaining
sensors (MTG, Himawari, TROPOMI, VIIRS, Sentinel-3, SEVIRI, MODIS) are
tracked under their per-sensor design issues.

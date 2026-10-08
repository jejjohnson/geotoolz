---
name: add-product-reader
description: Add a reader for a new Earth-observation product or sensor to geotoolz-products (geoproducts) — e.g. MTG FCI, SEVIRI, VIIRS, MODIS, TROPOMI, Sentinel-3. Use when asked to read, decode, support or ingest a new satellite product, file format or provider API.
---

# Add a product reader

Read `packages/geotoolz-products/AGENTS.md` (the reader contract, the sensor
subpackage layout, the shared toolkit) and `docs/products/product-readers.md`
first. `goes/` and `himawari/` are the most complete references;
`toy_sensor/` is the minimal one.

## 1. Research, then reuse

- Find the product's format spec and an independent reference reader to
  validate against (GDAL / rasterio, the provider's own tool, published
  values). Satpy may be **read** for reference; it is never a dependency.
- Map every piece of plumbing to the shared toolkit before writing any:
  HTTP → `_src/net.py`; public buckets → `_src/s3.py`; credentials →
  `_src/credentials.py`; atomic writes → `_src/files.py`; bbox / times →
  `_src/query.py`; NetCDF / HDF5 with CF packing → `_src/hdf.py`;
  geostationary fixed grid → `_src/geostationary.py`; several readers on
  one grid → `_src/stack.py`; RGB recipes / presets → `_src/presets.py`;
  packaged tables → `_src/constants.py`.
- Anything the new sensor needs that the next sensor will need too (a grid
  type, a decoder, a download pattern) goes into `_src/` **first**, with its
  own tests, then the sensor uses it.

## 2. Build the subpackage

```
geoproducts/<sensor>/
├── __init__.py   # public surface; docstring lists every name
├── reader.py     # Reader(ProductReader) — or l1b.py / l2.py per level
├── constants.py  # BANDS table, flag codes, platform constants
├── data/         # packaged calibration tables, if any
├── aws.py        # bucket listing / download, via _src/s3.py or _src/net.py
├── recipes.py    # RGB recipes as data (_src/presets.rgb_recipe)
├── presets.py    # geotoolz operators bound to band names ([operators])
└── _src/         # format parsing private to this sensor
```

- `Reader` subclasses `ProductReader`: implement `_read_window` (lazy —
  decode only the window asked for) and `_crs`, `_transform`, `_shape`,
  `_dtype`, `_bands`, `_fill_value`, `_track`.
- What a read returns is the `GeoTensor` carrier contract (root
  `AGENTS.md`):
  - bands on axis `-3`;
  - `fill_value_default` set to the product's nodata;
  - band names under `attrs["band_names"]`, which the base class sets from
    `_bands`;
  - per-band extras (units, `wavelengths` in nm) added through
    `_band_attrs`.

  That is what lets every geotoolz operator consume the read unchanged.
- Calibrate with the file's own coefficients; document units and the
  equation in the docstring.
- Heavy dependencies sit behind a per-sensor extra in
  `packages/geotoolz-products/pyproject.toml` (`[goes]` → h5py), imported
  lazily via `_src/extras.py`; anything that runs on the standard library
  (bucket helpers, HSD decoding) needs no extra.
- Register the subpackage in `geoproducts/__init__.py`.

## 3. Tests (`packages/geotoolz-products/tests/<sensor>/`)

- Build small synthetic files in a fixture module (`_<format>.py`, like
  `tests/goes/_goes_abi.py`) — no downloads in the fast tier.
- Check geolocation and calibrated values against the independent reference.
- Real downloads: `@pytest.mark.integration` (public data) or
  `@pytest.mark.live` (needs credentials).
- Coverage gate: 80 %.

## 4. Docs

- A page `docs/products/<sensor>.md` with a typed, shape-annotated example
  (`reader.load()  # (1, 5424, 5424) float32, reflectance`), added to the
  products nav in `mkdocs.yml`.
- Add the sensor to the list in `docs/products/index.md`, and its extra to
  the install lines and the extras table in
  `packages/geotoolz-products/README.md`.
- `make capabilities`, then the `pre-pr-check` skill.

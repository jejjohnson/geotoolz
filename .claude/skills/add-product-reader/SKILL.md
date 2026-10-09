---
name: add-product-reader
description: Add a reader for a new Earth-observation product or sensor to geotoolz-products (geoproducts) — e.g. MTG FCI, SEVIRI, VIIRS, MODIS, TROPOMI, Sentinel-3. Use when asked to read, decode, support or ingest a new satellite product, file format or provider API.
---

# Add a product reader

The subpackage layout, the `ProductReader` methods to implement and the
shared `_src/` toolkit table live in one place:
[`docs/products/product-readers.md`](../../../docs/products/product-readers.md).
Read it, plus `packages/geotoolz-products/AGENTS.md`, before writing code.
`goes/` and `himawari/` are the most complete references; `toy_sensor/` is
the minimal one.

## 1. Research, then reuse

- Find the product's format spec and an independent reference reader to
  validate against (GDAL / rasterio, the provider's own tool, published
  values). Satpy may be **read** for reference; it is never a dependency.
- Map every piece of plumbing to a row of the toolkit table before
  writing any. Object-store reads go through `geocloud.store`.
- Anything the new sensor needs that the next sensor will need too (a grid
  type, a decoder, a download pattern) goes into `_src/` **first**, with its
  own tests, then the sensor uses it. Add its row to the toolkit table.

## 2. Build the subpackage

Follow the layout on the docs page. What a read returns is the
`GeoTensor` carrier contract (root `AGENTS.md`):

- bands on axis `-3`;
- `fill_value_default` set to the product's nodata;
- band names under `attrs["band_names"]` (the base class sets them from
  `_bands`);
- per-band extras (units, `wavelengths` in nm) through `_band_attrs`.

That is what lets every geotoolz operator consume the read unchanged.
Calibrate with the file's own coefficients; document units and the
equation in the docstring. Put heavy dependencies behind a per-sensor
extra, imported lazily through `_src/extras.py`. Register the subpackage
in `geoproducts/__init__.py`.

## 3. Tests (`packages/geotoolz-products/tests/<sensor>/`)

- Build small synthetic files in a fixture module (`_<format>.py`, like
  `tests/goes/_goes_abi.py`) — no downloads in the fast tier.
- Check geolocation and calibrated values against the independent
  reference.
- Real downloads: `@pytest.mark.integration` (public data) or
  `@pytest.mark.live` (needs credentials).
- Coverage gate: 80 %.

## 4. Docs and checklist

- [ ] A page `docs/products/<sensor>.md` on the sensor template (see the
      docs page's "Tests and docs"), typed and shape-annotated, in the
      products nav of `mkdocs.yml`.
- [ ] The sensor in the sensor table of `docs/products/index.md`; its
      extra in that page's extras table and in
      `packages/geotoolz-products/README.md`.
- [ ] `docs/products/api.md` has a section for the new namespace.
- [ ] `make capabilities`, then the `pre-pr-check` skill.

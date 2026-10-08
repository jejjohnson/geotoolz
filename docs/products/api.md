# API reference

Product reader framework. See [Adding a new product reader](product-readers.md) for the
namespace contract (`Reader`, `BANDS`, `CONSTANTS`, `presets`) and the package-data layout.

- **Base class:** `ProductReader` — extends `georeader.GeoData` with the sensor surface (`_track`,
  `_bands`, lazy `_read_window`, …)
- **Reference reader:** `geoproducts.toy_sensor` — in-memory worked example exercising the
  full contract end-to-end
- **Stacking:** `geoproducts.stack` — any readers' bands on one reference grid
- **GOES-R ABI:** `geoproducts.goes` — L1b `Reader`, L2 `L2Reader` and `QualityReader` on
  the `+proj=geos` fixed grid, `goes.aws` bucket helpers, `recipes` and `presets`
  (see [GOES-R ABI](goes.md))
- **Himawari AHI:** `geoproducts.himawari` — HSD `Reader`, L2 `L2Reader`, `himawari.aws`
  bucket helpers, `recipes` and `presets` (see [Himawari AHI](himawari.md))
- Further per-sensor readers (MTG, TROPOMI, VIIRS, Sentinel-3, SEVIRI, MODIS) land
  alongside their design issues.

::: geoproducts

## GOES-R ABI

::: geoproducts.goes.l1b

::: geoproducts.goes.l2

::: geoproducts.goes.recipes

::: geoproducts.goes.aws

::: geoproducts.goes.presets

## Himawari AHI

::: geoproducts.himawari.reader

::: geoproducts.himawari.l2

::: geoproducts.himawari.recipes

::: geoproducts.himawari.aws

::: geoproducts.himawari.presets

# API reference

Product reader framework. See [Adding a new product reader](product-readers.md) for the
namespace contract (`Reader`, `BANDS`, `CONSTANTS`, `presets`) and the package-data layout.

- **Base class:** `ProductReader` — extends `georeader.GeoData` with the sensor surface (`_track`,
  `_bands`, lazy `_read_window`, …)
- **Reference reader:** `geoproducts.toy_sensor` — in-memory worked example exercising the
  full contract end-to-end
- **GOES-R ABI:** `geoproducts.goes` — L1b `Reader` / `QualityReader` on the `+proj=geos`
  fixed grid, `goes.aws` bucket helpers, `presets` (see [GOES-R ABI](goes.md))
- Further per-sensor readers (MTG, Himawari, TROPOMI, VIIRS, Sentinel-3, SEVIRI, MODIS) land
  alongside their design issues.

::: geoproducts

## GOES-R ABI

::: geoproducts.goes.reader

::: geoproducts.goes.aws

::: geoproducts.goes.presets

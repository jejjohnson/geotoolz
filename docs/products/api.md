# API reference

Product reader framework. See [Adding a new product reader](product-readers.md) for the
namespace contract (`Reader`, `BANDS`, `CONSTANTS`, `presets`) and the package-data layout.

- **Base class:** `ProductReader` — extends `georeader.GeoData` with the sensor surface (`_track`,
  `_bands`, lazy `_read_window`, …)
- **Reference reader:** `geoproducts.toy_sensor` — in-memory worked example exercising the
  full contract end-to-end
- Per-sensor implementations (MODIS, VIIRS, GOES, MTG, TROPOMI, S3, SEVIRI, Himawari) land alongside
  their design issues as the real format readers come online.

::: geoproducts

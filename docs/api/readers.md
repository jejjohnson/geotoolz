# Readers

Sensor reader framework. See the concept page [Adding a new sensor reader](../readers.md) for the
namespace contract (`Reader`, `BANDS`, `CONSTANTS`, `ops`, `presets`) and the package-data layout.

- **Base class:** `SensorReader` — extends `georeader.GeoData` with the sensor surface (`_track`,
  `_bands`, lazy `_read_window`, …)
- **Reference reader:** `geotoolz.readers.toy_sensor` — in-memory worked example exercising the
  full contract end-to-end
- Per-sensor implementations (MODIS, VIIRS, GOES, MTG, TROPOMI, S3, SEVIRI, Himawari) land alongside
  their design issues as the real format readers come online.

::: geotoolz.readers

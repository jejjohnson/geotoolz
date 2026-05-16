# QA

::: geotoolz.qa

## Sensor registry

| Preset | QA source | Default mask targets |
| --- | --- | --- |
| `S2QA60` | Sentinel-2 L1C `QA60` bitmask | cloud, cirrus |
| `S2SCL` | Sentinel-2 L2A `SCL` classes | mask everything except vegetation, soil, water |
| `LandsatQA_PIXEL` | Landsat Collection-2 `QA_PIXEL` bitmask | cloud, cloud shadow, cirrus |
| `MODISStateQA` | MODIS State QA bitmask | cloud, cloud shadow |

All QA mask operators return boolean `GeoTensor` masks with the original CRS and transform preserved and `fill_value_default=False`. The convention is `True` means "mask this pixel out".

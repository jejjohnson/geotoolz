# QA

`geotoolz.qa` provides **sensor-specific QA-bit decoders** layered on top of the generic primitives in [`geotoolz.cloud`](../cloud/). The two modules share one decoder implementation:

- `geotoolz.cloud._src.array.mask_from_qa_bits` — single-bit-flag decoding (OR of bits).
- `geotoolz.cloud._src.array.mask_from_scl` — categorical class membership.
- `geotoolz.qa._src.array.mask_from_bit_field` — contiguous multi-bit field decoding (needed for MODIS).

Pick `geotoolz.cloud.MaskFromQABits` / `MaskFromSCL` when you have an explicit list of bits / classes. Pick `geotoolz.qa.LandsatQA_PIXEL` / `S2QA60` / `S2SCL` / `MODISStateQA` when you want the published-spec defaults.

::: geotoolz.qa

## Sensor registry

| Preset | QA source | Default mask targets | Reference |
| --- | --- | --- | --- |
| `S2QA60` | Sentinel-2 L1C `QA60` bitmask (bit 10 cloud, 11 cirrus) | cloud + cirrus | ESA S2 L1C product spec |
| `S2SCL` | Sentinel-2 L2A `SCL` classes | mask everything except vegetation (4), soil (5), water (6) | Sen2Cor product spec |
| `LandsatQA_PIXEL` (sensor=`l89`) | Landsat 8/9 C2 `QA_PIXEL` | cloud (bit 3), cloud shadow (bit 4), cirrus (bit 2) | USGS LSDS-1619 |
| `LandsatQA_PIXEL` (sensor=`l7`) | Landsat 4-7 C2 `QA_PIXEL` (no cirrus bit) | cloud (bit 3), cloud shadow (bit 4) | USGS LSDS-1618 |
| `MODISStateQA` | MODIS `state_1km` / `state_500m` | cloud (bits [0,1] field, values 1,2) + cloud shadow (bit 2) | MOD09 User's Guide, Table 12 |

All QA mask operators return boolean `GeoTensor` masks with the original CRS and transform preserved and `fill_value_default=False`. The convention is **`True` means "mask this pixel out"**.

### MODIS bit-field semantics

MODIS `state_1km` packs categorical fields into multi-bit slots:

- Bits `[0, 1]` (2 bits): cloud state — `0`=clear, `1`=cloudy, `2`=mixed, `3`=not-set.
- Bit `2`: cloud shadow.
- Bits `[8, 9]` (2 bits): cirrus level — `0`=none, `1`=small, `2`=average, `3`=high.

OR-ing the bits individually (the standard Landsat semantics) would flag value `3` (not-set) as cloudy, which is wrong. `MODISStateQA` therefore decodes these as *field values* via `mask_from_bit_field`.

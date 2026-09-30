# QA

`geotoolz.qa` is the home of QA / cloud-mask **extraction**: generic decoders, sensor presets, and the decoding primitives they share:

- `mask_from_qa_bits` — single-bit-flag decoding (OR of bits).
- `mask_from_scl` — categorical class membership (Sentinel-2 SCL).
- `mask_from_bit_field` — contiguous multi-bit field decoding (needed for MODIS).

One vocabulary throughout: `qa_band` selects the QA band from a stack (`None`, the default for the generic decoders, means the carrier *is* the QA band), `bits` lists independent flag bits, `values` lists categorical class / field values, and the sensor presets take `targets` — registry names to mask out.

Pick the generic decoders (`MaskClouds`, `MaskCloudShadow`, `MaskCirrus`, `MaskSnow`, `MaskWater`, `MaskNoData` — one implementation under several semantic names) when you have an explicit list of `bits` / `values`; `invert=True` turns the list into a keep-list. `MaskInvalid` marks nodata pixels (the carrier fill, `NaN` fills included, or non-finite values); `MaskSaturated` flags pixels at or above a saturation value (`reduce_bands=False` for per-band flags). Pick `LandsatQA_PIXEL` / `S2QA60` / `S2SCL` / `MODISStateQA` when you want the published-spec defaults. To *apply* a mask, use [`geotoolz.mask.ApplyMask`](mask.md).

::: geotoolz.qa

## Sensor registry

| Preset | QA source | Default mask targets | Reference |
| --- | --- | --- | --- |
| `S2QA60` | Sentinel-2 L1C `QA60` bitmask (bit 10 cloud, 11 cirrus) | cloud + cirrus | ESA S2 L1C product spec |
| `S2SCL` | Sentinel-2 L2A `SCL` classes (registry derived from `SCL_TARGETS`) | every target except vegetation (4), soil (5), water (6) | Sen2Cor product spec |
| `LandsatQA_PIXEL` (sensor=`l89`) | Landsat 8/9 C2 `QA_PIXEL` | cloud (bit 3), cloud shadow (bit 4), cirrus (bit 2) | USGS LSDS-1619 |
| `LandsatQA_PIXEL` (sensor=`l7`) | Landsat 4-7 C2 `QA_PIXEL` (no cirrus bit) | cloud (bit 3), cloud shadow (bit 4) | USGS LSDS-1618 |
| `MODISStateQA` | MODIS `state_1km` / `state_500m` | cloud (bits [0,1] field, values 1,2) + cloud shadow (bit 2) | MOD09 User's Guide, Table 12 |

All QA mask operators return boolean masks matching the input carrier: GeoTensor in (CRS/transform preserved, `fill_value_default=False`), plain `np.ndarray` in, plain boolean array out. The convention is **`True` means "mask this pixel out"** (see [Mask polarity](../concepts.md#mask-polarity)).

### MODIS bit-field semantics

MODIS `state_1km` packs categorical fields into multi-bit slots:

- Bits `[0, 1]` (2 bits): cloud state — `0`=clear, `1`=cloudy, `2`=mixed, `3`=not-set.
- Bit `2`: cloud shadow.
- Bits `[8, 9]` (2 bits): cirrus level — `0`=none, `1`=small, `2`=average, `3`=high.

OR-ing the bits individually (the standard Landsat semantics) would flag value `3` (not-set) as cloudy, which is wrong. `MODISStateQA` therefore decodes these as *field values* via `mask_from_bit_field`.

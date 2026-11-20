# QA

`geoops.qa` is the home of QA / cloud-mask **extraction**: generic decoders, sensor presets, and the decoding primitives they share:

- `mask_from_qa_bits` — single-bit-flag decoding (OR of bits).
- `mask_from_scl` — categorical class membership (Sentinel-2 SCL).
- `mask_from_bit_field` — contiguous multi-bit field decoding (needed for MODIS).
- `flagged_band_count` and `mask_from_band_count` — EMIT's packed per-band mask: the bands flagged per pixel, and the dilated saturation mask above each detector column's baseline.

One vocabulary throughout: `qa_band` selects the QA band from a stack (`None`, the default for the generic decoders, means the carrier *is* the QA band), `bits` lists independent flag bits, `values` lists categorical class / field values, and the sensor presets take `targets` — registry names to mask out.

Pick the generic decoders (`MaskClouds`, `MaskCloudShadow`, `MaskCirrus`, `MaskSnow`, `MaskWater`, `MaskNoData` — one implementation under several semantic names) when you have an explicit list of `bits` / `values`; `invert=True` turns the list into a keep-list. `MaskInvalid` marks nodata pixels (the carrier fill, `NaN` fills included, or non-finite values); `MaskSaturated` flags pixels at or above a saturation value (`reduce_bands=False` for per-band flags). Pick `LandsatQA_PIXEL` / `S2QA60` / `S2SCL` / `MODISStateQA` / `EMITL2AMask` when you want the published-spec defaults, `EMITBandMask` for the saturation mask of an EMIT scene, and `DecodeBitmask` to unpack a QA bitmask into named boolean layers. `SCL` holds the Sentinel-2 SCL class IDs and `SCL_CLOUDS`, `SCL_CLOUDS_AND_INVALID`, `SCL_INVALID`, `SCL_LAND`, `SCL_WATER` the common class sets; `SENSOR_QA_REGISTRY` maps each sensor preset's `targets` names to bits / values. `S2Cloudless`, `CloudSEN12` and `OmniCloudMask` are placeholders reserved for optional ML-based detectors: they configure and serialise today but raise `ImportError` when called. To *apply* a mask, use [`geoops.mask.ApplyMask`](mask.md).

::: geoops.qa

## Sensor registry

| Preset | QA source | Default mask targets | Reference |
| --- | --- | --- | --- |
| `S2QA60` | Sentinel-2 L1C `QA60` bitmask (bit 10 cloud, 11 cirrus) | cloud + cirrus | ESA S2 L1C product spec |
| `S2SCL` | Sentinel-2 L2A `SCL` classes (registry derived from `SCL_TARGETS`) | every target except vegetation (4), soil (5), water (6) | Sen2Cor product spec |
| `LandsatQA_PIXEL` (sensor=`l89`) | Landsat 8/9 C2 `QA_PIXEL` | cloud (bit 3), cloud shadow (bit 4), cirrus (bit 2) | USGS LSDS-1619 |
| `LandsatQA_PIXEL` (sensor=`l7`) | Landsat 4-7 C2 `QA_PIXEL` (no cirrus bit) | cloud (bit 3), cloud shadow (bit 4) | USGS LSDS-1618 |
| `MODISStateQA` | MODIS `state_1km` / `state_500m` | cloud (bits [0,1] field, values 1,2) + cloud shadow (bit 2) | MOD09 User's Guide, Table 12 |
| `EMITL2AMask` | EMIT L2A `mask`, one flag per band (0 cloud, 1 cirrus, 2 water, 3 spacecraft, 4 dilated cloud, 7 aggregate; 5 and 6 are AOD and water vapour) | cloud + cirrus + water | EMIT L2A User Guide V1.0 (JPL D-107863), Table 3-4 |
| `EMITBandMask` | EMIT `band_mask`, one bit per spectral band in 36 bytes | flagged bands above the column baseline, dilated by 10 pixels | EMIT L2A User Guide V1.0, Table 3-3 |

All QA mask operators return boolean masks matching the input carrier: GeoArray in (CRS/transform preserved, `fill_value=False`), plain `np.ndarray` in, plain boolean array out. The convention is **`True` means "mask this pixel out"** (see [Mask polarity](../operators-concepts.md#mask-polarity)).

### MODIS bit-field semantics

MODIS `state_1km` packs categorical fields into multi-bit slots:

- Bits `[0, 1]` (2 bits): cloud state — `0`=clear, `1`=cloudy, `2`=mixed, `3`=not-set.
- Bit `2`: cloud shadow.
- Bits `[8, 9]` (2 bits): cirrus level — `0`=none, `1`=small, `2`=average, `3`=high.

OR-ing the bits individually (the standard Landsat semantics) would flag value `3` (not-set) as cloudy, which is wrong. `MODISStateQA` therefore decodes these as *field values* via `mask_from_bit_field`.

### EMIT saturation

EMIT is a push-broom instrument, so each column of a swath is one cross-track detector element. A dead or hot element flags the same bands in every row of its column, so `EMITBandMask` takes each column's smallest flagged-band count as its static baseline and flags what a pixel has above it, which in practice is saturation (flares, glint, bright roofs). It then grows the mask by `dilation` pixels in city-block distance. Compute it in swath geometry, before orthorectifying, while the columns still follow the detector.

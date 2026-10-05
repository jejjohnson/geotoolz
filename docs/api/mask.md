# Mask

Geometry-based masks, morphological mask ops, and boolean mask algebra. Convention: `True` means
"mask this pixel out" (consistent with `geotoolz.qa`; see [Mask polarity](../concepts.md#mask-polarity)).

- **Geometry / DEM masks:** `PolygonMask`, `BBoxMask`, `DistanceMask`, `LandMask`, `OceanMask`,
  `CountryMask`, `AltitudeMask`, `SlopeMask` — `keep="inside"` (default) keeps the region and masks
  everything else; `keep="outside"` drops the region. `LandMask` / `OceanMask` / `CountryMask`
  load Natural Earth lazily on their first call (never in the constructor, so building or
  hydrating a pipeline does no network I/O); the GeoDataFrames are then cached for the process
  lifetime — `mask.clear_natural_earth_cache()` releases them.
  `AltitudeMask` / `SlopeMask` take the DEM as a second positional carrier on the scene's grid:
  `SlopeMask(max_slope_deg=10.0)(scene, dem)`.
- **Morphology:** `DilateMask`, `ErodeMask`, `OpenMask`, `CloseMask`, `BufferMask`,
  `RemoveSmallObjects`, `RemoveSmallHoles`, `CleanMask`
- **Algebra:** `CombineMasks(op="or" | "and" | "xor")(m1, m2, ...)` (masks on one pixel grid), `InvertMask`
- **Apply:** `ApplyMask` (fill values where mask is `True`): `ApplyMask()(gt, mask)` with a mask
  carrier, or `ApplyMask(mask=BBoxMask(...))(gt)` with a mask-producing operator
- **Tier-A primitives:** `apply_mask`, `dilate_mask`, `erode_mask`, `open_mask`, `close_mask`,
  `buffer_mask`, `clean_mask`, `remove_small_objects`, `remove_small_holes`, `combine_masks`,
  `invert_mask`, `distance_mask`, `altitude_mask`, `slope_mask`, `slope_degrees`

::: geotoolz.mask

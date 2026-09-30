# Mask

Geometry-based masks, morphological mask ops, and boolean mask algebra. Convention: `True` means
"mask this pixel out" (consistent with `geotoolz.qa`; see [Mask polarity](../concepts.md#mask-polarity)).

- **Geometry / DEM masks:** `PolygonMask`, `BBoxMask`, `DistanceMask`, `LandMask`, `OceanMask`,
  `CountryMask`, `AltitudeMask`, `SlopeMask` — `keep="inside"` (default) keeps the region and masks
  everything else; `keep="outside"` drops the region. `LandMask` / `OceanMask` / `CountryMask`
  load Natural Earth lazily on their first call (never in the constructor, so building or
  hydrating a pipeline does no network I/O); the GeoDataFrames are then cached for the process
  lifetime — `mask.clear_natural_earth_cache()` releases them.
- **Morphology:** `DilateMask`, `ErodeMask`, `OpenMask`, `CloseMask`, `BufferMask`,
  `RemoveSmallObjects`, `RemoveSmallHoles`, `CleanMask`
- **Algebra:** `CombineMasks(op="or" | "and" | "xor")` (equally shaped masks), `InvertMask`
- **Apply:** `ApplyMask` (fill values where mask is `True`)

::: geotoolz.mask

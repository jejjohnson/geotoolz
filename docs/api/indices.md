# Indices

Spectral indices for vegetation, water, fire, urban, and mineral mapping. Each operator accepts band
identifiers as either integer axis indices or string band names (resolved against
`attrs` under `band_names`, then `descriptions`, then `bands` — see
[Band names](../concepts.md#band-names)).

- **Vegetation:** `NDVI`, `EVI`, `EVI2`, `SAVI`, `GCI`, `kNDVI`, `ARVI`
- **Water:** `NDWI`, `MNDWI`, `NDMI`
- **Fire / burn:** `NBR`, `NBR2`, `dNBR`
- **Snow / urban:** `NDSI`, `NDBI`, `BSI`, `CIRI`
- **Burned area (S2):** `BAIS2`
- **Minerals:** `ClayMinerals`, `IronOxide`
- **Custom:** `NormalizedDifference`, `AppendIndex`

::: geotoolz.indices

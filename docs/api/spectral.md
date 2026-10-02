# Spectral

Band-space operators — selection, math, spectral response, continuum removal — for hyperspectral
and multispectral cubes.

- **Band selection:** `SelectBands`, `ReorderBands`, `SplitBands`, `StackBands` — by index or
  name (`StackBands` requires every input on the same grid with the same `fill_value_default`)
- **Band math:** `BandMath` (restricted arithmetic expression over named bands), `BandRatio`
- **Spectral resampling:** `SpectralBinning`, `SpectralSmoothing`
- **Continuum removal:** `ContinuumRemoval`
- **Tier-A primitives:** `select_bands`, `reorder_bands`, `band_ratio`, `evaluate_band_math`,
  `spectral_binning`, `spectral_smoothing`, `continuum_removal`

The normalized difference is [`indices.NormalizedDifference`](indices.md) and the Gaussian
spectral-response convolution is [`radiometry.ApplySRF`](radiometry.md); the former
`spectral.NormalizedDifference`, `spectral.ApplySRF` and `spectral.GaussianSRF` duplicates were
removed.

::: geotoolz.spectral

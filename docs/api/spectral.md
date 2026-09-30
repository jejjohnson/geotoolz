# Spectral

Band-space operators — selection, math, spectral response, continuum removal — for hyperspectral
and multispectral cubes.

- Band selection / reordering / stacking / splitting by index or name (`StackBands` requires every
  input on the same grid with the same `fill_value_default`)
- Per-band math (linear combinations, ratios)
- Spectral binning and smoothing
- Continuum removal

The normalized difference is [`indices.NormalizedDifference`](indices.md) and the Gaussian
spectral-response convolution is [`radiometry.ApplySRF`](radiometry.md); the former
`spectral.NormalizedDifference`, `spectral.ApplySRF` and `spectral.GaussianSRF` duplicates were
removed.

::: geotoolz.spectral

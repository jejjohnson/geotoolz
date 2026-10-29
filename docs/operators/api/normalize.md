# Normalize

Per-band scaling and contrast operators for model inputs and display.

- **Per-band scalers:** `StandardScaler` (z-score), `RobustScaler` (median / IQR),
  `MinMaxScaler`, `ZeroOne`
- **Fixed statistics:** `Normalize` (z-score with given per-band mean / std), `PerBandStats`
  (compute and cache NaN-aware per-band statistics)
- **Non-linear scaling:** `LogScale`, `AsinhScale`, `PowerScale`
- **Contrast / histogram:** `HistogramStretch`, `HistogramMatch`, `CLAHE`
- **Tier-A primitives:** `standard_scale`, `robust_scale`, `minmax_scale`, `log_scale`,
  `asinh_scale`, `power_scale`, `histogram_match`, `clahe`, `per_band_stats`
- **Per-band statistics on every backend:** `band_stats`, `band_quantiles` (arrays, band axis by position) and `carrier_band_stats` (a carrier, band dim by name), mask-aware on NumPy, JAX (under `jax.jit` too) and dask

See the [Normalization guide](../normalization.md) for when to use which.

::: geoops.normalize

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

See the [Normalization guide](../normalization.md) for when to use which.

::: geotoolz.normalize

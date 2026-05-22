# Matched Filter

Pure-NumPy matched-filter family for hyperspectral retrieval (CH₄ / CO₂ / arbitrary trace gases).
Each piece — background mean, covariance, target spectrum, scoring — is a separate Operator so the
algebra is composable.

- **Core scoring:** `MatchedFilter`, `MatchedFilterPixel`, `MatchedFilterSNR`
- **Background statistics:**
  - `EstimateMean`, `EstimateCovEmpirical`, `EstimateCovLowRank`, `EstimateCovShrunk`
  - `StreamingBackground` — Welford accumulator + shrunk covariance across many cubes
  - `AdaptiveWindowBackground` — sliding-window local background
  - `GMMClusterBackground` / `ApplyClusterMF` — cluster-conditional background
- **Target construction:** `LinearTargetFromObs`, `NonlinearTargetFromObs`
- **Composed:** `ColumnEnhancement` — mean → cov → target → MF in one operator
- **Post-processing:** `DetectionThreshold`, `ValidateMFInputs`
- **Array primitives** (no GeoTensor): `apply_image`, `apply_pixel`, `matched_filter_snr`,
  `estimate_cov_empirical`, `estimate_cov_shrunk`, `estimate_cov_lowrank`, `shrink_covariance`,
  `detection_threshold`

The covariance shrinkage uses an analytical Ledoit-Wolf approximation that only needs the empirical
covariance and the sample count (`shrink_covariance(method="ledoit_wolf", ...)`); see the docstring
for the formula and properties.

::: geotoolz.matched_filter

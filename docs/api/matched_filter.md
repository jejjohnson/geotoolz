# Matched Filter

Pure-NumPy matched-filter family for hyperspectral retrieval (CH₄ / CO₂ / arbitrary trace gases).
Each piece — background mean, covariance, target spectrum, scoring — is a separate Operator so the
algebra is composable.

- **Core scoring:** `MatchedFilter`, `MatchedFilterPixel`, `MatchedFilterSNR`
- **Background statistics:**
  - `EstimateMean`, `EstimateCovEmpirical`, `EstimateCovLowRank`, `EstimateCovShrunk`
  - `StreamingBackground` — Welford accumulator + shrunk covariance across many cubes
  - `AdaptiveWindowBackground` / `ApplyAdaptiveMF` — sliding-window local mean + diagonal variance,
    scored per pixel with `α = tᵀΛ⁻¹(x − μ) / (tᵀΛ⁻¹t)`, `Λ = diag(σ² + ridge)`
  - `GMMClusterBackground` / `ApplyClusterMF` — cluster-conditional background
- **Target construction:** `LinearTargetFromObs`, `NonlinearTargetFromObs`
- **Composed:** `ColumnEnhancement` — mean → cov → target → MF in one operator
- **Result containers:** `NumpyLinearOperator` (dense covariance with `solve`), `AdaptiveBackground`,
  `ClusterBackground`, `StreamingBackgroundResult`, `WelfordAccumulator`
- **Post-processing:** `DetectionThreshold`, `ValidateMFInputs`
- **Array primitives** (no GeoTensor): `apply_image`, `apply_pixel`, `apply_adaptive_mf`, `matched_filter_snr`,
  `estimate_mean`, `estimate_cov_empirical`, `estimate_cov_shrunk`, `estimate_cov_lowrank`,
  `shrink_covariance`, `detection_threshold`, `validate_mf_inputs`

Covariance shrinkage (`"ledoit_wolf"`, the default, and `"oas"`) reproduces
`sklearn.covariance.ledoit_wolf` / `sklearn.covariance.oas`: the shrinkage intensity is identical on
the same centred samples, and because this package uses the `1/(n − 1)` sample covariance the shrunk
matrix is exactly `n/(n − 1)` times sklearn's (matched-filter scores are invariant to that scale).
Ledoit-Wolf needs the sample fourth moment `m₄ = mean‖xₖ − x̄‖⁴`: `estimate_cov_shrunk` computes it
from the pixels, `StreamingBackground` streams it through `WelfordAccumulator.m4`, and a direct
`shrink_covariance(method="ledoit_wolf", ...)` call must pass `fourth_moment=`.

::: geotoolz.matched_filter

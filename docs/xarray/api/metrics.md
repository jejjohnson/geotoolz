# Metrics

The `xr_toolz.metrics` package groups evaluation metrics by *scientific
diagnostic family*. Each submodule pairs Layer-0 functions (xarray-aware
pure functions) with Layer-1 `Operator` wrappers that compose into
`Sequential` pipelines and `Graph` networks.

For the design rationale, see
[`docs/design/validation.md`](../design/validation.md) and the
[validation API map](../design/api/validation.md).

## Taxonomy

| Submodule | Diagnostic family | Status |
|---|---|---|
| [`xr_toolz.metrics.pixel`](#pixel) | Pointwise scalar errors | shipped |
| [`xr_toolz.metrics.spectral`](#spectral) | Power-spectrum scores and resolved-scale | shipped |
| `xr_toolz.metrics.forecast` | Lead-time skill diagnostics | stub (V1) |
| `xr_toolz.metrics.multiscale` | Region-conditioned and band-limited skill | stub (V1) |
| `xr_toolz.metrics.structural` | Structural / geometric similarity (SSIM, …) | stub (V2) |
| `xr_toolz.metrics.probabilistic` | Ensemble calibration | stub (V2) |
| `xr_toolz.metrics.distributional` | Distributional distance (CRPS, Wasserstein) | stub (V2) |
| `xr_toolz.metrics.masked` | Masked / coverage-aware wrappers | stub (V2) |
| `xr_toolz.metrics.lagrangian` | Trajectory and transport metrics | stub (V3) |
| `xr_toolz.metrics.physical` | Physical-balance and conservation residuals | stub (V4) |
| `xr_toolz.metrics.object` | Event/object verification (POD, FAR, IoU, …) | stub (V5) |

Stub submodules are importable today and export nothing — they are populated by their respective view epics.

## Ergonomic re-exports

The Layer-1 `Operator` wrappers from `pixel` and `spectral` are
re-exported flat from the package root for ergonomic access:

```python
from xr_toolz.metrics import RMSE, MSE, MAE, NRMSE, Bias, Correlation, R2Score, PSDScore
```

Equivalent to:

```python
from xr_toolz.metrics.pixel import RMSE, MSE, MAE, NRMSE, Bias, Correlation, R2Score
from xr_toolz.metrics.spectral import PSDScore
```

A flat `xr_toolz.metrics.operators` module provides the same operator
surface for callers that prefer a dedicated module path.

## Pixel

::: xr_toolz.metrics.pixel

## Spectral

::: xr_toolz.metrics.spectral

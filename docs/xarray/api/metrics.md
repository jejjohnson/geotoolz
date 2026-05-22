# Metrics

The `xrtoolz.metrics` package groups evaluation metrics by *scientific
diagnostic family*. Each submodule pairs Layer-0 functions (xarray-aware
pure functions) with Layer-1 `Operator` wrappers that compose into
`Sequential` pipelines and `Graph` networks.

For the design rationale, see
[`docs/design/validation.md`](../design/validation.md) and the
[validation API map](../design/api/validation.md).

## Taxonomy

| Submodule | Diagnostic family | Status |
|---|---|---|
| [`xrtoolz.metrics.pixel`](#pixel) | Pointwise scalar errors | shipped |
| [`xrtoolz.metrics.spectral`](#spectral) | Power-spectrum scores and resolved-scale | shipped |
| `xrtoolz.metrics.forecast` | Lead-time skill diagnostics | stub (V1) |
| `xrtoolz.metrics.multiscale` | Region-conditioned and band-limited skill | stub (V1) |
| `xrtoolz.metrics.structural` | Structural / geometric similarity (SSIM, …) | stub (V2) |
| `xrtoolz.metrics.probabilistic` | Ensemble calibration | stub (V2) |
| `xrtoolz.metrics.distributional` | Distributional distance (CRPS, Wasserstein) | stub (V2) |
| `xrtoolz.metrics.masked` | Masked / coverage-aware wrappers | stub (V2) |
| `xrtoolz.metrics.lagrangian` | Trajectory and transport metrics | stub (V3) |
| `xrtoolz.metrics.physical` | Physical-balance and conservation residuals | stub (V4) |
| `xrtoolz.metrics.object` | Event/object verification (POD, FAR, IoU, …) | stub (V5) |

Stub submodules are importable today and export nothing — they are populated by their respective view epics.

## Ergonomic re-exports

The Layer-1 `Operator` wrappers from `pixel` and `spectral` are
re-exported flat from the package root for ergonomic access:

```python
from xrtoolz.metrics import RMSE, MSE, MAE, NRMSE, Bias, Correlation, R2Score, PSDScore
```

Equivalent to:

```python
from xrtoolz.metrics.pixel import RMSE, MSE, MAE, NRMSE, Bias, Correlation, R2Score
from xrtoolz.metrics.spectral import PSDScore
```

A flat `xrtoolz.metrics.operators` module provides the same operator
surface for callers that prefer a dedicated module path.

## Pixel

::: xrtoolz.metrics.pixel

## Spectral

::: xrtoolz.metrics.spectral

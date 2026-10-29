# Metrics

`xrops.metrics` groups evaluation metrics by *scientific diagnostic
family*. Each family pairs Layer-0 xarray functions with Layer-1 `Operator`
wrappers that compose into `Sequential` pipelines and `Graph` networks.

For the design rationale, see the [validation design
doc](../design/validation.md) and the [validation API
map](../design/api/validation.md).

## Diagnostic families

| Submodule | Family | Status |
|---|---|---|
| [`metrics.pixel`](#pixel-pointwise-errors) | Pointwise scalar errors | shipped |
| [`metrics.spectral`](#spectral-power-spectrum-skill) | Power-spectrum scores, resolved scale | shipped |
| [`metrics.multiscale`](#multiscale-region-band-conditioned) | Region- and band-conditioned skill | shipped |
| [`metrics.physical`](#physical-balance-residuals) | Physical-balance / conservation residuals | shipped |
| [`metrics.masked`](#masked-coverage-aware) | Coverage-aware masked wrappers | shipped |
| [`metrics.distributional`](#distributional-distances) | Distributional distances (CRPS, Wasserstein, …) | shipped |
| `metrics.structural` | Structural / geometric similarity (SSIM, …) | partial |
| `metrics.probabilistic` | Ensemble calibration | partial |
| `metrics.object` | Event / object verification (POD, FAR, IoU) | partial |
| `metrics.forecast` | Lead-time skill | stub |
| `metrics.lagrangian` | Trajectory / transport metrics | stub |

!!! tip "Ergonomic re-exports"
    The Layer-1 operators are re-exported flat from the package root, and a
    dedicated `xrops.metrics.operators` module mirrors them:

    ```python
    from xrops.metrics import RMSE, MSE, MAE, NRMSE, Bias, Correlation, R2Score, PSDScore
    # equivalently:
    from xrops.metrics.operators import RMSE, PSDScore
    ```

## Pixel — pointwise errors

Per-pixel scalar error and skill: MSE, RMSE, MAE, bias, normalized RMSE
(and its skill-score form), correlation, and R².

::: xrops.metrics.operators.MSE

::: xrops.metrics.operators.RMSE

::: xrops.metrics.operators.MAE

::: xrops.metrics.operators.Bias

::: xrops.metrics.operators.NRMSE

::: xrops.metrics.operators.NRMSEScore

::: xrops.metrics.operators.Correlation

::: xrops.metrics.operators.R2Score

### Functions

::: xrops.metrics.pixel

## Spectral — power-spectrum skill

Power-spectral-density scores and the resolved-scale diagnostic (the
wavelength at which a prediction's PSD score crosses a threshold).

::: xrops.metrics.operators.PSDScore

::: xrops.metrics.operators.SegmentedPSDScore

::: xrops.metrics.operators.WaveletPSDScore

### Functions

::: xrops.metrics.spectral

## Multiscale — region & band conditioned

Stratify skill by geographic region or by frequency band.

::: xrops.metrics.operators.EvaluateByRegion

::: xrops.metrics.operators.RegionScores

::: xrops.metrics.operators.FrequencyBandSkill

::: xrops.metrics.operators.BandLimitedRMSE

::: xrops.metrics.operators.SkillByLeadTime

## Physical — balance residuals

Conservation- and balance-law residuals used as physical evaluation
diagnostics (geostrophic balance, divergence, potential-vorticity
conservation, density inversions).

The derivative-based metrics default to lon/lat in degrees with
latitude-derived Coriolis. Idealised f/β-plane model output in metres
passes the horizontal dims, the geometry and an explicit Coriolis
parameter (same `f=` forms as [`xrops.ocn`](ocn.md)):

```python
from xrops.metrics import divergence_error, geostrophic_imbalance

cart = {"dims": ("x", "y"), "geometry": "cartesian"}
geostrophic_imbalance(run, ssh_var="eta", **cart, f=(1e-4, 1.6e-11))  # 0 = balanced
divergence_error(run, **cart, reduce="rms")                           # 1/s
```

`geostrophic_imbalance` is the scale-invariant ratio
$\mathrm{rms}(f\,\hat k\times\mathbf u + g\nabla\eta)\,/\,\mathrm{rms}(f\,\hat k\times\mathbf u)$:
0 for exact geostrophy, 1 for purely inertial flow.

::: xrops.metrics.operators.GeostrophicBalanceError

::: xrops.metrics.operators.GeostrophicImbalance

::: xrops.metrics.operators.DivergenceError

::: xrops.metrics.operators.PVConservationError

::: xrops.metrics.operators.DensityInversionFraction

## Masked — coverage aware

Wrap any metric so it ignores masked / missing cells and reports coverage.

::: xrops.metrics.operators.MaskedMetric

::: xrops.metrics.operators.BinnedResiduals2D

## Distributional distances

Distance between predicted and reference distributions.

::: xrops.metrics.operators.CRPS

::: xrops.metrics.operators.Wasserstein1

::: xrops.metrics.operators.EnergyDistance

## Structural & probabilistic

::: xrops.metrics.operators.SSIM

::: xrops.metrics.operators.GradientDifference

::: xrops.metrics.operators.PhaseShiftError

::: xrops.metrics.operators.EnsembleCoverage

::: xrops.metrics.operators.SpreadSkillRatio

::: xrops.metrics.operators.RankHistogram

::: xrops.metrics.operators.ReliabilityCurve

## Object / event verification

::: xrops.metrics.operators.InstanceMatcher

::: xrops.metrics.operators.InstanceF1AtIoU

::: xrops.metrics.operators.AveragePrecisionMatched

::: xrops.metrics.operators.CentroidDisplacement

::: xrops.metrics.operators.MaskIoU

## Forecast comparison

::: xrops.metrics.operators.DieboldMariano

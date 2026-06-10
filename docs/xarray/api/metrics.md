# Metrics

`xrtoolz.metrics` groups evaluation metrics by *scientific diagnostic
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
    dedicated `xrtoolz.metrics.operators` module mirrors them:

    ```python
    from xrtoolz.metrics import RMSE, MSE, MAE, NRMSE, Bias, Correlation, R2Score, PSDScore
    # equivalently:
    from xrtoolz.metrics.operators import RMSE, PSDScore
    ```

## Pixel — pointwise errors

Per-pixel scalar error and skill: MSE, RMSE, MAE, bias, normalized RMSE
(and its skill-score form), correlation, and R².

::: xrtoolz.metrics.operators.MSE

::: xrtoolz.metrics.operators.RMSE

::: xrtoolz.metrics.operators.MAE

::: xrtoolz.metrics.operators.Bias

::: xrtoolz.metrics.operators.NRMSE

::: xrtoolz.metrics.operators.NRMSEScore

::: xrtoolz.metrics.operators.Correlation

::: xrtoolz.metrics.operators.R2Score

### Functions

::: xrtoolz.metrics.pixel

## Spectral — power-spectrum skill

Power-spectral-density scores and the resolved-scale diagnostic (the
wavelength at which a prediction's PSD score crosses a threshold).

::: xrtoolz.metrics.operators.PSDScore

::: xrtoolz.metrics.operators.SegmentedPSDScore

::: xrtoolz.metrics.operators.WaveletPSDScore

### Functions

::: xrtoolz.metrics.spectral

## Multiscale — region & band conditioned

Stratify skill by geographic region or by frequency band.

::: xrtoolz.metrics.operators.EvaluateByRegion

::: xrtoolz.metrics.operators.RegionScores

::: xrtoolz.metrics.operators.FrequencyBandSkill

::: xrtoolz.metrics.operators.BandLimitedRMSE

::: xrtoolz.metrics.operators.SkillByLeadTime

## Physical — balance residuals

Conservation- and balance-law residuals used as physical evaluation
diagnostics (geostrophic balance, divergence, potential-vorticity
conservation, density inversions).

::: xrtoolz.metrics.operators.GeostrophicBalanceError

::: xrtoolz.metrics.operators.DivergenceError

::: xrtoolz.metrics.operators.PVConservationError

::: xrtoolz.metrics.operators.DensityInversionFraction

## Masked — coverage aware

Wrap any metric so it ignores masked / missing cells and reports coverage.

::: xrtoolz.metrics.operators.MaskedMetric

::: xrtoolz.metrics.operators.BinnedResiduals2D

## Distributional distances

Distance between predicted and reference distributions.

::: xrtoolz.metrics.operators.CRPS

::: xrtoolz.metrics.operators.Wasserstein1

::: xrtoolz.metrics.operators.EnergyDistance

## Structural & probabilistic

::: xrtoolz.metrics.operators.SSIM

::: xrtoolz.metrics.operators.GradientDifference

::: xrtoolz.metrics.operators.PhaseShiftError

::: xrtoolz.metrics.operators.EnsembleCoverage

::: xrtoolz.metrics.operators.SpreadSkillRatio

::: xrtoolz.metrics.operators.RankHistogram

::: xrtoolz.metrics.operators.ReliabilityCurve

## Object / event verification

::: xrtoolz.metrics.operators.InstanceMatcher

::: xrtoolz.metrics.operators.InstanceF1AtIoU

::: xrtoolz.metrics.operators.AveragePrecisionMatched

::: xrtoolz.metrics.operators.CentroidDisplacement

::: xrtoolz.metrics.operators.MaskIoU

## Forecast comparison

::: xrtoolz.metrics.operators.DieboldMariano

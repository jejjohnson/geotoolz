---
status: draft
version: 0.2.0
---

!!! note "Module paths shown are proposed design targets"
    The snippets below import from `xr_toolz.metrics.*`, `xr_toolz.budgets`,
    `xr_toolz.phenomena`, and `xr_toolz.lagrangian` — submodules that **do not
    exist in the current export surface**. Today, validators such as
    `ValidateCoords` live under `xr_toolz.geo.operators` (not
    `xr_toolz.geo.validation`), and metric primitives live under
    `xr_toolz.geo.metrics`. Treat the imports below as design-target aliases;
    once the modules ship, the snippets will be copy/paste-ready against the
    proposed layout.

# Validation Examples

This page shows how the proposed validation framework composes field, spectral, lead-time, structural, process, Lagrangian, and phenomena-based diagnostics without removing any existing examples.

---

## Example 1: Field + Spectral + Lead-Time Skill

### User Story

As an ML forecasting researcher, I want to evaluate a forecast with RMSE, spectral skill, and lead-time skill in one graph, so that I can diagnose both average error and scale-dependent degradation.

### Motivation

A model can perform well in global RMSE while losing small-scale variance or degrading rapidly with forecast horizon. These diagnostics should be computed together so that validation reports are internally consistent.

### Demo API

```python
from xr_toolz.core import Graph, Input
from xr_toolz.geo.operators import ValidateCoords  # current export path
from xr_toolz.metrics import RMSE, PSDScore
from xr_toolz.metrics.forecast import RMSEByLead
```

### Demo Example Usage

```python
pred = Input("prediction")
ref = Input("reference")

pred_clean = ValidateCoords()(pred)
ref_clean = ValidateCoords()(ref)

rmse = RMSE(variable="ssh", dims=("time", "lat", "lon"))(pred_clean, ref_clean)
psd = PSDScore(variable="ssh", dims=("lat", "lon"))(pred_clean, ref_clean)
lead = RMSEByLead(variable="ssh", dims=("lat", "lon"), lead_dim="lead_time")(pred_clean, ref_clean)

validation_graph = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={"rmse": rmse, "psd": psd, "lead_rmse": lead},
)

scores = validation_graph(prediction=ds_pred, reference=ds_ref)
```

---

## Example 2: Structural Skill for a Displaced Eddy

### User Story

As an ocean ML researcher, I want to evaluate whether a predicted eddy is structurally correct even when it is slightly displaced, so that the validation does not reduce every spatial phase error to a double-penalized RMSE.

### Motivation

Pointwise scores penalize a displaced feature twice: once where the observed object is missed and once where the predicted object appears. Structural metrics help separate amplitude error from displacement, morphology, and phase error.

### Demo API

```python
from xr_toolz.metrics import RMSE
from xr_toolz.metrics.structural import SSIM, PhaseShiftError, GradientDifference
```

### Demo Example Usage

```python
rmse = RMSE(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
ssim = SSIM(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
phase = PhaseShiftError(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
gradient = GradientDifference(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)

summary = {
    "pointwise_error": rmse,
    "structure": ssim,
    "phase_shift": phase,
    "gradient_error": gradient,
}
```

---

## Example 3: Regional and Regime-Aware Skill

### User Story

As a regional oceanographer, I want to compare model skill across coastal, open-ocean, eddy-rich, and frontal regimes, so that strong global performance does not hide regional failure modes.

### Motivation

Geophysical skill is often regime-dependent. Coastal regions, boundary currents, bathymetric gradients, and eddy-rich regions may have very different error behavior from open-ocean interiors.

### Demo API

```python
from xr_toolz.geo.masks import AddRegionMask
from xr_toolz.metrics import RMSE
from xr_toolz.metrics.scale import EvaluateByRegion
```

### Demo Example Usage

```python
metric = RMSE(variable="ssh", dims=("time",))
regional_skill = EvaluateByRegion(
    metric=metric,
    regions={
        "coastal": coastal_mask,
        "open_ocean": open_ocean_mask,
        "eddy_rich": eddy_rich_mask,
        "fronts": front_mask,
    },
)(ds_pred, ds_ref)
```

---

## Example 4: Probabilistic Ensemble Validation

### User Story

As a probabilistic forecasting researcher, I want to evaluate ensemble calibration and sharpness, so that a stochastic model is not judged only by the error of its ensemble mean.

### Motivation

For chaotic geophysical systems, a useful ensemble should be both calibrated and sharp. CRPS, spread-skill ratio, rank histograms, and coverage diagnostics expose underdispersion, overdispersion, and biased uncertainty.

### Demo API

```python
from xr_toolz.metrics.probabilistic import CRPS, SpreadSkillRatio, RankHistogram, EnsembleCoverage
```

### Demo Example Usage

```python
crps = CRPS(variable="ssh", ensemble_dim="member", dims=("time", "lat", "lon"))(ensemble_pred, ds_ref)
spread_skill = SpreadSkillRatio(variable="ssh", ensemble_dim="member", dims=("time", "lat", "lon"))(ensemble_pred, ds_ref)
ranks = RankHistogram(variable="ssh", ensemble_dim="member")(ensemble_pred, ds_ref)
coverage = EnsembleCoverage(variable="ssh", q=(0.05, 0.95), ensemble_dim="member")(ensemble_pred, ds_ref)
```

---

## Example 5: Process-Based Validation Graph

### User Story

As a physical oceanographer, I want to evaluate field skill and physical consistency in the same graph, so that I can detect models that score well statistically but violate balances or budgets.

### Motivation

Conventional error scores do not guarantee dynamical consistency. Balance residuals and budget checks expose physically implausible behavior, especially in long autoregressive rollouts.

### Demo API

```python
from xr_toolz.core import Graph, Input
from xr_toolz.metrics import RMSE
from xr_toolz.metrics.physical import GeostrophicBalanceError, DivergenceError
from xr_toolz.budgets import HeatBudgetResidual
```

### Demo Example Usage

```python
pred = Input("prediction")
ref = Input("reference")

rmse = RMSE(variable="ssh", dims=("time", "lat", "lon"))(pred, ref)
geostrophic = GeostrophicBalanceError(ssh_var="ssh", u_var="u", v_var="v")(pred)
divergence = DivergenceError(u_var="u", v_var="v", dims=("lat", "lon"))(pred)
heat = HeatBudgetResidual(temp_var="theta", u_var="u", v_var="v", surface_flux_var="qnet")(pred)

process_graph = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={
        "rmse": rmse,
        "geostrophic_balance": geostrophic,
        "divergence": divergence,
        "heat_budget": heat,
    },
)
```

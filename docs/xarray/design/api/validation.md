---
status: draft
version: 0.2.0
---

!!! note "Module paths shown are proposed design targets"
    The submodules referenced on this page (`xr_toolz.metrics.*`,
    `xr_toolz.budgets`, `xr_toolz.phenomena`, `xr_toolz.lagrangian`,
    `xr_toolz.viz.validation`, …) are **proposed** layouts and **are not part of
    the current export surface**. Today, most domain-agnostic functionality
    still lives under `xr_toolz.geo.*`. Treat the imports below as design-target
    aliases; once the proposed modules ship, the snippets become copy/paste-ready.

# Validation API Map

This page maps scientific validation questions to the proposed `xr_toolz` API surface. It complements [`primitives.md`](primitives.md), [`components.md`](components.md), and the top-level [`validation.md`](../validation.md) design note.

---

## User Story

As a user, I want one place to understand which validation tool to use for which scientific question, so that I can compose appropriate diagnostics without searching through every submodule.

## Motivation

Validation is broader than scalar metrics. A complete geoscience ML evaluation may need pointwise error, spectral fidelity, structural similarity, ensemble calibration, physical balances, conservation budgets, Lagrangian transport, and event verification.

---

## Which Diagnostic Should I Use?

| Scientific question | Module | Operators / functions |
|---|---|---|
| Are grid values accurate? | `xr_toolz.metrics.pixel` | `RMSE`, `MAE`, `Bias`, `Correlation`, `NashSutcliffe` |
| Are scales resolved? | `xr_toolz.metrics.spectral`, `xr_toolz.metrics.multiscale` | `PSDScore`, `ResolvedScale`, `CoherenceSkill`, `PerScaleRMSE`, `WaveletRMSE` |
| Does skill depend on forecast horizon? | `xr_toolz.metrics.forecast` | `SkillByLeadTime`, `RMSEByLead`, `AnomalyCorrelationByLead`, `SpectralSkillByLead` |
| Does skill vary by region or regime? | `xr_toolz.metrics.multiscale`, `xr_toolz.metrics.spectral`, `xr_toolz.geo.masks` | `EvaluateByRegion`, `FrequencyBandSkill`, `BandLimitedRMSE`, `AddRegionMask` |
| Is an ensemble calibrated? | `xr_toolz.metrics.probabilistic` | `CRPS`, `SpreadSkillRatio`, `RankHistogram`, `EnsembleCoverage`, `ReliabilityCurve` |
| Are structures displaced or blurred? | `xr_toolz.metrics.structural` | `SSIM`, `GradientDifference`, `PhaseShiftError`, `CentroidDisplacement` |
| Are physical balances respected? | `xr_toolz.metrics.physical`, `xr_toolz.kinematics` | `GeostrophicBalanceError`, `DivergenceError`, `VorticityError`, `KineticEnergyError` |
| Are budgets closed? | `xr_toolz.budgets` | `ControlVolumeIntegral`, `BoundaryFlux`, `HeatBudgetResidual`, `SaltBudgetResidual`, `VolumeBudgetResidual`, `KineticEnergyBudgetResidual` |
| Is material transport realistic? | `xr_toolz.lagrangian`, `xr_toolz.metrics.lagrangian` | `AdvectParticles`, `PairDispersion`, `ResidenceTime`, `ConnectivityMatrix`, `FTLE`, `EndpointError` |
| Are events captured? | `xr_toolz.phenomena`, `xr_toolz.metrics.object` | `DetectMarineHeatwaves`, `DetectEddies`, `DetectFronts`, `MatchObjects`, `ProbabilityOfDetection`, `FalseAlarmRatio`, `CriticalSuccessIndex`, `IntersectionOverUnion` |
| Do diagnostics need plots? | `xr_toolz.viz.validation` | `ScaleSkillPanel`, `SpectralSkillPanel`, `LeadTimeSkillPanel`, `ProcessBudgetPanel`, `EventVerificationPanel` |

---

## Recommended Validation Families

### 1. Baseline field scores

#### User Story

As an ML researcher, I want standard pointwise scores as a baseline, so that new diagnostics can be compared against familiar scalar metrics.

#### Motivation

RMSE, MAE, bias, correlation, and NSE remain useful, but they should be treated as baseline diagnostics rather than sufficient validation.

#### Demo API

```python
from xr_toolz.metrics import RMSE, MAE, Bias, Correlation, NashSutcliffe
```

#### Demo Example Usage

```python
rmse = RMSE(variable="ssh", dims=("time", "lat", "lon"))(ds_pred, ds_ref)
bias = Bias(variable="ssh", dims=("time", "lat", "lon"))(ds_pred, ds_ref)
```

---

### 2. Scale-aware validation

#### User Story

As a forecasting researcher, I want to evaluate by lead time, spatial scale, temporal scale, and region, so that aggregate scores do not hide scale-dependent failures.

#### Motivation

A model may have good global RMSE while underrepresenting high-wavenumber variance, failing in coastal regimes, or degrading rapidly with forecast horizon.

#### Demo API

```python
from xr_toolz.metrics.forecast import SkillByLeadTime, RMSEByLead, AnomalyCorrelationByLead
from xr_toolz.metrics.multiscale import EvaluateByRegion
from xr_toolz.metrics.spectral import FrequencyBandSkill
from xr_toolz.metrics.spectral import PSDScore, ResolvedScale
```

#### Demo Example Usage

```python
lead_rmse = RMSEByLead(variable="ssh", dims=("lat", "lon"), lead_dim="lead_time")(forecast, reference)
regional_skill = EvaluateByRegion(metric=RMSE(variable="ssh", dims=("time",)), regions=region_masks)(forecast, reference)
```

---

### 3. Probabilistic validation

#### User Story

As a probabilistic modeler, I want to evaluate calibration, sharpness, spread, and coverage, so that ensemble forecasts are judged as probability distributions rather than deterministic means.

#### Motivation

Chaotic geophysical systems require uncertainty-aware evaluation. CRPS, rank histograms, reliability curves, and coverage scores reveal underdispersion, overdispersion, and calibration errors.

#### Demo API

```python
from xr_toolz.metrics.probabilistic import CRPS, SpreadSkillRatio, RankHistogram, EnsembleCoverage, ReliabilityCurve
```

#### Demo Example Usage

```python
crps = CRPS(variable="ssh", ensemble_dim="member", dims=("time", "lat", "lon"))(ensemble, reference)
coverage = EnsembleCoverage(variable="ssh", q=(0.05, 0.95), ensemble_dim="member")(ensemble, reference)
```

---

### 4. Structural validation

#### User Story

As an ocean ML researcher, I want structural and displacement-aware metrics, so that coherent but shifted eddies, fronts, or plumes are not treated the same as missing features.

#### Motivation

Pointwise metrics have a double-penalty problem for displaced coherent structures. Structural metrics diagnose geometry, phase, and morphology.

#### Demo API

```python
from xr_toolz.metrics.structural import SSIM, GradientDifference, PhaseShiftError, CentroidDisplacement
```

#### Demo Example Usage

```python
ssim = SSIM(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
phase = PhaseShiftError(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
```

---

### 5. Physical and process validation

#### User Story

As a physical scientist, I want to evaluate geophysical balances, density structure, and conservation budgets, so that model skill reflects physical consistency.

#### Motivation

Good short-range errors do not guarantee physically plausible dynamics. Balance residuals, density checks, and control-volume budgets diagnose whether predictions obey the structure of the underlying system.

#### Demo API

```python
from xr_toolz.metrics.physical import GeostrophicBalanceError, DivergenceError, DensityInversionFraction
from xr_toolz.budgets import HeatBudgetResidual, SaltBudgetResidual, VolumeBudgetResidual, ControlVolumeIntegral
```

#### Demo Example Usage

```python
geo = GeostrophicBalanceError(ssh_var="ssh", u_var="u", v_var="v")(ds_pred)
heat = HeatBudgetResidual(temp_var="theta", u_var="u", v_var="v", surface_flux_var="qnet")(ds_pred)
```

---

### 6. Lagrangian validation

#### User Story

As a transport researcher, I want to advect particles through predicted and reference velocity fields, so that I can compare trajectories, dispersion, residence times, and connectivity.

#### Motivation

Eulerian field agreement does not guarantee material transport fidelity. Lagrangian diagnostics reveal errors that accumulate along parcel pathways.

#### Demo API

```python
from xr_toolz.lagrangian import SeedParticles, AdvectParticles, PairDispersion, ResidenceTime, ConnectivityMatrix, FTLE
from xr_toolz.metrics.lagrangian import EndpointError, TrajectoryRMSE, DispersionError
```

#### Demo Example Usage

```python
particles = SeedParticles(strategy="grid", spacing=0.25, region=med_mask)(ds_ref)
traj_pred = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_pred, particles)
traj_ref = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_ref, particles)
endpoint = EndpointError()(traj_pred, traj_ref)
```

---

### 7. Phenomena-based validation

#### User Story

As an applied scientist, I want to detect and verify events such as marine heatwaves, eddies, fronts, and upwelling events, so that model evaluation reflects scientifically meaningful phenomena.

#### Motivation

Event verification checks whether the model represents discrete structures in space and time, not only whether the continuous field has low average error.

#### Demo API

```python
from xr_toolz.phenomena import EventDefinition, DetectMarineHeatwaves, DetectEddies, DetectFronts, MatchObjects
from xr_toolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex, IntersectionOverUnion
```

#### Demo Example Usage

```python
events_pred = DetectMarineHeatwaves(sst_var="sst", climatology=sst_clim, percentile=90, min_duration=5)(ds_pred)
events_ref = DetectMarineHeatwaves(sst_var="sst", climatology=sst_clim, percentile=90, min_duration=5)(ds_ref)
matches = MatchObjects(method="iou", threshold=0.2)(events_pred, events_ref)

csi = CriticalSuccessIndex()(matches)
iou = IntersectionOverUnion()(matches)
```

---

## End-to-End Validation Graph

```python
from xr_toolz.core import Graph, Input
from xr_toolz.metrics import RMSE, PSDScore
from xr_toolz.metrics.forecast import RMSEByLead
from xr_toolz.metrics.structural import SSIM
from xr_toolz.budgets import HeatBudgetResidual

pred = Input("prediction")
ref = Input("reference")

rmse = RMSE(variable="ssh", dims=("time", "lat", "lon"))(pred, ref)
psd = PSDScore(variable="ssh", dims=("lat", "lon"))(pred, ref)
lead = RMSEByLead(variable="ssh", dims=("lat", "lon"), lead_dim="lead_time")(pred, ref)
structure = SSIM(variable="ssh", dims=("lat", "lon"))(pred, ref)
heat_budget = HeatBudgetResidual(temp_var="theta", u_var="u", v_var="v")(pred)

validation = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={
        "rmse": rmse,
        "psd": psd,
        "lead_rmse": lead,
        "ssim": structure,
        "heat_budget": heat_budget,
    },
)
```

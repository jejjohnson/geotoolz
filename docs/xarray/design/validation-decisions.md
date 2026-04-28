---
status: draft
version: 0.2.0
---

# Validation Design Decisions

This companion decision log records proposed validation-specific decisions. It extends [`decisions.md`](decisions.md) without modifying the accepted D1–D10 decisions.

---

## D11: Validation is organized by scientific diagnostic family

**Status:** proposed

### Context

Existing metrics cover pixel, spectral, multiscale, and distributional scores. Ocean and geoscience ML validation also requires structural, forecast, probabilistic, physical, Lagrangian, process, and phenomena-based diagnostics.

### Decision

Add a validation taxonomy with five conceptual families:

1. Scales of evaluation
2. Data representation
3. Physical representation
4. Process evaluation
5. Phenomena-based evaluation

Implementation is split across `metrics`, `lagrangian`, `budgets`, `phenomena`, and `viz`.

### Consequences

- `metrics` remains the home for scalar and array-valued skill scores.
- `lagrangian` owns particle and transport diagnostics.
- `budgets` owns conservation and control-volume residuals.
- `phenomena` owns event definitions, detection, labeling, matching, and properties.
- `viz` owns diagnostic panels.

### User Story

As a geoscience ML researcher, I want validation tools organized by scientific diagnostic question, so that I can choose the right metric family without reducing evaluation to a single global score.

### Motivation

A validation framework should reveal whether a model reproduces scales, uncertainty, structures, transport, processes, budgets, and events. These concepts are broader than pointwise error and should be discoverable in the API.

### Demo API

```python
from xr_toolz.metrics import RMSE, PSDScore
from xr_toolz.metrics.structural import SSIM
from xr_toolz.metrics.forecast import RMSEByLead
from xr_toolz.budgets import HeatBudgetResidual
from xr_toolz.lagrangian import AdvectParticles
from xr_toolz.phenomena import DetectMarineHeatwaves
```

### Demo Example Usage

```python
scores = {
    "rmse": RMSE(variable="ssh", dims=("time", "lat", "lon"))(ds_pred, ds_ref),
    "psd": PSDScore(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref),
    "ssim": SSIM(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref),
    "lead_rmse": RMSEByLead(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref),
    "heat_budget": HeatBudgetResidual(temp_var="theta", u_var="u", v_var="v")(ds_pred),
}
```

---

## D12: Lagrangian diagnostics live outside `metrics`

**Status:** proposed

### Context

Particle advection, trajectories, FTLE-like diagnostics, residence time, and connectivity are not just scalar metrics. They create intermediate trajectory datasets that can be reused across many analyses.

### Decision

Create `xr_toolz.lagrangian` as a first-class module. Scalar comparisons of trajectory outputs can live in `xr_toolz.metrics.lagrangian`, but trajectory generation and transport diagnostics live in `xr_toolz.lagrangian`.

### Consequences

- Users can compute trajectories once and reuse them across endpoint error, pair dispersion, residence time, and connectivity metrics.
- Lagrangian tools remain composable `Operator`s.
- Future optional acceleration can be added without changing the validation API.

### User Story

As a Lagrangian transport researcher, I want to create reusable particle trajectories from model velocity fields, so that I can evaluate transport pathways with several downstream diagnostics.

### Motivation

Eulerian field agreement does not guarantee correct material transport. Trajectory generation is a first-class transformation, not merely a metric implementation detail.

### Demo API

```python
from xr_toolz.lagrangian import SeedParticles, AdvectParticles, PairDispersion, ResidenceTime, ConnectivityMatrix, FTLE
from xr_toolz.metrics.lagrangian import EndpointError, TrajectoryRMSE, ConnectivityError
```

### Demo Example Usage

```python
particles = SeedParticles(strategy="grid", spacing=0.25, region=med_mask)(ds_ref)
traj_pred = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_pred, particles)
traj_ref = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_ref, particles)
endpoint = EndpointError()(traj_pred, traj_ref)
```

---

## D13: Conservation diagnostics live in `budgets`

**Status:** proposed

### Context

Budget residuals require tendencies, fluxes, control-volume integration, grid metrics, and optional source/sink terms. These are broader than simple field metrics.

### Decision

Create `xr_toolz.budgets` for generic and domain-specific budget residuals.

### Consequences

- Physical-process checks are reusable across ocean, atmosphere, tracers, and methane workflows.
- Budget residual outputs can be passed to `metrics`, `viz`, or `Graph` workflows.
- Grid metrics and control-volume definitions remain explicit instead of hidden in a metric implementation.

### User Story

As a physical scientist, I want to evaluate conservation residuals over finite volumes, so that long rollouts can be checked for physically meaningful drift.

### Motivation

A model can have good short-term RMSE while violating heat, salt, volume, tracer, or energy conservation. Budget checks identify where and how the prediction becomes dynamically inconsistent.

### Demo API

```python
from xr_toolz.budgets import (
    ControlVolumeIntegral,
    BoundaryFlux,
    BudgetResidual,
    HeatBudgetResidual,
    SaltBudgetResidual,
    VolumeBudgetResidual,
    KineticEnergyBudgetResidual,
)
```

### Demo Example Usage

```python
heat_residual = HeatBudgetResidual(
    temp_var="theta",
    u_var="u",
    v_var="v",
    surface_flux_var="qnet",
)(ds_pred)

regional_heat_drift = ControlVolumeIntegral(
    variable="heat_budget_residual",
    volume_metrics=grid_metrics,
    region=med_mask,
)(heat_residual.to_dataset(name="heat_budget_residual"))
```

---

## D14: Phenomena detection and object verification are separate

**Status:** proposed

### Context

Detecting an event and scoring an event forecast are different tasks. Marine heatwaves, eddies, fronts, plumes, and storms require event definitions; verification requires matching and scoring.

### Decision

Put event definitions, detection, labeling, matching, and properties in `xr_toolz.phenomena`. Put scores such as `ProbabilityOfDetection` (POD), `FalseAlarmRatio` (FAR), `CriticalSuccessIndex` (CSI), `IntersectionOverUnion` (IoU), duration error, and intensity bias in `xr_toolz.metrics.object`.

### Consequences

- The same detected events can be visualized, filtered, matched, or scored.
- Object metrics stay small and composable.
- Event definitions become reusable objects applied consistently to predictions and references.

### User Story

As an applied ocean scientist, I want to define, detect, match, and score events as separate steps, so that event verification is reproducible and scientifically interpretable.

### Motivation

Field-level scores can be good while important finite-amplitude events are missed, delayed, displaced, or weakened. Object-based verification evaluates the phenomena themselves.

### Demo API

```python
from xr_toolz.phenomena import EventDefinition, DetectMarineHeatwaves, DetectEddies, MatchObjects, ObjectProperties
from xr_toolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex, IntersectionOverUnion
```

### Demo Example Usage

```python
events_pred = DetectMarineHeatwaves(
    sst_var="sst",
    climatology=sst_clim,
    percentile=90,
    min_duration=5,
)(ds_pred)

events_ref = DetectMarineHeatwaves(
    sst_var="sst",
    climatology=sst_clim,
    percentile=90,
    min_duration=5,
)(ds_ref)

matches = MatchObjects(method="iou", threshold=0.2)(events_pred, events_ref)

scores = {
    "pod": ProbabilityOfDetection()(matches),
    "far": FalseAlarmRatio()(matches),
    "csi": CriticalSuccessIndex()(matches),
    "iou": IntersectionOverUnion()(matches),
}
```

---

## D15: Validation plots are terminal visualization operators

**Status:** proposed

### Context

Validation reports often require diagnostic panels rather than raw score arrays alone. Existing D10 already accepts viz operators as terminal `Operator`s returning `Figure` / `Axes`.

### Decision

Add validation-specific plot operators under `xr_toolz.viz.validation`.

### Consequences

- Validation graphs can emit both scores and figures.
- Diagnostic panels remain composable terminal outputs.
- Plotting does not mutate datasets or hide figures in attrs.

### User Story

As a researcher, I want one validation graph to return metrics and publication-ready diagnostic panels, so that scoring and visualization are reproducible together.

### Motivation

Scale skill, spectral skill, process budgets, Lagrangian trajectories, and event verification are easier to inspect with standardized panels.

### Demo API

```python
from xr_toolz.viz.validation import (
    ScaleSkillPanel,
    SpectralSkillPanel,
    LeadTimeSkillPanel,
    ProcessBudgetPanel,
    EulerianLagrangianPanel,
    EventVerificationPanel,
)
```

### Demo Example Usage

```python
fig = SpectralSkillPanel(variable="ssh", dims=("lat", "lon"))(psd_scores)
```

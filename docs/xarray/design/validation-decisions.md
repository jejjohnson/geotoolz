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
from xrtoolz.metrics import RMSE, PSDScore
from xrtoolz.metrics.structural import SSIM
from xrtoolz.metrics.forecast import RMSEByLead
from xrtoolz.budgets import HeatBudgetResidual
from xrtoolz.lagrangian import AdvectParticles
from xrtoolz.phenomena import DetectMarineHeatwaves
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

Create `xrtoolz.lagrangian` as a first-class module. Scalar comparisons of trajectory outputs can live in `xrtoolz.metrics.lagrangian`, but trajectory generation and transport diagnostics live in `xrtoolz.lagrangian`.

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
from xrtoolz.lagrangian import SeedParticles, AdvectParticles, PairDispersion, ResidenceTime, ConnectivityMatrix, FTLE
from xrtoolz.metrics.lagrangian import EndpointError, TrajectoryRMSE, ConnectivityError
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

Create `xrtoolz.budgets` for generic and domain-specific budget residuals.

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
from xrtoolz.budgets import (
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

Put event definitions, detection, labeling, matching, and properties in `xrtoolz.phenomena`. Put scores such as `ProbabilityOfDetection` (POD), `FalseAlarmRatio` (FAR), `CriticalSuccessIndex` (CSI), `IntersectionOverUnion` (IoU), duration error, and intensity bias in `xrtoolz.metrics.object`.

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
from xrtoolz.phenomena import EventDefinition, DetectMarineHeatwaves, DetectEddies, MatchObjects, ObjectProperties
from xrtoolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex, IntersectionOverUnion
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

Add validation-specific plot operators under `xrtoolz.viz.validation`.

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
from xrtoolz.viz.validation import (
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

## D16: Budget operators take grid metrics as explicit inputs

### Context

Conservation-budget operators (V4.2 / V4.3) need cell areas, widths, and volumes to compute control-volume integrals and boundary fluxes. These metrics depend on the grid (regular sphere, tripolar, NEMO C-grid, etc.) and cannot reliably be guessed from coordinate values alone — for example, a Dataset with `lat`/`lon` in degrees does not tell you whether the model used `R · cos φ · Δλ` or a tripolar overlay.

### Decision

**Integral / volume-weighted budget operators** — `ControlVolumeIntegral`, `BoundaryFlux`, and any future per-region closure operator that consumes `cell_volume` / face areas — **must** accept `volume_metrics` and `face_metrics` as constructor arguments. They never auto-derive cell volumes / face areas from coordinates. A user who wants spherical metrics from lon/lat coords calls `xrtoolz.calc.grid_metrics_from_coords` first to produce the metric Datasets. Models that already ship explicit grid metrics (CMEMS NEMO output, MOM6 horizontal grids) bypass the helper and pass the model's own metric Dataset.

**Per-cell residual operators** — `HeatBudgetResidual`, `SaltBudgetResidual`, `VolumeBudgetResidual`, `KineticEnergyBudgetResidual` — return the field ``∂φ/∂t + ∇·(u φ) − sources`` on the same grid as the input. These do not need cell volumes or face areas: the spherical-metric divergence ``∇·`` only requires the differential metric ``R cos φ``, which `xrtoolz.calc.divergence` derives from coordinates inside `_src.spherical`. Volume-integrated closure (residuals weighted by `cell_volume` and summed over a control volume) goes through `ControlVolumeIntegral`, which is where explicit metrics re-enter the pipeline.

### Consequences

- No silent failures from operators guessing the wrong metric.
- One canonical helper (`grid_metrics_from_coords`) lives in `xrtoolz.calc`; budgets stay metric-agnostic.
- Datasets without metrics get a clear `KeyError` at the budget call site rather than a wrong-by-percent answer.
- The convention aligns with xgcm's separation of grid metrics from data fields.

### Convention

`volume_metrics: xr.Dataset` carries:
- `dx` (m) — zonal cell width
- `dy` (m) — meridional cell width
- `dz` (m) — vertical cell thickness (optional, depth grids only)
- `cell_area` (m²) — `dx · dy`
- `cell_volume` (m³) — `cell_area · dz`, falling back to `cell_area` when `dz` is absent

`face_metrics: xr.Dataset` carries:
- `dx_e`, `dy_n` (m) — face widths between adjacent cell centres
- `dz_top` (m) — vertical face thickness (optional)
- `area_e`, `area_n`, `area_top` (m²) — corresponding face areas

### Helper

```python
from xrtoolz import calc

vol, face = calc.grid_metrics_from_coords(
    ds, lat="lat", lon="lon", depth="depth", sphere=True
)
```

Returns the two Datasets above. Edge cells extrapolate the nearest interior spacing — adequate for closure tests and demo notebooks; users who need exact boundary metrics should pass model-shipped metrics instead.

### Audit table — V4.5 kinematics

V4.1 / V4.3 reach into `xrtoolz.calc` and `xrtoolz.ocn` for derived quantities. Status of each, audited as part of the V4 epic:

| Quantity | Status | Location |
|----------|--------|----------|
| Vorticity (relative + absolute) | implemented | `xrtoolz.ocn.relative_vorticity`, `absolute_vorticity` (and `xrtoolz.calc.curl`) |
| Divergence (2-D + 3-D) | implemented | `xrtoolz.calc.divergence`, `xrtoolz.ocn.divergence` (3-D via `volume_budget_residual` with `w_var`) |
| Density (TEOS-10) | implemented (lazy `gsw`) | `xrtoolz.ocn.density_from_ts` — raises `ImportError` pointing at `xrtoolz[oceanography]` if `gsw` is missing |
| Mixed-layer depth | implemented | `xrtoolz.ocn.mixed_layer_depth` |
| Brunt–Väisälä frequency (N²) | implemented | `xrtoolz.ocn.brunt_vaisala_frequency` |
| Kinetic energy | implemented | `xrtoolz.ocn.kinetic_energy` |


---

## D16: Frequency-band specification uses physical units of the dim coords

**Status:** accepted (resolves validation.md §1 open question 2).

### Context

V1.3 introduces `FrequencyBandSkill` and `BandLimitedRMSE` for
scale-decomposed validation. The open question was how a user should
name a band: by physical frequency, by grid index, or by wavelength.

### Decision

Bands are a `dict[str, tuple[float, float]]` of `{name: (low, high)}`,
with `low` and `high` expressed in the **physical frequency units** of
the dim coord (cycles per coord-unit). The dim coordinate must carry
a `units` attribute; lat/lon coords whose units look like
`degrees_north` / `degrees_east` are converted to kilometres
internally (using a meridian-arc-length of `111.0` km/° plus a
`cos(mean_lat)` factor for longitude), so spatial bands can be
expressed in cycles/km. A per-dim `coord_spacing=` override is
available for high-precision work or non-CF coord units.

### Consequences

- Bands survive resolution changes — re-gridding the input does not
  shift the band edges.
- Bands above the per-dim Nyquist magnitude emit a `UserWarning` and
  produce `NaN` rather than raising, so a `Sequential` pipeline
  containing several bands is robust to one of them being too high.
- Inputs whose dim coord has no `units` attribute raise with a clear
  error message pointing at `xrtoolz.geo.validation` (or
  `coord_spacing=`).

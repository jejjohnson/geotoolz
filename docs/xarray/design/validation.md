---
status: draft
version: 0.2.0
---

!!! note "Module paths shown are proposed design targets"
    Throughout this page, paths such as `xr_toolz.metrics.*`, `xr_toolz.budgets`,
    and `xr_toolz.phenomena` refer to **proposed** subpackages and **are not part
    of the current export surface**. Most domain-agnostic functionality today
    still lives under `xr_toolz.geo` (for example `xr_toolz.geo.operators`,
    `xr_toolz.geo.metrics`). Treat the snippets below as architectural direction;
    once the proposed modules ship, the imports become copy/paste-ready.

# Validation Framework

`xr_toolz` validation should be broader than a list of scalar metrics. The design goal is to make evaluation workflows diagnose whether geoscience ML models reproduce values, scales, uncertainty, geometry, transport, physical processes, budgets, and identifiable events.

This page extends the existing `preprocess -> infer -> evaluate` vision with five complementary validation views:

1. Scales of evaluation
2. Data representation
3. Physical representation
4. Process evaluation
5. Phenomena-based evaluation

The additions follow the existing three-layer design:

- **Layer 0**: pure functions over `xarray` objects
- **Layer 1**: thin `Operator` wrappers
- **Layer 2**: `Graph` workflows combining prediction, reference, masks, climatology, grid metrics, and auxiliary data

---

## 1. Scales of Evaluation

### User Story

As an ocean ML researcher, I want to evaluate model skill by region, lead time, spatial scale, temporal scale, and spectral band, so that good aggregate RMSE does not hide failures at mesoscale, submesoscale, coastal, or long-lead regimes.

### Motivation

Oceanic and atmospheric variability are intrinsically multiscale. A global scalar score can be dominated by large-amplitude, large-scale variability while smoothing high-frequency structure. Validation should therefore support decomposition by spatial domain, temporal scale, forecast horizon, and frequency band.

### Design

Scale-aware validation should live primarily in `xr_toolz.metrics.forecast`, `xr_toolz.metrics.multiscale`, and `xr_toolz.metrics.spectral`, with masking and region utilities supplied by `xr_toolz.geo.masks` and `xr_toolz.geo.subset`.

Common partitions include:

- geographic regions and named masks
- coastal, open-ocean, eddy-rich, frontal, or bathymetry-defined regimes
- forecast lead time
- temporal bands such as sub-daily, synoptic, seasonal, and interannual
- spatial bands such as basin scale, mesoscale, and submesoscale
- Fourier or wavelet frequency bands

### Demo API

```python
# Layer 0 functions
def skill_by_lead_time(prediction, reference, *, metric_fn, lead_dim="lead_time") -> xr.DataArray: ...
def evaluate_by_region(prediction, reference, *, metric_fn, regions) -> xr.Dataset: ...
def evaluate_by_frequency_band(prediction, reference, *, variable, bands, dims) -> xr.Dataset: ...
def band_limited_rmse(prediction, reference, *, variable, bands, dims) -> xr.Dataset: ...

# Layer 1 operators
class SkillByLeadTime(Operator): ...
class EvaluateByRegion(Operator): ...
class FrequencyBandSkill(Operator): ...
class BandLimitedRMSE(Operator): ...
```

### Demo Example Usage

```python
from xr_toolz.core import Graph, Input
from xr_toolz.metrics import RMSE
from xr_toolz.metrics.forecast import SkillByLeadTime
from xr_toolz.metrics.multiscale import EvaluateByRegion
from xr_toolz.metrics.spectral import FrequencyBandSkill

pred = Input("prediction")
ref = Input("reference")

rmse = RMSE(variable="ssh", dims=("lat", "lon"))
lead_rmse = SkillByLeadTime(metric=rmse, lead_dim="lead_time")(pred, ref)
regional_rmse = EvaluateByRegion(metric=rmse, regions=region_masks)(pred, ref)
band_skill = FrequencyBandSkill(variable="ssh", dims=("lat", "lon"), bands=spatial_bands)(pred, ref)

graph = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={
        "lead_rmse": lead_rmse,
        "regional_rmse": regional_rmse,
        "band_skill": band_skill,
    },
)
```

### Notes / Open Questions

- Should `EvaluateByRegion` accept `regionmask.Regions`, an integer mask, or a dict of boolean masks? Prefer all three through normalization to an internal labeled mask.
- Should frequency bands be specified in physical units, grid indices, or wavelengths? Prefer named bands plus explicit coordinate metadata when available.

---

## 2. Data Representation

### User Story

As a geoscience ML researcher, I want to compare predictions using pointwise, probabilistic, spectral, and structural metrics, so that displaced but physically coherent features are not treated the same as completely wrong predictions.

### Motivation

The representation used for validation determines which errors are penalized. Pointwise L1/L2 metrics are useful baselines, but they suffer from double penalties when coherent structures are displaced. Probabilistic metrics are needed for ensemble or stochastic predictions. Spectral metrics reveal loss of scale-dependent variance. Structural metrics diagnose geometry, phase, and morphology.

### Design

Data-representation metrics should be organized by metric family:

```text
xr_toolz.metrics.pixel
xr_toolz.metrics.probabilistic
xr_toolz.metrics.spectral
xr_toolz.metrics.multiscale
xr_toolz.metrics.structural
xr_toolz.metrics.distributional
xr_toolz.metrics.masked
```

Pointwise metrics remain first-class but should be documented as baseline diagnostics rather than sufficient validation.

### Demo API

```python
# Structural metrics
def ssim(prediction, reference, *, variable, dims, window=None) -> xr.DataArray: ...
def gradient_difference(prediction, reference, *, variable, dims) -> xr.DataArray: ...
def phase_shift_error(prediction, reference, *, variable, dims) -> xr.Dataset: ...
def centroid_displacement(objects_pred, objects_ref, *, dims=("lat", "lon")) -> xr.Dataset: ...

# Probabilistic metrics
def spread_skill_ratio(ensemble, reference, *, variable, ensemble_dim="member", dims=None) -> xr.DataArray: ...
def rank_histogram(ensemble, reference, *, variable, ensemble_dim="member") -> xr.Dataset: ...
def ensemble_coverage(ensemble, reference, *, variable, q=(0.05, 0.95), ensemble_dim="member") -> xr.DataArray: ...
def reliability_curve(probability, event, *, probability_bins=None) -> xr.Dataset: ...

# Layer 1 operators
class SSIM(Operator): ...
class GradientDifference(Operator): ...
class PhaseShiftError(Operator): ...
class SpreadSkillRatio(Operator): ...
class RankHistogram(Operator): ...
class EnsembleCoverage(Operator): ...
class ReliabilityCurve(Operator): ...
```

### Demo Example Usage

```python
from xr_toolz.metrics.structural import SSIM, PhaseShiftError
from xr_toolz.metrics.probabilistic import SpreadSkillRatio, RankHistogram

structure = SSIM(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
phase = PhaseShiftError(variable="ssh", dims=("lat", "lon"))(ds_pred, ds_ref)
spread_skill = SpreadSkillRatio(variable="ssh", ensemble_dim="member")(ensemble_pred, ds_ref)
ranks = RankHistogram(variable="ssh", ensemble_dim="member")(ensemble_pred, ds_ref)
```

### Notes / Open Questions

- `SSIM` may require optional `scikit-image`; if unavailable, provide a documented fallback or raise an informative optional-dependency error.
- `phase_shift_error` should support both periodic and non-periodic spatial dimensions.
- Ensemble metrics should define expected input shape and metadata conventions for `member` dimensions.

---

## 3. Physical Representation

### User Story

As an oceanographer, I want to evaluate both Eulerian fields and Lagrangian transport, so that a model that looks good on a grid is not accepted if it produces unrealistic trajectories, mixing, residence times, or connectivity.

### Motivation

Most gridded model validation is Eulerian: predicted and reference fields are compared at fixed coordinates. This is necessary but not sufficient for transport. Small phase or gradient errors in a velocity field can integrate into large trajectory divergence, incorrect residence times, or biased regional exchange.

### Design

Eulerian diagnostics should remain in `xr_toolz.metrics` and `xr_toolz.kinematics`. Lagrangian diagnostics should be a first-class module because particle advection creates reusable trajectory datasets, not only scalar scores.

```text
xr_toolz.lagrangian          # trajectory generation and transport diagnostics
xr_toolz.metrics.lagrangian  # scalar comparisons of trajectories/statistics
```

### Demo API

```python
# Layer 0 functions
def seed_particles(ds, *, lon=None, lat=None, strategy="grid", spacing=None, region=None) -> xr.Dataset: ...
def sample_velocity(ds, particles, *, u_var="u", v_var="v", method="linear") -> xr.Dataset: ...
def advect_particles(ds, particles, *, u_var="u", v_var="v", dt="1h", steps=None, method="rk4") -> xr.Dataset: ...
def pair_dispersion(trajectories, *, pairs=None) -> xr.DataArray: ...
def residence_time(trajectories, *, regions) -> xr.Dataset: ...
def connectivity_matrix(trajectories, *, source_regions, target_regions) -> xr.DataArray: ...
def ftle(ds, particles, *, integration_time, u_var="u", v_var="v") -> xr.DataArray: ...

# Layer 1 operators
class SeedParticles(Operator): ...
class AdvectParticles(Operator): ...
class PairDispersion(Operator): ...
class ResidenceTime(Operator): ...
class ConnectivityMatrix(Operator): ...
class FTLE(Operator): ...

# Metrics over trajectory outputs
class TrajectoryRMSE(Operator): ...
class EndpointError(Operator): ...
class DispersionError(Operator): ...
class ResidenceTimeError(Operator): ...
class ConnectivityError(Operator): ...
```

### Demo Example Usage

```python
from xr_toolz.lagrangian import SeedParticles, AdvectParticles, PairDispersion
from xr_toolz.metrics.lagrangian import EndpointError

particles = SeedParticles(strategy="grid", spacing=0.25, region=med_mask)(ds_ref)

traj_pred = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_pred, particles)
traj_ref = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_ref, particles)

endpoint_error = EndpointError()(traj_pred, traj_ref)
pair_dispersion_bias = PairDispersion()(traj_pred) - PairDispersion()(traj_ref)
```

### Notes / Open Questions

- Particle integration should initially use numpy/scipy/xarray, with optional acceleration later.
- The trajectory schema should be documented carefully: recommended dimensions are `particle` and `time`, with variables such as `lon`, `lat`, and optional sampled fields.
- Drifter comparison should accept observed trajectory datasets using the same schema.

---

## 4. Process Evaluation

### User Story

As a physical oceanographer, I want to check balances, budgets, density structure, and material invariants, so that model skill reflects physical consistency rather than only statistical agreement.

### Motivation

A model can achieve favorable short-range error scores while violating conservation laws, static stability, geostrophic balance, or material transport constraints. Process-based validation helps distinguish physically plausible skill from learned correlations.

### Design

Process evaluation spans multiple modules:

```text
xr_toolz.kinematics        # derived quantities: vorticity, divergence, density, MLD, N2, KE
xr_toolz.metrics.physical  # balance and physical-consistency scores
xr_toolz.budgets           # control-volume budgets and residuals
xr_toolz.lagrangian        # material-frame diagnostics
```

### Demo API

```python
# Physical metrics
def geostrophic_balance_error(ds, *, ssh_var="ssh", u_var="u", v_var="v") -> xr.Dataset: ...
def divergence_error(ds, *, u_var="u", v_var="v", dims=("lat", "lon")) -> xr.DataArray: ...
def density_inversion_fraction(ds, *, density_var="rho", depth_dim="depth") -> xr.DataArray: ...
def pv_conservation_error(trajectories, *, pv_var="pv") -> xr.DataArray: ...

# Budgets
def control_volume_integral(ds, *, variable, volume_metrics, region=None, dims=("z", "lat", "lon")) -> xr.DataArray: ...
def boundary_flux(ds, *, variable, velocity_vars, face_metrics, region=None) -> xr.Dataset: ...
def budget_residual(tendency, flux_divergence, *, source=None, sink=None) -> xr.DataArray: ...
def heat_budget_residual(ds, *, temp_var="theta", u_var="u", v_var="v", w_var=None, surface_flux_var=None) -> xr.DataArray: ...
def salt_budget_residual(ds, *, salt_var="so", u_var="u", v_var="v", w_var=None, surface_flux_var=None) -> xr.DataArray: ...
def volume_budget_residual(ds, *, u_var="u", v_var="v", w_var=None) -> xr.DataArray: ...
def kinetic_energy_budget_residual(ds, *, u_var="u", v_var="v", forcing_vars=None) -> xr.DataArray: ...

# Layer 1 operators
class GeostrophicBalanceError(Operator): ...
class DivergenceError(Operator): ...
class DensityInversionFraction(Operator): ...
class ControlVolumeIntegral(Operator): ...
class HeatBudgetResidual(Operator): ...
class SaltBudgetResidual(Operator): ...
class VolumeBudgetResidual(Operator): ...
class KineticEnergyBudgetResidual(Operator): ...
```

### Demo Example Usage

```python
from xr_toolz.metrics.physical import GeostrophicBalanceError, DivergenceError
from xr_toolz.budgets import HeatBudgetResidual, ControlVolumeIntegral

geo_residual = GeostrophicBalanceError(ssh_var="ssh", u_var="u", v_var="v")(ds_pred)
div_residual = DivergenceError(u_var="u", v_var="v", dims=("lat", "lon"))(ds_pred)

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

### Notes / Open Questions

- Density and equation-of-state calculations should use `gsw` / TEOS-10 as an optional dependency when available.
- Derivative-based diagnostics should document filtering/coarse-graining choices because gradients amplify small-scale artifacts.
- Budget operators should make grid metrics explicit rather than guessing cell areas and volumes when possible.

---

## 5. Phenomena-Based Evaluation

### User Story

As an applied scientist, I want to evaluate specific events such as eddies, fronts, marine heatwaves, upwelling events, storms, or plumes, so that the model is judged by its ability to reproduce scientifically meaningful phenomena.

### Motivation

Many geoscience applications depend on identifiable finite-amplitude phenomena rather than continuous fields alone. A model may have low grid-level error while smoothing, displacing, delaying, weakening, or missing events. Object-based verification evaluates detection, spatial overlap, timing, intensity, duration, and lifecycle properties.

### Design

Separate event detection from event scoring:

```text
xr_toolz.phenomena       # event definitions, detection, labeling, matching, properties
xr_toolz.metrics.object  # contingency scores and object-property errors
xr_toolz.viz.validation  # event verification panels
```

### Demo API

```python
@dataclass
class EventDefinition:
    variable: str
    threshold: float | str
    baseline: xr.Dataset | None = None
    min_duration: int | None = None
    min_area: float | None = None
    connectivity: int = 8
    anomaly: bool = True

# Detection
def detect_anomaly_objects(ds, definition: EventDefinition) -> xr.Dataset: ...
def detect_marine_heatwaves(ds, *, sst_var="sst", climatology=None, percentile=90, min_duration=5) -> xr.Dataset: ...
def detect_eddies(ds, *, ssh_var="ssh", method="closed_contour", min_radius=None, min_lifetime=None) -> xr.Dataset: ...
def detect_fronts(ds, *, variable, gradient_threshold=None, min_length=None) -> xr.Dataset: ...
def label_objects(mask, *, dims=("lat", "lon"), connectivity=8) -> xr.Dataset: ...
def match_objects(objects_pred, objects_ref, *, method="iou", threshold=0.1) -> xr.Dataset: ...
def object_properties(objects, ds=None, *, variables=None) -> xr.Dataset: ...

# Object metrics
def contingency_table(matches) -> xr.Dataset: ...
def probability_of_detection(matches) -> xr.DataArray: ...
def false_alarm_ratio(matches) -> xr.DataArray: ...
def critical_success_index(matches) -> xr.DataArray: ...
def intersection_over_union(objects_pred, objects_ref) -> xr.DataArray: ...
def duration_error(matches) -> xr.DataArray: ...
def intensity_bias(matches, *, variable) -> xr.DataArray: ...
def centroid_distance(matches, *, dims=("lat", "lon")) -> xr.DataArray: ...
```

### Demo Example Usage

```python
from xr_toolz.phenomena import DetectMarineHeatwaves, MatchObjects
from xr_toolz.metrics.object import ProbabilityOfDetection, FalseAlarmRatio, CriticalSuccessIndex, IntersectionOverUnion

events_pred = DetectMarineHeatwaves(
    sst_var="sst",
    climatology=sst_climatology,
    percentile=90,
    min_duration=5,
)(ds_pred)

events_ref = DetectMarineHeatwaves(
    sst_var="sst",
    climatology=sst_climatology,
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

### Notes / Open Questions

- `xr_toolz.extremes` remains deferred to `xtremax`; `phenomena` is broader because it covers object/event definitions, detection, matching, and verification.
- Event definitions should be explicit and reusable so that prediction and reference fields are thresholded consistently.
- Object outputs should remain xarray-native and preserve event IDs, time bounds, geometry summaries, and matched-pair metadata.

---

## Validation Graph Pattern

### User Story

As an applied ML researcher, I want one graph to preprocess predictions and references, compute multiple diagnostics, and return both scores and figures.

### Motivation

Validation usually requires more than one score. A model run may need RMSE, spectral skill, lead-time skill, structural diagnostics, budget residuals, trajectory diagnostics, and event verification. The existing Graph API is the natural way to wire these multi-input and multi-output workflows.

### Demo API

```python
from xr_toolz.core import Graph, Input
from xr_toolz.metrics import RMSE, PSDScore
from xr_toolz.metrics.forecast import RMSEByLead
from xr_toolz.metrics.structural import SSIM
from xr_toolz.viz.validation import SpectralSkillPanel

pred = Input("prediction")
ref = Input("reference")

rmse = RMSE(variable="ssh", dims=("time", "lat", "lon"))(pred, ref)
psd = PSDScore(variable="ssh", dims=("lat", "lon"))(pred, ref)
lead = RMSEByLead(variable="ssh", dims=("lat", "lon"), lead_dim="lead_time")(pred, ref)
ssim = SSIM(variable="ssh", dims=("lat", "lon"))(pred, ref)
fig = SpectralSkillPanel(variable="ssh", dims=("lat", "lon"))(psd)

validation_graph = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={
        "rmse": rmse,
        "psd_score": psd,
        "lead_rmse": lead,
        "ssim": ssim,
        "spectral_panel": fig,
    },
)
```

### Demo Example Usage

```python
results = validation_graph(prediction=ds_pred, reference=ds_ref)

results["rmse"]
results["psd_score"]
results["lead_rmse"]
results["ssim"]
results["spectral_panel"]
```

---

## Recommended Module Additions

```text
xr_toolz.metrics.structural
xr_toolz.metrics.forecast
xr_toolz.metrics.probabilistic
xr_toolz.metrics.physical
xr_toolz.metrics.lagrangian
xr_toolz.metrics.object
xr_toolz.lagrangian
xr_toolz.budgets
xr_toolz.phenomena
xr_toolz.viz.validation
```

These additions should extend the existing API surface without removing the current examples or decisions.

---
status: draft
version: 0.2.0
---

!!! note "Module paths shown are proposed design targets"
    The snippets below import from `xr_toolz.lagrangian`, `xr_toolz.metrics.*`, and
    other submodules that **do not exist in the current export surface** — the
    current domain-agnostic functionality still lives under `xr_toolz.geo.*`.
    Treat these imports as design-target aliases; until the modules ship, map each
    `xr_toolz.<topic>` path to its equivalent under today's `xr_toolz.geo.<topic>`.

# Lagrangian Validation Examples

Lagrangian validation evaluates the material transport implied by predicted velocity fields. These examples complement Eulerian field metrics by focusing on trajectories, dispersion, residence times, and regional connectivity.

---

## Example 1: Particle Advection and Endpoint Error

### User Story

As a transport researcher, I want to compare particle trajectories under predicted and reference velocity fields, so that I can quantify accumulated pathway error.

### Motivation

Small Eulerian velocity errors can integrate into large Lagrangian trajectory errors. Endpoint error and trajectory RMSE provide direct diagnostics of transport realism.

### Demo API

```python
from xr_toolz.lagrangian import SeedParticles, AdvectParticles
from xr_toolz.metrics.lagrangian import EndpointError, TrajectoryRMSE
```

### Demo Example Usage

```python
particles = SeedParticles(strategy="grid", spacing=0.25, region=med_mask)(ds_ref)

traj_pred = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_pred, particles)
traj_ref = AdvectParticles(u_var="u", v_var="v", dt="1h", steps=240)(ds_ref, particles)

endpoint = EndpointError()(traj_pred, traj_ref)
trajectory_rmse = TrajectoryRMSE(dims=("particle", "time"))(traj_pred, traj_ref)
```

---

## Example 2: Pair Dispersion

### User Story

As a Lagrangian oceanographer, I want to compare pair dispersion statistics, so that I can evaluate whether the model reproduces scale-dependent spreading and mixing.

### Motivation

Pair dispersion measures how particle separations grow through time. It is sensitive to mesoscale and submesoscale transport errors that may be hidden in Eulerian RMSE.

### Demo API

```python
from xr_toolz.lagrangian import PairDispersion
from xr_toolz.metrics.lagrangian import DispersionError
```

### Demo Example Usage

```python
pair_pred = PairDispersion()(traj_pred)
pair_ref = PairDispersion()(traj_ref)

dispersion_error = DispersionError()(pair_pred, pair_ref)
```

---

## Example 3: Residence Time and Connectivity

### User Story

As a regional oceanographer, I want to compare residence times and connectivity matrices, so that I can evaluate whether a model reproduces exchange pathways between basins or coastal regions.

### Motivation

Transport applications often care about where water parcels go and how long they remain in a region. These diagnostics are especially relevant for pollutant transport, biogeochemistry, marine ecology, and regional ocean circulation.

### Demo API

```python
from xr_toolz.lagrangian import ResidenceTime, ConnectivityMatrix
from xr_toolz.metrics.lagrangian import ResidenceTimeError, ConnectivityError
```

### Demo Example Usage

```python
residence_pred = ResidenceTime(regions=basin_masks)(traj_pred)
residence_ref = ResidenceTime(regions=basin_masks)(traj_ref)
residence_error = ResidenceTimeError()(residence_pred, residence_ref)

connectivity_pred = ConnectivityMatrix(source_regions=basin_masks, target_regions=basin_masks)(traj_pred)
connectivity_ref = ConnectivityMatrix(source_regions=basin_masks, target_regions=basin_masks)(traj_ref)
connectivity_error = ConnectivityError()(connectivity_pred, connectivity_ref)
```

---

## Example 4: FTLE and Transport Barriers

### User Story

As a dynamical-systems researcher, I want to compute FTLE-like diagnostics from predicted velocity fields, so that I can compare stretching, mixing, and transport-barrier structure.

### Motivation

Finite-time stretching diagnostics reveal coherent transport geometry. They can identify dynamically meaningful structures that are not obvious from fixed-grid error fields.

### Demo API

```python
from xr_toolz.lagrangian import SeedParticles, FTLE
from xr_toolz.metrics import RMSE
```

### Demo Example Usage

```python
particles = SeedParticles(strategy="grid", spacing=0.1, region=domain_mask)(ds_ref)

ftle_pred = FTLE(integration_time="30D", u_var="u", v_var="v")(ds_pred, particles)
ftle_ref = FTLE(integration_time="30D", u_var="u", v_var="v")(ds_ref, particles)

ftle_error = RMSE(variable="ftle", dims=("lat", "lon"))(
    ftle_pred.to_dataset(name="ftle"),
    ftle_ref.to_dataset(name="ftle"),
)
```

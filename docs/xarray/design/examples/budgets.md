---
status: draft
version: 0.2.0
---

!!! note "Module paths shown are proposed design targets"
    The snippets below import from `xr_toolz.budgets`, `xr_toolz.metrics.*`, and other
    submodules that **do not exist in the current export surface** — the current
    domain-agnostic functionality still lives under `xr_toolz.geo.*`. Treat these
    imports as design-target aliases; once the modules ship the snippets become
    copy/paste-ready, but until then map each `xr_toolz.<topic>` path to its
    equivalent under today's `xr_toolz.geo.<topic>`.

# Budget Validation Examples

Budget validation checks whether predicted fields satisfy finite-volume conservation constraints. These diagnostics are especially useful for long autoregressive rollouts, coupled fields, and physically interpretable ML models.

---

## Example 1: Heat Budget Residual

### User Story

As a physical oceanographer, I want to know whether predicted temperature evolution is consistent with advection and surface heat fluxes, so that short-term field skill does not hide long-term heat drift.

### Motivation

A model can have good pointwise temperature skill while violating heat conservation. Budget residuals expose whether errors accumulate as physically meaningful drift.

### Demo API

```python
from xr_toolz.budgets import HeatBudgetResidual, ControlVolumeIntegral
```

### Demo Example Usage

```python
heat_residual = HeatBudgetResidual(
    temp_var="theta",
    u_var="u",
    v_var="v",
    surface_flux_var="qnet",
)(ds_pred)

regional_drift = ControlVolumeIntegral(
    variable="heat_budget_residual",
    volume_metrics=grid_metrics,
    region=med_mask,
)(heat_residual.to_dataset(name="heat_budget_residual"))
```

---

## Example 2: Salt Budget Residual

### User Story

As an ocean model evaluator, I want to check whether predicted salinity changes are consistent with transport and surface freshwater forcing, so that spurious water-mass modification is detected.

### Motivation

Salt budgets diagnose water-mass consistency and can reveal unphysical mixing or drift that is not captured by a single salinity RMSE.

### Demo API

```python
from xr_toolz.budgets import SaltBudgetResidual, ControlVolumeIntegral
```

### Demo Example Usage

```python
salt_residual = SaltBudgetResidual(
    salt_var="so",
    u_var="u",
    v_var="v",
    surface_flux_var="fwf",
)(ds_pred)

salt_drift = ControlVolumeIntegral(
    variable="salt_budget_residual",
    volume_metrics=grid_metrics,
    region=basin_mask,
)(salt_residual.to_dataset(name="salt_budget_residual"))
```

---

## Example 3: Volume Continuity Residual

### User Story

As a model developer, I want to check volume continuity, so that predicted velocity fields do not imply unphysical sources or sinks of water.

### Motivation

Continuity residuals diagnose mass/volume conservation and are a useful complement to velocity RMSE, especially for learned velocity fields.

### Demo API

```python
from xr_toolz.budgets import VolumeBudgetResidual
```

### Demo Example Usage

```python
volume_residual = VolumeBudgetResidual(
    u_var="u",
    v_var="v",
    w_var="w",
)(ds_pred)
```

---

## Example 4: Kinetic Energy Budget Residual

### User Story

As a dynamical oceanographer, I want to evaluate kinetic-energy production and dissipation consistency, so that model forecasts do not preserve low RMSE by introducing nonphysical energetics.

### Motivation

Energy diagnostics help determine whether the model reproduces realistic energy pathways, not merely visually plausible velocity snapshots.

### Demo API

```python
from xr_toolz.budgets import KineticEnergyBudgetResidual
from xr_toolz.viz.validation import ProcessBudgetPanel
```

### Demo Example Usage

```python
ke_residual = KineticEnergyBudgetResidual(
    u_var="u",
    v_var="v",
    forcing_vars=["wind_stress_x", "wind_stress_y"],
)(ds_pred)

fig = ProcessBudgetPanel(budget_var="ke_budget_residual")(
    ke_residual.to_dataset(name="ke_budget_residual")
)
```

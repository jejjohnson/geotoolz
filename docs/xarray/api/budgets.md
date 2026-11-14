# Budgets

Conservation-budget diagnostics. For a tracer $\theta$ with flux
$\mathbf{F}$ and source $Q$, the budget residual measures how far a dataset
departs from exact conservation:

$$
r = \frac{\partial\theta}{\partial t} + \nabla\cdot\mathbf{F} - Q.
$$

`ControlVolumeIntegral` and `BoundaryFlux` assemble the volume and surface
terms; the tracer-specific residuals specialise $\theta$ to heat, salt,
volume, or kinetic energy.

## Building blocks

::: xrops.budgets.operators.ControlVolumeIntegral

::: xrops.budgets.operators.BoundaryFlux

::: xrops.budgets.operators.BudgetResidual

## Tracer-specific residuals

::: xrops.budgets.operators.HeatBudgetResidual

::: xrops.budgets.operators.SaltBudgetResidual

::: xrops.budgets.operators.VolumeBudgetResidual

::: xrops.budgets.operators.KineticEnergyBudgetResidual

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.budgets.control_volume_integral

::: xrops.budgets.boundary_flux

::: xrops.budgets.budget_residual

::: xrops.budgets.heat_budget_residual

::: xrops.budgets.salt_budget_residual

::: xrops.budgets.volume_budget_residual

::: xrops.budgets.kinetic_energy_budget_residual

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

::: xrtoolz.budgets.operators.ControlVolumeIntegral

::: xrtoolz.budgets.operators.BoundaryFlux

::: xrtoolz.budgets.operators.BudgetResidual

## Tracer-specific residuals

::: xrtoolz.budgets.operators.HeatBudgetResidual

::: xrtoolz.budgets.operators.SaltBudgetResidual

::: xrtoolz.budgets.operators.VolumeBudgetResidual

::: xrtoolz.budgets.operators.KineticEnergyBudgetResidual

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.budgets.control_volume_integral

::: xrtoolz.budgets.boundary_flux

::: xrtoolz.budgets.budget_residual

::: xrtoolz.budgets.heat_budget_residual

::: xrtoolz.budgets.salt_budget_residual

::: xrtoolz.budgets.volume_budget_residual

::: xrtoolz.budgets.kinetic_energy_budget_residual

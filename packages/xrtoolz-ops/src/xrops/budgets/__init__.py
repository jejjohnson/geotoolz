"""Conservation-budget diagnostics — V4.2 / V4.3.

Two layers, mirroring the rest of xrops:

- **Layer 0** — pure-function primitives:
  :func:`control_volume_integral`, :func:`boundary_flux`,
  :func:`budget_residual`, plus the tracer-/momentum-specific
  reductions :func:`heat_budget_residual`,
  :func:`salt_budget_residual`, :func:`volume_budget_residual`,
  :func:`kinetic_energy_budget_residual`.
- **Layer 1** — :class:`Operator` wrappers under
  :mod:`xrops.budgets.operators`.

**Control-volume integrals** (:class:`ControlVolumeIntegral`) take explicit
``volume_metrics``, never auto-derived (V4.4 / D16). Use
:func:`xrops.grad.grid_metrics_from_coords` to build them from a Dataset's
coords if the model output does not already ship them. **Boundary fluxes**
(:class:`BoundaryFlux`) take a region, and the faces of the grid that the
coordinates describe.

**Per-cell residual operators** (:class:`HeatBudgetResidual`,
:class:`SaltBudgetResidual`, :class:`VolumeBudgetResidual`,
:class:`KineticEnergyBudgetResidual`) return a per-cell residual field
``∂φ/∂t + ∇·(u φ) − sources``. They use the flux through the faces of
each cell (``diagnostix.budgets.flux_divergence``), on the sphere that
the coordinates describe — they do **not** consume the
``volume_metrics`` / ``face_metrics`` Datasets. The edge cells of the
domain, which have a face outside the data, are NaN. Multiply the residual
by ``cell_volume`` and integrate with :class:`ControlVolumeIntegral`
if you need a control-volume-integrated closure.
"""

from xrops.budgets._src.flux import boundary_flux
from xrops.budgets._src.heat import heat_budget_residual
from xrops.budgets._src.ke import kinetic_energy_budget_residual
from xrops.budgets._src.residual import budget_residual
from xrops.budgets._src.salt import salt_budget_residual
from xrops.budgets._src.volume import control_volume_integral
from xrops.budgets._src.volume_budget import volume_budget_residual
from xrops.budgets.operators import (
    BoundaryFlux,
    BudgetResidual,
    ControlVolumeIntegral,
    HeatBudgetResidual,
    KineticEnergyBudgetResidual,
    SaltBudgetResidual,
    VolumeBudgetResidual,
)


__all__ = [
    "BoundaryFlux",
    "BudgetResidual",
    "ControlVolumeIntegral",
    "HeatBudgetResidual",
    "KineticEnergyBudgetResidual",
    "SaltBudgetResidual",
    "VolumeBudgetResidual",
    "boundary_flux",
    "budget_residual",
    "control_volume_integral",
    "heat_budget_residual",
    "kinetic_energy_budget_residual",
    "salt_budget_residual",
    "volume_budget_residual",
]

"""Oceanography physics operators.

Layer-0 primitives are implemented in :mod:`xr_toolz.ocn._src` and
re-exported here.

Content:

- Geostrophic quantities: Coriolis parameter, stream function,
  geostrophic velocities, kinetic energy, relative / absolute vorticity,
  divergence, enstrophy, strain components, Okubo–Weiss parameter.
- SSH composition from altimetry products (``calculate_ssh_alongtrack``).
- Variable attribute harmonization (``validate_ssh``, ``validate_velocity``).

All finite-differencing of lon/lat fields goes through :mod:`metpy.calc`,
which handles the unit / distance conversion under the hood.
"""

from xr_toolz.ocn._src.geostrophic import (
    absolute_vorticity,
    coriolis_normalized,
    coriolis_parameter,
    divergence,
    enstrophy,
    geostrophic_velocities,
    kinetic_energy,
    okubo_weiss,
    relative_vorticity,
    shear_strain,
    strain_magnitude,
    streamfunction,
    tensor_strain,
)
from xr_toolz.ocn._src.ssh import (
    calculate_ssh_alongtrack,
    calculate_ssh_unfiltered,
)
from xr_toolz.ocn._src.validation import (
    validate_ssh,
    validate_velocity,
)


__all__ = [
    "absolute_vorticity",
    "calculate_ssh_alongtrack",
    "calculate_ssh_unfiltered",
    "coriolis_normalized",
    "coriolis_parameter",
    "divergence",
    "enstrophy",
    "geostrophic_velocities",
    "kinetic_energy",
    "okubo_weiss",
    "relative_vorticity",
    "shear_strain",
    "strain_magnitude",
    "streamfunction",
    "tensor_strain",
    "validate_ssh",
    "validate_velocity",
]

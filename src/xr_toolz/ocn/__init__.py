"""Oceanography physics operators.

Layer-0 primitives are implemented in :mod:`xr_toolz.ocn._src` and
re-exported here.

Content:

- Kinematic and geostrophic quantities: Coriolis parameter, stream
  function, geostrophic / ageostrophic velocities, (eddy) kinetic
  energy, velocity magnitudes, relative / absolute / shear /
  curvature vorticity, divergence, enstrophy, strain components,
  Okubo–Weiss parameter, tracer advection, frontogenesis,
  barotropic potential vorticity, Brunt–Väisälä frequency, mixed
  layer depth.
- SSH composition from altimetry products (``calculate_ssh_alongtrack``).
- Variable attribute harmonization (``validate_ssh``, ``validate_velocity``).

All finite-differencing of lon/lat fields goes through
:mod:`xr_toolz.calc`, which converts lon/lat (degrees) to metric
``∂/∂x`` and ``∂/∂y`` and applies the spherical curvature corrections.
"""

from xr_toolz.ocn._src.kinematics import (
    absolute_vorticity,
    advection,
    ageostrophic_velocities,
    brunt_vaisala_frequency,
    coriolis_normalized,
    coriolis_parameter,
    curvature_vorticity,
    divergence,
    eddy_kinetic_energy,
    enstrophy,
    frontogenesis,
    geostrophic_velocities,
    horizontal_velocity_magnitude,
    kinetic_energy,
    lapse_rate,
    mixed_layer_depth,
    okubo_weiss,
    potential_vorticity_barotropic,
    relative_vorticity,
    shear_strain,
    shear_vorticity,
    strain_magnitude,
    streamfunction,
    tensor_strain,
    velocity_magnitude,
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
    "advection",
    "ageostrophic_velocities",
    "brunt_vaisala_frequency",
    "calculate_ssh_alongtrack",
    "calculate_ssh_unfiltered",
    "coriolis_normalized",
    "coriolis_parameter",
    "curvature_vorticity",
    "divergence",
    "eddy_kinetic_energy",
    "enstrophy",
    "frontogenesis",
    "geostrophic_velocities",
    "horizontal_velocity_magnitude",
    "kinetic_energy",
    "lapse_rate",
    "mixed_layer_depth",
    "okubo_weiss",
    "potential_vorticity_barotropic",
    "relative_vorticity",
    "shear_strain",
    "shear_vorticity",
    "strain_magnitude",
    "streamfunction",
    "tensor_strain",
    "validate_ssh",
    "validate_velocity",
    "velocity_magnitude",
]

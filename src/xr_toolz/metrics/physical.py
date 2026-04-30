"""V4.1 physical-consistency metrics — re-exports from ``_src.physical``.

See :mod:`xr_toolz.metrics._src.physical` for full documentation.
"""

from xr_toolz.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    PVConservationError,
    density_inversion_fraction,
    divergence_error,
    geostrophic_balance_error,
    pv_conservation_error,
)


__all__ = [
    "DensityInversionFraction",
    "DivergenceError",
    "GeostrophicBalanceError",
    "PVConservationError",
    "density_inversion_fraction",
    "divergence_error",
    "geostrophic_balance_error",
    "pv_conservation_error",
]

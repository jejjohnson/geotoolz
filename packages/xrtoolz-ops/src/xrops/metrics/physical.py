"""V4.1 physical-consistency metrics — re-exports from ``_src.physical``.

See :mod:`xrops.metrics._src.physical` for full documentation.
"""

from xrops.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    GeostrophicImbalance,
    PVConservationError,
    density_inversion_fraction,
    divergence_error,
    geostrophic_balance_error,
    geostrophic_imbalance,
    pv_conservation_error,
)


__all__ = [
    "DensityInversionFraction",
    "DivergenceError",
    "GeostrophicBalanceError",
    "GeostrophicImbalance",
    "PVConservationError",
    "density_inversion_fraction",
    "divergence_error",
    "geostrophic_balance_error",
    "geostrophic_imbalance",
    "pv_conservation_error",
]

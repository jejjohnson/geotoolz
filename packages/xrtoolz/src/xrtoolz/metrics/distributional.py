"""Distributional evaluation metrics — public re-export.

Layer-0 functions: :func:`crps_ensemble`, :func:`energy_distance`,
:func:`wasserstein_1`.

Layer-1 operators: :class:`CRPS`, :class:`EnergyDistance`,
:class:`Wasserstein1`.

Implementation lives in :mod:`xrtoolz.metrics._src.distributional`.
"""

from xrtoolz.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
    crps_ensemble,
    energy_distance,
    wasserstein_1,
)


__all__ = [
    "CRPS",
    "EnergyDistance",
    "Wasserstein1",
    "crps_ensemble",
    "energy_distance",
    "wasserstein_1",
]

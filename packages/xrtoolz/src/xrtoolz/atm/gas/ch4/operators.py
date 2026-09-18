"""Layer-1 ``Operator`` wrappers around :mod:`xrtoolz.atm.gas.ch4._src`.

Each operator captures the variable / dimension names in ``__init__`` and
forwards the dataset to the pure-function primitive in ``_apply``. See
:mod:`xrtoolz.atm.gas.ch4._src.column` for the underlying implementations.
"""

from __future__ import annotations

from typing import Any

from xrcore import Operator
from xrtoolz.atm.gas.ch4._src import column as _column


class ApplyColumnAveragingKernel(Operator):
    """Smooth a model profile with a satellite column averaging kernel.

    ``ŷ = hᵀ x_a + Σ_l h_l A_l (x_l − x_{a,l})``. See
    :func:`xrtoolz.atm.gas.ch4.apply_column_averaging_kernel`.

    Args:
        profile: Model profile ``x`` per layer.
        prior: Retrieval prior ``x_a`` per layer.
        averaging_kernel: Column averaging kernel ``A`` per layer.
        pressure_weights: Pressure weights ``h`` per layer.
        level: Layer dimension.
        name: Name of the output variable.

    Returns:
        The input dataset with ``name`` added (``level`` reduced away).
    """

    def __init__(
        self,
        profile: str,
        prior: str,
        averaging_kernel: str,
        pressure_weights: str,
        level: str = "layer",
        name: str = "xch4_smoothed",
    ):
        self.profile = profile
        self.prior = prior
        self.averaging_kernel = averaging_kernel
        self.pressure_weights = pressure_weights
        self.level = level
        self.name = name

    def _apply(self, ds):
        return _column.apply_column_averaging_kernel(
            ds,
            profile=self.profile,
            prior=self.prior,
            averaging_kernel=self.averaging_kernel,
            pressure_weights=self.pressure_weights,
            level=self.level,
            name=self.name,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "prior": self.prior,
            "averaging_kernel": self.averaging_kernel,
            "pressure_weights": self.pressure_weights,
            "level": self.level,
            "name": self.name,
        }


class DryAirColumn(Operator):
    """Dry-air column number density ``N_A (p_s / g − W) / M_dry``.

    See :func:`xrtoolz.atm.gas.ch4.dry_air_column`.

    Args:
        surface_pressure: Surface pressure [Pa].
        water_vapour_column: Optional total column water vapour [kg m⁻²].
        name: Name of the output variable.

    Returns:
        The input dataset with ``name`` added [molecules m⁻²].
    """

    def __init__(
        self,
        surface_pressure: str = "sp",
        water_vapour_column: str | None = None,
        name: str = "dry_air_column",
    ):
        self.surface_pressure = surface_pressure
        self.water_vapour_column = water_vapour_column
        self.name = name

    def _apply(self, ds):
        return _column.dry_air_column(
            ds,
            surface_pressure=self.surface_pressure,
            water_vapour_column=self.water_vapour_column,
            name=self.name,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "surface_pressure": self.surface_pressure,
            "water_vapour_column": self.water_vapour_column,
            "name": self.name,
        }


class MixingRatioToColumn(Operator):
    """Column number density ``Ω = N_A / (g M_dry) ∫ χ (1 − q) dp``.

    See :func:`xrtoolz.atm.gas.ch4.mixing_ratio_to_column`.

    Args:
        vmr: Dry-air mole fraction per level.
        pressure: Pressure per level [Pa].
        specific_humidity: Optional specific humidity per level.
        level: Vertical dimension.
        name: Name of the output variable.

    Returns:
        The input dataset with ``name`` added (``level`` reduced away).
    """

    def __init__(
        self,
        vmr: str,
        pressure: str = "pressure",
        specific_humidity: str | None = None,
        level: str = "level",
        name: str = "column",
    ):
        self.vmr = vmr
        self.pressure = pressure
        self.specific_humidity = specific_humidity
        self.level = level
        self.name = name

    def _apply(self, ds):
        return _column.mixing_ratio_to_column(
            ds,
            vmr=self.vmr,
            pressure=self.pressure,
            specific_humidity=self.specific_humidity,
            level=self.level,
            name=self.name,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "vmr": self.vmr,
            "pressure": self.pressure,
            "specific_humidity": self.specific_humidity,
            "level": self.level,
            "name": self.name,
        }

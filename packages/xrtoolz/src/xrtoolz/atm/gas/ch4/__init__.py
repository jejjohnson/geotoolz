"""Methane-specific physics operators.

Layer-0 primitives are implemented in :mod:`xrtoolz.atm.gas.ch4._src`
and re-exported here.

Content:

- ``apply_column_averaging_kernel``: TROPOMI / OCO-style smoothing of a
  model profile with a column averaging kernel and pressure weights.
- ``dry_air_column``: dry-air column number density from surface
  pressure (and optional total column water vapour).
- ``mixing_ratio_to_column``: dry-air mole-fraction profile → column
  number density; ``column_mass_to_delta_vmr``: column mass
  enhancement → mixing-ratio enhancement over a well-mixed layer.
- Layer-1 ``Operator`` wrappers (:mod:`xrtoolz.atm.gas.ch4.operators`),
  all re-exported here.
"""

from xrtoolz.atm.gas.ch4._src.column import (
    apply_column_averaging_kernel,
    column_mass_to_delta_vmr,
    dry_air_column,
    mixing_ratio_to_column,
)
from xrtoolz.atm.gas.ch4.operators import (
    ApplyColumnAveragingKernel,
    DryAirColumn,
    MixingRatioToColumn,
)


__all__ = [
    "ApplyColumnAveragingKernel",
    "DryAirColumn",
    "MixingRatioToColumn",
    "apply_column_averaging_kernel",
    "column_mass_to_delta_vmr",
    "dry_air_column",
    "mixing_ratio_to_column",
]

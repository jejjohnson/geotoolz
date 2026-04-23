"""Shared typed primitives for ``xr_toolz``.

These types are deliberately agnostic of any particular submodule
(``data``, ``geo``, ``ocn``, ``atm``, ...). They exist so every module
speaks the same language for things like "a lon/lat bounding box",
"a time window", "a CF variable".

Exports:

- Spatial: :class:`BBox`.
- Temporal: :class:`TimeRange`.
- Vertical: :class:`DepthRange`, :class:`PressureLevels`.
- Request: :class:`Request` (typed composition of the above).
- Variables: :class:`Variable`, :data:`REGISTRY`, :func:`resolve`,
  :func:`register`.
- Validation: :func:`validate_variable`, :func:`validate_dataset`,
  :func:`apply_cf_attrs`, :class:`ValidationReport`, :class:`Issue`,
  :class:`Severity`.
"""

from xr_toolz.types._src.geometry import BBox
from xr_toolz.types._src.levels import DepthRange, PressureLevels
from xr_toolz.types._src.request import Request
from xr_toolz.types._src.time import TimeRange
from xr_toolz.types._src.validation import (
    Issue,
    Severity,
    ValidationReport,
    apply_cf_attrs,
    validate_dataset,
    validate_variable,
)
from xr_toolz.types._src.variable import (
    ADT,
    BBP443,
    CHL,
    D2M,
    DENS,
    ICE_CONC,
    KD490,
    MDT,
    MSL,
    NO3,
    O2,
    PH,
    PHYC,
    PO4,
    PP,
    REGISTRY,
    RRS412,
    RRS443,
    RRS490,
    RRS510,
    RRS555,
    RRS670,
    SI,
    SLA,
    SO,
    SOS,
    SP,
    SPCO2,
    SPM,
    SSH,
    SSRD,
    SST,
    T2M,
    TP,
    U10,
    UGOS,
    UO,
    V10,
    VGOS,
    VO,
    ZOOC,
    ZSD,
    Variable,
    register,
    resolve,
)


__all__ = [
    "ADT",
    "BBP443",
    "CHL",
    "D2M",
    "DENS",
    "ICE_CONC",
    "KD490",
    "MDT",
    "MSL",
    "NO3",
    "O2",
    "PH",
    "PHYC",
    "PO4",
    "PP",
    "REGISTRY",
    "RRS412",
    "RRS443",
    "RRS490",
    "RRS510",
    "RRS555",
    "RRS670",
    "SI",
    "SLA",
    "SO",
    "SOS",
    "SP",
    "SPCO2",
    "SPM",
    "SSH",
    "SSRD",
    "SST",
    "T2M",
    "TP",
    "U10",
    "UGOS",
    "UO",
    "V10",
    "VGOS",
    "VO",
    "ZOOC",
    "ZSD",
    "BBox",
    "DepthRange",
    "Issue",
    "PressureLevels",
    "Request",
    "Severity",
    "TimeRange",
    "ValidationReport",
    "Variable",
    "apply_cf_attrs",
    "register",
    "resolve",
    "validate_dataset",
    "validate_variable",
]

"""GOES-R ABI band table, quality-flag meanings and orbital constants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoproducts._src.constants import load_csv


# ABI channel names, as they appear in the L1b file names (``...-M6C13_...``).
CHANNELS: tuple[str, ...] = tuple(f"C{n:02d}" for n in range(1, 17))
REFLECTIVE_CHANNELS: tuple[str, ...] = CHANNELS[:6]
EMISSIVE_CHANNELS: tuple[str, ...] = CHANNELS[6:]

# ``DQF`` values of an L1b radiance file (PUG Vol. 3, ``flag_meanings``).
DQF_GOOD = 0
DQF_CONDITIONALLY_USABLE = 1
DQF_OUT_OF_RANGE = 2
DQF_NO_VALUE = 3
DQF_FOCAL_PLANE_TEMPERATURE_EXCEEDED = 4
DQF_FLAGS: dict[int, str] = {
    DQF_GOOD: "good",
    DQF_CONDITIONALLY_USABLE: "conditionally_usable",
    DQF_OUT_OF_RANGE: "out_of_range",
    DQF_NO_VALUE: "no_value",
    DQF_FOCAL_PLANE_TEMPERATURE_EXCEEDED: "focal_plane_temperature_exceeded",
}
# Fill of the uint8 quality layer outside the file's extent.
DQF_FILL = 255

# Operational slots (sub-satellite longitude, degrees east). Each L1b file
# carries its own ``nominal_satellite_subpoint_lon``; prefer
# ``Reader.satellite_lon_deg`` when a file is at hand.
GOES_EAST_LON_DEG = -75.2
GOES_WEST_LON_DEG = -137.0
# Nominal geostationary height above the equator (metres).
SATELLITE_HEIGHT_M = 35_786_023.0

# CIMSS synthetic-green recipe for ABI true colour (ABI has no green band).
SYNTHETIC_GREEN_WEIGHTS: dict[str, float] = {"C02": 0.45, "C03": 0.10, "C01": 0.45}

_CACHE: dict[str, Any] = {}

if TYPE_CHECKING:
    BANDS: tuple[dict[str, str], ...]


def __getattr__(name: str) -> Any:
    if name == "BANDS":
        if name not in _CACHE:
            _CACHE[name] = load_csv(__package__, "data/bands.csv")
        return _CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BANDS",
    "CHANNELS",
    "DQF_CONDITIONALLY_USABLE",
    "DQF_FILL",
    "DQF_FLAGS",
    "DQF_FOCAL_PLANE_TEMPERATURE_EXCEEDED",
    "DQF_GOOD",
    "DQF_NO_VALUE",
    "DQF_OUT_OF_RANGE",
    "EMISSIVE_CHANNELS",
    "GOES_EAST_LON_DEG",
    "GOES_WEST_LON_DEG",
    "REFLECTIVE_CHANNELS",
    "SATELLITE_HEIGHT_M",
    "SYNTHETIC_GREEN_WEIGHTS",
]

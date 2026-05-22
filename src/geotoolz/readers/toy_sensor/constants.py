"""Toy-sensor calibration constants and lazy table accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geotoolz.readers._constants import load_csv


BAND_BLUE = "blue"
BAND_GREEN = "green"
BAND_RED = "red"
BAND_NIR = "nir"
_CACHE: dict[str, Any] = {}

if TYPE_CHECKING:
    BANDS: tuple[dict[str, str], ...]
    CONSTANTS: dict[str, object]


def __getattr__(name: str) -> Any:
    if name == "BANDS":
        if name not in _CACHE:
            _CACHE[name] = load_csv(__package__, "data/bands.csv")
        return _CACHE[name]
    if name == "CONSTANTS":
        if name not in _CACHE:
            _CACHE[name] = {
                "solar_irradiance": load_csv(__package__, "data/solar_irradiance.csv")
            }
        return _CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BANDS",
    "BAND_BLUE",
    "BAND_GREEN",
    "BAND_NIR",
    "BAND_RED",
    "CONSTANTS",
]

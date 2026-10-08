"""toy reference reader namespace."""

from __future__ import annotations

from typing import Any

from geoproducts.toy_sensor import constants, presets
from geoproducts.toy_sensor.presets import NDVI
from geoproducts.toy_sensor.reader import Reader


def __getattr__(name: str) -> Any:
    if name in {"BANDS", "CONSTANTS"}:
        return getattr(constants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BANDS", "CONSTANTS", "NDVI", "Reader", "constants", "presets"]

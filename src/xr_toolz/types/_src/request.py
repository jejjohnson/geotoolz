"""Canonical request payload composed of the typed primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xr_toolz.types._src.geometry import BBox
from xr_toolz.types._src.levels import DepthRange, PressureLevels
from xr_toolz.types._src.time import TimeRange


@dataclass(frozen=True)
class Request:
    """Canonical request payload consumed by data adapters.

    Not every field applies to every source; adapters pick what they
    need. Keeping one struct means the catalog and the UI speak the
    same shape.
    """

    variables: tuple[str, ...]
    bbox: BBox | None = None
    time: TimeRange | None = None
    depth: DepthRange | None = None
    levels: PressureLevels | None = None
    extras: dict[str, Any] | None = None

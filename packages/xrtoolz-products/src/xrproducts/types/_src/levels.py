"""Vertical coordinate types: continuous depth, discrete pressure levels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthRange:
    """Continuous depth window (positive-down, metres)."""

    min: float
    max: float

    def __post_init__(self) -> None:
        if self.min < 0 or self.max < 0:
            raise ValueError(f"depth must be >= 0 m, got ({self.min}, {self.max})")
        if self.min > self.max:
            raise ValueError(f"depth min ({self.min}) must be <= max ({self.max})")

    def as_cmems(self) -> dict[str, float]:
        return {"minimum_depth": self.min, "maximum_depth": self.max}


@dataclass(frozen=True)
class PressureLevels:
    """Discrete pressure levels in hPa (CDS-style)."""

    levels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("PressureLevels must contain at least one level")
        if any(level <= 0 for level in self.levels):
            raise ValueError(f"pressure levels must be positive hPa: {self.levels}")

    def as_cds_form(self) -> list[str]:
        return [str(level) for level in self.levels]

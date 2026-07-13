"""Spatial matchup strategies.

A `SpatialStrategy` decides "does the secondary row spatially match
the primary?" given two footprints. Strategies are dataclasses so
they round-trip through the persisted ``matchups.parquet`` table's
``tolerance_json`` column without bespoke serialization code.

All bodies operate over `shapely` geometries; the matchup engine
pre-filters candidates via STRtree (envelope overlap) before calling
``match()``, so a strategy's job is the precise truth gate.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    import shapely.geometry.base


@runtime_checkable
class SpatialStrategy(Protocol):
    """A predicate over two footprints.

    The engine calls ``match(primary, secondary)`` for each candidate
    pair after the STRtree pre-filter narrows by envelope overlap.
    """

    def match(
        self,
        primary: shapely.geometry.base.BaseGeometry,
        secondary: shapely.geometry.base.BaseGeometry,
    ) -> bool: ...


@dataclasses.dataclass(frozen=True)
class Intersects:
    """Non-zero geometric intersection (the cheapest predicate).

    Matches any pair whose footprints share at least one point.
    Use when the matchup definition is "any overlap counts" — a
    common choice for opportunistic fusion where partial coverage
    is still useful.
    """

    def match(
        self,
        primary: shapely.geometry.base.BaseGeometry,
        secondary: shapely.geometry.base.BaseGeometry,
    ) -> bool:
        return primary.intersects(secondary)


@dataclasses.dataclass(frozen=True)
class IouAtLeast:
    """Intersection-over-union ≥ ``threshold``.

    Stricter than plain intersection — useful when you want the
    secondary to *substantially* overlap the primary (e.g. for
    training-data quality gates where a sliver-overlap is noise).

    Args:
        threshold: Minimum IoU in ``[0, 1]``.

    Notes:
        IoU on raw geographic CRS coordinates is degree-area, which
        is fine as a ratio (numerator and denominator scale the same
        way) but loses meaning if you compare across latitudes. For
        equal-area comparisons project both inputs to a meter CRS
        first.
    """

    threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"IouAtLeast.threshold must be in [0, 1]; got {self.threshold!r}"
            )

    def match(
        self,
        primary: shapely.geometry.base.BaseGeometry,
        secondary: shapely.geometry.base.BaseGeometry,
    ) -> bool:
        intersection = primary.intersection(secondary)
        if intersection.is_empty:
            return False
        union = primary.union(secondary)
        if union.area == 0.0:
            # Degenerate: both are zero-area (e.g. two points or
            # crossing LineStrings). Area-based IoU is undefined
            # for these — treat them as "matching" only if the
            # geometries are equal, otherwise fail rather than
            # silently ignoring the threshold.
            return primary.equals(secondary)
        return (intersection.area / union.area) >= self.threshold


@dataclasses.dataclass(frozen=True)
class CentroidWithin:
    """Secondary centroid falls within a buffered primary footprint.

    Useful when matching against point-like secondaries (in-situ
    stations, drifters) where a buffered-polygon test is more
    forgiving than strict polygon containment.

    Args:
        buffer: Distance (in the geometry's CRS units) by which the
            primary footprint is enlarged before the point-in-polygon
            test. ``0.0`` = strict centroid-in-polygon; larger values
            tolerate near-misses. String-with-units forms
            (e.g. ``"5km"``) are reserved for a future revision —
            for now use floats and reproject upfront if needed.
    """

    buffer: float | str

    def match(
        self,
        primary: shapely.geometry.base.BaseGeometry,
        secondary: shapely.geometry.base.BaseGeometry,
    ) -> bool:
        if isinstance(self.buffer, str):
            raise NotImplementedError(
                "String-with-units buffer (e.g. '5km') is not yet "
                "supported. Pass a float in CRS units, or reproject "
                "the inputs into a meter-based CRS upfront."
            )
        buffered = primary.buffer(self.buffer) if self.buffer > 0 else primary
        return buffered.contains(secondary.centroid)


@dataclasses.dataclass(frozen=True)
class Contains:
    """Secondary footprint is fully contained in the primary.

    The strictest spatial predicate — use when you need the
    secondary to be a true subset of the primary's footprint
    (e.g. a station tile inside an L2 swath).
    """

    def match(
        self,
        primary: shapely.geometry.base.BaseGeometry,
        secondary: shapely.geometry.base.BaseGeometry,
    ) -> bool:
        return primary.contains(secondary)

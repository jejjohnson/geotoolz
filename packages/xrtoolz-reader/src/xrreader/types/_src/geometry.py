"""Spatial geometry types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned lon/lat bounding box.

    Longitude is accepted in either ``[-180, 180]`` or ``[0, 360]``
    convention; use :meth:`to_180` / :meth:`to_360` to normalize. A
    ``lon_min > lon_max`` box is interpreted as crossing the
    antimeridian.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_min <= 90.0:
            raise ValueError(f"lat_min out of [-90, 90]: {self.lat_min}")
        if not -90.0 <= self.lat_max <= 90.0:
            raise ValueError(f"lat_max out of [-90, 90]: {self.lat_max}")
        if self.lat_min > self.lat_max:
            raise ValueError(
                f"lat_min ({self.lat_min}) must be <= lat_max ({self.lat_max})"
            )

    # ---- constructors -----------------------------------------------------

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> BBox:
        """Build from ``(lon_min, lon_max, lat_min, lat_max)``."""
        lon_min, lon_max, lat_min, lat_max = t
        return cls(lon_min, lon_max, lat_min, lat_max)

    @classmethod
    def global_(cls) -> BBox:
        """Whole-globe bounding box in ``[-180, 180]`` convention."""
        return cls(-180.0, 180.0, -90.0, 90.0)

    # ---- derived ----------------------------------------------------------

    @property
    def crosses_antimeridian(self) -> bool:
        return self.lon_min > self.lon_max

    @property
    def width(self) -> float:
        """Longitude extent (handles antimeridian wrap)."""
        if self.crosses_antimeridian:
            return (360.0 - self.lon_min) + self.lon_max
        return self.lon_max - self.lon_min

    @property
    def height(self) -> float:
        """Latitude extent in degrees."""
        return self.lat_max - self.lat_min

    # ---- normalization ----------------------------------------------------

    def to_180(self) -> BBox:
        """Return a copy with longitudes normalized to ``[-180, 180]``."""

        def _w(x: float) -> float:
            return ((x + 180.0) % 360.0) - 180.0

        # A full-globe box has identical endpoints under modulo (both map
        # to -180); preserve the canonical span instead of collapsing it.
        if self.lon_max - self.lon_min == 360.0:
            return BBox(-180.0, 180.0, self.lat_min, self.lat_max)
        return BBox(_w(self.lon_min), _w(self.lon_max), self.lat_min, self.lat_max)

    def to_360(self) -> BBox:
        """Return a copy with longitudes normalized to ``[0, 360]``."""
        # ``-180 % 360 == 180 == 180 % 360`` would collapse a full-globe box
        # to a single meridian; preserve the canonical span instead.
        if self.lon_max - self.lon_min == 360.0:
            return BBox(0.0, 360.0, self.lat_min, self.lat_max)
        return BBox(
            self.lon_min % 360.0, self.lon_max % 360.0, self.lat_min, self.lat_max
        )

    # ---- serializers ------------------------------------------------------

    def as_cmems(self) -> dict[str, float]:
        """``copernicusmarine.subset`` kwargs.

        Raises:
            ValueError: if the box crosses the antimeridian — passing
                ``lon_min > lon_max`` would send CMEMS an inverted window.
                Call :meth:`to_360` first, or split the box at ±180°.
        """
        if self.crosses_antimeridian:
            raise ValueError(
                "BBox crosses the antimeridian; CMEMS would receive an "
                "inverted minimum/maximum longitude. Call .to_360() first "
                "or split the box at ±180° and request each half."
            )
        return {
            "minimum_longitude": self.lon_min,
            "maximum_longitude": self.lon_max,
            "minimum_latitude": self.lat_min,
            "maximum_latitude": self.lat_max,
        }

    def as_cds_area(self) -> list[float]:
        """CDS ``area`` key: ``[North, West, South, East]``.

        Raises:
            ValueError: if the box crosses the antimeridian — ``West > East``
                is an inverted window. Call :meth:`to_360` first, or split
                the box at ±180°.
        """
        if self.crosses_antimeridian:
            raise ValueError(
                "BBox crosses the antimeridian; CDS would receive an inverted "
                "West/East window. Call .to_360() first or split the box at "
                "±180° and request each half."
            )
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]

    def as_xarray_sel(
        self,
        lon: str = "lon",
        lat: str = "lat",
        lat_descending: bool = False,
    ) -> dict[str, slice]:
        """``ds.sel(**bbox.as_xarray_sel())`` selector.

        Args:
            lon: Name of the longitude coordinate.
            lat: Name of the latitude coordinate.
            lat_descending: Set when the dataset's latitude index runs
                north-to-south (ERA5 and most CDS reanalysis output do).
                ``sel`` requires slice bounds in index order, so an
                ascending ``slice(lat_min, lat_max)`` selects **zero**
                points on such a grid. Infer it with
                ``bool(ds[lat][0] > ds[lat][-1])``.

        Raises:
            ValueError: if the box crosses the antimeridian — a single
                ``slice(lon_min, lon_max)`` would select zero points in
                that case. Split the box at ±180° and select the halves
                separately, or call :meth:`to_360` first.
        """
        if self.crosses_antimeridian:
            raise ValueError(
                "BBox crosses the antimeridian; a single slice cannot "
                "represent the wrap. Split the box at ±180° and sel() "
                "each half, or call .to_360() first."
            )
        lat_slice = (
            slice(self.lat_max, self.lat_min)
            if lat_descending
            else slice(self.lat_min, self.lat_max)
        )
        return {
            lon: slice(self.lon_min, self.lon_max),
            lat: lat_slice,
        }

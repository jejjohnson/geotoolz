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

        return BBox(_w(self.lon_min), _w(self.lon_max), self.lat_min, self.lat_max)

    def to_360(self) -> BBox:
        """Return a copy with longitudes normalized to ``[0, 360]``."""
        return BBox(
            self.lon_min % 360.0, self.lon_max % 360.0, self.lat_min, self.lat_max
        )

    # ---- serializers ------------------------------------------------------

    def as_cmems(self) -> dict[str, float]:
        """``copernicusmarine.subset`` kwargs."""
        return {
            "minimum_longitude": self.lon_min,
            "maximum_longitude": self.lon_max,
            "minimum_latitude": self.lat_min,
            "maximum_latitude": self.lat_max,
        }

    def as_cds_area(self) -> list[float]:
        """CDS ``area`` key: ``[North, West, South, East]``."""
        return [self.lat_max, self.lon_min, self.lat_min, self.lon_max]

    def as_xarray_sel(self, lon: str = "lon", lat: str = "lat") -> dict[str, slice]:
        """``ds.sel(**bbox.as_xarray_sel())`` selector.

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
        return {
            lon: slice(self.lon_min, self.lon_max),
            lat: slice(self.lat_min, self.lat_max),
        }

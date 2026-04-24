"""Fixed-point observation station type and collection.

``Station`` describes a single stationary platform (weather station,
tide gauge, buoy, mooring). ``StationCollection`` is a lightweight
frozen container with filtering, spatial selection, and nearest-
neighbour helpers — the same operations most station catalogues
expose, kept adapter-agnostic so every source (AEMET, NOAA ISD,
tide gauges, ...) can return the same shape.

The station dimension this maps to in xarray is CF's
``featureType = "timeSeries"`` discrete sampling geometry (CF 9.3):
a ``(station, time)`` dataset with per-station ``lon``/``lat``/
``altitude`` coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from xr_toolz.types._src.geometry import BBox


if TYPE_CHECKING:
    import pandas as pd


_EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Station:
    """One fixed observation station.

    Coordinates are decimal degrees in the ``[-180, 180]`` / ``[-90, 90]``
    convention. ``city`` / ``province`` / ``community`` reflect the
    administrative hierarchy exposed by AEMET and most national services;
    adapters that don't split this way can leave them ``None``.
    """

    id: str
    name: str
    lon: float
    lat: float
    altitude: float | None = None
    wmo_id: str | None = None
    source: str | None = None
    city: str | None = None
    province: str | None = None
    community: str | None = None
    timezone: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    active: bool | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon out of [-180, 180]: {self.lon}")
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat out of [-90, 90]: {self.lat}")
        if not isinstance(self.attrs, MappingProxyType):
            frozen = MappingProxyType(dict(self.attrs))
            object.__setattr__(self, "attrs", frozen)

    def distance_to(self, lon: float, lat: float) -> float:
        """Great-circle distance to ``(lon, lat)`` in kilometres."""
        return _haversine_km(self.lon, self.lat, lon, lat)

    def in_bbox(self, bbox: BBox) -> bool:
        """Whether the station falls inside ``bbox`` (inclusive)."""
        if not bbox.lat_min <= self.lat <= bbox.lat_max:
            return False
        if bbox.crosses_antimeridian:
            return self.lon >= bbox.lon_min or self.lon <= bbox.lon_max
        return bbox.lon_min <= self.lon <= bbox.lon_max


@dataclass(frozen=True)
class StationCollection:
    """Immutable collection of :class:`Station` with set-like filters.

    Supports iteration, ``len``, containment, tuple-style indexing, and
    conversion to :class:`pandas.DataFrame`. All filter methods return a
    new collection rather than mutating in place, so chains like
    ``stations.filter(province="Cantabria").within(bbox)`` are safe.
    """

    stations: tuple[Station, ...] = ()

    def __post_init__(self) -> None:
        # Accept any iterable and normalize to a tuple so callers can
        # pass ``list[Station]`` without losing immutability guarantees.
        if not isinstance(self.stations, tuple):
            object.__setattr__(self, "stations", tuple(self.stations))

    # ---- container dunders -----------------------------------------------

    def __iter__(self) -> Iterator[Station]:
        return iter(self.stations)

    def __len__(self) -> int:
        return len(self.stations)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Station):
            return item in self.stations
        if isinstance(item, str):
            return any(s.id == item for s in self.stations)
        return False

    def __getitem__(self, key: int | slice | str) -> Station | StationCollection:
        if isinstance(key, str):
            for s in self.stations:
                if s.id == key:
                    return s
            raise KeyError(f"no station with id {key!r}")
        if isinstance(key, slice):
            return StationCollection(self.stations[key])
        return self.stations[key]

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_iter(cls, stations: Iterable[Station]) -> StationCollection:
        """Build from any iterable of :class:`Station`."""
        return cls(tuple(stations))

    # ---- filters ---------------------------------------------------------

    def filter(
        self,
        *,
        bbox: BBox | None = None,
        city: str | Iterable[str] | None = None,
        province: str | Iterable[str] | None = None,
        community: str | Iterable[str] | None = None,
        source: str | Iterable[str] | None = None,
        active: bool | None = None,
        has_wmo: bool | None = None,
    ) -> StationCollection:
        """Return a new collection keeping stations matching *all* filters.

        String-valued filters accept either a single value or an iterable
        of values (OR-semantics within a filter, AND across filters).
        Matching is case-insensitive on the admin fields so users don't
        have to remember whether AEMET capitalises "Cantabria" or not.
        """
        city_set = _norm_set(city)
        province_set = _norm_set(province)
        community_set = _norm_set(community)
        source_set = _norm_set(source)

        def keep(s: Station) -> bool:
            if bbox is not None and not s.in_bbox(bbox):
                return False
            if city_set is not None and _norm(s.city) not in city_set:
                return False
            if province_set is not None and _norm(s.province) not in province_set:
                return False
            if community_set is not None and _norm(s.community) not in community_set:
                return False
            if source_set is not None and _norm(s.source) not in source_set:
                return False
            if active is not None and s.active != active:
                return False
            if has_wmo is True and not s.wmo_id:
                return False
            return not (has_wmo is False and s.wmo_id)

        return StationCollection(tuple(s for s in self.stations if keep(s)))

    def within(self, bbox: BBox) -> StationCollection:
        """Shorthand for ``filter(bbox=bbox)``."""
        return self.filter(bbox=bbox)

    def nearest(self, point: tuple[float, float], n: int = 1) -> StationCollection:
        """Return the ``n`` stations closest to ``point = (lon, lat)``."""
        if n <= 0:
            return StationCollection(())
        lon, lat = point
        ranked = sorted(self.stations, key=lambda s: s.distance_to(lon, lat))
        return StationCollection(tuple(ranked[:n]))

    # ---- listing helpers -------------------------------------------------

    def ids(self) -> tuple[str, ...]:
        """Tuple of station IDs in collection order."""
        return tuple(s.id for s in self.stations)

    def cities(self) -> tuple[str, ...]:
        """Sorted unique non-empty ``city`` values."""
        return _sorted_unique(s.city for s in self.stations)

    def provinces(self) -> tuple[str, ...]:
        """Sorted unique non-empty ``province`` values."""
        return _sorted_unique(s.province for s in self.stations)

    def communities(self) -> tuple[str, ...]:
        """Sorted unique non-empty ``community`` values."""
        return _sorted_unique(s.community for s in self.stations)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the collection as a :class:`pandas.DataFrame`."""
        import pandas as pd

        if not self.stations:
            return pd.DataFrame(
                columns=[
                    "id",
                    "name",
                    "lon",
                    "lat",
                    "altitude",
                    "wmo_id",
                    "source",
                    "city",
                    "province",
                    "community",
                    "timezone",
                    "start_time",
                    "end_time",
                    "active",
                ]
            )
        rows = [
            {
                "id": s.id,
                "name": s.name,
                "lon": s.lon,
                "lat": s.lat,
                "altitude": s.altitude,
                "wmo_id": s.wmo_id,
                "source": s.source,
                "city": s.city,
                "province": s.province,
                "community": s.community,
                "timezone": s.timezone,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "active": s.active,
            }
            for s in self.stations
        ]
        return pd.DataFrame(rows)


# ---- internals ----------------------------------------------------------


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * asin(sqrt(a))


def _norm(value: str | None) -> str | None:
    return value.strip().casefold() if value else None


def _norm_set(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value.strip().casefold()}
    return {v.strip().casefold() for v in value}


def _sorted_unique(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(sorted({v for v in values if v}))

"""Query-parameter helpers shared by provider clients: bounding boxes and times.

Catalogue APIs (REST, STAC, bucket listings) all take a lon/lat bounding
box and a time window; these helpers validate and format them the same
way everywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime


__all__ = ["BBox", "as_utc", "rfc3339_utc", "time_interval", "validate_lonlat_bbox"]

#: ``(west, south, east, north)`` in WGS-84 degrees.
BBox = tuple[float, float, float, float]


def validate_lonlat_bbox(bbox: BBox) -> None:
    """Reject malformed ``(W, S, E, N)`` boxes before they reach an API.

    Raises:
        ValueError: ``bbox`` is not length 4, a latitude is outside
            ``[-90, 90]``, ``S > N``, or ``W > E``. Antimeridian-crossing
            boxes (``W > E``) are rejected rather than silently misread —
            split them into two boxes, one on each side of ±180°.

    Examples:
        >>> validate_lonlat_bbox((-104.5, 31.5, -103.5, 32.5))
        >>> validate_lonlat_bbox((10.0, 0.0, -10.0, 1.0))
        Traceback (most recent call last):
        ...
        ValueError: bbox west > east ((10.0, 0.0, -10.0, 1.0)) — antimeridian-crossing boxes are not supported; split into two boxes on either side of ±180°.
    """  # noqa: E501
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (W, S, E, N); got {bbox!r}")
    w, s, e, n = (float(v) for v in bbox)
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValueError(f"bbox latitudes must lie in [-90, 90]; got {bbox!r}")
    if s > n:
        raise ValueError(f"bbox south > north; got {bbox!r}")
    if w > e:
        raise ValueError(
            f"bbox west > east ({bbox!r}) — antimeridian-crossing boxes are "
            "not supported; split into two boxes on either side of ±180°."
        )


def as_utc(value: datetime) -> datetime:
    """``value`` in UTC: naive datetimes are taken as UTC, aware ones converted.

    Examples:
        >>> as_utc(datetime(2026, 1, 1, 12)).isoformat()
        '2026-01-01T12:00:00+00:00'
    """
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def rfc3339_utc(value: datetime) -> str:
    """``value`` as RFC 3339 in UTC with a ``Z`` suffix (see :func:`as_utc`).

    Examples:
        >>> rfc3339_utc(datetime(2026, 1, 1, 12))
        '2026-01-01T12:00:00Z'
    """
    return as_utc(value).isoformat().replace("+00:00", "Z")


def time_interval(start: datetime | None, end: datetime | None) -> str | None:
    """An RFC 3339 interval ``"start/end"`` with ``..`` for an open side.

    The STAC / OGC datetime-range form; ``None`` when both sides are open.

    Examples:
        >>> time_interval(datetime(2026, 1, 1), None)
        '2026-01-01T00:00:00Z/..'
    """
    if start is None and end is None:
        return None
    lo = rfc3339_utc(start) if start else ".."
    hi = rfc3339_utc(end) if end else ".."
    return f"{lo}/{hi}"

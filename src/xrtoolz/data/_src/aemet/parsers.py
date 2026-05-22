"""Low-level parsers for AEMET OpenData payloads.

AEMET returns station metadata in a quirky format: coordinates as
sexagesimal DMS strings (``"402358N"``), decimals with commas
(``"1,2"``), and timestamps with the literal suffix ``UTC``. These
helpers normalise all of that to Python / pandas friendly types.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime


_DMS_RE = re.compile(
    r"^\s*(?P<deg>\d{2,3})(?P<min>\d{2})(?P<sec>\d{2})(?P<hemi>[NSEWnsew])\s*$"
)


def parse_dms(text: str) -> float:
    """Parse AEMET's ``DDMMSS[NSEW]`` coordinate string into decimal degrees.

    Example: ``"402358N"`` → ``40.399444`` (40° 23' 58" N).
    """
    match = _DMS_RE.match(text)
    if match is None:
        raise ValueError(f"not a DDMMSS[NSEW] coordinate: {text!r}")
    deg = int(match.group("deg"))
    minutes = int(match.group("min"))
    sec = int(match.group("sec"))
    hemi = match.group("hemi").upper()
    # AEMET occasionally reports seconds == 60 (e.g. ``162860W``) as a
    # rounded boundary; accept that tolerance for seconds so we don't
    # reject real stations. Minutes must still be in the normal 0..59
    # range — anything else is genuinely malformed.
    if not 0 <= minutes < 60 or not 0 <= sec <= 60:
        raise ValueError(f"invalid minutes/seconds in {text!r}")
    decimal = deg + minutes / 60.0 + sec / 3600.0
    if hemi in ("S", "W"):
        decimal = -decimal
    return decimal


def parse_spanish_float(value: str | float | int | None) -> float | None:
    """Convert an AEMET numeric string (with comma decimals) to ``float``.

    Returns ``None`` for missing / sentinel values (``""``, ``None``,
    ``"Ip"`` — AEMET's "trace precipitation" flag — and ``"-"``).

    AEMET's monthly-extreme fields annotate the numeric with a
    parenthesised day-of-month (``"-1.8(23)"`` = -1.8 °C on the 23rd);
    we strip any parenthesised suffix so the numeric value still
    parses.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    stripped = value.strip()
    if not stripped or stripped.lower() in {"-", "ip", "acum"}:
        return None
    # Drop ``(DD)`` / ``(23)`` day-of-extreme annotation if present.
    paren = stripped.find("(")
    if paren > 0:
        stripped = stripped[:paren].strip()
    try:
        return float(stripped.replace(",", "."))
    except ValueError:
        return None


def format_aemet_datetime(dt: datetime) -> str:
    """Format a ``datetime`` as AEMET's ``YYYY-MM-DDTHH:MM:SSUTC``.

    The literal suffix ``UTC`` (not ``Z``) is required by the daily
    climatology endpoint — a known footgun.
    """
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "UTC"


def parse_aemet_datetime(text: str) -> datetime:
    """Parse AEMET's ``YYYY-MM-DDTHH:MM:SS(UTC|Z)?`` datetime.

    Both the literal ``UTC`` suffix and the ISO ``Z`` are accepted.
    """
    cleaned = text.strip()
    if cleaned.endswith("UTC"):
        cleaned = cleaned[:-3]
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

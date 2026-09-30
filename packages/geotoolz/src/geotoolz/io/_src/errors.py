"""Exception type shared by the io primitives and operators."""

from __future__ import annotations


class GeoToolzIOError(RuntimeError):
    """Raised when a geotoolz I/O operator cannot read or write data."""


__all__ = ["GeoToolzIOError"]

"""Refuse georeader carriers at geoproducts' public boundary.

geoproducts takes and returns `geoarray.GeoArray` / `geoarray.RasterSource`;
georeader is only an internal engine. A georeader object passed in by a
user is refused with a `TypeError` naming the geoarray adapter that
converts it. The check reads the class hierarchy only, so it never imports
georeader.
"""

from __future__ import annotations

from typing import Any

from geoarray import RasterSource


__all__ = ["reject_georeader", "require_source"]


def _georeader_classes(x: Any) -> list[type]:
    return [
        cls
        for cls in type(x).__mro__
        if cls.__module__ == "georeader" or cls.__module__.startswith("georeader.")
    ]


def reject_georeader(x: Any, where: str) -> None:
    """Raise `TypeError` when ``x`` is a georeader ``GeoTensor`` or reader.

    Args:
        x: The value a public geoproducts function received.
        where: The function's name, for the message.

    Raises:
        TypeError: ``x`` is a georeader object; the message names
            ``geoarray.interop.georeader.from_georeader`` for a ``GeoTensor``
            and ``source_from_georeader`` for a lazy reader (``GeoData``).
    """
    classes = _georeader_classes(x)
    if not classes:
        return
    if any(cls.__name__ == "GeoTensor" for cls in classes):
        raise TypeError(
            f"{where} takes a geoarray.GeoArray, got a georeader GeoTensor; "
            "convert it with geoarray.interop.georeader.from_georeader(gt)"
        )
    raise TypeError(
        f"{where} takes a geoarray.GeoArray or RasterSource, got georeader's "
        f"{type(x).__name__}; wrap it with "
        "geoarray.interop.georeader.source_from_georeader(reader)"
    )


def require_source(x: Any, where: str) -> RasterSource:
    """``x`` if it is a `geoarray.RasterSource`; else a `TypeError`.

    Args:
        x: The value a public geoproducts function received.
        where: The function's name, for the message.

    Raises:
        TypeError: ``x`` is a georeader object (see `reject_georeader`) or
            not a `RasterSource`.
    """
    reject_georeader(x, where)
    if not isinstance(x, RasterSource):
        raise TypeError(
            f"{where} takes a geoarray.RasterSource (a geoproducts reader or a "
            f"GeoArray), got {type(x).__name__}"
        )
    return x

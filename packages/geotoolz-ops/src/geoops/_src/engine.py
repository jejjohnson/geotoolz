"""The georeader engine boundary: geotoolz carriers in, georeader out, and back.

Some operators still compute through georeader's functions (reprojection,
mosaics, rasterize / vectorize, georeferencing, windowed reads, solar
geometry). Those take and return georeader ``GeoTensor`` / ``GeoData``, so
every such call goes through this module: `to_engine` on the way in,
`from_engine` on the way out. Nothing else in geoops touches a
``GeoTensor``, and replacing an engine call later changes no public API.
"""

from __future__ import annotations

from typing import Any

from geoarray import GeoArray, RasterSource

from geoops._src.geo import is_geotensor, reject_georeader
from geoops._src.optional import import_optional


__all__ = ["from_engine", "to_engine"]


def to_engine(x: Any, op_name: str = "geotoolz") -> Any:
    """``x`` as georeader expects it: a ``GeoTensor`` or ``GeoData``.

    A `GeoArray` becomes a ``GeoTensor`` sharing its memory (JAX data is
    copied to the host first); a `RasterSource` becomes a ``GeoData``
    view read window by window. A georeader object handed in by a user
    (a ``GeoTensor``, a reader) is refused: only geotoolz carriers cross
    into the engine, so a constructor-held ``GeoTensor`` cannot slip
    through. Anything else is returned unchanged.

    Raises:
        TypeError: ``x`` is already a georeader object; the message names
            the ``geoarray.interop.georeader`` bridge to convert it with.
        ImportError: georeader is not installed; the message names
            ``geotoolz-ops[sensors]``.
    """
    if isinstance(x, GeoArray | RasterSource):
        import_optional("georeader", "sensors", feature=op_name)
    from geoarray.interop.georeader import source_to_georeader, to_georeader

    if isinstance(x, GeoArray):
        return to_georeader(x.to_numpy())
    if isinstance(x, RasterSource):
        return source_to_georeader(x)
    reject_georeader(x, op_name)
    return x


def from_engine(result: Any) -> Any:
    """A georeader ``GeoTensor`` as a `GeoArray` (zero-copy); else unchanged."""
    if is_geotensor(result):
        from geoarray.interop.georeader import from_georeader

        return from_georeader(result)
    return result

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

from geoops._src.geo import is_geotensor


__all__ = ["from_engine", "to_engine"]


def to_engine(x: Any) -> Any:
    """``x`` as georeader expects it: a ``GeoTensor`` or ``GeoData``.

    A `GeoArray` becomes a ``GeoTensor`` sharing its memory (JAX data is
    copied to the host first); a `RasterSource` becomes a ``GeoData``
    view read window by window. Anything else is returned unchanged.
    """
    from geoarray.interop.georeader import source_to_georeader, to_georeader

    if isinstance(x, GeoArray):
        return to_georeader(x.to_numpy())
    if isinstance(x, RasterSource):
        return source_to_georeader(x)
    return x


def from_engine(result: Any) -> Any:
    """A georeader ``GeoTensor`` as a `GeoArray` (zero-copy); else unchanged."""
    if is_geotensor(result):
        from geoarray.interop.georeader import from_georeader

        return from_georeader(result)
    return result

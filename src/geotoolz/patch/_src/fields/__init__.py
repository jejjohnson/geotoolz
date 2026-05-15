"""`Field` adapters — one per substrate.

Each adapter is a thin shim that exposes the unified `Field` Protocol
(`domain`, `select`, `with_data`) on top of a backend-specific carrier.
The raster adapter is essentially free — georeader's `GeoData` already
covers it, the wrapper just renames `read_from_window` → `select`.

Non-raster adapters guard their optional-extra import at top-of-module:
import the adapter and you get a friendly error pointing at the right
``pip install`` extra if the backend library is missing.
"""

from __future__ import annotations

from geotoolz.patch._src.fields.raster import (
    AsyncRasterField,
    RasterField,
)


__all__ = [
    "AsyncRasterField",
    "RasterField",
]


def __getattr__(name: str):
    """Lazy load adapters that depend on optional extras.

    Importing the adapter triggers the `_missing_extra` error path if
    the backend library isn't installed — but importing
    ``geotoolz.patch.fields`` itself shouldn't, hence the lazy hook.
    """
    if name == "XarrayField":
        from geotoolz.patch._src.fields.xarray import XarrayField

        return XarrayField
    if name == "GeoPandasField":
        from geotoolz.patch._src.fields.geopandas import GeoPandasField

        return GeoPandasField
    if name == "XvecField":
        from geotoolz.patch._src.fields.xvec import XvecField

        return XvecField
    if name == "RioXarrayField":
        from geotoolz.patch._src.fields.rio_xarray import RioXarrayField

        return RioXarrayField
    raise AttributeError(name)

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

import importlib
from typing import Any

from geopatcher._src.fields.raster import (
    AsyncRasterField,
    RasterField,
)
from geopatcher._src.fields.reproject import ReprojectingRasterField


# Adapters whose backend is an optional extra: name -> defining module. The
# one list `geopatcher.fields` and the root `geopatcher` namespace resolve
# lazily from, so the three never drift apart.
LAZY_ADAPTERS: dict[str, str] = {
    "CogField": "geopatcher._src.fields.cog",
    "DaskField": "geopatcher._src.fields.dask",
    "GeoPandasField": "geopatcher._src.fields.geopandas",
    "RioXarrayField": "geopatcher._src.fields.rio_xarray",
    "XarrayField": "geopatcher._src.fields.xarray",
    "XvecField": "geopatcher._src.fields.xvec",
}

__all__ = [
    "AsyncRasterField",
    "CogField",
    "DaskField",
    "GeoPandasField",
    "RasterField",
    "ReprojectingRasterField",
    "RioXarrayField",
    "XarrayField",
    "XvecField",
]


def __getattr__(name: str) -> Any:
    """Lazy load adapters that depend on optional extras.

    Constructing an adapter whose backend library isn't installed raises
    the `missing_extra` error — but importing ``geopatcher.fields`` itself
    shouldn't, hence the lazy hook.
    """
    if name in LAZY_ADAPTERS:
        return getattr(importlib.import_module(LAZY_ADAPTERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

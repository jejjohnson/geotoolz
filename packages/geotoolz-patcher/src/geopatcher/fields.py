"""`geopatcher.fields` — Field adapters and domains, one per substrate.

A `Field` is what a patcher reads from: a ``domain`` (the I/O-free
metadata a sampler plans anchors over) plus ``select(window)``. The
GeoTIFF / georeader adapter, `RasterField`, lives at the package root;
this module holds the rest:

- `AsyncRasterField` / `ReprojectingRasterField` — georeader readers read
  asynchronously, or warped onto another grid as they are read.
- `CogField` — a Cloud-Optimized GeoTIFF on object storage, with batched
  tile fetches (``[cog]`` extra, on geotoolz-cloud).
- `XarrayField`, `RioXarrayField`, `DaskField` — gridded N-D data
  (``[grid]`` / ``[xarray-raster]`` / ``[dask]``).
- `GeoPandasField`, `XvecField` — vector features and point cubes
  (``[vector]`` / ``[point]``).
- `RasterDomain`, `GridDomain`, `VectorDomain`, `PointDomain` — the
  domain shapes the samplers understand.

The extras-gated adapters resolve lazily: importing this module never
needs an optional dependency, and using an adapter whose extra is missing
raises its install hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geopatcher._src import fields as _fields
from geopatcher._src.domains import (
    GridDomain,
    PointDomain,
    RasterDomain,
    VectorDomain,
)
from geopatcher._src.fields import AsyncRasterField, ReprojectingRasterField


if TYPE_CHECKING:  # bound lazily at runtime; named here for type checkers / docs
    from geopatcher._src.fields.cog import CogField
    from geopatcher._src.fields.dask import DaskField
    from geopatcher._src.fields.geopandas import GeoPandasField
    from geopatcher._src.fields.rio_xarray import RioXarrayField
    from geopatcher._src.fields.xarray import XarrayField
    from geopatcher._src.fields.xvec import XvecField


__all__ = [
    "AsyncRasterField",
    "CogField",
    "DaskField",
    "GeoPandasField",
    "GridDomain",
    "PointDomain",
    "RasterDomain",
    "ReprojectingRasterField",
    "RioXarrayField",
    "VectorDomain",
    "XarrayField",
    "XvecField",
]


def __getattr__(name: str) -> Any:
    """Resolve the extras-gated adapters on first access."""
    if name in _fields.LAZY_ADAPTERS:
        return getattr(_fields, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

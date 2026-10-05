"""Public alias for `geopatcher._src.fields`.

Re-exports `RasterField` / `AsyncRasterField` / `ReprojectingRasterField`
eagerly and the extras-gated adapters (`XarrayField`, `GeoPandasField`,
`XvecField`, `RioXarrayField`, `DaskField`, `ObstoreCogField`) lazily — so
``from geopatcher.fields import XarrayField`` only triggers the
optional-extra import path when the name is actually accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geopatcher._src import fields as _fields
from geopatcher._src.fields import (
    AsyncRasterField,
    RasterField,
    ReprojectingRasterField,
)


if TYPE_CHECKING:  # bound lazily at runtime; named here for type checkers / docs
    from geopatcher._src.fields.dask import DaskField
    from geopatcher._src.fields.geopandas import GeoPandasField
    from geopatcher._src.fields.obstore_cog import ObstoreCogField
    from geopatcher._src.fields.rio_xarray import RioXarrayField
    from geopatcher._src.fields.xarray import XarrayField
    from geopatcher._src.fields.xvec import XvecField


# The extras-gated names below are resolved on attribute access via
# `__getattr__`; we list them in `__all__` so static-analysis tooling sees
# the public surface even though they aren't bound at module top-level.
__all__ = [
    "AsyncRasterField",
    "DaskField",
    "GeoPandasField",
    "ObstoreCogField",
    "RasterField",
    "ReprojectingRasterField",
    "RioXarrayField",
    "XarrayField",
    "XvecField",
]


def __getattr__(name: str) -> Any:
    """Defer to the private package's lazy loader for the extras-gated adapters."""
    if name in _fields.LAZY_ADAPTERS:
        return getattr(_fields, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

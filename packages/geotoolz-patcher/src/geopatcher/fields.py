"""Public alias for `geopatcher._src.fields`.

Re-exports `RasterField` / `AsyncRasterField` eagerly and the extras-gated
adapters (`XarrayField`, `GeoPandasField`, `XvecField`, `RioXarrayField`,
`DaskField`, `ObstoreCogField`) lazily — so
``from geopatcher.fields import XarrayField`` only triggers the
optional-extra import path when the name is actually accessed.
"""

from __future__ import annotations

from typing import Any

from geopatcher._src.fields import AsyncRasterField, RasterField


# The extras-gated names below are resolved on attribute access via
# `__getattr__`; we list them in `__all__` so static-analysis tooling sees
# the public surface even though they aren't bound at module top-level.
__all__ = [  # noqa: F822 - extras-gated names resolved via __getattr__
    "AsyncRasterField",
    "DaskField",
    "GeoPandasField",
    "ObstoreCogField",
    "RasterField",
    "RioXarrayField",
    "XarrayField",
    "XvecField",
]


def __getattr__(name: str) -> Any:
    """Defer to the private package's lazy loader for the extras-gated adapters."""
    if name in {
        "XarrayField",
        "GeoPandasField",
        "XvecField",
        "RioXarrayField",
        "DaskField",
        "ObstoreCogField",
    }:
        from geopatcher._src import fields as _f

        return getattr(_f, name)
    raise AttributeError(name)

"""External data-source adapters.

See ``docs/design/query-matchup.md`` §4 for the full design. This
module is the bridge between *remote* catalogs (NASA earthaccess,
STAC endpoints, Google Earth Engine, CMR) and the local
``GeoCatalog``: each adapter knows how to ask its upstream service
"what's in this bbox and interval?" and produce normalized rows
(`SourceRow`) that downstream code can ingest into a local catalog.

The Protocol is intentionally narrow — `bounds + interval + filters
dict` — so adapters can be added without redesigning the surface.
"""

from __future__ import annotations

from geocatalog._src.sources._base import (
    AuthStatus,
    Source,
    SourceRow,
)


__all__ = [
    "AuthStatus",
    "Source",
    "SourceRow",
]


def __getattr__(name: str) -> object:
    """Lazy-import adapters so optional extras stay opt-in.

    Adapters live in dedicated modules but are referenced from this
    package's top-level namespace; importing them here would pull in
    every optional dependency on a bare ``import geocatalog.sources``,
    defeating the extras gating. Instead we resolve the import on
    first attribute access.
    """
    if name == "EarthAccessSource":
        from geocatalog._src.sources.earthaccess import EarthAccessSource

        return EarthAccessSource
    if name == "STACSource":
        from geocatalog._src.sources.stac import STACSource

        return STACSource
    if name == "CMRSource":
        from geocatalog._src.sources.cmr import CMRSource

        return CMRSource
    if name == "GEESource":
        from geocatalog._src.sources.gee import GEESource

        return GEESource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

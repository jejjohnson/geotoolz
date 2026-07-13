"""`geocatalog.sources` — external data-source adapters.

Hybrid-layout sub-namespace mirroring `geocatalog.catalog` and
`geocatalog.types`. Re-exports the `Source` Protocol, the
`SourceRow` carrier, and the concrete adapters (each extras-gated).

Adapters are imported lazily on attribute access so a bare
``import geocatalog.sources`` does not pull in `earthaccess`,
`pystac-client`, or `earthengine-api`. See ``docs/design/query-matchup.md``
§4.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geocatalog._src.sources import AuthStatus, Source, SourceRow


if TYPE_CHECKING:
    from geocatalog._src.sources.cmr import CMRSource
    from geocatalog._src.sources.earthaccess import EarthAccessSource
    from geocatalog._src.sources.gee import GEESource
    from geocatalog._src.sources.stac import STACSource


__all__ = [
    "AuthStatus",
    "CMRSource",
    "EarthAccessSource",
    "GEESource",
    "STACSource",
    "Source",
    "SourceRow",
]


def __getattr__(name: str) -> Any:
    """Lazy-import adapters so optional extras stay opt-in."""
    if name in {"EarthAccessSource", "STACSource", "CMRSource", "GEESource"}:
        from geocatalog._src import sources as _sources

        return getattr(_sources, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

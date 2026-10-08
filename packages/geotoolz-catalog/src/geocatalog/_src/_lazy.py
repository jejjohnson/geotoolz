"""One table of lazily imported public names, shared by every facade.

The extras-gated modules (DuckDB, xarray, vector, STAC, the source
adapters) import their optional dependencies softly, so importing them
never fails; they are loaded lazily only to keep ``import geocatalog``
cheap. Using a name whose extra is missing raises the extra's install
hint at the point of use (see `geocatalog._src._extras`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any


LAZY: dict[str, str] = {
    "CMRSource": "geocatalog._src.sources.cmr",
    "DuckDBGeoCatalog": "geocatalog._src.backends.duckdb_backend",
    "EarthAccessSource": "geocatalog._src.sources.earthaccess",
    "GEESource": "geocatalog._src.sources.gee",
    "STACSource": "geocatalog._src.sources.stac",
    "build_vector_catalog": "geocatalog._src.formats.vector",
    "build_xarray_catalog": "geocatalog._src.formats.xarray_backend",
    "from_stac_items": "geocatalog._src.storage.stac",
    "from_stac_search": "geocatalog._src.storage.stac",
    "load_vector": "geocatalog._src.formats.vector",
    "load_xarray": "geocatalog._src.formats.xarray_backend",
    "to_stac_collection": "geocatalog._src.storage.stac",
}
"""Public name → module that defines it."""


def lazy_getattr(
    namespace: dict[str, Any], names: Iterable[str]
) -> Callable[[str], Any]:
    """A module ``__getattr__`` resolving ``names`` from `LAZY`.

    The resolved object is cached in ``namespace`` (the facade's
    ``globals()``), and an unknown name raises `AttributeError` naming
    the facade itself.
    """
    module_name = namespace["__name__"]
    wanted = frozenset(names)
    unknown = wanted - LAZY.keys()
    if unknown:
        raise ValueError(f"not lazy names: {sorted(unknown)}")

    def __getattr__(name: str) -> Any:
        if name not in wanted:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        attr = getattr(importlib.import_module(LAZY[name]), name)
        namespace[name] = attr
        return attr

    return __getattr__


__all__ = ["LAZY", "lazy_getattr"]

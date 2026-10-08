"""`geocatalog.sources` — discover scenes in external archives.

Every adapter satisfies the `Source` protocol and yields `SourceRow`
records (`source_row_to_gdf_row` flattens one into a catalog row):

- `STACSource` — any STAC API (``[stac]`` extra).
- `CMRSource` — NASA's Common Metadata Repository.
- `EarthAccessSource` — NASA Earthdata via earthaccess
  (``[earthaccess]`` extra).
- `GEESource` — Google Earth Engine (``[gee]`` extra).

`AuthStatus` reports whether an adapter's credentials work.
`from_stac_items` / `from_stac_search` build a catalog straight from
STAC. Adapters load lazily, so a bare ``import geocatalog.sources``
needs no extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src.bundle import source_row_to_gdf_row
from geocatalog._src.sources import (
    AuthStatus,
    Source,
    SourceRow,
)


if TYPE_CHECKING:
    from geocatalog._src.sources.cmr import CMRSource
    from geocatalog._src.sources.earthaccess import EarthAccessSource
    from geocatalog._src.sources.gee import GEESource
    from geocatalog._src.sources.stac import STACSource
    from geocatalog._src.stac import from_stac_items, from_stac_search


__all__ = [
    "AuthStatus",
    "CMRSource",
    "EarthAccessSource",
    "GEESource",
    "STACSource",
    "Source",
    "SourceRow",
    "from_stac_items",
    "from_stac_search",
    "source_row_to_gdf_row",
]


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "CMRSource",
        "EarthAccessSource",
        "GEESource",
        "STACSource",
        "from_stac_items",
        "from_stac_search",
    ],
)

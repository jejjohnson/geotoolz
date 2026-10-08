"""`geocatalog.storage` — save, share and reopen catalogs.

- GeoParquet: `to_geoparquet` / `from_geoparquet`, `migrate_geoparquet`
  (older files to `SCHEMA_VERSION_CURRENT`), `sort_geoparquet`, and
  `StreamingParquetWriter` for catalogs built row by row.
- STAC: `to_stac_collection` (``[stac]`` extra; loads lazily).
- Provenance: `CatalogBundle` keeps a catalog together with the queries
  (`QueryRecord`) and matchups that produced it.

```python
from geocatalog.storage import from_geoparquet, to_geoparquet

to_geoparquet(catalog, "scenes.parquet")
same = from_geoparquet("scenes.parquet")
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src.storage.bundle import (
    CatalogBundle,
    QueryRecord,
)
from geocatalog._src.storage.parquet import (
    SCHEMA_VERSION_CURRENT,
    from_geoparquet,
    migrate_geoparquet,
    to_geoparquet,
)
from geocatalog._src.storage.streaming import (
    StreamingParquetWriter,
    sort_geoparquet,
)


if TYPE_CHECKING:
    from geocatalog._src.storage.stac import to_stac_collection


__all__ = [
    "SCHEMA_VERSION_CURRENT",
    "CatalogBundle",
    "QueryRecord",
    "StreamingParquetWriter",
    "from_geoparquet",
    "migrate_geoparquet",
    "sort_geoparquet",
    "to_geoparquet",
    "to_stac_collection",
]


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "to_stac_collection",
    ],
)

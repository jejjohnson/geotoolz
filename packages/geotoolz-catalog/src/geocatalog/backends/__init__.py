"""`geocatalog.backends` — the catalog implementations and their errors.

Every backend satisfies the `geocatalog.GeoCatalog` protocol, so code
written against one runs on the other:

- `InMemoryGeoCatalog` — a GeoDataFrame in memory; the default.
- `DuckDBGeoCatalog` — DuckDB-spatial over GeoParquet, for catalogs
  larger than memory (``[duckdb]`` extra; loads lazily).

`CatalogRow` is the row ``iter_rows`` yields, and `GeoCatalogError`
(with `CatalogMetadataError`, `CatalogSchemaError`,
`CatalogClosedError`) is what they raise. `geocatalog.open_catalog`
picks a backend from the file for you.

```python
from geocatalog import open_catalog
from geocatalog.backends import CatalogRow, InMemoryGeoCatalog

catalog = open_catalog("scenes.parquet", engine="memory")
assert isinstance(catalog, InMemoryGeoCatalog)
row: CatalogRow = next(catalog.iter_rows())
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src.base import (
    CatalogClosedError,
    CatalogMetadataError,
    CatalogRow,
    CatalogSchemaError,
    GeoCatalogError,
)
from geocatalog._src.memory import InMemoryGeoCatalog


if TYPE_CHECKING:
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog


__all__ = [
    "CatalogClosedError",
    "CatalogMetadataError",
    "CatalogRow",
    "CatalogSchemaError",
    "DuckDBGeoCatalog",
    "GeoCatalogError",
    "InMemoryGeoCatalog",
]


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "DuckDBGeoCatalog",
    ],
)

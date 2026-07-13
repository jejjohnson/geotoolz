"""`geocatalog.bundle` — directory-of-Parquet catalog persistence.

Public alias for `geocatalog._src.bundle`. See
``docs/design/query-matchup.md`` §4.4 for the directory layout and
§4.5 for the ingest workflow.
"""

from __future__ import annotations

from geocatalog._src.bundle import (
    CatalogBundle,
    QueryRecord,
    source_row_to_gdf_row,
)


__all__ = [
    "CatalogBundle",
    "QueryRecord",
    "source_row_to_gdf_row",
]

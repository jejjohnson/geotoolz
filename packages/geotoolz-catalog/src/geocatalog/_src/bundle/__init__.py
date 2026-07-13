"""`CatalogBundle` — a directory of catalog + queries + matchups.

A bundle is the persisted form of the ingest / matchup workflow. It
wraps an `InMemoryGeoCatalog` (the items table) with two sibling
tables — ``queries.parquet`` (provenance: which call produced which
items) and ``matchups.parquet`` (matched-row tuples) — plus a
``_meta.json`` carrying the target CRS and schema version.

Three entry points:

* `CatalogBundle.empty(target_crs, backend)` — create a fresh bundle.
* `CatalogBundle.from_directory(path)` — load an existing one.
* `CatalogBundle.from_catalog(catalog)` — wrap an in-memory
  catalog when the user already built one without using the bundle
  API.

Mutating methods:

* `bundle.ingest(source, **query_kwargs) -> str` queries an external
  `Source`, appends each `SourceRow` to the items table, and records
  the call in ``queries``. Returns the new query's UUID.
* `bundle.write_matchups(rows, tag=None)` stores a stream of
  `MatchupRow` instances in the matchups table.
* `bundle.to_directory(path)` persists the bundle as three Parquet
  files + a JSON metadata sidecar.

The bundle is meant to be the user-facing artifact — share it,
version it in DVC, mount it on another machine. Round-trips through
`from_directory(...).to_directory(...)` are bit-stable on the catalog
side; ``queries`` and ``matchups`` round-trip as their respective
dataclass schemas.
"""

from __future__ import annotations

from geocatalog._src.bundle._catalog_bundle import (
    CatalogBundle,
    QueryRecord,
    source_row_to_gdf_row,
)


__all__ = [
    "CatalogBundle",
    "QueryRecord",
    "source_row_to_gdf_row",
]

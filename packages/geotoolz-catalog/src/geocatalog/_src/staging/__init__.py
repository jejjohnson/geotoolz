"""Staging layer — resolve remote URIs into local files.

Catalog ingestion (`Source.query` → `SourceRow`) does *not* download
data: it records URIs. Staging is the explicit step that pulls bytes
into a local cache and rewrites a catalog to point at those local
copies, ready to be opened by `load_raster` / `load_vector` /
`load_xarray`.

Implemented surface (see ``docs/design/query-matchup.md`` §4.7):

* `stage` — orchestrator entry point: fetches each row's remote
  assets into the cache (parallel, with retry) and returns a catalog
  whose rows point at the local copies.
* `LocalCache` — cache configuration carrier (root directory, TTL);
  content is keyed by ``(uri, asset)``.
* `field_for` — bridge a staged catalog to a `geopatcher.Field`
  (soft-imports geopatcher; requires the ``[patch]`` extra).
"""

from __future__ import annotations

from geocatalog._src.staging._base import LocalCache, stage
from geocatalog._src.staging._field_for import field_for


__all__ = [
    "LocalCache",
    "field_for",
    "stage",
]

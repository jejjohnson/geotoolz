"""`geocatalog.staging` — resolve remote URIs into local files.

Hybrid-layout sub-namespace over `geocatalog._src.staging`:

* `stage` — fetch every remote URI referenced by a catalog (via fsspec,
  in parallel, with retries; local paths are used in place) into a
  `LocalCache` and return a new catalog whose ``filepath`` / asset map
  point at the cached copies.
  Asset-aware: rows carrying the JSON-encoded asset map written by
  `CatalogBundle.ingest` have each named asset staged independently.
* `LocalCache` — a cache keyed by URI (expiring signature parameters
  removed), laid out ``{root}/{key[:2]}/{key}{ext}``, with an optional
  TTL and `prune`; root defaults to ``$GEOCATALOG_CACHE`` or
  ``~/.cache/geocatalog``.
* `field_for` — bridge from a staged catalog to a `geopatcher` Field
  (requires the ``[patch]`` extra).

See ``docs/catalog/design/query-matchup.md`` §4.7.
"""

from __future__ import annotations

from geocatalog._src.staging import LocalCache, field_for, stage


__all__ = [
    "LocalCache",
    "field_for",
    "stage",
]

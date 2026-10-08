---
name: add-catalog-source
description: Add a discovery adapter (a geocatalog Source) for a remote archive or catalogue API — a STAC endpoint, NASA CMR / earthaccess collection, Earth Engine, a provider REST API — to geotoolz-catalog. Use when asked to search, discover or ingest scenes from a new archive into a catalog.
---

# Add a catalog source

Read `packages/geotoolz-catalog/AGENTS.md` first. The `Source` protocol lives
in `geocatalog/_src/sources/_base.py`; `stac.py` and `cmr.py` are the
complete references.

## 1. Check it is really new

- Many archives already speak STAC: try `geocatalog.sources.STACSource` (or
  `from_stac_search`) against the endpoint before writing an adapter. NASA
  collections are covered by `CMRSource` / `EarthAccessSource`.
- A provider client that also downloads or reads products belongs in
  geotoolz-products; the catalog adapter only *discovers* (footprint, time,
  asset URLs) and never opens data.

## 2. Implement the adapter (`_src/sources/<name>.py`)

- A class with `name: str` (the stable `SourceRow.source` value),
  `query(bounds, interval=None, *, collection=None, filters=None, limit=None)
  -> Iterator[SourceRow]` and `auth_status() -> AuthStatus` (never raises).
- `bounds` are EPSG:4326 `(xmin, ymin, xmax, ymax)`; times are naive UTC —
  convert with `geocatalog.utils` (`to_utc_ts`, `to_rfc3339`).
- Honour the `limit` contract with `wants_no_rows(limit)`: `None` = all,
  `0` = no request, negative = `ValueError`. Stream pages; don't collect.
- Retries: `geocatalog.utils.retry_transient_io`.
- Optional dependency: import it softly and raise `missing_extra(...)`
  from `_src/_extras.py` at use; add the extra to
  `packages/geotoolz-catalog/pyproject.toml` (and to `[sources-all]`,
  `[full]`).

## 3. Wire it up

- Add the class to `LAZY` in `_src/_lazy.py` and to the lazy list and
  `__all__` of `geocatalog/sources/__init__.py`.
- `tests/test_public_surface.py` and `tests/test_packaging.py` must still
  pass (star import with every extra missing; every extra's dependencies
  imported somewhere).

## 4. Tests and docs

- Mock the upstream HTTP responses (see `tests/test_cmr_source.py`); a test
  against the real service is `@pytest.mark.live`.
- Cover pagination, `limit=0`, auth failure, and the `SourceRow` fields.
- Document it on `docs/catalog/api/sources.md` (an `:::` entry and its
  extra), then `make capabilities` and the `pre-pr-check` skill.

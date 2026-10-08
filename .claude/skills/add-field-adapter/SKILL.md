---
name: add-field-adapter
description: Add a geopatcher Field adapter (a new data source the patchers can split and merge — an array library, a vector format, a lazy store) or a new patcher axis (geometry, sampler, window, aggregation). Use when asked to make geopatcher work with a new data type or to add a tiling / weighting / merging strategy.
---

# Add a field adapter or a patcher axis

Read `packages/geotoolz-patcher/AGENTS.md` first, and the ADRs in
`docs/patcher/decisions.md` that touch the contract you are extending.

## Before writing one

- Check `geopatcher.fields` (RasterField, XarrayField, RioXarrayField,
  DaskField, GeoPandasField, XvecField, CogField, …) and the axes in
  `geopatcher.spatial.*` / `geopatcher.temporal.*` — most requests are a
  configuration of what exists.
- Data on object storage is read through geotoolz-cloud
  (`geocloud.cog.CogSource`, `geocloud.store`); extend those rather than
  adding I/O here.

## A field adapter (`_src/fields/<name>.py`)

- Implement the `Field` protocol (`_src/protocols.py`): `domain` (I/O-free
  metadata: CRS, bounds, shape — reuse the domain types in
  `_src/domains.py`), `select(indexer)`, `with_data(array)`.
- Optional, when the source supports it: `select_many(indexers)` (batched
  reads; `run.parallel_map` uses it), `aselect` for `AsyncSpatialPatcher`,
  `cache_id()` for `PatchCache` (raise `UnstableIdentityError` rather than
  return an identity that can drift).
- Export it from `geopatcher/fields.py`; if it needs an extra, register it in
  `LAZY_ADAPTERS` (`_src/fields/__init__.py`) and raise `missing_extra` from
  `_src/_extras.py` at use.

## A patcher axis (`_src/spatial/<axis>.py` or `_src/temporal/`)

- Subclass the axis base, name it **without** a `Spatial` / `Temporal`
  prefix, export it from `geopatcher/spatial/<axis>.py`.
- Make it round-trip: `from_config(obj.get_config())` rebuilds an equal
  object (config envelopes name it `"spatial.<axis>.<Name>"`).

## Tests and docs

- `tests/test_exports.py` (one home, no prefixes) and
  `tests/test_docs_paths.py` must pass; add behaviour tests that split and
  merge a small field and compare with a direct computation.
- Document it on the matching page in `docs/patcher/api/`; record a contract
  change as a new ADR. Then `make capabilities` and the `pre-pr-check`
  skill.

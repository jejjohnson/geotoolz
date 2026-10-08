# geotoolz-patcher (`geopatcher`) — agent rules

Split a geospatial field into patches, process them, stitch them back. The
root [`AGENTS.md`](../../AGENTS.md) applies too; this file adds what is
specific to the patcher. Design decisions are recorded as ADRs in
`docs/patcher/decisions.md` — read the relevant one before changing a
contract, and add one for a new decision.

## Public layout — one home per name

| Namespace | Holds |
|---|---|
| `geopatcher` (root) | the patchers (`SpatialPatcher`, `AsyncSpatialPatcher`, `TemporalPatcher`, `SpatioTemporalPatcher`), the carriers (`Patch`, `TemporalPatch`, `SpatioTemporalPatch`), the `Field` / `AsyncField` / `Domain` protocols, `RasterField` |
| `geopatcher.spatial.{geometry,sampler,window,aggregation}` | the four spatial axes, **unprefixed** (`spatial.window.Hann`) |
| `geopatcher.temporal.{geometry,sampler,window,aggregation,stencils}` | the temporal axes and stencils |
| `geopatcher.fields` | every other `Field` adapter and the domain types; extras-gated adapters resolve lazily |
| `geopatcher.matched` | co-registered multi-source patching |
| `geopatcher.run` | `parallel_map`, batching, Dask / JAX bridges, `PatchCache`, random-access views |
| `geopatcher.observe` | hooks, the resumable journal, error records, strict mode |
| `geopatcher.config` | `axis_envelope`, `from_config`, `config_from_fields` |
| `geopatcher.integrations.pipekit` | pipekit operator wrappers (`[pipekit]` extra) |

Facades re-export only; implementation lives in `_src/` (`_src/spatial/`,
`_src/temporal/`, `_src/fields/`, `_src/runners.py`, …). Enforced by
`tests/test_exports.py` (facades consistent, one home, no `Spatial*` /
`Temporal*` prefixes, removed modules stay removed) and
`tests/test_docs_paths.py` (every `geopatcher.…` path in the docs resolves;
retired spellings are rejected).

## Contracts

- **`Field`** (`_src/protocols.py`): `domain` (I/O-free metadata), `select(indexer)`,
  `with_data(array)`. Optional: `select_many(indexers)` — `run.parallel_map`
  uses it for batched reads; `aselect` for `AsyncSpatialPatcher`;
  `cache_id()` for `PatchCache` identity (raise `UnstableIdentityError`
  rather than return an identity that can change under the same id).
- **New field adapter**: implement it in `_src/fields/<name>.py`, export it
  from `geopatcher.fields`; if it needs an extra, register it in
  `LAZY_ADAPTERS` (`_src/fields/__init__.py`) so importing the namespace
  never needs the extra.
- **New axis class**: subclass the axis base in `_src/spatial/<axis>.py` (or
  `_src/temporal/`), export it unprefixed from `spatial.<axis>`, and make it
  round-trip: `from_config(obj.get_config())` must rebuild an equal object.
  Config envelopes name classes by public path (`"spatial.window.Hann"`,
  built from `_PUBLIC_MODULES` in `_src/_serialize.py`); user subclasses are
  named `module.qualname`.
- **`PatchCache` format id** `"geopatcher.PatchCache/3"` must never change
  silently — bumping it orphans every existing cache; it needs a migration
  and an ADR.
- **Object storage and COGs** come from geotoolz-cloud (`geocloud.store`,
  `geocloud.cog.CogSource`; `fields.CogField` subclasses it). Never build an
  obstore client here.
- **Extras**: raise `missing_extra(...)` from `_src/_extras.py` so the error
  names the extra; `tests/test_packaging.py` checks every extra's
  dependencies are actually used.

## Tests

- Run from this directory: `uv run pytest tests/test_sampler.py -v`.
- Coverage gate: 65 %.
- Docstring examples are illustrative (not collected as doctests).

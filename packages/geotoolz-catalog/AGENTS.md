# geotoolz-catalog (`geocatalog`) — agent rules

Find, index, join, load and stage geospatial files. The root
[`AGENTS.md`](../../AGENTS.md) applies too; this file adds what is specific
to the catalog.

## Public layout — one home per name

The root holds `GeoCatalog`, `GeoSlice`, `open_catalog` and
`query` / `intersect` / `union`; every other name lives in exactly one
namespace, named for the workflow step it serves:

| Step | Namespace | Private implementation |
|---|---|---|
| discover | `geocatalog.sources` | `_src/sources/` |
| index | `geocatalog.build` | `_src/formats/` (+ `append_files` in `_src/storage/streaming.py`) |
| hold | `geocatalog.backends` | `_src/backends/` |
| join | `geocatalog.matchup` | `_src/matchup/` |
| read | `geocatalog.load` | `_src/formats/` |
| model states | `geocatalog.states` | `_src/states.py` (queries, name maps), `_src/formats/states.py` (builder + loader) |
| save / share | `geocatalog.storage` | `_src/storage/` |
| stage | `geocatalog.staging` | `_src/staging/` |
| patch | `geocatalog.patch` | `_src/patch/` |
| grids | `geocatalog.grid` | `_src/grid.py`, `_src/geoslice.py` |
| helpers | `geocatalog.utils` | `_src/utils/` |

A data format's builder and loader share private helpers, so they live
together in `_src/formats/<format>.py`. `geocatalog.matchup` is an ordinary
module (`from geocatalog.matchup import matchup`).

Enforced by `tests/test_public_surface.py` (sorted unique `__all__`, the
root surface, one home per name, removed modules stay removed, star imports
work with every extra missing), `tests/test_docs_paths.py` (every
`geocatalog.…` path and every `from geocatalog… import …` in the docs,
README and sources resolves) and `tests/test_packaging.py` (every extra's
dependencies are actually imported).

## Contracts

- **`GeoSlice`** (`_src/geoslice.py`) is the wire format between catalog,
  loaders and patchers: `bounds` (xmin < xmax, ymin < ymax), `interval`
  (a `pd.Interval` with `closed="both"`; tz-aware endpoints become naive
  UTC), `resolution`, `crs`, optional `align`; `grid` (a `geoarray.Grid`
  of `transform`, `crs` and `shape`) is the output pixel grid. It is
  frozen — change a slice with `dataclasses.replace`.
- **The carrier** is `geoarray.GeoArray`. Every raster loader
  (`load_raster`, `aload_raster`, `load_raster_timeseries`, `load_vector`)
  returns one on `slice_.grid`, built from rasterio's NumPy output with an
  explicit `fill_value` (the nodata the mosaic used, 0 when no source
  declares one; `fill` for `load_vector`). `field_for` wraps that GeoArray,
  or, with `materialize=False`, a lazy `geoarray.RasterSource` per file. No
  public name takes or returns a georeader `GeoTensor` / `GeoData`.
  georeader is not a base dependency: it is only the engine behind the lazy
  `field_for` path (`RasterioReader` wrapped with
  `geoarray.interop.georeader.source_from_georeader`), from the `[patch]`
  extra, imported inside the function.
- **Time** is naive UTC everywhere in a catalog; use the helpers in
  `geocatalog.utils` (`to_utc_ts`, `to_naive_utc`, `to_rfc3339`) rather than
  converting by hand.
- **URIs and retries**: `geocatalog.utils.parse_uri` for every path or URI
  (local, `s3://`, `gs://`, `az://`, `https://`, `hf://`);
  `retry_transient_io` for transient network failures.
- **Extras-gated names** (DuckDB, xarray, vector, STAC, the source adapters)
  are registered in `_src/_lazy.py` `LAZY` and resolved lazily by the
  facades; a missing extra raises via `require_extra` / `missing_extra`
  (`_src/_extras.py`). Import an optional dependency softly
  (`try: import duckdb` / `except ImportError: duckdb = None`) or inside the
  function that needs it, so every module imports on a base install.
- **Persisted catalogs**: GeoParquet carries `SCHEMA_VERSION_CURRENT`; a
  schema change needs a migration in `migrate_geoparquet` and a
  `docs/catalog/schema-versions.md` entry.
- **State columns** (schema v1): `kind`, `run_time`, `member`, `producer`,
  `parents` (`STATE_COLUMNS` in `_src/_schema.py`) have fixed types. A
  frame is a state catalog only when it has `run_time` (`is_state_frame`);
  elsewhere the names are ordinary extras. Coerce them with
  `coerce_state_columns` wherever rows are read back (Parquet, DuckDB),
  and never let the streaming writer infer their types.
- **Object storage** (the `[cloud]` extra; credentials from
  `geocloud.credentials`): `stage()` downloads remote URIs with
  `geocloud.files.download` — never a copy loop of its own — and builders
  and loaders read remote files through rasterio / GDAL (`/vsi*/`) or
  `geocloud.fs` (`_resolve_uri`), geotoolz-cloud's fsspec filesystem on the
  shared pool. Never import s3fs / gcsfs / adlfs or build an fsspec
  filesystem of our own. Tests fake the download with the `cloud_stub`
  fixture, and mount a `MemoryStore` (`geocloud.store.mount`) for remote
  reads.

## Tests

- Run from this directory: `uv run pytest tests/test_geoslice.py -v`.
- `live` marks tests against real external APIs (always deselected);
  `slow` runs in the all-extras CI job. `tests/bench` is an opt-in
  benchmark suite (`pytest tests/bench --benchmark-only`).
- CI's all-extras run fails if any test is *skipped*, so gate optional
  tests on markers, not on `importorskip` of an extra CI installs.
- Coverage gate: 65 %.
- The `geocatalog` CLI (`_cli.py`, cyclopts) is covered by `tests/test_cli.py`.

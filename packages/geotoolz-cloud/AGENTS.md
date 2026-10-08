# geotoolz-cloud (`geocloud`) — agent rules

Cloud object storage for the whole stack. The root
[`AGENTS.md`](../../AGENTS.md) applies too; this file adds what is specific
to `geocloud`. It depends on georeader and obstore only — never on another
workspace package.

## What lives here

- **`geocloud.store`** — the one process-wide obstore client pool:
  `get_obstore(uri)` (one client per bucket / container / host, LRU-bounded),
  `object_key`, `get_range_bytes`, `clear_obstore_pool`,
  `set_obstore_pool_maxsize`, `SUPPORTED_SCHEMES`, and `mount` / `unmount`
  (serve one root from a store the caller built). Every package that reads
  from a bucket takes its client from here, so a process talking to one
  bucket opens one HTTP/2 connection pool. **Never construct an obstore store
  anywhere else in the workspace.**
- **`geocloud.files`** — whole-object verbs by URI or local path, on the
  pool: `ls`, `info`, `exists`, `read_bytes`, `write_bytes`, `open`,
  `download`, `upload`, `copy`, `sync`, `rm`, `sign`, `ObjectInfo`. A
  package that moves files (staging, product downloads) calls these rather
  than streaming bytes itself. Local paths go through an obstore
  `LocalStore` built in `_src/files.py`, the one other place a store is
  constructed.
- **`geocloud.credentials`** — the per-root registry `get_obstore` merges
  under `storage_options` (`set_credentials`, `remove_credentials`,
  `credential_roots`, `load_credentials` of a TOML file, auto-loaded from
  `credentials_path()`), `gdal_access` (GDAL
  path + config options from the same registry) and `redact`. Code never
  writes credentials to `os.environ`; error messages that show a URI pass
  it through `redact` (`_r` in `_src/store.py` / `_src/files.py`).
- **`geocloud.cog`** (`[cog]` extra, async-geotiff) — `CogSource`
  (`open` / `aopen`, `read_window(s)` / `aread_window(s)`, `identity()`,
  `object_version()`; pickles by URL so it ships to process pools),
  `CogDomain`, `AsyncCogReader` and the async `read_*` mirrors of
  `georeader.read`. Batched reads fetch every tile a set of windows touches
  exactly once.

`geopatcher.fields.CogField` subclasses `CogSource`; extend `CogSource` here
rather than adding read logic to the patcher.

## Upstream gaps

`_src/cog_source.py` carries two local corrections over async-geotiff,
tracked in issues #414 (non-zero raster tiepoint) and #415 (CRS upstream
cannot build). Keep their regression tests unchanged; remove a correction
only when an async-geotiff release fixes it, raising the `[cog]` floor in the
same change.

## Tests

- Run from this directory: `uv run pytest tests/test_store.py -v`; tests use
  `obstore.store.LocalStore` / `MemoryStore` (`mount` one at an `s3://` or
  `az://` root to exercise remote URIs), no network.
- Sync code that needs an async obstore call (a streamed cross-store copy)
  drives it with `_src/aio.py`, which also works under a running loop.
- `tests/conftest.py` sets `GEOCLOUD_CREDENTIALS=` and clears the registry
  around every test, so no test reads a developer's credentials file.
- Coverage gate: 80 %.
- CI's `geotoolz-cloud-base` job installs the package without extras: the
  pool must work, and anything needing async-geotiff must fail at use with
  an install hint naming `geotoolz-cloud[cog]` (`_src/extras.py`).

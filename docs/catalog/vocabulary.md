# Parameter vocabulary

Each concept has one keyword across every public `geocatalog` function,
class and CLI flag, so you can guess a name from any other call. Old
spellings still work for one minor release with a `DeprecationWarning`;
passing both the old and the new name is a `TypeError`.

## Names

| Concept | Name | Values / notes | Was |
|---|---|---|---|
| Kind of data indexed | `kind` / `.kind` | `"raster"`, `"xarray"`, `"vector"` (`CatalogKind`) | `backend=`, `.backend` |
| Where the catalog lives | `engine` | `"memory"`, `"duckdb"` (`StorageEngine`); `open_catalog` adds `"auto"` | `backend=` on the builders, STAC and CLI |
| Spatial-join method | `join` | `InMemoryGeoCatalog.intersect(join="sjoin")` | `engine=` |
| Catalog CRS | `crs` | the CRS the footprints are stored in (builders, `CatalogBundle`, `open_catalog`) | `target_crs=`, `CatalogBundle.target_crs` |
| CRS of a query box | `crs` | `query(bounds=..., crs=...)` — the CRS `bounds` is expressed in | — |
| Spatial extent | `bounds` | `(minx, miny, maxx, maxy)` | `bbox=` (`from_stac_search`, CLI `--bbox`) |
| Result cap | `limit` | | `max_items=` (`from_stac_search`) |
| Artifact read | `source` | path or URI of an existing artifact | `path=` (`from_geoparquet`), `src=` |
| Artifact written | `out_path` | | `path=` (`to_geoparquet`), `dst=` |
| Files to index | `filepaths` | the builders' positional input | — |
| File-open threads | `max_open_workers` | `load_raster`, `aload_raster` | `concurrency=` (`aload_raster`) |
| Extraction strategy | `concurrency` | `"sequential"` / `"async"` (`build_raster_catalog`) | — |

The CLI follows the same table: `--crs`, `--engine`, `--bounds`. The old
flags (`--target-crs`, `--backend`, `--bbox`) work for one minor release,
print a deprecation notice and are hidden from `--help`. Passing an old and
a new flag together is an error.

`stats --json` reports `kind`, and `query --json` reports `bounds`. For one
release they also report the old `backend` and `bbox` keys.

**On-disk names do not change,** so artifacts written by any release stay
readable. That covers the reserved `_backend` column, the `get_config()`
`"engine"` key, and the bundle `_meta.json` keys `target_crs` / `backend`. `get_config()` reports the data kind under
`"kind"` (and, for one release, the old `"backend"` key).

## Defaults that look different on purpose

| Parameter | Default | Why |
|---|---|---|
| `iter_rows(batch_size=)` | `1024` | rows fetched per round trip while *iterating*; small so the first row arrives quickly |
| builders' / writers' `batch_size` | `10_000` | rows per Parquet row group while *writing*; large so row groups compress well |
| builders' `n_workers` | `1` | worker *processes* for metadata extraction; parallelism is opt-in because forking has a fixed cost |
| `load_raster_timeseries(n_workers=)` | `4` | *threads* reading daily mosaics; rasterio releases the GIL, so threads are cheap |

## Errors

Errors about a catalog's *state or artifacts* derive from
`GeoCatalogError` *and* from the builtin a caller would have caught
before the hierarchy existed:

| Class | Also a | Raised when |
|---|---|---|
| `CatalogMetadataError` | `ValueError` | `strict=True` and an artifact's metadata is missing or unreadable |
| `CatalogSchemaError` | `ValueError` | an artifact's `_schema_version` cannot be read by this release |
| `CatalogClosedError` | `RuntimeError` | a closed `DuckDBGeoCatalog` (or one derived from it) is used; also a `duckdb.ConnectionException` |

Bad *arguments* are not `GeoCatalogError`s. A wrong value (an unknown
`kind`, malformed `bounds`) raises `ValueError`, a wrong type raises
`TypeError`, and an unparsable CRS raises `pyproj.exceptions.CRSError`. Each case raises the same type on both
backends (`tests/test_vocabulary.py`). To catch every failure, catch
`GeoCatalogError`, `ValueError`, `TypeError` and `CRSError`.

## Third-party catalogs

The `GeoCatalog` protocol names the attribute `kind`. For one minor
release, a catalog that still exposes `backend` instead passes
`isinstance(obj, GeoCatalog)` with a `DeprecationWarning`, and
`field_for` reads its `backend`.

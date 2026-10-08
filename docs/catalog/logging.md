# Logging

`geocatalog` uses [`loguru`](https://loguru.readthedocs.io/) for all
internal logging. Following loguru's [library
recipe](https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application),
the package disables its own logger at import time:

```python
# src/geocatalog/__init__.py
from loguru import logger as _logger
_logger.disable("geocatalog")
```

so just importing the library produces no output. This is the default
because most consumers don't want a third-party package writing to
stderr unprompted.

## Opting in

A consumer app turns the package's logs back on with one call:

```python
from loguru import logger
logger.enable("geocatalog")
```

After this, every `INFO` / `WARNING` / `ERROR` / `DEBUG` record emitted
from inside `geocatalog.*` reaches loguru's default stderr sink (or any
sink the consumer added with `logger.add(...)`).

## Routing logs to a file

`loguru.logger.add(...)` is the entry point for all sink configuration —
files, rotation, formatting, structured (JSON) output. A typical
catalog-build setup that wants to keep a record of every skipped or
fallback file looks like:

```python
from loguru import logger
import geocatalog as gc

logger.enable("geocatalog")
logger.add(
    "catalog-build.log",
    rotation="50 MB",          # roll the file when it crosses 50 MB
    retention=10,              # keep the 10 most recent rolled files
    backtrace=True,            # include locals on exception
    diagnose=True,
)

cat = gc.build.build_raster_catalog(paths, ...)
```

## Where logs come from

`geocatalog` emits records from these public entry points today (the
records' logger names are the internal modules under `geocatalog`, so
`logger.enable("geocatalog")` covers all of them):

- `build_raster_catalog` — `WARNING` when a filename doesn't match the
  date regex (the file is skipped).
- `build_vector_catalog` — `WARNING` on empty vector files and regex
  misses.
- `build_raster_catalog` / `build_vector_catalog` — `INFO` when the
  `duckdb` backend is asked to canonicalise footprints to EPSG:4326
  because no `crs` was passed.
- `StreamingParquetWriter` — `ERROR` (with traceback) when closing the
  parquet handle fails while a partial file is being discarded after an
  error.
- `append_files` — `INFO` when input files are skipped because their
  `filepath` is already indexed in the archive.
- `sort_geoparquet` — `DEBUG` after a Hilbert-sorted rewrite completes.

All call sites use loguru's `{}` placeholder style, not stdlib's `%s`.

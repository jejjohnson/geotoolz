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

cat = gc.build_raster_catalog(paths, ...)
```

## Where logs come from

`geocatalog` emits records from these paths today:

- `geocatalog._src.raster._filepath_to_row` — `WARNING` when a filename
  doesn't match the date regex (the file is skipped).
- `geocatalog._src.vector._vector_row_for_stream` — `WARNING` on empty
  vector files and regex misses.
- `geocatalog._src.raster.build_raster_catalog` /
  `geocatalog._src.vector.build_vector_catalog` — `INFO` when the
  `duckdb` backend is asked to canonicalise footprints to EPSG:4326
  because no `target_crs` was passed.
- `geocatalog._src.streaming.StreamingParquetWriter.__exit__` —
  `ERROR` (with traceback) when the writer's `close()` itself fails
  during an unwind.
- `geocatalog._src.streaming.sort_geoparquet` — `DEBUG` after a
  Hilbert-sorted rewrite completes.

All call sites use loguru's `{}` placeholder style, not stdlib's `%s`.

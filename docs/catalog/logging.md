# Logging

Turn on `geocatalog`'s logs with `logger.enable("geocatalog")` to see which
files a build skipped and why. The package logs through
[loguru](https://loguru.readthedocs.io/) and, following loguru's
[library recipe](https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application),
disables its logger at import, so importing it prints nothing.

## Log a build to a file

```python
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog

root: Path = Path(tempfile.mkdtemp())
for name in ("s2_20240605.tif", "notes.tif"):                        # the second has no date
    with rasterio.open(
        root / name, "w", driver="GTiff", width=10, height=10, count=1, dtype="uint8",
        crs="EPSG:32611", transform=from_origin(500_000, 4_300_000, 10, 10),
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.uint8))            # (1, 10, 10) uint8

logger.enable("geocatalog")
sink: int = logger.add(root / "catalog-build.log", rotation="50 MB", retention=10)
catalog: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    sorted(root.glob("*.tif")), filename_regex=r"s2_(?P<date>\d{8})\.tif"
)                                                                    # 1 row; WARNING: skipping notes.tif
logger.remove(sink)
```

`logger.add` takes any loguru sink: a file with rotation, a JSON
serializer or your own function.

## What gets logged

| Where | Level | When |
| --- | --- | --- |
| `build_raster_catalog` | `WARNING` | a file name does not match `filename_regex`; the file is skipped |
| `build_vector_catalog` | `WARNING` | an empty vector file or a regex miss |
| `build_raster_catalog` / `build_vector_catalog` | `INFO` | `engine="duckdb"` without `crs=` falls back to EPSG:4326 |
| `append_files` | `INFO` | files already indexed in the archive are skipped |
| `StreamingParquetWriter` | `ERROR` | closing the file fails while a partial write is discarded |
| `sort_geoparquet` | `DEBUG` | a Hilbert-sorted rewrite finished |

Grid-alignment notices are Python `warnings`, not log records, so they
show without `logger.enable`; see
[Grid alignment](how-to/grid-alignment.md).

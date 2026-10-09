# Index large archives

Past ~10⁵ files, build into GeoParquet instead of RAM, and query it with
`DuckDBGeoCatalog`. All three patterns below need the `[duckdb]` extra;
[Backends](../concepts.md#backends) explains the trade-off.

## Stream a build into GeoParquet

`engine="duckdb"` on any builder extracts rows in worker processes and
streams them into one GeoParquet 1.1 file. Peak RAM is about
`batch_size × row size`, not the whole catalog:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import DuckDBGeoCatalog

if __name__ == "__main__":                                           # guard needed for n_workers > 1
    # 20 dated tiles stand in for an archive of millions.
    root: Path = Path(tempfile.mkdtemp())
    for day in range(1, 21):
        with rasterio.open(
            root / f"s2_202406{day:02d}.tif", "w", driver="GTiff", width=100, height=100, count=1,
            dtype="uint16", crs="EPSG:32611", transform=from_origin(500_000 + 500 * day, 4_303_000, 10, 10),
        ) as dst:
            dst.write(np.zeros((1, 100, 100), dtype=np.uint16))     # (1, 100, 100) uint16

    catalog: DuckDBGeoCatalog = gc.build.build_raster_catalog(
        sorted(root.glob("*.tif")),
        filename_regex=r"s2_(?P<date>\d{8})\.tif",
        engine="duckdb",
        out_path=root / "archive.parquet",                           # required with engine="duckdb"
        crs="EPSG:4326",                                             # the default for shared artifacts
        n_workers=4,                                                 # processes opening files
        sort_by=("start_time", "geometry_hilbert"),                  # clusters rows for pruning
    )                                                                # 20 rows, opened lazily
    week: gc.GeoSlice = gc.GeoSlice(
        bounds=(-118.0, 38.0, -116.0, 40.0),                         # lon/lat around the tiles
        interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-07"), closed="both"),
        resolution=(0.0001, 0.0001),
        crs="EPSG:4326",
    )
    hits: DuckDBGeoCatalog = catalog.query(week)                     # 7 rows
```

The file has a per-row `bbox` column and Hilbert-sorted rows, so a small
query reads row-group statistics instead of every geometry. Without `crs=`
the streaming build stores lon/lat footprints. `build_xarray_catalog` is
the exception: it needs the native `crs=`.

## Append to a partitioned archive

For an archive that grows, write a directory of Hive-partitioned shards
with `geocatalog.build.append_files`. You supply `extract_fn`: a
module-level (picklable) function that maps one file to a row dict, or
`None` to skip it:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import DuckDBGeoCatalog


def extract_row(path: str | Path) -> dict | None:
    """One catalog row per file: footprint, day-long interval, CRS, path."""
    with rasterio.open(path) as src:
        day: pd.Timestamp = pd.Timestamp(Path(path).stem.split("_")[1])
        return {
            "filepath": str(path),
            "geometry": shapely.box(*src.bounds),                    # in the archive CRS
            "start_time": day,
            "end_time": day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
            "crs": str(src.crs),
        }


def write_tiles(root: Path, dates: list[str]) -> list[Path]:
    """One 1 km tile per date, UTM 11N."""
    paths: list[Path] = []
    for date in dates:
        path: Path = root / f"s2_{date}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=100, height=100, count=1, dtype="uint16",
            crs="EPSG:32611", transform=from_origin(500_000, 4_303_000, 10, 10),
        ) as dst:
            dst.write(np.zeros((1, 100, 100), dtype=np.uint16))     # (1, 100, 100) uint16
        paths.append(path)
    return paths


root: Path = Path(tempfile.mkdtemp())
archive: Path = root / "archive"                                     # a local directory, not a .parquet file
first: list[Path] = write_tiles(root, ["20240601", "20240715"])
later: list[Path] = write_tiles(root, ["20240716", "20240801"])

options: dict = {"crs": "EPSG:32611", "kind": "raster", "partition_by": ("year", "month")}
catalog: DuckDBGeoCatalog = gc.build.append_files(archive, first, extract_row, **options)          # 2 rows
catalog = gc.build.append_files(archive, first + later, extract_row, **options)                    # 4 rows: the 2 known files are skipped
july: DuckDBGeoCatalog = gc.open_catalog(archive, engine="duckdb").sql(where="month = 7")          # 2 rows
```

How `append_files` behaves:

- **New rows only.** Existing shards are never rewritten; `year`, `month`
  and `day` are derived from `start_time`.
- **Idempotent.** Files whose `filepath` is already indexed are skipped,
  so re-running an interrupted call completes it without duplicates.
- **Atomic shards.** Each shard is written to a hidden temp file and
  renamed into place.
- **One layout.** A `partition_by` that differs from the archive's raises
  `ValueError`.
- **Local only.** Build the directory locally, then sync it to object
  storage.

`open_catalog` reads a directory or a glob as one table. Partition columns
in a `sql(where=…)` predicate prune whole directories.

## Query a remote artifact

DuckDB reads GeoParquet over HTTP range requests and fetches only the row
groups a query touches. The CRS of a remote file cannot be read from its
metadata yet, so pass it. This sketch needs your own bucket:

<!-- docs-check: skip -->
```python
import pandas as pd

import geocatalog as gc
from geocatalog.backends import DuckDBGeoCatalog

archive: DuckDBGeoCatalog = gc.open_catalog("s3://<your-bucket>/s2_archive.parquet", crs="EPSG:4326")
shards: DuckDBGeoCatalog = gc.open_catalog("s3://<your-bucket>/s2_archive/**/*.parquet", crs="EPSG:4326")

june: gc.GeoSlice = gc.GeoSlice(
    bounds=(-120.25, 38.85, -119.85, 39.30),
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(0.0001, 0.0001),
    crs="EPSG:4326",
)
hits: DuckDBGeoCatalog = archive.query(june)                         # reads ~MB of row groups, not GB
```

A remote directory of shards needs the glob; a bare remote directory is not
expanded.

## Performance knobs

| Knob | Raise it when | Cost |
| --- | --- | --- |
| `batch_size` (10 000) | many small rows leave the writer idle | more peak RAM |
| `n_workers` (1) | opening files is the bottleneck | disk and network load |
| `sort_by=("start_time", "geometry_hilbert")` | most queries are AOI + time | a slower build |
| `ordered=True` | you need byte-identical output | less parallelism |
| rows per shard | queries touch few partitions | aim for 10⁵–10⁶ rows per shard |

Under ~10⁵ files, skip all of this: the in-memory catalog is faster end to
end, and both backends share one protocol.

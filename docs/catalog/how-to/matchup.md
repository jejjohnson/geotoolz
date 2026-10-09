# Match observations across sources

`geocatalog.matchup.matchup` pairs rows from two or more catalogs that
overlap in space and fall close in time. Use it to find Sentinel-2 /
Landsat pairs, satellite / in-situ collocations or any multi-sensor
training set.

## Pair two catalogs

Pick a spatial strategy and a temporal strategy, then load each pair onto
one shared grid:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog
from geocatalog.matchup import Intersects, MatchupRow, NearestInTime


def write_scenes(root: Path, prefix: str, dates: list[str], res: float, x0: float) -> list[Path]:
    """One 6 km × 6 km single-band scene per date at `res` metres, UTM 11N."""
    n: int = int(6_000 / res)
    paths: list[Path] = []
    for date in dates:
        path: Path = root / f"{prefix}_{date}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=n, height=n, count=1, dtype="uint16",
            crs="EPSG:32611", transform=from_origin(x0, 4_308_000, res, res),
        ) as dst:
            dst.write(np.full((1, n, n), 1_000, dtype=np.uint16))   # (1, n, n) uint16
        paths.append(path)
    return paths


root: Path = Path(tempfile.mkdtemp())
s2: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    write_scenes(root, "s2", ["20240605", "20240615", "20240625"], 10.0, 501_000),
    filename_regex=r"_(?P<date>\d{8})\.tif", crs="EPSG:32611",
)                                                                    # 3 rows, 10 m
landsat: InMemoryGeoCatalog = gc.build.build_raster_catalog(
    write_scenes(root, "ls", ["20240606", "20240622"], 30.0, 504_000),
    filename_regex=r"_(?P<date>\d{8})\.tif", crs="EPSG:32611",
)                                                                    # 2 rows, 30 m, 3 km further east

pairs: list[MatchupRow] = list(
    gc.matchup.matchup(s2, landsat, spatial=Intersects(), temporal=NearestInTime(dt="3D"))
)                                                                    # 2 pairs: 06-05 ↔ 06-06, 06-25 ↔ 06-22

pair: MatchupRow = pairs[0]
s2_path, ls_path = pair.member_ids                                   # a plain catalog is keyed by filepath
grid: gc.GeoSlice = gc.GeoSlice(
    bounds=pair.geometry_intersect.bounds,                           # the 3 km × 6 km overlap
    interval=pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both"),
    resolution=(30.0, 30.0),                                         # the coarser sensor's grid
    crs="EPSG:32611",
)
s2_chip: GeoTensor = gc.load.load_raster(s2.where(f"filepath == '{s2_path}'"), grid)        # (1, 200, 100) uint16
ls_chip: GeoTensor = gc.load.load_raster(landsat.where(f"filepath == '{ls_path}'"), grid)   # (1, 200, 100) uint16
```

The 06-15 scene has no Landsat partner within three days, so it drops out.
Each `MatchupRow` carries:

- `member_ids` and `member_roles`, primary first. Plain catalogs use
  `filepath` as the id; bundle rows use their STAC or CMR id.
- `geometry_intersect`, the shared footprint in the working CRS.
- `time_offset_sec`, each member's offset from the primary's midpoint
  (`time_reference`).
- `matchup_id`, a content hash: the same inputs always give the same ids.

## Strategies

| Spatial | Matches when |
| --- | --- |
| `Intersects()` | the footprints overlap at all |
| `IouAtLeast(threshold)` | intersection over union ≥ `threshold` |
| `CentroidWithin(buffer)` | the secondary's centroid falls in the buffered primary |
| `Contains()` | the primary fully contains the secondary |

| Temporal | Keeps |
| --- | --- |
| `NearestInTime(dt)` | the one nearest secondary within `dt` |
| `WithinWindow(start, end)` | every secondary whose midpoint falls in the window around the primary's |
| `Synchronous(tolerance)` | every secondary whose interval overlaps the primary's |

## Options

- **N-way joins.** Pass a mapping, `secondary={"landsat": ls, "buoys": b}`.
  `join="all"` (default) needs every role; `join="any"` keeps partial
  tuples.
- **Working CRS.** `crs=` defaults to the primary's CRS. Distances such as
  `CentroidWithin.buffer` are in its units, so use a projected CRS for
  metres.
- **Filter first.** Narrow the inputs with `query` or `where` before the
  call; the join runs in memory.
- **Persist.** `CatalogBundle.write_matchups(rows, tag=...)` writes
  `matchups.parquet`; see
  [Record provenance with a bundle](stac-ingestion.md#record-provenance-with-a-bundle).

To cut matched pairs into patches across sensors, use
`geopatcher.matched.MatchedField`; the
[design record](../design/query-matchup.md) walks through it.

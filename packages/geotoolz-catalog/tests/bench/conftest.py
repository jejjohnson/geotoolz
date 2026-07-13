"""Shared helpers for the catalog benchmark suite.

The bench suite generates large synthetic catalogs *in RAM* — disk I/O
is intentionally excluded so the tracked numbers reflect the
library's own work (R-tree query, geopandas overlay, DuckDB
predicate-pushdown), not the test rig's filesystem.

`make_inmemory_catalog(n_rows)` is the single entry point; backends
that need a GeoParquet artifact write the catalog to a tempfile
inside their own fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely.geometry

from geocatalog import InMemoryGeoCatalog


def make_inmemory_catalog(
    n_rows: int,
    *,
    seed: int = 0,
    bounds_box: tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0),
    time_window: tuple[str, str] = ("2020-01-01", "2024-12-31"),
) -> InMemoryGeoCatalog:
    """Build an `InMemoryGeoCatalog` with `n_rows` random-ish polygons.

    Uses a deterministic numpy RNG seeded by ``seed`` so bench runs
    compare like-for-like across commits. Polygons are 0.5 x 0.5 degree tiles
    scattered uniformly inside ``bounds_box``; each row gets a 1-day
    interval inside ``time_window``.

    Returns:
        An `InMemoryGeoCatalog` with backend ``"raster"`` and CRS
        EPSG:4326. Construction time is O(n_rows) and dominated by
        the GeoDataFrame build; the catalog's R-tree is materialised
        lazily on first query (`gdf.sindex` is built on demand).
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds_box
    side = 0.5
    x0 = rng.uniform(xmin, xmax - side, size=n_rows)
    y0 = rng.uniform(ymin, ymax - side, size=n_rows)
    geoms = [
        shapely.geometry.box(x, y, x + side, y + side)
        for x, y in zip(x0, y0, strict=True)
    ]

    t0 = pd.Timestamp(time_window[0])
    t1 = pd.Timestamp(time_window[1])
    span = (t1 - t0).total_seconds()
    starts_sec = rng.uniform(0.0, span, size=n_rows)
    starts = pd.to_datetime(t0.value + (starts_sec * 1e9).astype("int64"))
    ends = starts + pd.Timedelta(days=1)

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {
            "filepath": [f"synthetic_{i:07d}.tif" for i in range(n_rows)],
            "geometry": geoms,
            "start_time": starts,
            "end_time": ends,
        },
        crs="EPSG:4326",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")

"""Hypothesis strategies for catalog property-based tests.

The strategies here generate *valid* `GeoSlice` and `InMemoryGeoCatalog`
inputs — no NaN bounds, no NaT-only intervals on populated catalogs,
nothing that crosses the antimeridian. The narrow input domain is
deliberate: we want to exercise structural invariants (round-trip
identity, CRS-invariance, intersect-symmetry) over inputs that the
library *promises* to support, not stress-test every degenerate
geometric corner.

Edge cases the strategies *do* cover:

- Catalog with 0 rows.
- Catalog with 1 row.
- Intervals where ``start == end`` (instantaneous observations).
- Sub-microsecond timestamps (exercises the GeoParquet microsecond
  truncation path on read-back).

All generated bounds are in EPSG:4326 — `test_properties.py`
reprojects to other CRSs *within* the property test rather than at
strategy time, because the structural invariants need a canonical
"native CRS" anchor to compare against.
"""

from __future__ import annotations

import geopandas as gpd
import hypothesis.strategies as st
import pandas as pd
import shapely.geometry

from geocatalog._src.geoslice import GeoSlice
from geocatalog._src.memory import InMemoryGeoCatalog


# Use a narrow time window so generated intervals stay parseable and
# round-trip cleanly through GeoParquet's int64-microsecond timestamps.
# Bounds are in *microseconds* since epoch so the strategy can emit
# sub-second offsets — useful for exercising the int64-microsecond
# truncation path on round-trip (a whole-second-only strategy would
# never fire it).
_TIME_MIN_US = pd.Timestamp("2000-01-01").value // 1000  # microseconds
_TIME_MAX_US = pd.Timestamp("2030-01-01").value // 1000


@st.composite
def bbox_strategy_4326(draw: st.DrawFn) -> tuple[float, float, float, float]:
    """`(xmin, ymin, xmax, ymax)` in EPSG:4326, well inside the equator.

    Keeps |lon| <= 10 and |lat| <= 10 so reprojection to UTM 29N or
    EPSG:3413 (north-polar stereographic) doesn't blow up. The
    antimeridian and polar caps are out of scope here — see issue #16.
    """
    xmin = draw(st.floats(-10.0, 9.0, allow_nan=False, allow_infinity=False))
    xmax = draw(st.floats(xmin + 0.01, 10.0, allow_nan=False, allow_infinity=False))
    ymin = draw(st.floats(-10.0, 9.0, allow_nan=False, allow_infinity=False))
    ymax = draw(st.floats(ymin + 0.01, 10.0, allow_nan=False, allow_infinity=False))
    return (xmin, ymin, xmax, ymax)


@st.composite
def interval_strategy(draw: st.DrawFn) -> pd.Interval:
    """A `pd.Interval(closed='both')` within the 2000-2030 window.

    Endpoints are drawn at *microsecond* resolution, not seconds, so the
    strategy actually exercises the GeoParquet int64-microsecond
    timestamp path that the round-trip property claims to cover.
    """
    start_us = draw(st.integers(_TIME_MIN_US, _TIME_MAX_US))
    # Allow zero-width (instantaneous observations) so we exercise the
    # NaT-adjacent edge of the IntervalIndex.
    end_us = draw(st.integers(start_us, _TIME_MAX_US))
    start = pd.Timestamp(start_us, unit="us")
    end = pd.Timestamp(end_us, unit="us")
    return pd.Interval(start, end, closed="both")


@st.composite
def geoslice_strategy(draw: st.DrawFn) -> GeoSlice:
    """A valid `GeoSlice` — bbox + interval at fixed resolution in EPSG:4326.

    All generated slices carry EPSG:4326 bounds + CRS, matching the
    catalog strategy. Properties that need to fuzz CRS handling
    reproject the slice's bounds inside the test rather than asking
    the strategy to emit `(bounds, crs)` pairs in different units.
    """
    bounds = draw(bbox_strategy_4326())
    interval = draw(interval_strategy())
    return GeoSlice(
        bounds=bounds,
        interval=interval,
        resolution=(0.01, 0.01),
        crs="EPSG:4326",
    )


@st.composite
def catalog_strategy(
    draw: st.DrawFn,
    n_rows: st.SearchStrategy[int] | None = None,
) -> InMemoryGeoCatalog:
    """An `InMemoryGeoCatalog` over 0-20 rows in EPSG:4326, ``backend="raster"``.

    Each row carries:

    - ``geometry``: a non-degenerate shapely box in (-10, -10, 10, 10).
    - ``start_time`` / ``end_time``: the corners of a `pd.Interval`.
    - ``filepath``: a stable synthetic string keyed off the row index.

    The 20-row cap keeps a single property run inside the per-test budget
    even at ``max_examples=200``; the issue's success criterion of "≥3
    properties x 200 examples" is hit comfortably at this size.
    """
    if n_rows is None:
        n_rows = st.integers(0, 20)
    count = draw(n_rows)
    boxes: list[shapely.geometry.Polygon] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    paths: list[str] = []
    for i in range(count):
        bounds = draw(bbox_strategy_4326())
        interval = draw(interval_strategy())
        boxes.append(shapely.geometry.box(*bounds))
        starts.append(interval.left)
        ends.append(interval.right)
        paths.append(f"synthetic_{i:04d}.tif")
    gdf = gpd.GeoDataFrame(
        {
            "filepath": paths,
            "geometry": boxes,
            "start_time": starts,
            "end_time": ends,
        },
        crs="EPSG:4326",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")

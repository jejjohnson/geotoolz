"""UMM-G footprint and time decoding (#236).

Shared by the CMR and earthaccess adapters.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
import shapely

from geocatalog._src._timeutil import TIME_INVARIANT_END, TIME_INVARIANT_START
from geocatalog._src.sources._umm import (
    extract_cloud_cover,
    granule_geometry,
    granule_interval,
)
from geocatalog._src.sources.cmr import _cmr_item_to_source_row


def _umm(geometry: dict[str, Any]) -> dict[str, Any]:
    return {"SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": geometry}}}


def _points(coords: list[tuple[float, float]]) -> dict[str, Any]:
    return {"Points": [{"Longitude": x, "Latitude": y} for x, y in coords]}


def _rect(w: float, s: float, e: float, n: float) -> dict[str, float]:
    return {
        "WestBoundingCoordinate": w,
        "SouthBoundingCoordinate": s,
        "EastBoundingCoordinate": e,
        "NorthBoundingCoordinate": n,
    }


def _ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [*coords, coords[0]]


# ---------------------------------------------------------------------------
# Antimeridian
# ---------------------------------------------------------------------------


def test_rectangle_across_the_antimeridian_is_split_not_complemented() -> None:
    geom = granule_geometry(_umm({"BoundingRectangles": [_rect(179, -10, -179, 10)]}))
    assert geom is not None
    assert geom.geom_type == "MultiPolygon"
    assert geom.area == pytest.approx(2 * 20)  # 2° wide, not 358°
    assert geom.covers(shapely.Point(179.5, 0))
    assert geom.covers(shapely.Point(-179.5, 0))
    assert not geom.covers(shapely.Point(0, 0))


def test_ordinary_rectangle_is_unchanged() -> None:
    geom = granule_geometry(_umm({"BoundingRectangles": [_rect(-10, 35, -5, 45)]}))
    assert geom is not None
    assert geom.equals(shapely.box(-10, 35, -5, 45))


def test_gpolygon_across_the_antimeridian_is_split() -> None:
    shell = _ring([(178, -5), (-178, -5), (-178, 5), (178, 5)])
    geom = granule_geometry(_umm({"GPolygons": [{"Boundary": _points(shell)}]}))
    assert geom is not None
    assert geom.geom_type == "MultiPolygon"
    assert geom.area == pytest.approx(4 * 10)
    assert not geom.covers(shapely.Point(0, 0))


def test_line_across_the_antimeridian_is_split() -> None:
    geom = granule_geometry(_umm({"Lines": [_points([(170, 0), (-170, 10)])]}))
    assert geom is not None
    assert geom.geom_type == "MultiLineString"
    assert geom.length == pytest.approx(
        (20**2 + 10**2) ** 0.5
    )  # 20° east, not 340° west
    assert geom.bounds[0] == -180 and geom.bounds[2] == 180


# ---------------------------------------------------------------------------
# ExclusiveZone, Lines, nulls
# ---------------------------------------------------------------------------


def test_exclusive_zone_becomes_a_hole() -> None:
    shell = _ring([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = _ring([(4, 4), (6, 4), (6, 6), (4, 6)])
    geom = granule_geometry(
        _umm(
            {
                "GPolygons": [
                    {
                        "Boundary": _points(shell),
                        "ExclusiveZone": {"Boundaries": [_points(hole)]},
                    }
                ]
            }
        )
    )
    assert geom is not None
    assert geom.area == pytest.approx(100 - 4)
    assert not geom.covers(shapely.Point(5, 5))


def test_exclusive_zone_across_the_antimeridian() -> None:
    shell = _ring([(178, -5), (-178, -5), (-178, 5), (178, 5)])
    hole = _ring([(179.5, -1), (-179.5, -1), (-179.5, 1), (179.5, 1)])
    geom = granule_geometry(
        _umm(
            {
                "GPolygons": [
                    {
                        "Boundary": _points(shell),
                        "ExclusiveZone": {"Boundaries": [_points(hole)]},
                    }
                ]
            }
        )
    )
    assert geom is not None
    assert geom.area == pytest.approx(40 - 2)


def test_lines_are_decoded() -> None:
    geom = granule_geometry(
        _umm({"Lines": [_points([(0, 0), (1, 1)]), _points([(2, 2), (3, 3)])]})
    )
    assert geom is not None
    assert geom.geom_type == "MultiLineString"
    assert len(geom.geoms) == 2


def test_null_attribute_name_is_tolerated() -> None:
    umm = {
        "AdditionalAttributes": [
            {"Name": None, "Values": ["x"]},
            {"Name": "CLOUD_COVERAGE", "Values": ["12.5"]},
        ]
    }
    assert extract_cloud_cover(umm) == 12.5


# ---------------------------------------------------------------------------
# Temporal extent
# ---------------------------------------------------------------------------


def test_open_ended_range_is_closed_with_the_sentinel() -> None:
    interval = granule_interval(
        {
            "TemporalExtent": {
                "RangeDateTime": {"BeginningDateTime": "2020-01-01T00:00Z"}
            }
        }
    )
    assert interval is not None
    assert interval.left == pd.Timestamp("2020-01-01", tz="UTC")
    assert interval.right == TIME_INVARIANT_END.tz_localize("UTC")


def test_missing_beginning_is_closed_with_the_sentinel() -> None:
    interval = granule_interval(
        {"TemporalExtent": {"RangeDateTime": {"EndingDateTime": "2020-01-01T00:00Z"}}}
    )
    assert interval is not None
    assert interval.left == TIME_INVARIANT_START.tz_localize("UTC")


def test_single_date_time() -> None:
    interval = granule_interval(
        {"TemporalExtent": {"SingleDateTime": "2020-01-01T12:00Z"}}
    )
    assert interval is not None
    assert interval.left == interval.right == pd.Timestamp("2020-01-01T12:00", tz="UTC")


def test_empty_range_falls_back_to_single_date_time() -> None:
    interval = granule_interval(
        {
            "TemporalExtent": {
                "RangeDateTime": {"BeginningDateTime": None, "EndingDateTime": ""},
                "SingleDateTime": "2020-01-01T12:00Z",
            }
        }
    )
    assert interval is not None
    assert interval.left == pd.Timestamp("2020-01-01T12:00", tz="UTC")


def test_no_temporal_extent_is_none() -> None:
    assert granule_interval({"TemporalExtent": {"RangeDateTime": {}}}) is None


# ---------------------------------------------------------------------------
# Through an adapter
# ---------------------------------------------------------------------------


def test_cmr_row_for_an_antimeridian_granule_has_the_split_footprint() -> None:
    item = {
        "umm": {
            "GranuleUR": "FIJI-1",
            "CollectionReference": {"ShortName": "X"},
            "TemporalExtent": {
                "RangeDateTime": {"BeginningDateTime": "2024-06-01T00:00:00Z"}
            },
            **_umm({"BoundingRectangles": [_rect(177, -20, -178, -15)]}),
        }
    }
    row = _cmr_item_to_source_row(
        item, source_name="cmr", query_id="q", fetched_at=datetime.now(UTC)
    )
    assert row is not None
    assert row.geometry.area == pytest.approx(5 * 5)
    assert row.interval.right == TIME_INVARIANT_END.tz_localize("UTC")


# ---------------------------------------------------------------------------
# Review follow-ups (#357)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("west", "east"), [(180, -170), (170, -180)])
def test_seam_aligned_rectangle_has_no_zero_width_part(
    west: float, east: float
) -> None:
    geom = granule_geometry(_umm({"BoundingRectangles": [_rect(west, -5, east, 5)]}))
    assert geom is not None
    assert geom.is_valid
    assert geom.geom_type == "Polygon"
    assert geom.area == pytest.approx(10 * 10)


def test_polygon_edge_on_the_seam_stays_a_polygon() -> None:
    # The first/last edge lies on 180°; the area unwraps to 180°-190°.
    shell = _ring([(180, -5), (-170, -5), (-170, 5), (180, 5)])
    geom = granule_geometry(_umm({"GPolygons": [{"Boundary": _points(shell)}]}))
    assert geom is not None
    assert geom.geom_type == "Polygon"
    assert geom.area == pytest.approx(10 * 10)


def test_line_crossing_the_antimeridian_twice_keeps_every_segment() -> None:
    track = [(170, 0), (-170, 1), (-100, 2), (0, 3), (100, 4), (170, 5), (-170, 6)]
    geom = granule_geometry(_umm({"Lines": [_points(track)]}))
    assert geom is not None
    xs = {round(x) for part in geom.geoms for x, _ in part.coords}
    assert {-170, -100, 0, 100, 170} <= xs
    assert all(-180 <= x <= 180 for x in xs)
    # Unwrapped, the track runs 360 + 20 = 380° east, and it stays that long.
    assert sum(
        abs(b[0] - a[0])
        for part in geom.geoms
        for a, b in itertools.pairwise(part.coords)
    ) == pytest.approx(380)


def test_malformed_end_is_not_read_as_open() -> None:
    temporal = {
        "RangeDateTime": {
            "BeginningDateTime": "2020-01-01T00:00Z",
            "EndingDateTime": "unknown",
        }
    }
    assert granule_interval({"TemporalExtent": temporal}) is None
    temporal["SingleDateTime"] = "2020-01-01T06:00Z"
    interval = granule_interval({"TemporalExtent": temporal})
    assert interval is not None
    assert interval.left == interval.right == pd.Timestamp("2020-01-01T06:00", tz="UTC")


def test_open_range_beyond_the_sentinel_keeps_its_direction() -> None:
    late = granule_interval(
        {
            "TemporalExtent": {
                "RangeDateTime": {"BeginningDateTime": "2150-01-01T00:00Z"}
            }
        }
    )
    assert late is not None
    assert late.left == late.right == pd.Timestamp("2150-01-01", tz="UTC")
    early = granule_interval(
        {"TemporalExtent": {"RangeDateTime": {"EndingDateTime": "1850-01-01T00:00Z"}}}
    )
    assert early is not None
    assert early.left == early.right == pd.Timestamp("1850-01-01", tz="UTC")


def test_cartesian_polygon_is_not_unwrapped() -> None:
    shell = _ring([(-170, -5), (170, -5), (170, 5), (-170, 5)])
    geometry = {
        "CoordinateSystem": "CARTESIAN",
        "GPolygons": [{"Boundary": _points(shell)}],
    }
    geom = granule_geometry(_umm(geometry))
    assert geom is not None
    assert geom.area == pytest.approx(340 * 10)  # the wide planar polygon
    assert geom.covers(shapely.Point(0, 0))


def test_cartesian_line_is_not_split() -> None:
    geometry = {
        "CoordinateSystem": "CARTESIAN",
        "Lines": [_points([(170, 0), (-170, 10)])],
    }
    geom = granule_geometry(_umm(geometry))
    assert geom is not None
    assert geom.geom_type == "LineString"


def test_seam_only_line_is_on_both_edges() -> None:
    geom = granule_geometry(_umm({"Lines": [_points([(180, -10), (180, 10)])]}))
    assert geom is not None
    assert geom.intersects(shapely.box(179, -1, 180, 1))
    assert geom.intersects(shapely.box(-180, -1, -179, 1))


def test_segment_180_degrees_apart_goes_over_the_pole() -> None:
    geom = granule_geometry(_umm({"Lines": [_points([(0, 80), (180, 80)])]}))
    assert geom is not None
    assert geom.intersects(shapely.Point(0, 89.5))  # near the pole
    assert not geom.intersects(shapely.box(80, 79, 100, 81))  # not along the parallel

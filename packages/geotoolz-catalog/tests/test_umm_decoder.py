"""UMM-G footprint and time decoding (#236).

Shared by the CMR and earthaccess adapters.
"""

from __future__ import annotations

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

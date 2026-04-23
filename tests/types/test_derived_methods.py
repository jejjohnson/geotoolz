"""Helpers and derived methods added alongside the core types."""

from __future__ import annotations

import pandas as pd
import pytest

from xr_toolz.types import BBox, DepthRange, PressureLevels, Request, TimeRange


# ---- BBox derived methods -----------------------------------------------


def test_bbox_width_and_height_simple():
    b = BBox(-10.0, 40.0, 30.0, 60.0)
    assert b.width == pytest.approx(50.0)
    assert b.height == pytest.approx(30.0)


def test_bbox_width_handles_antimeridian_wrap():
    wrap = BBox(170.0, -170.0, -10.0, 10.0)
    # 170° E -> 180° (10°) + -180° -> -170° (10°) = 20° wide.
    assert wrap.width == pytest.approx(20.0)


def test_bbox_to_360_basic():
    b = BBox(-10.0, 40.0, 30.0, 60.0).to_360()
    assert b.lon_min == pytest.approx(350.0)
    assert b.lon_max == pytest.approx(40.0)


def test_bbox_as_xarray_sel_uses_custom_dim_names():
    b = BBox(-10.0, 40.0, 30.0, 60.0)
    sel = b.as_xarray_sel(lon="longitude", lat="latitude")
    assert sel["longitude"] == slice(-10.0, 40.0)
    assert sel["latitude"] == slice(30.0, 60.0)


# ---- TimeRange derived methods ------------------------------------------


def test_time_range_as_xarray_sel_default_dim():
    t = TimeRange.parse("2020-01-01", "2020-01-05")
    sel = t.as_xarray_sel()
    assert sel["time"].start == t.start
    assert sel["time"].stop == t.end


def test_time_range_preserves_tzaware_input():
    aware = pd.Timestamp("2020-01-01T00:00:00", tz="US/Pacific")
    t = TimeRange.parse(aware, aware)
    assert t.start.tzinfo is not None
    # The internal UTC conversion must have run (tz_convert branch).
    assert t.start == pd.Timestamp("2020-01-01T08:00:00", tz="UTC")


def test_time_range_parse_rejects_unparseable():
    with pytest.raises(ValueError):
        TimeRange.parse("not a date", "also not a date")


# ---- Request composite --------------------------------------------------


def test_request_stores_all_fields_and_is_frozen():
    r = Request(
        variables=("sst", "ssh"),
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-31"),
        depth=DepthRange(0.0, 100.0),
        levels=PressureLevels((500, 1000)),
        extras={"grid": "0.25"},
    )
    assert r.variables == ("sst", "ssh")
    assert r.depth is not None
    assert r.depth.max == 100.0
    assert r.extras == {"grid": "0.25"}
    # Frozen (immutable).
    with pytest.raises(Exception):  # noqa: B017
        r.variables = ("foo",)  # type: ignore[misc]


def test_request_defaults_are_none():
    r = Request(variables=("sst",))
    assert r.bbox is None
    assert r.time is None
    assert r.depth is None
    assert r.levels is None
    assert r.extras is None


# ---- Levels edge cases --------------------------------------------------


def test_depth_range_accepts_equal_bounds_as_degenerate_point():
    d = DepthRange(50.0, 50.0)
    assert d.as_cmems() == {"minimum_depth": 50.0, "maximum_depth": 50.0}


def test_pressure_levels_single_level_is_valid():
    pl = PressureLevels((500,))
    assert pl.as_cds_form() == ["500"]

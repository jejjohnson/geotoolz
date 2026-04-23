"""Typed request primitives — ``BBox``, ``TimeRange``, ``DepthRange``,
``PressureLevels``."""

from __future__ import annotations

import pandas as pd
import pytest

from xr_toolz.types import BBox, DepthRange, PressureLevels, TimeRange


# ---- BBox ----------------------------------------------------------------


def test_bbox_from_tuple_round_trip():
    b = BBox.from_tuple((-10.0, 40.0, 30.0, 60.0))
    assert (b.lon_min, b.lon_max, b.lat_min, b.lat_max) == (-10.0, 40.0, 30.0, 60.0)


def test_bbox_rejects_invalid_latitude():
    with pytest.raises(ValueError, match="lat_min"):
        BBox(0.0, 10.0, -95.0, 0.0)
    with pytest.raises(ValueError, match="lat_min"):
        BBox(0.0, 10.0, 20.0, 10.0)


def test_bbox_detects_antimeridian_crossing():
    wrap = BBox(170.0, -170.0, -10.0, 10.0)
    assert wrap.crosses_antimeridian
    straight = BBox(-10.0, 40.0, 30.0, 60.0)
    assert not straight.crosses_antimeridian


def test_bbox_lon_normalization():
    b = BBox(350.0, 370.0, 0.0, 10.0)
    assert b.to_180().lon_min == pytest.approx(-10.0)
    assert b.to_180().lon_max == pytest.approx(10.0)

    b2 = BBox(-10.0, 10.0, 0.0, 10.0)
    assert b2.to_360().lon_min == pytest.approx(350.0)
    assert b2.to_360().lon_max == pytest.approx(10.0)


def test_bbox_as_cmems_keys_match_client_signature():
    b = BBox(-10.0, 40.0, 30.0, 60.0)
    expected_keys = {
        "minimum_longitude",
        "maximum_longitude",
        "minimum_latitude",
        "maximum_latitude",
    }
    assert set(b.as_cmems().keys()) == expected_keys


def test_bbox_as_cds_area_order_is_NWSE():
    b = BBox(-10.0, 40.0, 30.0, 60.0)
    # CDS expects [North, West, South, East].
    assert b.as_cds_area() == [60.0, -10.0, 30.0, 40.0]


# ---- TimeRange -----------------------------------------------------------


def test_time_range_parses_strings_as_utc():
    t = TimeRange.parse("2020-01-01", "2020-01-31")
    assert t.start == pd.Timestamp("2020-01-01", tz="UTC")
    assert t.end == pd.Timestamp("2020-01-31", tz="UTC")


def test_time_range_rejects_reversed():
    with pytest.raises(ValueError, match="start"):
        TimeRange.parse("2020-06-01", "2020-01-01")


def test_time_range_to_index_defaults_daily():
    t = TimeRange.parse("2020-01-01", "2020-01-03")
    idx = t.to_index()
    assert len(idx) == 3


def test_time_range_as_cmems_iso_format():
    t = TimeRange.parse("2020-01-01", "2020-01-31")
    payload = t.as_cmems()
    assert payload["start_datetime"].startswith("2020-01-01")
    assert payload["end_datetime"].startswith("2020-01-31")


def test_time_range_as_cds_form_explodes_ymd():
    t = TimeRange.parse("2020-01-29", "2020-02-02")
    form = t.as_cds_form()
    assert form["year"] == ["2020"]
    assert set(form["month"]) == {"01", "02"}
    assert set(form["day"]) == {"29", "30", "31", "01", "02"}


# ---- DepthRange ----------------------------------------------------------


def test_depth_range_validates_non_negative_and_ordered():
    d = DepthRange(0.0, 100.0)
    assert d.as_cmems() == {"minimum_depth": 0.0, "maximum_depth": 100.0}
    with pytest.raises(ValueError):
        DepthRange(-1.0, 10.0)
    with pytest.raises(ValueError):
        DepthRange(100.0, 10.0)


# ---- PressureLevels ------------------------------------------------------


def test_pressure_levels_rejects_empty_or_non_positive():
    with pytest.raises(ValueError):
        PressureLevels(())
    with pytest.raises(ValueError):
        PressureLevels((0, 500))


def test_pressure_levels_as_cds_form_stringifies():
    pl = PressureLevels((500, 850, 1000))
    assert pl.as_cds_form() == ["500", "850", "1000"]

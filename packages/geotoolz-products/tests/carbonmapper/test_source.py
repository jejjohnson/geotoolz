"""Tests for ``geoproducts.carbonmapper.source``."""

from datetime import UTC, datetime

import pytest

from geoproducts.carbonmapper.source import CMSource, _strip_query_suffix


def test_strip_query_suffix_with_suffix():
    assert (
        _strip_query_suffix("CH4_6A_500m_-117.26768_34.59375?plume_gas=CH4&bbox=1")
        == "CH4_6A_500m_-117.26768_34.59375"
    )


def test_strip_query_suffix_without_suffix():
    assert _strip_query_suffix("CH4_1B2_100m_-104_32") == "CH4_1B2_100m_-104_32"


def test_strip_query_suffix_empty():
    assert _strip_query_suffix("") == ""


def test_from_geojson_feature_strips_suffix_and_parses_fields():
    feature = {
        "properties": {
            "source_name": "CH4_1B2_100m_-104.17525_32.49125?plume_gas=CH4",
            "sector": "1B2",
            "gas": "CH4",
            "plume_count": 12,
            "persistence": 0.42,
            "emission_auto": 250.0,
            "emission_uncertainty_auto": 35.0,
            "first_observation": "2024-03-01T12:00:00Z",
            "last_observation": "2024-09-15T18:30:00Z",
        },
        "geometry": {
            "type": "Point",
            "coordinates": [-104.17525, 32.49125],
        },
    }
    src = CMSource.from_geojson_feature(feature)
    assert src.source_name == "CH4_1B2_100m_-104.17525_32.49125"
    assert src.sector == "1B2"
    assert src.gas == "CH4"
    assert src.plume_count == 12
    assert src.persistence == pytest.approx(0.42)
    assert src.emission_auto == pytest.approx(250.0)
    assert src.point.x == pytest.approx(-104.17525)
    assert src.point.y == pytest.approx(32.49125)
    assert src.first_observation == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
    assert src.last_observation == datetime(2024, 9, 15, 18, 30, tzinfo=UTC)


def test_from_geojson_feature_missing_geometry_raises():
    feature = {"properties": {"source_name": "x"}, "geometry": {}}
    with pytest.raises(ValueError):
        CMSource.from_geojson_feature(feature)


def test_cmsource_is_hashable():
    """Documented as hashable, but the `raw` dict made `hash()` raise."""
    src = CMSource.from_geojson_feature(
        {
            "properties": {
                "source_name": "CH4_1B2_100m_-104_32",
                "sector": "1B2",
                "gas": "CH4",
                "plume_count": 1,
                "persistence": 0.1,
            },
            "geometry": {"type": "Point", "coordinates": [-104.0, 32.0]},
        }
    )
    assert src in {src}


def test_from_geojson_feature_reads_published_timestamp_keys():
    feature = {
        "properties": {
            "source_name": "CH4_1B2_100m_-100.0_30.0",
            "timestamp_min": "2024-01-02T03:04:05Z",
            "timestamp_max": "2025-06-07T08:09:10Z",
        },
        "geometry": {"type": "Point", "coordinates": [-100.0, 30.0]},
    }
    src = CMSource.from_geojson_feature(feature)
    assert src.first_observation == datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert src.last_observation == datetime(2025, 6, 7, 8, 9, 10, tzinfo=UTC)

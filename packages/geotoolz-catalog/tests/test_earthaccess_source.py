"""Unit tests for `geocatalog._src.sources.earthaccess.EarthAccessSource`.

Two tiers:

* **Offline unit tests** build synthetic UMM granule dicts by hand
  and drive `EarthAccessSource.query` via a faked
  ``earthaccess.search_data`` so we can assert the mapping logic
  without network access. Runs in every CI invocation.
* **Live tests** (`@pytest.mark.live`) hit NASA CMR through the real
  `earthaccess` library. Skipped by default
  (``addopts = "-m 'not live'"``); opt in with
  ``uv run pytest -m live --no-cov``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


# `earthaccess` is an optional extra; skip the whole module when
# it's not installed.
earthaccess = pytest.importorskip("earthaccess")

from geocatalog._src.sources._base import SourceRow
from geocatalog._src.sources.earthaccess import (
    EarthAccessSource,
    _asset_key_from_url,
    _extract_cloud_cover,
    _granule_geometry,
    _granule_interval,
    _granule_to_source_row,
)


# ---------------------------------------------------------------------------
# Synthetic UMM fixtures
# ---------------------------------------------------------------------------


def _polygon_umm(
    *,
    granule_ur: str = "MOD09GA.A2024153.h17v05.061.2024155033945",
    short_name: str = "MOD09GA",
    bbox: tuple[float, float, float, float] = (-10.0, 35.0, -5.0, 45.0),
    start: datetime = datetime(2024, 6, 1, tzinfo=UTC),
    end: datetime = datetime(2024, 6, 1, 23, 59, tzinfo=UTC),
    cloud_cover: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal UMM dict shaped like CMR's umm_json items."""
    xmin, ymin, xmax, ymax = bbox
    ring = [
        {"Longitude": xmin, "Latitude": ymin},
        {"Longitude": xmax, "Latitude": ymin},
        {"Longitude": xmax, "Latitude": ymax},
        {"Longitude": xmin, "Latitude": ymax},
        {"Longitude": xmin, "Latitude": ymin},
    ]
    umm: dict[str, Any] = {
        "GranuleUR": granule_ur,
        "CollectionReference": {"ShortName": short_name, "Version": "061"},
        "TemporalExtent": {
            "RangeDateTime": {
                "BeginningDateTime": start.isoformat(),
                "EndingDateTime": end.isoformat(),
            }
        },
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {"GPolygons": [{"Boundary": {"Points": ring}}]}
            }
        },
    }
    if cloud_cover is not None:
        umm["CloudCover"] = cloud_cover
    if extra:
        umm.update(extra)
    return umm


def _bbox_umm(bbox: tuple[float, float, float, float], **kw: Any) -> dict[str, Any]:
    """UMM with BoundingRectangles instead of GPolygons."""
    base = _polygon_umm(bbox=bbox, **kw)
    base["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"] = {
        "BoundingRectangles": [
            {
                "WestBoundingCoordinate": bbox[0],
                "SouthBoundingCoordinate": bbox[1],
                "EastBoundingCoordinate": bbox[2],
                "NorthBoundingCoordinate": bbox[3],
            }
        ]
    }
    return base


class _FakeGranule:
    """Minimal `earthaccess.results.DataGranule` stand-in."""

    def __init__(self, umm: dict[str, Any], links: list[str] | None = None) -> None:
        self._umm = umm
        self._links = links or []

    def get(self, key: str, default: Any = None) -> Any:
        if key == "umm":
            return self._umm
        if key == "meta":
            return {"concept-id": f"G-{self._umm.get('GranuleUR', '?')}"}
        return default

    def data_links(self) -> list[str]:
        return list(self._links)


# ---------------------------------------------------------------------------
# Helpers — geometry / interval / cloud-cover extraction
# ---------------------------------------------------------------------------


class TestGranuleGeometry:
    def test_polygon_to_shapely(self) -> None:
        umm = _polygon_umm(bbox=(-10, 35, -5, 45))
        geom = _granule_geometry(umm)
        assert geom is not None
        assert geom.geom_type == "Polygon"
        assert geom.bounds == (-10.0, 35.0, -5.0, 45.0)

    def test_bbox_to_shapely(self) -> None:
        umm = _bbox_umm(bbox=(0, 0, 1, 1))
        geom = _granule_geometry(umm)
        assert geom is not None
        assert geom.bounds == (0.0, 0.0, 1.0, 1.0)

    def test_point_to_shapely(self) -> None:
        umm = _polygon_umm()
        umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"] = {
            "Points": [{"Longitude": 1.0, "Latitude": 2.0}]
        }
        geom = _granule_geometry(umm)
        assert geom is not None
        assert geom.geom_type == "Point"
        assert (geom.x, geom.y) == (1.0, 2.0)

    def test_missing_geometry_returns_none(self) -> None:
        umm = {"SpatialExtent": {}}
        assert _granule_geometry(umm) is None
        assert _granule_geometry({}) is None

    def test_gpolygon_missing_coordinates_skipped(self) -> None:
        # A polygon point missing Latitude/Longitude must be skipped
        # silently rather than raising KeyError (UMM occasionally
        # carries partial point objects).
        umm = {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "GPolygons": [
                            {
                                "Boundary": {
                                    "Points": [
                                        {"Longitude": 0.0, "Latitude": 0.0},
                                        {"Longitude": 1.0, "Latitude": 0.0},
                                        {"Longitude": 1.0},  # no Latitude
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
        # Two surviving points < 3 → polygon dropped, geometry is None.
        assert _granule_geometry(umm) is None

    def test_points_missing_coordinates_skipped(self) -> None:
        umm = {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "Points": [
                            {"Longitude": 1.0, "Latitude": 2.0},
                            {"Latitude": 3.0},  # no Longitude
                        ]
                    }
                }
            }
        }
        geom = _granule_geometry(umm)
        assert geom is not None
        assert geom.geom_type == "Point"
        assert (geom.x, geom.y) == (1.0, 2.0)


class TestGranuleInterval:
    def test_range_to_interval(self) -> None:
        umm = _polygon_umm(
            start=datetime(2024, 6, 1, tzinfo=UTC),
            end=datetime(2024, 6, 2, tzinfo=UTC),
        )
        iv = _granule_interval(umm)
        assert iv is not None
        assert iv.left == pd.Timestamp("2024-06-01T00:00:00Z")
        assert iv.right == pd.Timestamp("2024-06-02T00:00:00Z")

    def test_single_datetime_to_instant_interval(self) -> None:
        umm = {"TemporalExtent": {"SingleDateTime": "2024-06-15T12:00:00Z"}}
        iv = _granule_interval(umm)
        assert iv is not None
        assert iv.left == iv.right == pd.Timestamp("2024-06-15T12:00:00Z")

    def test_naive_datetime_normalized_to_utc(self) -> None:
        umm = {
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": "2024-06-15T08:00:00-07:00",
                    "EndingDateTime": "2024-06-15T17:00:00-07:00",
                }
            }
        }
        iv = _granule_interval(umm)
        assert iv is not None
        assert iv.left == pd.Timestamp("2024-06-15T15:00:00Z")
        assert iv.right == pd.Timestamp("2024-06-16T00:00:00Z")

    def test_missing_temporal_returns_none(self) -> None:
        assert _granule_interval({}) is None


class TestCloudCoverExtraction:
    def test_direct_cloud_cover_field(self) -> None:
        assert _extract_cloud_cover({"CloudCover": "12.4"}) == 12.4

    def test_additional_attributes(self) -> None:
        umm = {
            "AdditionalAttributes": [
                {"Name": "CLOUD_COVERAGE", "Values": ["8.5"]},
            ]
        }
        assert _extract_cloud_cover(umm) == 8.5

    def test_missing_returns_none(self) -> None:
        assert _extract_cloud_cover({}) is None


class TestAssetKey:
    def test_short_stem(self) -> None:
        url = "https://example.com/data/MOD09GA.A2024.h17v05.tif"
        assert _asset_key_from_url(url) == "MOD09GA.A2024.h17v05"

    def test_extension_when_stem_too_long(self) -> None:
        # Very long stem → fall back to extension.
        long_stem = "x" * 100
        url = f"https://example.com/data/{long_stem}.nc"
        assert _asset_key_from_url(url) == "nc"


# ---------------------------------------------------------------------------
# Granule → SourceRow mapping
# ---------------------------------------------------------------------------


class TestGranuleToSourceRow:
    def _provenance_args(self) -> dict[str, Any]:
        return {
            "source_name": "earthaccess",
            "query_id": "abc123",
            "fetched_at": datetime(2026, 5, 25, tzinfo=UTC),
            "source_version": "earthaccess/0.18.0",
        }

    def test_well_formed_granule(self) -> None:
        granule = _FakeGranule(
            _polygon_umm(),
            links=[
                "https://example.com/data/MOD09GA.A2024.h17v05.061.hdf",
                "https://example.com/data/MOD09GA.A2024.h17v05.061.xml",
            ],
        )
        row = _granule_to_source_row(granule, **self._provenance_args())
        assert isinstance(row, SourceRow)
        assert row.source == "earthaccess"
        assert row.collection == "MOD09GA"
        assert row.id.startswith("MOD09GA.A2024153")
        # Two assets resolved from data_links().
        assert set(row.assets) == {
            "MOD09GA.A2024.h17v05.061",  # both URLs share this stem
            "MOD09GA.A2024.h17v05.061__1",  # disambiguated
        }
        assert row.provenance["query_id"] == "abc123"

    def test_cloud_cover_propagated_to_properties(self) -> None:
        granule = _FakeGranule(_polygon_umm(cloud_cover=12.4))
        row = _granule_to_source_row(granule, **self._provenance_args())
        assert row is not None
        assert row.properties["eo:cloud_cover"] == 12.4

    def test_missing_geometry_returns_none(self) -> None:
        umm = _polygon_umm()
        umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"] = {}
        granule = _FakeGranule(umm)
        assert _granule_to_source_row(granule, **self._provenance_args()) is None

    def test_missing_temporal_returns_none(self) -> None:
        umm = _polygon_umm()
        del umm["TemporalExtent"]
        granule = _FakeGranule(umm)
        assert _granule_to_source_row(granule, **self._provenance_args()) is None


# ---------------------------------------------------------------------------
# EarthAccessSource.query with a faked search_data
# ---------------------------------------------------------------------------


class TestEarthAccessQuery:
    def test_query_forwards_args_and_emits_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_search_data(*, count: int, **kwargs: Any) -> list[_FakeGranule]:
            captured["count"] = count
            captured["kwargs"] = kwargs
            return [
                _FakeGranule(_polygon_umm(granule_ur="a", bbox=(-10, 35, -5, 45))),
                _FakeGranule(_polygon_umm(granule_ur="b", bbox=(-10, 35, -5, 45))),
            ]

        from geocatalog._src.sources import earthaccess as ea_mod

        monkeypatch.setattr(ea_mod.earthaccess, "search_data", fake_search_data)
        src = EarthAccessSource()
        rows = list(
            src.query(
                bounds=(-10, 35, -5, 45),
                interval=pd.Interval(
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-30", tz="UTC"),
                    closed="both",
                ),
                collection="MOD09GA",
                filters={"cloud_cover": (0, 20)},
                limit=10,
            )
        )
        assert len(rows) == 2
        assert {r.id for r in rows} == {"a", "b"}
        # `count=10` honoured (matches `limit`).
        assert captured["count"] == 10
        # Forwarded kwargs include both built-ins and user filters.
        assert captured["kwargs"]["short_name"] == "MOD09GA"
        assert captured["kwargs"]["bounding_box"] == (-10, 35, -5, 45)
        assert captured["kwargs"]["cloud_cover"] == (0, 20)
        assert "temporal" in captured["kwargs"]

    def test_limit_none_maps_to_count_minus_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_search_data(*, count: int, **kwargs: Any) -> list[_FakeGranule]:
            captured["count"] = count
            return []

        from geocatalog._src.sources import earthaccess as ea_mod

        monkeypatch.setattr(ea_mod.earthaccess, "search_data", fake_search_data)
        list(EarthAccessSource().query(bounds=(-10, 35, -5, 45)))
        assert captured["count"] == -1

    def test_skips_granules_without_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken_umm = _polygon_umm(granule_ur="broken")
        broken_umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"] = {}

        def fake_search_data(*, count: int, **kwargs: Any) -> list[_FakeGranule]:
            return [
                _FakeGranule(_polygon_umm(granule_ur="good")),
                _FakeGranule(broken_umm),
            ]

        from geocatalog._src.sources import earthaccess as ea_mod

        monkeypatch.setattr(ea_mod.earthaccess, "search_data", fake_search_data)
        rows = list(EarthAccessSource().query(bounds=(-10, 35, -5, 45)))
        assert [r.id for r in rows] == ["good"]


class TestEarthAccessAuthStatus:
    def test_unauthenticated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from geocatalog._src.sources import earthaccess as ea_mod

        # Stand up a fake auth object that says "not authenticated".
        fake_auth = MagicMock()
        fake_auth.authenticated = False
        monkeypatch.setattr(ea_mod.earthaccess, "__auth__", fake_auth)
        monkeypatch.setattr(
            ea_mod.earthaccess, "get_requests_https_session", lambda: MagicMock()
        )
        status = EarthAccessSource().auth_status()
        assert status.authenticated is False
        assert "not authenticated" in (status.detail or "").lower()

    def test_authenticated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from geocatalog._src.sources import earthaccess as ea_mod

        fake_auth = MagicMock()
        fake_auth.authenticated = True
        monkeypatch.setattr(ea_mod.earthaccess, "__auth__", fake_auth)
        monkeypatch.setattr(
            ea_mod.earthaccess, "get_requests_https_session", lambda: MagicMock()
        )
        status = EarthAccessSource().auth_status()
        assert status.authenticated is True


# ---------------------------------------------------------------------------
# Live tests — opt-in, requires network access + EDL credentials.
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestEarthAccessLive:
    """End-to-end against the real NASA CMR via earthaccess.

    Skipped by default. Run with `uv run pytest -m live --no-cov`.
    """

    def test_search_mod09ga_iberia(self) -> None:
        src = EarthAccessSource()
        rows = list(
            src.query(
                bounds=(-10, 35, 5, 45),
                interval=pd.Interval(
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-03", tz="UTC"),
                    closed="both",
                ),
                collection="MOD09GA",
                limit=3,
            )
        )
        assert 0 < len(rows) <= 3
        for row in rows:
            assert row.source == "earthaccess"
            assert row.collection == "MOD09GA"

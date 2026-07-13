"""Unit tests for `geocatalog._src.sources.cmr.CMRSource`.

Drives `CMRSource.query` against a monkey-patched
``urllib.request.urlopen`` so the UMM-JSON → SourceRow mapping can
be tested without network access. A small `@pytest.mark.live`
suite hits real CMR when opt-in.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from geocatalog._src.sources._base import SourceRow
from geocatalog._src.sources.cmr import CMRSource, _cmr_item_to_source_row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_item(
    *,
    granule_ur: str = "MOD09GA.A2024.h17v05",
    short_name: str = "MOD09GA",
    bbox: tuple[float, float, float, float] = (-10.0, 35.0, -5.0, 45.0),
    start: datetime = datetime(2024, 6, 1, tzinfo=UTC),
    end: datetime = datetime(2024, 6, 1, 23, 59, tzinfo=UTC),
    data_urls: list[str] | None = None,
    cloud_cover: float | None = None,
) -> dict[str, Any]:
    """One element of CMR's `items` array."""
    xmin, ymin, xmax, ymax = bbox
    item = {
        "umm": {
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
                    "Geometry": {
                        "BoundingRectangles": [
                            {
                                "WestBoundingCoordinate": xmin,
                                "SouthBoundingCoordinate": ymin,
                                "EastBoundingCoordinate": xmax,
                                "NorthBoundingCoordinate": ymax,
                            }
                        ]
                    }
                }
            },
            "RelatedUrls": [{"URL": u, "Type": "GET DATA"} for u in (data_urls or [])],
        }
    }
    if cloud_cover is not None:
        item["umm"]["CloudCover"] = cloud_cover
    return item


class _FakeResponse:
    """Minimal urllib HTTPResponse stand-in."""

    def __init__(
        self,
        body: dict[str, Any],
        *,
        status: int = 200,
        search_after: str | None = None,
    ) -> None:
        self._body = body
        self.status = status
        self.headers: dict[str, str] = {}
        if search_after:
            self.headers["CMR-Search-After"] = search_after

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Item → SourceRow mapping
# ---------------------------------------------------------------------------


class TestCMRItemMapping:
    def _provenance(self) -> dict[str, Any]:
        return {
            "source_name": "cmr",
            "query_id": "q1",
            "fetched_at": datetime(2026, 5, 25, tzinfo=UTC),
        }

    def test_well_formed_item(self) -> None:
        item = _make_item(
            data_urls=[
                "https://example.com/data/MOD09GA.A2024.h17v05.hdf",
                "https://example.com/data/MOD09GA.A2024.h17v05.xml",
            ],
            cloud_cover=8.0,
        )
        row = _cmr_item_to_source_row(item, **self._provenance())
        assert isinstance(row, SourceRow)
        assert row.source == "cmr"
        assert row.collection == "MOD09GA"
        assert row.id == "MOD09GA.A2024.h17v05"
        assert row.geometry.bounds == (-10.0, 35.0, -5.0, 45.0)
        assert row.properties["eo:cloud_cover"] == 8.0
        assert len(row.assets) == 2

    def test_filters_non_data_urls(self) -> None:
        item = _make_item(data_urls=["https://example.com/x.hdf"])
        # Add a non-GET-DATA URL (the kind CMR sometimes attaches —
        # browse images, metadata pages, etc.).
        item["umm"]["RelatedUrls"].append(
            {"URL": "https://example.com/x.png", "Type": "GET RELATED VISUALIZATION"}
        )
        row = _cmr_item_to_source_row(item, **self._provenance())
        assert row is not None
        assert len(row.assets) == 1

    def test_missing_geometry_skipped(self) -> None:
        item = _make_item()
        item["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"] = {}
        assert _cmr_item_to_source_row(item, **self._provenance()) is None

    def test_missing_temporal_skipped(self) -> None:
        item = _make_item()
        del item["umm"]["TemporalExtent"]
        assert _cmr_item_to_source_row(item, **self._provenance()) is None


# ---------------------------------------------------------------------------
# CMRSource.query via monkey-patched urlopen
# ---------------------------------------------------------------------------


class TestCMRQuery:
    def test_query_single_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, *, timeout: float = 60.0) -> _FakeResponse:
            captured["url"] = req.full_url
            return _FakeResponse(
                {"items": [_make_item(granule_ur="x"), _make_item(granule_ur="y")]}
            )

        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen", fake_urlopen
        )
        src = CMRSource()
        rows = list(
            src.query(
                bounds=(-10, 35, 5, 45),
                interval=pd.Interval(
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-30", tz="UTC"),
                    closed="both",
                ),
                collection="MOD09GA",
                limit=5,
            )
        )
        assert [r.id for r in rows] == ["x", "y"]
        assert "bounding_box=-10%2C35%2C5%2C45" in captured["url"]
        assert "short_name=MOD09GA" in captured["url"]
        assert "temporal=" in captured["url"]

    def test_query_paginated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two pages; CMR's search-after header strings together.
        call_log: list[Any] = []

        def fake_urlopen(req: Any, *, timeout: float = 60.0) -> _FakeResponse:
            # urllib normalises header keys to capitalised-first-letter
            # form, so `CMR-Search-After` becomes `Cmr-search-after`.
            call_log.append(req.headers.get("Cmr-search-after"))
            n = len(call_log)
            if n == 1:
                return _FakeResponse(
                    {"items": [_make_item(granule_ur=f"p1-{i}") for i in range(3)]},
                    search_after="cursor-1",
                )
            if n == 2:
                return _FakeResponse(
                    {"items": [_make_item(granule_ur=f"p2-{i}") for i in range(2)]},
                    # No more pages signalled.
                )
            raise AssertionError("too many calls")

        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen", fake_urlopen
        )
        rows = list(CMRSource().query(bounds=(-10, 35, 5, 45)))
        assert len(rows) == 5
        # Page 1 had no search-after header; page 2 echoed `cursor-1`.
        assert call_log == [None, "cursor-1"]

    def test_limit_caps_emitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req: Any, *, timeout: float = 60.0) -> _FakeResponse:
            return _FakeResponse(
                {"items": [_make_item(granule_ur=f"x-{i}") for i in range(10)]},
            )

        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen", fake_urlopen
        )
        rows = list(CMRSource().query(bounds=(-10, 35, 5, 45), limit=3))
        assert len(rows) == 3

    def test_limit_zero_emits_nothing_without_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A `limit=0` query must short-circuit before issuing any
        # request. Patch `urlopen` to a sentinel that would fail the
        # test if it ever got called.
        def fail(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError("urlopen should not be called when limit=0")

        monkeypatch.setattr("geocatalog._src.sources.cmr.urllib.request.urlopen", fail)
        assert list(CMRSource().query(bounds=(-10, 35, 5, 45), limit=0)) == []
        assert list(CMRSource().query(bounds=(-10, 35, 5, 45), limit=-1)) == []

    def test_filters_forwarded_as_url_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, *, timeout: float = 60.0) -> _FakeResponse:
            captured["url"] = req.full_url
            return _FakeResponse({"items": []})

        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen", fake_urlopen
        )
        list(
            CMRSource().query(
                bounds=(-10, 35, 5, 45),
                filters={"provider": "LPDAAC_ECS", "platform": "Terra"},
            )
        )
        assert "provider=LPDAAC_ECS" in captured["url"]
        assert "platform=Terra" in captured["url"]


class TestCMRAuthStatus:
    def test_reachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen",
            lambda req, *, timeout=10.0: _FakeResponse({"items": []}),
        )
        status = CMRSource().auth_status()
        assert status.authenticated is True
        assert "anonymous" in (status.detail or "").lower()

    def test_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(req: Any, *, timeout: float = 10.0) -> Any:
            raise OSError("connection refused")

        monkeypatch.setattr("geocatalog._src.sources.cmr.urllib.request.urlopen", boom)
        status = CMRSource().auth_status()
        assert status.authenticated is False
        assert "could not reach" in (status.detail or "").lower()

    def test_token_reflected_in_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen",
            lambda req, *, timeout=10.0: _FakeResponse({"items": []}),
        )
        status = CMRSource(token="abc").auth_status()
        assert status.authenticated is True
        assert "token set" in (status.detail or "")

    def test_non_200_marks_unauthenticated_with_status_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even though urlopen returned, a non-200 status should not
        # report "reachable" — instead surface the status code.
        monkeypatch.setattr(
            "geocatalog._src.sources.cmr.urllib.request.urlopen",
            lambda req, *, timeout=10.0: _FakeResponse({"items": []}, status=503),
        )
        status = CMRSource().auth_status()
        assert status.authenticated is False
        assert "status 503" in (status.detail or "")


class TestCMRGeometryHardening:
    """Regression — malformed UMM should be skipped, not crash."""

    def _spatial(self, geom: dict[str, Any]) -> dict[str, Any]:
        return {"HorizontalSpatialDomain": {"Geometry": geom}}

    def test_gpolygon_missing_coordinates_skipped(self) -> None:
        from geocatalog._src.sources.cmr import _granule_geometry

        # Two valid points + one missing Latitude → the bad point is
        # dropped silently. With the bad one filtered out, only 2
        # vertices remain (< 3 required for a polygon), so the whole
        # GPolygon is skipped — no KeyError raised.
        umm = {
            "SpatialExtent": self._spatial(
                {
                    "GPolygons": [
                        {
                            "Boundary": {
                                "Points": [
                                    {"Longitude": 0.0, "Latitude": 0.0},
                                    {"Longitude": 1.0, "Latitude": 0.0},
                                    {"Longitude": 1.0},  # missing Latitude
                                ]
                            }
                        }
                    ]
                }
            )
        }
        assert _granule_geometry(umm) is None

    def test_points_missing_coordinates_skipped(self) -> None:
        from geocatalog._src.sources.cmr import _granule_geometry

        umm = {
            "SpatialExtent": self._spatial(
                {
                    "Points": [
                        {"Longitude": 0.0, "Latitude": 0.0},
                        {"Longitude": 1.0},  # missing Latitude → skipped
                    ]
                }
            )
        }
        geom = _granule_geometry(umm)
        assert geom is not None
        assert geom.geom_type == "Point"


# ---------------------------------------------------------------------------
# Live tests — opt-in, requires network access to NASA CMR.
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCMRLive:
    """End-to-end against real NASA CMR. Skipped by default."""

    def test_mod09ga_iberia(self) -> None:
        src = CMRSource()
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
            assert row.source == "cmr"
            assert row.collection == "MOD09GA"

    def test_auth_status_real(self) -> None:
        status = CMRSource().auth_status()
        assert status.authenticated is True


# Module-level helper — pull StringIO into scope for typing.
def _consume(reader: io.StringIO) -> str:
    return reader.read()

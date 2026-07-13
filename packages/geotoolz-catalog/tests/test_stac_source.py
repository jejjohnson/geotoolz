"""Unit tests for `geocatalog._src.sources.stac.STACSource`.

Two tiers:

* **Offline unit tests** build synthetic `pystac.Item`s by hand and
  feed them through `_item_to_source_row`, plus drive `STACSource.query`
  via a faked client. Runs in every CI invocation.
* **Live tests** (`@pytest.mark.live`) hit Planetary Computer. Skipped
  by default (``addopts = "-m 'not live'"``); opt in with
  ``uv run pytest -m live --no-cov``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


# The `[stac]` extra is optional; without it the whole STAC test
# module skips cleanly rather than failing at collection. This is
# the same pattern the rest of the suite uses for extras-gated tests
# (DuckDB, xarray, etc).
pystac = pytest.importorskip("pystac")
pytest.importorskip("pystac_client")

from shapely.geometry import box, mapping

from geocatalog._src.sources._base import SourceRow
from geocatalog._src.sources.stac import (
    STACSource,
    _interval_to_stac_datetime,
    _item_to_source_row,
    _to_utc_timestamp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_item(
    *,
    id_: str = "S2A_MSIL2A_20240615T103631_N0510_R008_T29SQB_20240615T141432",
    collection: str = "sentinel-2-l2a",
    datetime_: datetime | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    bbox: tuple[float, float, float, float] = (-9.0, 38.0, -8.0, 39.0),
    extra_props: dict[str, Any] | None = None,
    assets: dict[str, str] | None = None,
) -> pystac.Item:
    geom = mapping(box(*bbox))
    properties: dict[str, Any] = {"eo:cloud_cover": 12.4}
    if start_datetime is not None:
        properties["start_datetime"] = start_datetime.isoformat()
    if end_datetime is not None:
        properties["end_datetime"] = end_datetime.isoformat()
    if extra_props:
        properties.update(extra_props)
    item = pystac.Item(
        id=id_,
        geometry=geom,
        bbox=list(bbox),
        datetime=datetime_,
        properties=properties,
        collection=collection,
    )
    assets = assets or {
        "B04": "https://example.blob/red.tif",
        "B08": "https://example.blob/nir.tif",
    }
    for key, href in assets.items():
        item.add_asset(key, pystac.Asset(href=href, media_type="image/tiff"))
    return item


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestIntervalToStacDatetime:
    def test_utc_interval(self) -> None:
        iv = pd.Interval(
            pd.Timestamp("2024-06-01", tz="UTC"),
            pd.Timestamp("2024-06-30", tz="UTC"),
            closed="both",
        )
        s = _interval_to_stac_datetime(iv)
        assert s == "2024-06-01T00:00:00Z/2024-06-30T00:00:00Z"

    def test_naive_interval_assumed_utc(self) -> None:
        # Naive timestamps are coerced to UTC — STAC requires
        # timezone-aware ISO 8601 with `Z`.
        iv = pd.Interval(
            pd.Timestamp("2024-06-01"),
            pd.Timestamp("2024-06-02"),
            closed="both",
        )
        assert _interval_to_stac_datetime(iv) == (
            "2024-06-01T00:00:00Z/2024-06-02T00:00:00Z"
        )

    def test_non_utc_tz_converted(self) -> None:
        # Use a fixed UTC offset rather than a named zone so the test
        # doesn't depend on the host having tzdata installed for that
        # named region (some minimal Linux images don't).
        iv = pd.Interval(
            pd.Timestamp("2024-06-01T08:00-07:00"),
            pd.Timestamp("2024-06-01T17:00-07:00"),
            closed="both",
        )
        assert _interval_to_stac_datetime(iv) == (
            "2024-06-01T15:00:00Z/2024-06-02T00:00:00Z"
        )


class TestItemToSourceRow:
    def _provenance(self) -> dict[str, Any]:
        return {
            "query_id": "abc123",
            "fetched_at": datetime(2026, 5, 25, tzinfo=UTC).isoformat(),
            "source_version": "pystac-client/0.9.0",
        }

    def test_single_datetime(self) -> None:
        when = datetime(2024, 6, 15, 10, 36, 31, tzinfo=UTC)
        item = _make_item(datetime_=when)
        row = _item_to_source_row(
            item,
            source_name="stac.pc",
            query_id="abc123",
            fetched_at=datetime(2026, 5, 25, tzinfo=UTC),
            source_version="pystac-client/0.9.0",
        )
        assert isinstance(row, SourceRow)
        assert row.source == "stac.pc"
        assert row.collection == "sentinel-2-l2a"
        assert row.interval.left == row.interval.right == pd.Timestamp(when)
        assert row.geometry.bounds == (-9.0, 38.0, -8.0, 39.0)
        assert row.assets == {
            "B04": "https://example.blob/red.tif",
            "B08": "https://example.blob/nir.tif",
        }
        assert row.properties["eo:cloud_cover"] == 12.4
        assert row.provenance["query_id"] == "abc123"

    def test_datetime_range_preferred(self) -> None:
        # When both `start_datetime` and `end_datetime` are present in
        # properties, they take precedence over a single `datetime`.
        start = datetime(2024, 6, 1, tzinfo=UTC)
        end = datetime(2024, 6, 8, tzinfo=UTC)
        item = _make_item(
            datetime_=None,
            start_datetime=start,
            end_datetime=end,
        )
        row = _item_to_source_row(
            item,
            source_name="stac.pc",
            query_id="q",
            fetched_at=datetime(2026, 5, 25, tzinfo=UTC),
            source_version="v",
        )
        assert row.interval.left == pd.Timestamp(start)
        assert row.interval.right == pd.Timestamp(end)

    def test_missing_geometry_raises(self) -> None:
        # An item without geometry can't be ingested into the
        # geo-indexed catalog; we fail loudly rather than emit a
        # broken row. pystac validates on construction so we mutate
        # post-hoc.
        item = _make_item(datetime_=datetime(2024, 6, 15, tzinfo=UTC))
        item.geometry = None
        with pytest.raises(ValueError, match="no geometry"):
            _item_to_source_row(
                item,
                source_name="stac.pc",
                query_id="q",
                fetched_at=datetime(2026, 5, 25, tzinfo=UTC),
                source_version="v",
            )

    def test_missing_datetime_raises(self) -> None:
        # pystac.Item itself refuses to construct without temporal
        # info, so mutate after the fact: drop start_datetime /
        # end_datetime and clear `datetime`.
        item = _make_item(
            datetime_=datetime(2024, 6, 15, tzinfo=UTC),
        )
        item.datetime = None
        item.properties.pop("start_datetime", None)
        item.properties.pop("end_datetime", None)
        with pytest.raises(ValueError, match="cannot build interval"):
            _item_to_source_row(
                item,
                source_name="stac.pc",
                query_id="q",
                fetched_at=datetime(2026, 5, 25, tzinfo=UTC),
                source_version="v",
            )

    def test_datetime_range_normalized_to_utc(self) -> None:
        # STAC items can carry any tz. Downstream pandas operations
        # (IntervalIndex, merge_asof) refuse to compare tz-aware
        # against naive Timestamps, so we coerce everything to UTC
        # at the ingest boundary. A `-07:00` start should land at
        # 15:00Z in the row's interval.
        start = datetime(
            2024,
            6,
            1,
            8,
            0,
            tzinfo=__import__("datetime").timezone(
                __import__("datetime").timedelta(hours=-7)
            ),
        )
        end = datetime(
            2024,
            6,
            1,
            17,
            0,
            tzinfo=__import__("datetime").timezone(
                __import__("datetime").timedelta(hours=-7)
            ),
        )
        item = _make_item(
            datetime_=None,
            start_datetime=start,
            end_datetime=end,
        )
        row = _item_to_source_row(
            item,
            source_name="stac.pc",
            query_id="q",
            fetched_at=datetime(2026, 5, 25, tzinfo=UTC),
            source_version="v",
        )
        assert row.interval.left == pd.Timestamp("2024-06-01T15:00:00Z")
        assert row.interval.right == pd.Timestamp("2024-06-02T00:00:00Z")
        # Both endpoints are tz-aware UTC — guards against naive
        # mixing in downstream IntervalIndex operations.
        assert row.interval.left.tzinfo is not None
        assert row.interval.right.tzinfo is not None

    def test_to_utc_timestamp_naive_assumed_utc(self) -> None:
        ts = _to_utc_timestamp("2024-06-15T10:00:00")
        assert ts == pd.Timestamp("2024-06-15T10:00:00Z")
        assert ts.tzinfo is not None

    def test_to_utc_timestamp_tz_aware_converted(self) -> None:
        ts = _to_utc_timestamp("2024-06-15T10:00:00-04:00")
        assert ts == pd.Timestamp("2024-06-15T14:00:00Z")

    def test_empty_collection_id_handled(self) -> None:
        item = _make_item(datetime_=datetime(2024, 6, 15, tzinfo=UTC))
        item.collection_id = None
        row = _item_to_source_row(
            item,
            source_name="stac.pc",
            query_id="q",
            fetched_at=datetime(2026, 5, 25, tzinfo=UTC),
            source_version="v",
        )
        assert row.collection == ""


# ---------------------------------------------------------------------------
# Unit tests: STACSource.query (with a faked client)
# ---------------------------------------------------------------------------


class _FakeSearch:
    """Mimics pystac_client's ItemSearch — only the `items()` method is used."""

    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items

    def items(self) -> list[pystac.Item]:
        return self._items


class _FakeClient:
    """Just enough to satisfy STACSource.query."""

    def __init__(self, items: list[pystac.Item]) -> None:
        self._items = items
        self.last_search_kwargs: dict[str, Any] | None = None

    def search(self, **kwargs: Any) -> _FakeSearch:
        self.last_search_kwargs = kwargs
        return _FakeSearch(self._items)

    def get_self_href(self) -> str:
        return "https://fake/api/stac/v1"


class TestSTACSourceQuery:
    def test_query_emits_source_rows(self) -> None:
        items = [
            _make_item(
                id_=f"item_{i}",
                datetime_=datetime(2024, 6, i + 1, tzinfo=UTC),
            )
            for i in range(3)
        ]
        src = STACSource(endpoint="https://fake", name="stac.fake")
        src._client = _FakeClient(items)  # type: ignore[assignment]

        rows = list(
            src.query(
                bounds=(-9, 38, -8, 39),
                interval=pd.Interval(
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-30", tz="UTC"),
                    closed="both",
                ),
                collection="sentinel-2-l2a",
            )
        )
        assert len(rows) == 3
        assert {r.id for r in rows} == {"item_0", "item_1", "item_2"}
        # Every row shares a query_id because they came from one call.
        assert len({r.provenance["query_id"] for r in rows}) == 1

    def test_search_kwargs_forwarded(self) -> None:
        src = STACSource(endpoint="https://fake", name="stac.fake")
        client = _FakeClient([])
        src._client = client  # type: ignore[assignment]

        list(
            src.query(
                bounds=(-10, 35, 5, 45),
                interval=pd.Interval(
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-30", tz="UTC"),
                    closed="both",
                ),
                collection="sentinel-2-l2a",
                filters={"eo:cloud_cover": {"lt": 20}},
                limit=5,
            )
        )
        kw = client.last_search_kwargs
        assert kw is not None
        assert kw["bbox"] == [-10, 35, 5, 45]
        assert kw["collections"] == ["sentinel-2-l2a"]
        assert kw["datetime"] == "2024-06-01T00:00:00Z/2024-06-30T00:00:00Z"
        assert kw["query"] == {"eo:cloud_cover": {"lt": 20}}
        assert kw["max_items"] == 5

    def test_cql2_filter_routed_separately(self) -> None:
        # `filter` key in the filters dict goes to pystac-client's
        # `filter=` (CQL-2), not the legacy `query=` channel.
        src = STACSource(endpoint="https://fake", name="stac.fake")
        client = _FakeClient([])
        src._client = client  # type: ignore[assignment]

        list(
            src.query(
                bounds=(-10, 35, 5, 45),
                filters={"filter": "eo:cloud_cover < 20"},
            )
        )
        kw = client.last_search_kwargs
        assert kw is not None
        assert kw["filter"] == "eo:cloud_cover < 20"
        assert "query" not in kw

    def test_optional_args_omitted(self) -> None:
        # Without interval / collection / filters / limit, the
        # passthroughs are omitted entirely rather than sent as `None`
        # (which some STAC servers reject).
        src = STACSource(endpoint="https://fake", name="stac.fake")
        client = _FakeClient([])
        src._client = client  # type: ignore[assignment]

        list(src.query(bounds=(-10, 35, 5, 45)))
        kw = client.last_search_kwargs
        assert kw is not None
        assert set(kw) == {"bbox"}

    def test_pc_factory_signs_assets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `STACSource.planetary_computer()` flips sign_assets=True, so
        # each item passes through `planetary_computer.sign` before
        # being mapped to a SourceRow.
        item = _make_item(datetime_=datetime(2024, 6, 15, tzinfo=UTC))
        signed_marker_asset = pystac.Asset(href="signed://red.tif")
        signed = _make_item(
            id_=item.id,
            datetime_=datetime(2024, 6, 15, tzinfo=UTC),
            assets={"B04": "signed://red.tif"},
        )

        sign_mock = MagicMock(return_value=signed)
        from geocatalog._src.sources import stac as stac_mod

        monkeypatch.setattr(stac_mod, "planetary_computer", MagicMock(sign=sign_mock))

        src = STACSource.planetary_computer()
        src._client = _FakeClient([item])  # type: ignore[assignment]

        rows = list(src.query(bounds=(-9, 38, -8, 39)))
        sign_mock.assert_called_once_with(item)
        assert rows[0].assets["B04"] == "signed://red.tif"
        # Mark the lint-only fixture as used.
        assert signed_marker_asset.href == "signed://red.tif"


class TestSTACSourceConstruction:
    def test_factories_set_endpoint_and_name(self) -> None:
        pc = STACSource.planetary_computer()
        assert pc.endpoint == "https://planetarycomputer.microsoft.com/api/stac/v1"
        assert pc.name == "stac.pc"
        assert pc.sign_assets is True

        es = STACSource.earth_search()
        assert es.endpoint == "https://earth-search.aws.element84.com/v1"
        assert es.name == "stac.es"
        assert es.sign_assets is False

    def test_auth_status_reachable(self) -> None:
        src = STACSource(endpoint="https://fake", name="stac.fake")
        src._client = _FakeClient([])  # type: ignore[assignment]
        status = src.auth_status()
        assert status.authenticated is True
        assert status.source == "stac.fake"

    def test_auth_status_unreachable(self) -> None:
        src = STACSource(endpoint="https://fake", name="stac.fake")
        broken = _FakeClient([])
        broken.get_self_href = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        src._client = broken  # type: ignore[assignment]
        status = src.auth_status()
        assert status.authenticated is False
        assert "boom" in (status.detail or "")


# ---------------------------------------------------------------------------
# Live tests — opt-in, requires network access to Planetary Computer.
# Run with: uv run pytest -m live --no-cov
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestSTACSourceLive:
    """End-to-end against the real Planetary Computer.

    Skipped by default. Each test caps `limit` so it stays under a
    couple of seconds.
    """

    def test_sentinel_2_iberia(self) -> None:
        src = STACSource.planetary_computer()
        rows = list(
            src.query(
                bounds=(-9.5, 38.5, -8.5, 39.5),
                interval=pd.Interval(
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-30", tz="UTC"),
                    closed="both",
                ),
                collection="sentinel-2-l2a",
                filters={"eo:cloud_cover": {"lt": 30}},
                limit=3,
            )
        )
        assert 0 < len(rows) <= 3
        for row in rows:
            assert row.source == "stac.pc"
            assert row.collection == "sentinel-2-l2a"
            # PC signs blob URLs — they carry a SAS token in the query string.
            assert any("sig=" in href for href in row.assets.values())

    def test_auth_status_real(self) -> None:
        src = STACSource.planetary_computer()
        status = src.auth_status()
        assert status.authenticated is True

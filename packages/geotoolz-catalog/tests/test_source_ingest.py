"""Source adapters and bundle ingest behave consistently (#240)."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import shapely
from tenacity import wait_none

from geocatalog._src import retry as retry_module
from geocatalog._src.bundle import CatalogBundle
from geocatalog._src.sources import cmr as cmr_mod
from geocatalog._src.sources._base import AuthStatus, Source, SourceRow
from geocatalog._src.sources._umm import granule_assets, granule_to_source_row
from geocatalog._src.sources.cmr import CMRSource, _cmr_item_to_source_row
from geocatalog._src.sources.earthaccess import _granule_to_source_row


FETCHED = datetime(2026, 1, 1, tzinfo=UTC)


def _granule(
    ur: str = "G1",
    *,
    urls: list[tuple[str, str]] = (),  # type: ignore[assignment]
    geometry: bool = True,
) -> dict[str, Any]:
    umm: dict[str, Any] = {
        "GranuleUR": ur,
        "CollectionReference": {"ShortName": "MOD09GA"},
        "TemporalExtent": {
            "RangeDateTime": {
                "BeginningDateTime": "2024-06-01T00:00:00Z",
                "EndingDateTime": "2024-06-01T23:59:59Z",
            }
        },
        "RelatedUrls": [{"URL": u, "Type": t} for u, t in urls],
        "CloudCover": 12.0,
    }
    if geometry:
        umm["SpatialExtent"] = {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "BoundingRectangles": [
                        {
                            "WestBoundingCoordinate": -10,
                            "SouthBoundingCoordinate": 35,
                            "EastBoundingCoordinate": -5,
                            "NorthBoundingCoordinate": 45,
                        }
                    ]
                }
            }
        }
    return {"umm": umm, "meta": {"concept-id": f"C-{ur}"}}


HTTPS = ("https://data.example/MOD09GA.A2024.hdf", "GET DATA")
S3 = ("s3://bucket/MOD09GA.A2024.hdf", "GET DATA VIA DIRECT ACCESS")


# ---------------------------------------------------------------------------
# One granule → row mapper
# ---------------------------------------------------------------------------


def test_cmr_and_earthaccess_rows_are_identical() -> None:
    item = _granule(urls=[HTTPS, S3])
    via_cmr = _cmr_item_to_source_row(
        item, source_name="x", query_id="q", fetched_at=FETCHED
    )
    via_ea = _granule_to_source_row(
        item,
        source_name="x",
        query_id="q",
        fetched_at=FETCHED,
        source_version="cmr/umm_json",
    )
    assert via_cmr == via_ea


def test_https_links_come_before_s3_whatever_the_order() -> None:
    assets = granule_assets(_granule(urls=[S3, HTTPS])["umm"])
    assert list(assets) == ["MOD09GA.A2024", "MOD09GA.A2024__s3"]


def test_https_and_s3_links_get_distinct_keys() -> None:
    assets = granule_assets(_granule(urls=[HTTPS, S3, HTTPS])["umm"])
    assert assets == {
        "MOD09GA.A2024": HTTPS[0],
        "MOD09GA.A2024__s3": S3[0],
        "MOD09GA.A2024__1": HTTPS[0],
    }


def test_concept_id_is_the_fallback_id() -> None:
    item = _granule()
    del item["umm"]["GranuleUR"]
    row = granule_to_source_row(
        item, source_name="x", query_id="q", fetched_at=FETCHED, source_version="v"
    )
    assert row is not None and row.id == "C-G1"


def test_granule_without_a_footprint_is_skipped() -> None:
    row = granule_to_source_row(
        _granule(geometry=False),
        source_name="x",
        query_id="q",
        fetched_at=FETCHED,
        source_version="v",
    )
    assert row is None


# ---------------------------------------------------------------------------
# CMR query string, retry
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.status = 200
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return json.dumps(self._body).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _capture_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def urlopen(req: Any, timeout: float = 0) -> _Response:
        urls.append(req.full_url)
        return _Response({"items": []})

    monkeypatch.setattr(cmr_mod.urllib.request, "urlopen", urlopen)
    return urls


def _params(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


def test_temporal_is_utc_z_at_second_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _capture_urls(monkeypatch)
    interval = pd.Interval(
        pd.Timestamp("2024-06-01T02:00:00.123456789+02:00"),
        pd.Timestamp("2024-06-02T02:00:00+02:00"),
        closed="both",
    )
    list(CMRSource().query((0, 0, 1, 1), interval, collection="X"))
    assert _params(urls[0])["temporal"] == ["2024-06-01T00:00:00Z,2024-06-02T00:00:00Z"]


def test_list_filters_repeat_the_key_as_given(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _capture_urls(monkeypatch)
    list(
        CMRSource().query(
            (0, 0, 1, 1),
            collection="X",
            filters={"provider": ["ASF", "LARC"], "platform[]": ["Terra", "Aqua"]},
        )
    )
    params = _params(urls[0])
    assert params["provider"] == ["ASF", "LARC"]  # CMR's documented form
    assert params["platform[]"] == ["Terra", "Aqua"]  # bracket kept when given


def test_temporal_upper_bound_rounds_up(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _capture_urls(monkeypatch)
    interval = pd.Interval(
        pd.Timestamp("2024-06-01T00:00:00.900", tz="UTC"),
        pd.Timestamp("2024-06-01T00:00:10.100", tz="UTC"),
        closed="both",
    )
    list(CMRSource().query((0, 0, 1, 1), interval, collection="X"))
    assert _params(urls[0])["temporal"] == ["2024-06-01T00:00:00Z,2024-06-01T00:00:11Z"]


def test_filters_cannot_override_built_in_params() -> None:
    with pytest.raises(ValueError, match="bounding_box"):
        list(CMRSource().query((0, 0, 1, 1), filters={"bounding_box": "1,2,3,4"}))


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://cmr", code, "err", {}, io.BytesIO(b""))  # type: ignore[arg-type]


def test_transient_http_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_module, "_RETRY_WAIT", wait_none())
    calls = []

    def urlopen(req: Any, timeout: float = 0) -> _Response:
        calls.append(req)
        if len(calls) < 3:
            raise _http_error(503 if len(calls) == 1 else 429)
        return _Response({"items": [_granule()]})

    monkeypatch.setattr(cmr_mod.urllib.request, "urlopen", urlopen)
    rows = list(CMRSource(retries=3).query((0, 0, 1, 1), collection="X"))
    assert len(calls) == 3
    assert [r.id for r in rows] == ["G1"]


def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_module, "_RETRY_WAIT", wait_none())
    calls = []

    def urlopen(req: Any, timeout: float = 0) -> _Response:
        calls.append(req)
        raise _http_error(404)

    monkeypatch.setattr(cmr_mod.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        list(CMRSource(retries=3).query((0, 0, 1, 1), collection="X"))
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# STAC: undecodable items are skipped like CMR granules
# ---------------------------------------------------------------------------


def test_stac_item_without_geometry_or_bbox_is_skipped() -> None:
    pystac = pytest.importorskip("pystac")
    pytest.importorskip("pystac_client")
    from geocatalog._src.sources.stac import STACSource

    def item(item_id: str, *, located: bool) -> Any:
        it = pystac.Item(
            id=item_id,
            geometry=shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 6, 1, tzinfo=UTC),
            properties={},
        )
        if not located:
            it.geometry = None
            it.bbox = None
        return it

    class _Search:
        def items(self) -> Iterator[Any]:
            yield item("bad", located=False)
            yield item("good", located=True)

    class _Client:
        def search(self, **kwargs: Any) -> _Search:
            return _Search()

    src = STACSource(endpoint="https://fake", name="stac.fake")
    src._client = _Client()  # type: ignore[assignment]
    assert [r.id for r in src.query((0, 0, 1, 1))] == ["good"]


# ---------------------------------------------------------------------------
# Bundle: id is a primary key; persistence
# ---------------------------------------------------------------------------


class _FixedSource(Source):
    name = "fixed"

    def __init__(self, rows: list[SourceRow]) -> None:
        self.rows = rows

    def query(
        self, bounds, interval=None, *, collection=None, filters=None, limit=None
    ):  # type: ignore[no-untyped-def]
        yield from self.rows

    def auth_status(self) -> AuthStatus:
        return AuthStatus(source=self.name, authenticated=True)


def _row(
    row_id: str,
    href: str = "https://x/a.tif",
    *,
    collection: str = "c",
    asset_key: str = "data",
) -> SourceRow:
    ts = pd.Timestamp("2024-06-01", tz="UTC")
    return SourceRow(
        id=row_id,
        source="fixed",
        collection=collection,
        geometry=shapely.box(0, 0, 1, 1),
        interval=pd.Interval(ts, ts, closed="both"),
        assets={asset_key: href},
    )


def _bundle_with(*row_ids: str) -> CatalogBundle:
    bundle = CatalogBundle.empty(target_crs="EPSG:4326")
    bundle.ingest(_FixedSource([_row(i) for i in row_ids]), bounds=(0, 0, 1, 1))
    return bundle


def test_reingest_raises_by_default() -> None:
    bundle = _bundle_with("a", "b")
    with pytest.raises(ValueError, match="already in the bundle"):
        bundle.ingest(_FixedSource([_row("b"), _row("c")]), bounds=(0, 0, 1, 1))
    assert sorted(bundle.catalog.gdf["id"]) == ["a", "b"]
    assert len(bundle.queries) == 1  # the failed ingest left no trace


def test_reingest_with_skip_is_a_no_op() -> None:
    bundle = _bundle_with("a", "b")
    bundle.ingest(
        _FixedSource([_row("a"), _row("b")]), bounds=(0, 0, 1, 1), on_duplicate="skip"
    )
    assert sorted(bundle.catalog.gdf["id"]) == ["a", "b"]


def test_reingest_with_replace_keeps_the_new_row() -> None:
    bundle = _bundle_with("a", "b")
    bundle.ingest(
        _FixedSource([_row("b", href="https://x/new.tif")]),
        bounds=(0, 0, 1, 1),
        on_duplicate="replace",
    )
    gdf = bundle.catalog.gdf.set_index("id")
    assert sorted(gdf.index) == ["a", "b"]
    assert gdf.loc["b", "filepath"] == "https://x/new.tif"


def test_duplicates_within_one_ingest_are_caught() -> None:
    bundle = CatalogBundle.empty(target_crs="EPSG:4326")
    with pytest.raises(ValueError, match="repeated"):
        bundle.ingest(_FixedSource([_row("a"), _row("a")]), bounds=(0, 0, 1, 1))


def test_created_at_survives_resaves(tmp_path: Path) -> None:
    bundle = _bundle_with("a")
    bundle.to_directory(tmp_path)
    first = json.loads((tmp_path / "_meta.json").read_text())

    reloaded = CatalogBundle.from_directory(tmp_path)
    reloaded.to_directory(tmp_path)
    second = json.loads((tmp_path / "_meta.json").read_text())

    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]


def test_failed_save_keeps_the_previous_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from geocatalog._src.bundle import _catalog_bundle as bundle_mod

    bundle = _bundle_with("a")
    bundle.to_directory(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    # The catalog changes too: a half-published save would pair the new
    # items table with the old queries.
    bundle.ingest(_FixedSource([_row("b")]), bounds=(0, 0, 1, 1))

    def crash(queries: Any, path: Path) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(bundle_mod, "_queries_to_parquet", crash)
    bundle.queries.append(bundle.queries[0])
    with pytest.raises(OSError, match="disk full"):
        bundle.to_directory(tmp_path)

    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert after["items.parquet"] == before["items.parquet"]
    assert after["queries.parquet"] == before["queries.parquet"]
    assert after["_meta.json"] == before["_meta.json"]
    assert not [n for n in after if n.startswith(".")]


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"backend": "rastr"}, "unknown backend"),
        ({"target_crs": "not a crs"}, "valid `target_crs`"),
        ({"target_crs": "EPSG:3857"}, "does not match"),
    ],
)
def test_meta_is_validated(tmp_path: Path, patch: dict[str, Any], match: str) -> None:
    _bundle_with("a").to_directory(tmp_path)
    meta_path = tmp_path / "_meta.json"
    meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), **patch}))
    with pytest.raises(ValueError, match=match):
        CatalogBundle.from_directory(tmp_path)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_sources_all_lists_every_adapter() -> None:
    import geocatalog._src.sources as inner

    for name in ("CMRSource", "EarthAccessSource", "GEESource", "STACSource"):
        assert name in inner.__all__


def test_same_id_in_another_collection_is_not_a_duplicate() -> None:
    bundle = _bundle_with("a")
    bundle.ingest(_FixedSource([_row("a", collection="other")]), bounds=(0, 0, 1, 1))
    assert sorted(bundle.catalog.gdf["collection"]) == ["c", "other"]


def test_skipped_duplicates_are_not_validated() -> None:
    bundle = CatalogBundle.empty(target_crs="EPSG:4326")
    src = _FixedSource([_row("a", asset_key="B04")])
    bundle.ingest(src, bounds=(0, 0, 1, 1), primary_asset="B04")
    # The re-ingested row lacks B04, but `skip` discards it unexamined.
    bundle.ingest(
        _FixedSource([_row("a", asset_key="other")]),
        bounds=(0, 0, 1, 1),
        primary_asset="B04",
        on_duplicate="skip",
    )
    assert len(bundle.catalog) == 1


def test_replace_drops_matchups_of_replaced_items() -> None:
    from geocatalog._src.matchup.engine import MatchupRow

    bundle = _bundle_with("a", "b")
    ts = pd.Timestamp("2024-06-01", tz="UTC")

    def matchup(*ids: str) -> MatchupRow:
        return MatchupRow(
            matchup_id="-".join(ids),
            strategy="test",
            member_ids=ids,
            member_sources=("fixed",) * len(ids),
            member_roles=("primary",) + ("secondary",) * (len(ids) - 1),
            geometry_intersect=shapely.box(0, 0, 1, 1),
            time_reference=ts.to_pydatetime(),
            time_offset_sec=(0.0,) * len(ids),
        )

    bundle.matchups = [matchup("a", "b"), matchup("b")]
    bundle.ingest(
        _FixedSource([_row("a")]), bounds=(0, 0, 1, 1), on_duplicate="replace"
    )
    assert [m.matchup_id for m in bundle.matchups] == ["b"]


def test_stac_items_with_bad_projection_are_not_silently_skipped() -> None:
    pystac = pytest.importorskip("pystac")
    from geocatalog import from_stac_items

    item = pystac.Item(
        id="x",
        geometry=shapely.geometry.mapping(shapely.box(0, 0, 1, 1)),
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 6, 1, tzinfo=UTC),
        properties={"proj:epsg": "unknown"},
    )
    item.add_asset("data", pystac.Asset(href="https://x/a.tif"))
    with pytest.raises(ValueError):
        from_stac_items([item])


def test_empty_bundle_crs_is_validated(tmp_path: Path) -> None:
    CatalogBundle.empty(target_crs="EPSG:4326").to_directory(tmp_path)
    meta_path = tmp_path / "_meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "target_crs": "EPSG:3857"}))
    with pytest.raises(ValueError, match="does not match"):
        CatalogBundle.from_directory(tmp_path)


@pytest.mark.parametrize(("code", "retried"), [(507, True), (520, True), (404, False)])
def test_every_5xx_is_retried(code: int, retried: bool) -> None:
    from geocatalog._src.retry import _is_transient

    assert _is_transient(_http_error(code)) is retried


def test_replace_keeps_matchups_of_a_same_id_item_in_another_collection(
    tmp_path: Path,
) -> None:
    from geocatalog._src.matchup import Intersects, Synchronous, matchup

    bundle = _bundle_with("a")
    bundle.ingest(_FixedSource([_row("a", collection="other")]), bounds=(0, 0, 1, 1))
    station = SourceRow(
        id="st",
        source="insitu",
        collection="stations",
        geometry=shapely.box(0.4, 0.4, 0.5, 0.5),
        interval=pd.Interval(
            pd.Timestamp("2024-06-01", tz="UTC"),
            pd.Timestamp("2024-06-01", tz="UTC"),
            closed="both",
        ),
    )
    rows = list(
        matchup(
            [_row("a", collection="c"), _row("a", collection="other")],
            [station],
            spatial=Intersects(),
            temporal=Synchronous(),
        )
    )
    assert [r.member_collections for r in rows] == [
        ("c", "stations"),
        ("other", "stations"),
    ]
    bundle.write_matchups(rows)
    bundle.to_directory(tmp_path)
    bundle = CatalogBundle.from_directory(tmp_path)  # collections persist

    bundle.ingest(
        _FixedSource([_row("a", collection="c")]),
        bounds=(0, 0, 1, 1),
        on_duplicate="replace",
    )
    assert [m.member_collections[0] for m in bundle.matchups] == ["other"]

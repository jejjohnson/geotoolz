"""`matchup()` is deterministic, catalog-aware and tz/NaT/CRS-safe (#242)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from geocatalog import InMemoryGeoCatalog
from geocatalog._src.bundle import CatalogBundle
from geocatalog._src.matchup import (
    CentroidWithin,
    Contains,
    Intersects,
    IouAtLeast,
    NearestInTime,
    Synchronous,
    WithinWindow,
    matchup,
)
from geocatalog._src.matchup.engine import MatchupRow
from geocatalog._src.sources._base import SourceRow


T0 = datetime(2024, 6, 1, 12, tzinfo=UTC)


def _row(
    id_: str,
    *,
    source: str = "a",
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1),
    time: Any = T0,
    end: Any = None,
) -> SourceRow:
    left = pd.Timestamp(time)
    right = pd.Timestamp(end) if end is not None else left
    return SourceRow(
        id=id_,
        source=source,
        collection="c",
        geometry=box(*bbox),
        interval=pd.Interval(left, right, closed="both"),
    )


def _run(primary: Any, secondary: Any, **kwargs: Any) -> list[MatchupRow]:
    kwargs.setdefault("spatial", Intersects())
    kwargs.setdefault("temporal", NearestInTime(dt="6h"))
    return list(matchup(primary, secondary, **kwargs))


def _catalog(rows: list[dict[str, Any]], crs: str = "EPSG:4326") -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return InMemoryGeoCatalog(gdf, kind="raster")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_runs_give_identical_rows() -> None:
    primary = [_row(f"p{i}", bbox=(i, 0, i + 1, 1)) for i in range(3)]
    secondary = [
        _row(f"s{i}", source="b", bbox=(i + 0.5, 0, i + 1.5, 1)) for i in range(3)
    ]
    assert _run(primary, secondary) == _run(primary, secondary)


def test_ids_do_not_depend_on_secondary_order() -> None:
    primary = [_row("p")]
    secondary = [
        _row(f"s{i}", source="b", time=T0 + timedelta(hours=i)) for i in range(5)
    ]
    forward = _run(primary, secondary, temporal=WithinWindow(start="-6h", end="6h"))
    backward = _run(
        primary, secondary[::-1], temporal=WithinWindow(start="-6h", end="6h")
    )
    assert forward == backward
    assert [r.member_ids[1] for r in forward] == [f"s{i}" for i in range(5)]


def test_nearest_in_time_tie_is_broken_by_source_and_id() -> None:
    primary = [_row("p")]
    tied = [
        _row("z", source="b", time=T0 + timedelta(hours=1)),
        _row("y", source="b", time=T0 - timedelta(hours=1)),
        _row("x", source="c", time=T0 + timedelta(hours=1)),
    ]
    for order in (tied, tied[::-1], [tied[2], tied[0], tied[1]]):
        (row,) = _run(primary, order)
        assert row.member_ids == ("p", "y")


def test_matchup_id_is_a_content_hash() -> None:
    (row,) = _run([_row("p")], [_row("s", source="b")])
    # Golden: changing the id recipe breaks every persisted matchup.
    assert row.matchup_id == "c7060ed5017330fba36fd4ba4d77c39e"
    other = _run([_row("p")], [_row("s", source="b")], temporal=NearestInTime(dt="1h"))
    assert other[0].matchup_id != row.matchup_id


def test_duplicate_rows_and_primaries_emit_one_matchup() -> None:
    primary = [_row("p"), _row("p")]
    secondary = [_row("s", source="b"), _row("s", source="b")]
    assert len(_run(primary, secondary)) == 1


def test_self_pairs_are_skipped_unless_asked_for() -> None:
    rows = [_row("a"), _row("b", bbox=(0.5, 0, 1.5, 1))]
    pairs = {r.member_ids for r in _run(rows, rows, temporal=Synchronous())}
    assert pairs == {("a", "b"), ("b", "a")}
    pairs = {
        r.member_ids
        for r in _run(rows, rows, temporal=Synchronous(), include_self=True)
    }
    assert ("a", "a") in pairs and ("b", "b") in pairs


def test_one_row_in_two_roles_is_not_a_matchup() -> None:
    shared = _row("s", source="b")
    rows = _run(
        [_row("p")],
        {"x": [shared], "y": [shared]},
        temporal=WithinWindow(start="-1h", end="1h"),
    )
    assert rows == []


def test_primary_role_name_is_reserved() -> None:
    with pytest.raises(ValueError, match="primary"):
        _run([_row("p")], {"primary": [_row("s")]})


# ---------------------------------------------------------------------------
# Time zones and missing times
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "temporal",
    [NearestInTime(dt="2h"), WithinWindow(start="-2h", end="2h"), Synchronous("2h")],
    ids=["nearest", "window", "synchronous"],
)
def test_naive_and_aware_intervals_mix(temporal: Any) -> None:
    naive = _row("p", time=pd.Timestamp("2024-06-01T12:00"))  # naive = UTC
    tokyo = _row(
        "s", source="b", time=pd.Timestamp("2024-06-01T22:00", tz="Asia/Tokyo")
    )  # 13:00 UTC
    (row,) = _run([naive], [tokyo], temporal=temporal)
    assert row.time_offset_sec == (0.0, 3600.0)
    assert row.time_reference == T0


def test_rows_without_times_never_match() -> None:
    nat = _catalog(
        [
            {
                "geometry": box(0, 0, 1, 1),
                "start_time": pd.NaT,
                "end_time": pd.NaT,
                "filepath": "nat.tif",
            },
            {
                "geometry": box(0, 0, 1, 1),
                "start_time": pd.Timestamp("2024-06-01T12:00"),
                "end_time": pd.Timestamp("2024-06-01T12:00"),
                "filepath": "ok.tif",
            },
        ]
    )
    assert [r.member_ids for r in _run([_row("p")], nat)] == [("p", "ok.tif")]
    assert [r.member_ids for r in _run(nat, [_row("s", source="b")])] == [
        ("ok.tif", "s")
    ]


def test_shipped_strategies_tolerate_nat_directly() -> None:
    primary = pd.Interval(pd.Timestamp(T0), pd.Timestamp(T0), closed="both")
    nat = pd.IntervalIndex.from_arrays(
        pd.to_datetime([None, T0]), pd.to_datetime([None, T0]), closed="both"
    )
    for strategy in (NearestInTime(dt="1h"), WithinWindow("-1h", "1h"), Synchronous()):
        assert strategy.select(primary, nat) == [1]


# ---------------------------------------------------------------------------
# Every shipped strategy through matchup()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spatial", "expected"),
    [
        (Intersects(), {"inside", "overlap"}),
        (IouAtLeast(0.2), {"inside", "overlap"}),
        (IouAtLeast(0.5), {"overlap"}),
        (CentroidWithin(buffer=0.0), {"inside", "overlap"}),
        (Contains(), {"inside"}),
    ],
    ids=["intersects", "iou0.2", "iou0.5", "centroid", "contains"],
)
def test_every_spatial_strategy_through_the_engine(
    spatial: Any, expected: set[str]
) -> None:
    primary = [_row("p", bbox=(0, 0, 2, 2))]
    secondary = [
        _row("inside", source="b", bbox=(0.5, 0.5, 1.5, 1.5)),  # IoU 0.25
        _row("overlap", source="b", bbox=(0.5, 0, 2.5, 2)),  # IoU 0.6
        _row("apart", source="b", bbox=(5, 5, 6, 6)),
    ]
    got = {
        r.member_ids[1]
        for r in _run(
            primary, secondary, spatial=spatial, temporal=WithinWindow("-1h", "1h")
        )
    }
    assert got == expected


@pytest.mark.parametrize(
    ("temporal", "expected"),
    [
        (NearestInTime(dt="3h"), {"h1"}),
        (WithinWindow(start="-3h", end="3h"), {"h1", "h2"}),
        (WithinWindow(start="0h", end="5h"), {"h1", "h4"}),
        (Synchronous(), {"span"}),
    ],
    ids=["nearest", "window", "forward-window", "synchronous"],
)
def test_every_temporal_strategy_through_the_engine(
    temporal: Any, expected: set[str]
) -> None:
    secondary = [
        _row("h1", source="b", time=T0 + timedelta(hours=1)),
        _row("h2", source="b", time=T0 - timedelta(hours=2)),
        _row("h4", source="b", time=T0 + timedelta(hours=4)),
        _row(
            "span",
            source="b",
            time=T0 - timedelta(hours=10),
            end=T0 + timedelta(minutes=1),
        ),
    ]
    got = {r.member_ids[1] for r in _run([_row("p")], secondary, temporal=temporal)}
    # `span`'s midpoint is ~5h before T0, outside every midpoint window.
    assert got == expected


def test_custom_filter_only_strategy_with_shared_intervals() -> None:
    @dataclasses.dataclass(frozen=True)
    class KeepAll:
        def filter(
            self, primary: pd.Interval, candidates: pd.IntervalIndex
        ) -> pd.IntervalIndex:
            return candidates[::-1]  # order is not part of the contract

    secondary = [_row(f"s{i}", source="b") for i in range(3)]  # identical times
    rows = _run([_row("p")], secondary, temporal=KeepAll())
    assert [r.member_ids[1] for r in rows] == ["s0", "s1", "s2"]


def test_custom_filter_returning_a_foreign_interval_raises() -> None:
    @dataclasses.dataclass(frozen=True)
    class Invents:
        def filter(
            self, primary: pd.Interval, candidates: pd.IntervalIndex
        ) -> pd.IntervalIndex:
            return pd.IntervalIndex([pd.Interval(0, 1, closed="both")])

    with pytest.raises(ValueError, match="not one of the candidate"):
        _run([_row("p")], [_row("s", source="b")], temporal=Invents())


def test_tolerance_is_plain_json_in_seconds() -> None:
    (row,) = _run(
        [_row("p")],
        [_row("s", source="b")],
        spatial=IouAtLeast(0.2),
        temporal=WithinWindow(start="-1h", end="30min"),
        join="all",
    )
    assert row.tolerance == {
        "spatial": {"type": "IouAtLeast", "threshold": 0.2},
        "temporal": {"type": "WithinWindow", "start_sec": -3600.0, "end_sec": 1800.0},
        "join": "all",
        "crs": "EPSG:4326",
    }
    assert "IouAtLeast" in row.strategy and "WithinWindow" in row.strategy


# ---------------------------------------------------------------------------
# Catalogs, bundles and CRS
# ---------------------------------------------------------------------------


def _utm_scenes() -> InMemoryGeoCatalog:
    # Two 10 km tiles in UTM 31N near (3°E, 0°N); the second is 20 km east.
    return _catalog(
        [
            {
                "geometry": box(500_000, 0, 510_000, 10_000),
                "start_time": pd.Timestamp("2024-06-01T12:00"),
                "end_time": pd.Timestamp("2024-06-01T12:00"),
                "filepath": "near.tif",
            },
            {
                "geometry": box(530_000, 0, 540_000, 10_000),
                "start_time": pd.Timestamp("2024-06-01T12:00"),
                "end_time": pd.Timestamp("2024-06-01T12:00"),
                "filepath": "far.tif",
            },
        ],
        crs="EPSG:32631",
    )


def test_catalog_primary_with_source_row_secondary() -> None:
    # A lon/lat station footprint inside `near.tif`.
    station = _row("st", source="insitu", bbox=(3.04, 0.04, 3.05, 0.05))
    (row,) = _run(_utm_scenes(), [station])
    assert row.member_ids == ("near.tif", "st")
    assert row.member_sources == ("catalog", "insitu")
    # The intersect is in the primary catalog's CRS (metres).
    assert row.tolerance["crs"] == "EPSG:32631"
    assert row.geometry_intersect.bounds[0] > 500_000


def test_matchup_ids_keep_the_legacy_crs_serialisation() -> None:
    """A CRS without an authority keys matchups by its PROJ string, as before.

    The tolerance feeds every persisted `MatchupRow` id; re-serialising it
    (e.g. as WKT2) would re-key existing bundles and duplicate rows on
    the next `write_matchups`.
    """
    import pyproj

    laea = "+proj=laea +lat_0=0 +lon_0=3 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    scenes = _utm_scenes()
    gdf = scenes.gdf.to_crs(laea)
    station = _row("st", source="insitu", bbox=(3.04, 0.04, 3.05, 0.05))
    (row,) = _run(InMemoryGeoCatalog(gdf, kind="raster"), [station])
    assert row.tolerance["crs"] == pyproj.CRS(laea).to_string()


def test_bundle_items_keep_their_source_ids() -> None:
    bundle = CatalogBundle.empty(crs="EPSG:4326", kind="raster")

    class _Fake:
        name = "fake"

        def query(self, *args: Any, **kwargs: Any) -> Any:
            yield _row("granule-1", source="fake")

    bundle.ingest(_Fake(), bounds=(-1, -1, 2, 2))
    (row,) = _run([_row("p")], bundle)
    assert row.member_ids == ("p", "granule-1")
    assert row.member_sources == ("a", "fake")


def test_crs_makes_the_buffer_metres() -> None:
    station = [_row("st", source="insitu", bbox=(3.149, 0.04, 3.15, 0.041))]
    # `near.tif` ends at 510 km easting; the station is ~7 km east of it.
    strict = _run(_utm_scenes(), station, spatial=CentroidWithin(buffer=0.0))
    buffered = _run(_utm_scenes(), station, spatial=CentroidWithin(buffer=10_000.0))
    assert strict == []
    assert [r.member_ids for r in buffered] == [("near.tif", "st")]


def test_degree_buffer_warns() -> None:
    with pytest.warns(UserWarning, match="degrees"):
        _run([_row("p")], [_row("s", source="b")], spatial=CentroidWithin(0.1))


def test_explicit_crs_rejoins_an_antimeridian_split_footprint() -> None:
    # A granule split at ±180° into two halves; its true centroid is on
    # the seam, but in lon/lat the parts' centroid is near 0°.
    split = SourceRow(
        id="g",
        source="b",
        collection="c",
        geometry=shapely.MultiPolygon([box(179, -1, 180, 1), box(-180, -1, -179, 1)]),
        interval=pd.Interval(pd.Timestamp(T0), pd.Timestamp(T0), closed="both"),
    )
    swath = [_row("p", bbox=(179.5, -0.5, 180, 0.5))]
    assert _run(swath, [split], spatial=CentroidWithin(buffer=0.0)) == []
    rows = _run(
        swath, [split], spatial=CentroidWithin(buffer=60_000.0), crs="EPSG:3832"
    )
    assert [r.member_ids for r in rows] == [("p", "g")]


def test_same_id_in_two_collections_are_two_members() -> None:
    a = dataclasses.replace(_row("x", source="b"), collection="c1")
    b = dataclasses.replace(_row("x", source="b"), collection="c2")
    rows = _run([_row("p")], [a, b], temporal=Synchronous())
    assert sorted(r.member_collections[1] for r in rows) == ["c1", "c2"]
    assert len({r.matchup_id for r in rows}) == 2


# ---------------------------------------------------------------------------
# Review follow-ups (#364)
# ---------------------------------------------------------------------------


def test_missing_ids_in_string_columns_fall_back_to_the_filepath() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["a.tif", "b.tif"],
            "id": pd.array([None, None], dtype="string"),
            "source": pd.array([None, None], dtype="string"),
        },
        geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1)],
        crs="EPSG:4326",
        index=pd.IntervalIndex.from_arrays(
            [pd.Timestamp("2024-06-01T12:00")] * 2,
            [pd.Timestamp("2024-06-01T12:00")] * 2,
            closed="both",
        ),
    )
    catalog = InMemoryGeoCatalog(gdf, kind="raster")
    rows = _run([_row("p")], catalog, temporal=Synchronous())
    assert sorted(r.member_ids[1] for r in rows) == ["a.tif", "b.tif"]
    assert {r.member_sources[1] for r in rows} == {"catalog"}


class _PlainStrategy:
    """A custom strategy with no dataclass and no ``__repr__``."""

    def __init__(self, hours: int) -> None:
        self.hours = hours

    def filter(
        self, primary: pd.Interval, candidates: pd.IntervalIndex
    ) -> pd.IntervalIndex:
        return candidates


def test_plain_custom_strategy_ids_do_not_depend_on_the_instance() -> None:
    one = _run([_row("p")], [_row("s", source="b")], temporal=_PlainStrategy(2))
    two = _run([_row("p")], [_row("s", source="b")], temporal=_PlainStrategy(2))
    other = _run([_row("p")], [_row("s", source="b")], temporal=_PlainStrategy(3))
    assert one[0].matchup_id == two[0].matchup_id != other[0].matchup_id
    assert "0x" not in one[0].strategy


def test_rewriting_a_matchup_replaces_it_in_the_bundle() -> None:
    bundle = CatalogBundle.empty(crs="EPSG:4326", kind="raster")
    rows = _run(
        [_row("p")],
        [_row("s", source="b"), _row("t", source="b")],
        temporal=Synchronous(),
    )
    assert bundle.write_matchups(rows, tag="first") == 2
    assert (
        bundle.write_matchups(
            _run([_row("p")], [_row("s", source="b")], temporal=Synchronous()),
            tag="second",
        )
        == 1
    )
    assert [(m.member_ids[1], m.query_set) for m in bundle.matchups] == [
        ("s", "second"),
        ("t", "first"),
    ]

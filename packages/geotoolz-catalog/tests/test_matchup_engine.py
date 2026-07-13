"""End-to-end tests for the matchup engine.

Exercises the matchup() pipeline against synthetic `SourceRow`
inputs spread across a small Iberia bbox. Covers:

* Spatial strategies (Intersects / IouAtLeast / CentroidWithin / Contains)
* Temporal strategies (NearestInTime / WithinWindow / Synchronous)
* Pairwise and N-way matchups, with "all" and "any" join policies.
* STRtree pre-filter correctness — secondaries far outside the
  primary footprint are skipped without ever calling `.match()`.

Synthetic fixtures (no pystac, no network) so the suite runs in
every CI invocation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from shapely.geometry import box

import geocatalog.matchup as matchup_ns
from geocatalog._src.matchup import (
    CentroidWithin,
    Contains,
    Intersects,
    IouAtLeast,
    MatchupRow,
    NearestInTime,
    Synchronous,
    WithinWindow,
    matchup,
)
from geocatalog._src.sources._base import SourceRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(
    id_: str,
    *,
    source: str = "test",
    collection: str = "synthetic",
    bbox: tuple[float, float, float, float] = (-9.0, 38.0, -8.0, 39.0),
    time: datetime,
    duration: timedelta = timedelta(0),
    properties: dict | None = None,
) -> SourceRow:
    """Build a minimal `SourceRow` for matchup tests."""
    geom = box(*bbox)
    interval = pd.Interval(
        pd.Timestamp(time), pd.Timestamp(time + duration), closed="both"
    )
    return SourceRow(
        id=id_,
        source=source,
        collection=collection,
        geometry=geom,
        interval=interval,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# Spatial strategies
# ---------------------------------------------------------------------------


class TestSpatialStrategies:
    def test_intersects_overlapping(self) -> None:
        a = box(0, 0, 1, 1)
        b = box(0.5, 0.5, 1.5, 1.5)
        assert Intersects().match(a, b)

    def test_intersects_disjoint(self) -> None:
        a = box(0, 0, 1, 1)
        b = box(2, 2, 3, 3)
        assert not Intersects().match(a, b)

    def test_iou_above_threshold(self) -> None:
        # Two identical squares → IoU = 1.0.
        a = box(0, 0, 1, 1)
        assert IouAtLeast(0.5).match(a, a)

    def test_iou_below_threshold(self) -> None:
        # Two 2x1 rectangles sharing a 1x1 overlap:
        # intersection area = 1, union area = 3, IoU = 1/3 ~= 0.333.
        a = box(0, 0, 2, 1)
        b = box(1, 0, 3, 1)
        assert IouAtLeast(0.30).match(a, b)
        assert not IouAtLeast(0.40).match(a, b)

    def test_iou_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            IouAtLeast(-0.1)
        with pytest.raises(ValueError):
            IouAtLeast(1.1)

    def test_iou_disjoint_is_false(self) -> None:
        # Even with threshold=0, disjoint geoms produce empty
        # intersection → no match (handled by the early bailout).
        a = box(0, 0, 1, 1)
        b = box(2, 2, 3, 3)
        assert not IouAtLeast(0.0).match(a, b)

    def test_iou_zero_area_geoms_only_match_when_equal(self) -> None:
        # Two zero-area geometries (e.g. crossing LineStrings,
        # coincident points) can't be measured by an area ratio.
        # The degenerate branch treats them as matching only when
        # the geometries are equal — otherwise the threshold would
        # be silently ignored.
        from shapely.geometry import LineString, Point

        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)  # equal to p1
        p3 = Point(3.0, 4.0)  # disjoint
        assert IouAtLeast(0.5).match(p1, p2)
        assert not IouAtLeast(0.5).match(p1, p3)
        # Crossing LineStrings: intersection is a non-empty Point
        # (area 0), union area 0 too. Not equal as full geometries
        # → no match, even at threshold=0.
        crossing_a = LineString([(0, 0), (2, 2)])
        crossing_b = LineString([(0, 2), (2, 0)])
        assert not IouAtLeast(0.0).match(crossing_a, crossing_b)

    def test_centroid_within_no_buffer(self) -> None:
        # `secondary` is the unit square (centroid at (0.5, 0.5)),
        # primary is [0, 1]^2 → centroid is inside.
        primary = box(0, 0, 1, 1)
        secondary = box(0.4, 0.4, 0.6, 0.6)
        assert CentroidWithin(buffer=0.0).match(primary, secondary)

    def test_centroid_within_with_buffer(self) -> None:
        # Without buffer the centroid is outside; with a buffer of
        # 0.5 the buffered primary reaches it.
        primary = box(0, 0, 1, 1)
        secondary = box(1.3, 0.4, 1.5, 0.6)  # centroid at (1.4, 0.5)
        assert not CentroidWithin(buffer=0.0).match(primary, secondary)
        assert CentroidWithin(buffer=0.5).match(primary, secondary)

    def test_centroid_within_string_buffer_unsupported(self) -> None:
        # "5km" etc. is reserved for a future revision; today we
        # only accept floats in CRS units.
        primary = box(0, 0, 1, 1)
        secondary = box(0.4, 0.4, 0.6, 0.6)
        with pytest.raises(NotImplementedError, match="String-with-units"):
            CentroidWithin(buffer="5km").match(primary, secondary)

    def test_contains_true(self) -> None:
        primary = box(0, 0, 2, 2)
        secondary = box(0.5, 0.5, 1.5, 1.5)
        assert Contains().match(primary, secondary)

    def test_contains_false(self) -> None:
        # Partial overlap doesn't count as contained.
        primary = box(0, 0, 1, 1)
        secondary = box(0.5, 0.5, 1.5, 1.5)
        assert not Contains().match(primary, secondary)


# ---------------------------------------------------------------------------
# Temporal strategies
# ---------------------------------------------------------------------------


def _iv(t: datetime, duration: timedelta = timedelta(0)) -> pd.Interval:
    return pd.Interval(pd.Timestamp(t), pd.Timestamp(t + duration), closed="both")


def _ividx(items: list[pd.Interval]) -> pd.IntervalIndex:
    return pd.IntervalIndex.from_tuples(
        [(iv.left, iv.right) for iv in items], closed="both"
    )


class TestNearestInTime:
    def test_picks_closest_within_dt(self) -> None:
        primary = _iv(datetime(2024, 6, 15, 12, 0, tzinfo=UTC))
        candidates = _ividx(
            [
                _iv(datetime(2024, 6, 15, 8, 0, tzinfo=UTC)),  # 4h before
                _iv(datetime(2024, 6, 15, 14, 0, tzinfo=UTC)),  # 2h after — nearest
                _iv(datetime(2024, 6, 15, 18, 0, tzinfo=UTC)),  # 6h after
            ]
        )
        result = NearestInTime(dt="6h").filter(primary, candidates)
        assert len(result) == 1
        assert result[0].left == pd.Timestamp(datetime(2024, 6, 15, 14, 0, tzinfo=UTC))

    def test_empty_when_nothing_in_range(self) -> None:
        primary = _iv(datetime(2024, 6, 15, 12, 0, tzinfo=UTC))
        candidates = _ividx(
            [
                _iv(datetime(2024, 6, 20, 12, 0, tzinfo=UTC)),  # 5 days later
            ]
        )
        result = NearestInTime(dt="1h").filter(primary, candidates)
        assert len(result) == 0

    def test_handles_empty_candidates(self) -> None:
        primary = _iv(datetime(2024, 6, 15, tzinfo=UTC))
        candidates = pd.IntervalIndex([], closed="both")
        result = NearestInTime(dt="6h").filter(primary, candidates)
        assert len(result) == 0

    def test_accepts_timedelta(self) -> None:
        primary = _iv(datetime(2024, 6, 15, 12, 0, tzinfo=UTC))
        candidates = _ividx([_iv(datetime(2024, 6, 15, 13, 0, tzinfo=UTC))])
        result = NearestInTime(dt=timedelta(hours=2)).filter(primary, candidates)
        assert len(result) == 1


class TestWithinWindow:
    def test_filters_to_window(self) -> None:
        primary = _iv(datetime(2024, 6, 15, 12, 0, tzinfo=UTC))
        candidates = _ividx(
            [
                _iv(datetime(2024, 6, 14, 0, 0, tzinfo=UTC)),  # 36h before
                _iv(datetime(2024, 6, 15, 6, 0, tzinfo=UTC)),  # 6h before
                _iv(datetime(2024, 6, 15, 18, 0, tzinfo=UTC)),  # 6h after
                _iv(datetime(2024, 6, 17, 0, 0, tzinfo=UTC)),  # 36h after
            ]
        )
        # ±12h window: only the middle two survive.
        result = WithinWindow(start="-12h", end="12h").filter(primary, candidates)
        assert len(result) == 2

    def test_empty_candidates(self) -> None:
        primary = _iv(datetime(2024, 6, 15, tzinfo=UTC))
        candidates = pd.IntervalIndex([], closed="both")
        result = WithinWindow(start="-1h", end="1h").filter(primary, candidates)
        assert len(result) == 0


class TestSynchronous:
    def test_overlapping_intervals(self) -> None:
        # Primary is a 1-hour window; candidate overlaps the second half.
        primary = _iv(
            datetime(2024, 6, 15, 12, 0, tzinfo=UTC), duration=timedelta(hours=1)
        )
        candidates = _ividx(
            [
                _iv(
                    datetime(2024, 6, 15, 12, 30, tzinfo=UTC),
                    duration=timedelta(hours=1),
                ),
                _iv(
                    datetime(2024, 6, 15, 14, 0, tzinfo=UTC),
                    duration=timedelta(hours=1),
                ),  # no overlap
            ]
        )
        result = Synchronous().filter(primary, candidates)
        assert len(result) == 1

    def test_tolerance_allows_near_miss(self) -> None:
        # Primary [12:00, 13:00]; candidate [13:30, 14:30] doesn't
        # overlap directly, but a 1h tolerance expands the window
        # to [11:00, 14:00] which catches it.
        primary = _iv(
            datetime(2024, 6, 15, 12, 0, tzinfo=UTC), duration=timedelta(hours=1)
        )
        candidates = _ividx(
            [
                _iv(
                    datetime(2024, 6, 15, 13, 30, tzinfo=UTC),
                    duration=timedelta(hours=1),
                ),
            ]
        )
        assert len(Synchronous().filter(primary, candidates)) == 0
        assert len(Synchronous(tolerance="1h").filter(primary, candidates)) == 1


# ---------------------------------------------------------------------------
# Matchup engine — pairwise
# ---------------------------------------------------------------------------


class TestMatchupPairwise:
    def _build(self) -> tuple[list[SourceRow], list[SourceRow]]:
        # 3 primary rows over Iberia at noon on consecutive days.
        primaries = [
            _row(
                f"p_{i}",
                source="modis",
                collection="MOD09GA",
                bbox=(-9.0, 38.0, -8.0, 39.0),
                time=datetime(2024, 6, 14 + i, 12, 0, tzinfo=UTC),
            )
            for i in range(3)
        ]
        secondaries = [
            # 1 closely matching S2 per primary day, plus 1 far away
            # in space and 1 far away in time (should both be skipped).
            _row(
                "s_close_0",
                source="s2",
                bbox=(-8.5, 38.5, -7.5, 39.5),
                time=datetime(2024, 6, 14, 14, 0, tzinfo=UTC),  # 2h after
            ),
            _row(
                "s_close_1",
                source="s2",
                bbox=(-8.5, 38.5, -7.5, 39.5),
                time=datetime(2024, 6, 15, 14, 0, tzinfo=UTC),
            ),
            _row(
                "s_close_2",
                source="s2",
                bbox=(-8.5, 38.5, -7.5, 39.5),
                time=datetime(2024, 6, 16, 14, 0, tzinfo=UTC),
            ),
            _row(
                "s_far_spatial",
                source="s2",
                bbox=(40.0, 40.0, 41.0, 41.0),  # central Asia — disjoint
                time=datetime(2024, 6, 15, 14, 0, tzinfo=UTC),
            ),
            _row(
                "s_far_temporal",
                source="s2",
                bbox=(-8.5, 38.5, -7.5, 39.5),  # overlaps spatially
                time=datetime(2025, 6, 15, 14, 0, tzinfo=UTC),  # +1 year
            ),
        ]
        return primaries, secondaries

    def test_one_match_per_primary(self) -> None:
        primaries, secondaries = self._build()
        rows = list(
            matchup(
                primary=primaries,
                secondary=secondaries,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
            )
        )
        assert len(rows) == 3
        assert {r.member_ids[0] for r in rows} == {"p_0", "p_1", "p_2"}
        # Each primary should match its same-day "close" secondary.
        pairs = {(r.member_ids[0], r.member_ids[1]) for r in rows}
        assert pairs == {
            ("p_0", "s_close_0"),
            ("p_1", "s_close_1"),
            ("p_2", "s_close_2"),
        }

    def test_strict_iou_drops_partial_overlaps(self) -> None:
        # Primary [-9, -8] x [38, 39] (1x1 deg); secondary
        # [-8.5, -7.5] x [38.5, 39.5] (1x1 deg, 50% overlap).
        # IoU = 0.25 / 1.75 = ~0.143 — passes threshold 0.1, fails 0.5.
        primaries, secondaries = self._build()
        permissive = list(
            matchup(
                primary=primaries[:1],
                secondary=secondaries[:1],
                spatial=IouAtLeast(0.10),
                temporal=NearestInTime(dt="6h"),
            )
        )
        strict = list(
            matchup(
                primary=primaries[:1],
                secondary=secondaries[:1],
                spatial=IouAtLeast(0.50),
                temporal=NearestInTime(dt="6h"),
            )
        )
        assert len(permissive) == 1
        assert len(strict) == 0

    def test_matchup_row_metadata(self) -> None:
        primaries, secondaries = self._build()
        rows = list(
            matchup(
                primary=primaries[:1],
                secondary=secondaries,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
                tag="iberia_test",
            )
        )
        row = rows[0]
        assert isinstance(row, MatchupRow)
        assert row.member_sources == ("modis", "s2")
        assert row.member_roles == ("primary", "secondary")
        assert row.query_set == "iberia_test"
        # Tolerance carries the strategy summary for re-running.
        assert "IouAtLeast" in row.strategy or "Intersects" in row.strategy
        assert "NearestInTime" in row.strategy
        # 2h secondary lag → offsets ≈ [0, 7200].
        assert row.time_offset_sec[0] == pytest.approx(0.0)
        assert row.time_offset_sec[1] == pytest.approx(7200.0, abs=1.0)

    def test_geometry_intersect_is_common_footprint(self) -> None:
        # Primary 1x1; secondary 1x1 shifted by (0.5, 0.5).
        # Intersection is 0.5x0.5 = 0.25.
        primaries, secondaries = self._build()
        rows = list(
            matchup(
                primary=primaries[:1],
                secondary=secondaries[:1],
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
            )
        )
        assert rows[0].geometry_intersect.area == pytest.approx(0.25, abs=1e-9)

    def test_empty_when_no_temporal_match(self) -> None:
        primaries, _ = self._build()
        far_secondaries = [
            _row(
                "far",
                source="s2",
                bbox=(-9, 38, -8, 39),
                time=datetime(2025, 1, 1, tzinfo=UTC),
            )
        ]
        rows = list(
            matchup(
                primary=primaries,
                secondary=far_secondaries,
                spatial=Intersects(),
                temporal=NearestInTime(dt="1h"),
            )
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Matchup engine — N-way
# ---------------------------------------------------------------------------


class TestMatchupNWay:
    def _build(self) -> tuple[list[SourceRow], dict[str, list[SourceRow]]]:
        primary = [
            _row(
                "p",
                source="modis",
                bbox=(-9, 38, -8, 39),
                time=datetime(2024, 6, 15, 12, tzinfo=UTC),
            )
        ]
        s2 = [
            _row(
                "s2_a",
                source="s2",
                bbox=(-8.5, 38.5, -7.5, 39.5),
                time=datetime(2024, 6, 15, 13, tzinfo=UTC),
            )
        ]
        landsat = [
            _row(
                "landsat_a",
                source="landsat",
                bbox=(-9, 38, -8, 39),
                time=datetime(2024, 6, 15, 11, tzinfo=UTC),
            )
        ]
        return primary, {"s2": s2, "landsat": landsat}

    def test_three_member_row_emitted(self) -> None:
        primary, secs = self._build()
        rows = list(
            matchup(
                primary=primary,
                secondary=secs,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
            )
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.member_ids == ("p", "s2_a", "landsat_a")
        assert row.member_roles == ("primary", "s2", "landsat")
        assert row.member_sources == ("modis", "s2", "landsat")

    def test_all_join_drops_when_role_empty(self) -> None:
        # `landsat` has no temporal candidates → "all" drops the primary.
        primary, secs = self._build()
        secs["landsat"][0] = _row(
            "landsat_far",
            source="landsat",
            bbox=(-9, 38, -8, 39),
            time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        rows = list(
            matchup(
                primary=primary,
                secondary=secs,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
                join="all",
            )
        )
        assert rows == []

    def test_any_join_emits_partial(self) -> None:
        primary, secs = self._build()
        secs["landsat"][0] = _row(
            "landsat_far",
            source="landsat",
            bbox=(-9, 38, -8, 39),
            time=datetime(2025, 1, 1, tzinfo=UTC),
        )
        rows = list(
            matchup(
                primary=primary,
                secondary=secs,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
                join="any",
            )
        )
        assert len(rows) == 1
        # Only primary + s2 — landsat was dropped.
        assert rows[0].member_ids == ("p", "s2_a")
        assert rows[0].member_roles == ("primary", "s2")


# ---------------------------------------------------------------------------
# Position-preserving temporal filter (regression for the P1 review)
# ---------------------------------------------------------------------------


class TestDuplicateTimestampPositions:
    """`temporal.filter` returns a subset in input position order; the
    engine must preserve **both multiplicity and position** when
    mapping back to rows. A set-based key lookup would silently
    return all rows with the chosen timestamp, breaking selector
    strategies like `NearestInTime`.
    """

    def test_nearest_in_time_picks_single_row_among_duplicates(self) -> None:
        # Two secondaries with identical timestamps (same satellite,
        # different tiles). NearestInTime should pick exactly one;
        # the engine's position-preserving recovery should yield
        # exactly one matchup row, not two.
        primary = [
            _row(
                "p",
                source="modis",
                bbox=(-9, 38, -8, 39),
                time=datetime(2024, 6, 15, 12, tzinfo=UTC),
            )
        ]
        secondaries = [
            _row(
                "s2_a",
                source="s2",
                bbox=(-9, 38, -8, 39),
                time=datetime(2024, 6, 15, 14, tzinfo=UTC),
            ),
            _row(
                "s2_b",
                source="s2",
                bbox=(-9, 38, -8, 39),
                # Same timestamp as s2_a.
                time=datetime(2024, 6, 15, 14, tzinfo=UTC),
            ),
        ]
        rows = list(
            matchup(
                primary=primary,
                secondary=secondaries,
                spatial=Intersects(),
                temporal=NearestInTime(dt="6h"),
            )
        )
        # NearestInTime returns one position; the engine must
        # respect that (not blow up to two rows because the
        # timestamps happen to be equal).
        assert len(rows) == 1
        # Order-preservation: the first matching row is the one
        # picked (consistent with NearestInTime's tie-break on
        # first occurrence).
        assert rows[0].member_ids == ("p", "s2_a")


# ---------------------------------------------------------------------------
# join="any" empty-members guard
# ---------------------------------------------------------------------------


class TestAnyJoinEmptyGuard:
    """When every secondary role misses under `join="any"`, the
    engine must not emit a primary-only "matchup" — a row with a
    single member has no joinable content.
    """

    def test_any_join_skips_when_all_roles_miss(self) -> None:
        primary = [
            _row(
                "p",
                source="modis",
                bbox=(-9, 38, -8, 39),
                time=datetime(2024, 6, 15, 12, tzinfo=UTC),
            )
        ]
        # Far in time → NearestInTime misses; the empty-role
        # branch under join="any" must not let a primary-only
        # row through.
        far_secondaries = [
            _row(
                "s2_far",
                source="s2",
                bbox=(-9, 38, -8, 39),
                time=datetime(2025, 6, 15, 12, tzinfo=UTC),
            )
        ]
        rows = list(
            matchup(
                primary=primary,
                secondary=far_secondaries,
                spatial=Intersects(),
                temporal=NearestInTime(dt="1h"),
                join="any",
            )
        )
        assert rows == []


# ---------------------------------------------------------------------------
# STRtree pre-filter behaviour
# ---------------------------------------------------------------------------


class TestSpatialIndexPrefilter:
    def test_far_away_candidates_skipped(self) -> None:
        # 100 secondaries in a non-overlapping bbox; pre-filter
        # should skip all of them without ever calling the truth
        # gate. Indirect test: the matchup returns nothing in
        # under reasonable time.
        primaries = [
            _row(
                "p",
                bbox=(0, 0, 1, 1),
                time=datetime(2024, 6, 15, 12, tzinfo=UTC),
            )
        ]
        secondaries = [
            _row(
                f"s_{i}",
                bbox=(100 + i * 0.01, 100, 100 + (i + 1) * 0.01, 101),
                time=datetime(2024, 6, 15, 12, tzinfo=UTC),
            )
            for i in range(100)
        ]
        rows = list(
            matchup(
                primary=primaries,
                secondary=secondaries,
                spatial=Intersects(),
                temporal=Synchronous(tolerance="1d"),
            )
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReexports:
    def test_matchup_namespace(self) -> None:
        assert matchup_ns.matchup is matchup
        assert matchup_ns.IouAtLeast is IouAtLeast
        assert matchup_ns.NearestInTime is NearestInTime

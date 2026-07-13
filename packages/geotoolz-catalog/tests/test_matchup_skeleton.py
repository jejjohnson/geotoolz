"""Smoke tests for the scaffolded `geocatalog.matchup` surface."""

from __future__ import annotations

from datetime import datetime, timedelta

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
    SpatialStrategy,
    Synchronous,
    TemporalStrategy,
    WithinWindow,
    matchup,
)


class TestReexports:
    def test_subnamespace_reexports(self) -> None:
        assert matchup_ns.matchup is matchup
        assert matchup_ns.MatchupRow is MatchupRow
        assert matchup_ns.IouAtLeast is IouAtLeast
        assert matchup_ns.NearestInTime is NearestInTime


class TestSpatialStrategies:
    def test_iou_threshold_bounds(self) -> None:
        with pytest.raises(ValueError):
            IouAtLeast(threshold=-0.1)
        with pytest.raises(ValueError):
            IouAtLeast(threshold=1.1)
        IouAtLeast(threshold=0.0)
        IouAtLeast(threshold=1.0)

    def test_is_runtime_protocol(self) -> None:
        # Every concrete strategy duck-types `SpatialStrategy`.
        for strat in [Intersects(), IouAtLeast(0.2), Contains()]:
            assert isinstance(strat, SpatialStrategy)

    def test_strategy_match_returns_bool(self) -> None:
        # Bodies are implemented; behaviour coverage lives in
        # tests/test_matchup_engine.py. Skeleton just locks the
        # bool-return contract.
        b = box(0, 0, 1, 1)
        for strat in [Intersects(), IouAtLeast(0.2), CentroidWithin(0.0), Contains()]:
            assert isinstance(strat.match(b, b), bool)


class TestTemporalStrategies:
    def test_is_runtime_protocol(self) -> None:
        for strat in [
            NearestInTime(dt=timedelta(hours=6)),
            WithinWindow(start="-6h", end="6h"),
            Synchronous(),
        ]:
            assert isinstance(strat, TemporalStrategy)

    def test_default_synchronous_tolerance(self) -> None:
        # Default tolerance is "0s" — strict overlap.
        assert Synchronous().tolerance == "0s"


class TestMatchupRow:
    def test_construction(self) -> None:
        mr = MatchupRow(
            matchup_id="abc",
            strategy="iou>=0.2 & nearest_in_time<=6h",
            member_ids=("a", "b"),
            member_sources=("earthaccess", "stac.pc"),
            member_roles=("primary", "secondary"),
            geometry_intersect=box(0, 0, 1, 1),
            time_reference=datetime(2024, 6, 1),
            time_offset_sec=(0.0, 600.0),
        )
        assert mr.matchup_id == "abc"
        assert len(mr.member_ids) == 2
        assert mr.tolerance == {}


class TestMatchupEngine:
    def test_signature_accepts_iterables(self) -> None:
        # `matchup` now takes iterables of SourceRow (not a Selector
        # against a GeoCatalog — that wrapper lands once ingest is
        # wired up). Smoke-check the signature accepts the documented
        # kwargs without raising; behaviour coverage lives in
        # tests/test_matchup_engine.py.
        result = list(
            matchup(
                primary=[],
                secondary=[],
                spatial=IouAtLeast(0.2),
                temporal=NearestInTime(dt="6h"),
            )
        )
        assert result == []

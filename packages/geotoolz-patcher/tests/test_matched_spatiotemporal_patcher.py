"""End-to-end tests for `MatchedField` + `MatchedSpatioTemporalPatcher`.

Exercises split / merge in both ``"product"`` and ``"coupled"``
coupling modes against a stub primary `SpatialPatcher` driven by a
`MatchedField`. The temporal slicing logic in the matched-spatio-
temporal patcher mirrors `SpatioTemporalPatcher`; these tests pin
down per-source lockstep slicing without depending on georeader.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from _helpers import StubDomain as _StubDomain

from geopatcher._src.matched import (
    MatchedField,
    MatchedSpatioTemporalPatch,
    MatchedSpatioTemporalPatcher,
)
from geopatcher._src.matched.patch import PRIMARY_KEY
from geopatcher._src.patch import Patch, SpatioTemporalPatch
from geopatcher._src.spatial_time import SpatioTemporalPatcher
from geopatcher._src.time.aggregation import TemporalAggregation, TemporalMean
from geopatcher._src.time.geometry import TemporalFixedLookback
from geopatcher._src.time.patcher import TemporalPatcher
from geopatcher._src.time.sampler import TemporalRegularStride
from geopatcher._src.time.window import TemporalCausalBoxcar


# ---------------------------------------------------------------------------
# Stub Field / Domain / SpatialPatcher minimal enough to drive the matched
# spatio-temporal patcher without touching georeader.
# ---------------------------------------------------------------------------


class _ArrField:
    """A `Field` whose `select(indexer)` returns a backing 3-D numpy chunk.

    Deliberately NOT the shared `_helpers.ArrField`: ``select`` here
    returns the array unchanged for *any* indexer (including slices) so
    the matched spatial patcher's per-anchor reads see the full time
    series at each spatial chip.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self._domain = _StubDomain()

    @property
    def domain(self) -> Any:
        return self._domain

    def select(self, indexer: Any) -> Any:
        # Return the full time series for whatever spatial indexer; the
        # matched patcher relies on this shape for product-mode slicing.
        return self._values

    def with_data(self, array: Any) -> Any:
        return array


class _StubSpatialSampler:
    """Sampler with a fixed `anchors_` list (used in coupled mode too)."""

    def __init__(self, anchors_: list[Any]) -> None:
        self.anchors_ = anchors_

    def anchors(self, *args: Any, **kwargs: Any) -> list[Any]:
        return list(self.anchors_)


class _StubSpatialGeometry:
    def neighborhood(self, domain: Any, anchor: Any) -> Any:
        return f"nbhd[{anchor}]"


class _StubSpatialWindow:
    def weights(self, geometry: Any) -> None:
        return None


class _StubSpatialPatcher:
    """Stand-in for `SpatialPatcher` that yields one `Patch` per anchor.

    `Patch.data` is taken straight from `field.select(indexer)`, so when
    `field` is a `MatchedField`, the data is the dict the matched
    patcher unpacks.
    """

    def __init__(self, anchors: list[Any]) -> None:
        self.sampler = _StubSpatialSampler(anchors)
        self.geometry = _StubSpatialGeometry()
        self.window = _StubSpatialWindow()

    def split(self, field: Any) -> Iterator[Patch]:
        for anchor in self.sampler.anchors_:
            indexer = f"nbhd[{anchor}]"
            data = field.select(indexer)
            yield Patch(data=data, anchor=anchor, indices=indexer, weights=None)


def _make_temporal(
    aggregation: TemporalAggregation | None = None,
) -> TemporalPatcher:
    return TemporalPatcher(
        geometry=TemporalFixedLookback(length=2),
        sampler=TemporalRegularStride(step=2),
        window=TemporalCausalBoxcar(),
        aggregation=aggregation if aggregation is not None else TemporalMean(),
    )


class _RecordingTemporalAgg(TemporalAggregation):
    """TemporalAggregation stub — returns ``("merged", name, n)``."""

    streaming_safe = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[int] = []

    def merge(self, patches: Any) -> Any:
        materialised = list(patches)
        self.calls.append(len(materialised))
        return ("merged", self.name, len(materialised))


# ---------------------------------------------------------------------------
# Product coupling
# ---------------------------------------------------------------------------


class TestProductCoupling:
    def _build(
        self,
        *,
        with_secondary: bool = True,
    ) -> tuple[MatchedSpatioTemporalPatcher, MatchedField]:
        primary_arr = np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2)
        secondary_arr = primary_arr * 10.0
        secondaries: dict[str, _ArrField] = {}
        coreg: dict[str, Any] = {}
        if with_secondary:
            secondaries["s2"] = _ArrField(secondary_arr)
            coreg["s2"] = lambda raw, prim: raw
        mf = MatchedField(
            primary=_ArrField(primary_arr),
            secondaries=secondaries,
            coreg=coreg,
        )
        spatial_stub = _StubSpatialPatcher(anchors=[(0, 0), (0, 1)])
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="product",
            time_axis=0,
        )
        mstp = MatchedSpatioTemporalPatcher(primary=primary)
        return mstp, mf

    def test_yields_matched_spatiotemporal_patches(self) -> None:
        mstp, mf = self._build()
        patches = list(mstp.split(mf))
        # 2 spatial anchors x 2 temporal anchors (time_len=4, stride=2) = 4
        assert len(patches) == 4
        for mp in patches:
            assert isinstance(mp, MatchedSpatioTemporalPatch)

    def test_members_keyed_by_source(self) -> None:
        mstp, mf = self._build()
        first = next(iter(mstp.split(mf)))
        assert set(first.members) == {PRIMARY_KEY, "s2"}
        assert isinstance(first.members[PRIMARY_KEY], SpatioTemporalPatch)

    def test_secondary_sliced_in_lockstep(self) -> None:
        mstp, mf = self._build()
        for mp in mstp.split(mf):
            prim = np.asarray(mp.members[PRIMARY_KEY].data)
            sec = np.asarray(mp.members["s2"].data)
            np.testing.assert_allclose(sec, prim * 10.0)

    def test_inner_patches_share_spatial_and_temporal_anchors(self) -> None:
        mstp, mf = self._build()
        first = next(iter(mstp.split(mf)))
        for inner in first.members.values():
            assert inner.space == first.space
            assert inner.time == first.time

    def test_primary_only_field(self) -> None:
        mstp, mf = self._build(with_secondary=False)
        first = next(iter(mstp.split(mf)))
        assert set(first.members) == {PRIMARY_KEY}

    def test_non_dict_data_raises(self) -> None:
        plain = _ArrField(np.zeros((4, 2, 2)))
        spatial_stub = _StubSpatialPatcher(anchors=[(0, 0)])
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="product",
        )
        mstp = MatchedSpatioTemporalPatcher(primary=primary)
        with pytest.raises(TypeError, match=r"dict.*MatchedField"):
            list(mstp.split(plain))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Coupled coupling
# ---------------------------------------------------------------------------


class TestCoupledCoupling:
    def _build(self) -> tuple[MatchedSpatioTemporalPatcher, MatchedField]:
        primary_arr = np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2)
        secondary_arr = primary_arr + 1.0
        mf = MatchedField(
            primary=_ArrField(primary_arr),
            secondaries={"s2": _ArrField(secondary_arr)},
            coreg={"s2": lambda raw, prim: raw},
        )
        # Coupled mode reads sampler.anchors_ as (space, time) pairs.
        spatial_stub = _StubSpatialPatcher(anchors=[((0, 0), 1), ((0, 1), 2)])
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="coupled",
            time_axis=0,
        )
        mstp = MatchedSpatioTemporalPatcher(primary=primary)
        return mstp, mf

    def test_yields_one_patch_per_pair(self) -> None:
        mstp, mf = self._build()
        patches = list(mstp.split(mf))
        assert len(patches) == 2
        for mp in patches:
            assert isinstance(mp, MatchedSpatioTemporalPatch)

    def test_pairs_pin_space_and_time(self) -> None:
        mstp, mf = self._build()
        patches = list(mstp.split(mf))
        spaces = [(p.space, p.time) for p in patches]
        assert spaces == [((0, 0), 1), ((0, 1), 2)]

    def test_coupled_requires_anchors_attribute(self) -> None:
        # A sampler missing `anchors_` triggers the documented TypeError.
        class _NoAnchors:
            def anchors(self) -> list[Any]:
                return []

        class _BadSpatial:
            def __init__(self) -> None:
                self.sampler = _NoAnchors()
                self.geometry = _StubSpatialGeometry()
                self.window = _StubSpatialWindow()

            def split(self, field: Any) -> Iterator[Patch]:
                yield from ()

        mf = MatchedField(primary=_ArrField(np.zeros((4, 2, 2))))
        primary = SpatioTemporalPatcher(
            spatial=_BadSpatial(),  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="coupled",
        )
        mstp = MatchedSpatioTemporalPatcher(primary=primary)
        with pytest.raises(TypeError, match="coupled coupling requires"):
            list(mstp.split(mf))


# ---------------------------------------------------------------------------
# Merge — per-source dispatch
# ---------------------------------------------------------------------------


class TestMatchedSpatioTemporalPatcherMerge:
    def _build(
        self, *, with_secondary_agg: bool = True
    ) -> tuple[
        MatchedSpatioTemporalPatcher,
        MatchedField,
        _RecordingTemporalAgg,
        _RecordingTemporalAgg | None,
    ]:
        primary_arr = np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2)
        mf = MatchedField(
            primary=_ArrField(primary_arr),
            secondaries={"s2": _ArrField(primary_arr * 2)},
            coreg={"s2": lambda raw, prim: raw},
        )
        spatial_stub = _StubSpatialPatcher(anchors=[(0, 0), (0, 1)])
        primary_agg = _RecordingTemporalAgg("primary_agg")
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(aggregation=primary_agg),
            coupling="product",
            time_axis=0,
        )
        secondary_aggregators: dict[str, TemporalAggregation] = {}
        secondary_agg = _RecordingTemporalAgg("s2_agg") if with_secondary_agg else None
        if secondary_agg is not None:
            secondary_aggregators["s2"] = secondary_agg
        mstp = MatchedSpatioTemporalPatcher(
            primary=primary,
            secondary_aggregators=secondary_aggregators,
        )
        return mstp, mf, primary_agg, secondary_agg

    def test_merge_returns_dict_keyed_by_source(self) -> None:
        mstp, mf, _, _ = self._build()
        patches = list(mstp.split(mf))
        out = mstp.merge(patches, mf)
        assert set(out) == {PRIMARY_KEY, "s2"}

    def test_merge_results_are_anchor_lists(self) -> None:
        # Mirrors SpatioTemporalPatcher.merge — each source's value is
        # [(spatial_anchor, temporal_merge), …].
        mstp, mf, _, _ = self._build()
        patches = list(mstp.split(mf))
        out = mstp.merge(patches, mf)
        for _name, by_anchor in out.items():
            assert isinstance(by_anchor, list)
            # 2 spatial anchors → 2 entries per source
            assert len(by_anchor) == 2
            for anchor, _ in by_anchor:
                assert anchor in {(0, 0), (0, 1)}

    def test_merge_skips_secondary_without_aggregator(self) -> None:
        mstp, mf, _, _ = self._build(with_secondary_agg=False)
        out = mstp.merge(list(mstp.split(mf)), mf)
        assert set(out) == {PRIMARY_KEY}

    def test_merge_dispatches_to_secondary_aggregator(self) -> None:
        mstp, mf, primary_agg, secondary_agg = self._build()
        patches = list(mstp.split(mf))
        mstp.merge(patches, mf)
        assert secondary_agg is not None
        # Secondary's agg is invoked once per spatial anchor group;
        # 2 spatial anchors → 2 invocations, each over the per-anchor list.
        assert len(secondary_agg.calls) == 2
        # The primary's temporal aggregator is invoked the same number
        # of times since it shares the spatial grouping shape.
        assert len(primary_agg.calls) == 2


# ---------------------------------------------------------------------------
# Typo guard / construction
# ---------------------------------------------------------------------------


class TestUnknownAggregatorNamesRejected:
    def _build(
        self, *, agg_name: str
    ) -> tuple[MatchedSpatioTemporalPatcher, MatchedField]:
        mf = MatchedField(
            primary=_ArrField(np.zeros((4, 2, 2))),
            secondaries={"s2": _ArrField(np.zeros((4, 2, 2)))},
            coreg={"s2": lambda raw, prim: raw},
        )
        spatial_stub = _StubSpatialPatcher(anchors=[(0, 0)])
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="product",
        )
        mstp = MatchedSpatioTemporalPatcher(
            primary=primary,
            secondary_aggregators={agg_name: _RecordingTemporalAgg("typo")},
        )
        return mstp, mf

    def test_split_rejects_unknown_aggregator_name(self) -> None:
        mstp, mf = self._build(agg_name="s22")
        with pytest.raises(ValueError, match=r"not in mfield\.secondaries"):
            list(mstp.split(mf))

    def test_merge_rejects_unknown_aggregator_name(self) -> None:
        mstp, mf = self._build(agg_name="s22")
        with pytest.raises(ValueError, match=r"not in mfield\.secondaries"):
            mstp.merge([], mf)


class TestConstructionAndCoupling:
    def test_default_secondary_aggregators_empty(self) -> None:
        spatial_stub = _StubSpatialPatcher(anchors=[(0, 0)])
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="product",
        )
        mstp = MatchedSpatioTemporalPatcher(primary=primary)
        assert mstp.secondary_aggregators == {}

    def test_module_namespace_exposes_class(self) -> None:
        import geopatcher.matched as matched_ns

        assert matched_ns.MatchedSpatioTemporalPatcher is MatchedSpatioTemporalPatcher
        assert matched_ns.MatchedSpatioTemporalPatch is MatchedSpatioTemporalPatch

    def test_unknown_coupling_raises(self) -> None:
        # The matched patcher inherits coupling from primary; an invalid
        # value surfaces the same error the primary would.
        spatial_stub = _StubSpatialPatcher(anchors=[(0, 0)])
        primary = SpatioTemporalPatcher(
            spatial=spatial_stub,  # type: ignore[arg-type]
            temporal=_make_temporal(),
            coupling="product",
        )
        primary.coupling = "bogus"  # type: ignore[assignment]
        mstp = MatchedSpatioTemporalPatcher(primary=primary)
        mf = MatchedField(primary=_ArrField(np.zeros((4, 2, 2))))
        with pytest.raises(ValueError, match="unknown coupling"):
            list(mstp.split(mf))

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

from geopatcher import spatial, temporal
from geopatcher._src.matched import (
    MatchedField,
    MatchedSpatioTemporalPatch,
    MatchedSpatioTemporalPatcher,
)
from geopatcher._src.matched.patch import PRIMARY_KEY
from geopatcher._src.patch import Patch, SpatioTemporalPatch
from geopatcher._src.spatial_time import SpatioTemporalPatcher
from geopatcher._src.temporal.patcher import TemporalPatcher


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
    aggregation: temporal.aggregation.Aggregation | None = None,
) -> TemporalPatcher:
    # "shrink" keeps anchor 0's short [0, 1) window: two temporal anchors.
    return TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=2, boundary="shrink"),
        sampler=temporal.sampler.RegularStride(step=2),
        window=temporal.window.CausalBoxcar(),
        aggregation=aggregation
        if aggregation is not None
        else temporal.aggregation.Mean(),
    )


class _RecordingTemporalAgg(temporal.aggregation.Aggregation):
    """temporal.aggregation.Aggregation stub — returns ``("merged", name, n)``."""

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
        secondary_aggregators: dict[str, temporal.aggregation.Aggregation] = {}
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


# ---------------------------------------------------------------------------
# Parity with SpatioTemporalPatcher (#203): coord-aware geometries,
# coord_value / member-summed hook payloads, cadence check, prefetch
# ---------------------------------------------------------------------------


def _time_raster(n_time: int, scale: float = 1.0) -> Any:
    import rasterio
    from georeader.geotensor import GeoTensor

    from geopatcher._src.fields.raster import RasterField

    arr = np.arange(n_time * 16 * 16, dtype=np.float32).reshape(n_time, 16, 16)
    return RasterField(
        GeoTensor(
            values=arr * scale,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
            fill_value_default=np.nan,
        )
    )


def _stencil_patcher(coupling: str = "product", anchors_: Any = None) -> Any:
    from geopatcher import SpatialPatcher
    from geopatcher.temporal.stencils import TimeStencil

    stencil = TimeStencil("-3h", "3h", "3h", closed="both")
    sampler: Any = (
        spatial.sampler.RegularStride(step=8)
        if anchors_ is None
        else spatial.sampler.Explicit(anchors_=anchors_)
    )
    return SpatioTemporalPatcher(
        spatial=SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(8, 8)),
            sampler=sampler,
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        ),
        temporal=TemporalPatcher(
            geometry=temporal.geometry.StencilGeometry(
                stencil=stencil, source_step=np.timedelta64(3, "h")
            ),
            sampler=temporal.sampler.StencilSampler(stencil=stencil),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Forecast(horizon=1),
        ),
        coupling=coupling,  # type: ignore[arg-type]
    )


def _coord() -> np.ndarray:
    return np.arange("2020-01-01T00", "2020-01-02T00", 3, dtype="datetime64[h]").astype(
        "datetime64[ns]"
    )


class _CoordHook:
    def __init__(self) -> None:
        self.done: list[tuple[Any, int, Any]] = []

    def on_patch_done(
        self, anchor: Any, elapsed: float, bytes_: int, coord_value: Any = None
    ) -> None:
        self.done.append((anchor, bytes_, coord_value))


@pytest.mark.parametrize("coupling", ["product", "coupled"])
def test_stencil_geometry_matches_single_source(coupling: str) -> None:
    anchors_ = None if coupling == "product" else [((0, 0), 3), ((8, 8), 5)]
    stp = _stencil_patcher(coupling, anchors_)
    mf = MatchedField(
        primary=_time_raster(8),
        secondaries={"s": _time_raster(8, scale=2.0)},
        coreg={"s": lambda raw, prim: raw},
    )
    coord = _coord()
    msp = MatchedSpatioTemporalPatcher(primary=stp)
    hook = _CoordHook()
    matched = list(msp.split(mf, hooks=[hook], coord=coord, prefetch=2))
    single = list(stp.split(mf.primary, coord=coord))
    assert len(matched) == len(single) > 0
    for mp, ref, (anchor, nbytes, coord_value) in zip(
        matched, single, hook.done, strict=True
    ):
        assert (mp.space, mp.time) == (ref.space, ref.time) == anchor
        prim = np.asarray(mp.members[PRIMARY_KEY].data)
        np.testing.assert_array_equal(prim, np.asarray(ref.data))
        np.testing.assert_array_equal(np.asarray(mp.members["s"].data), prim * 2)
        assert nbytes == 2 * prim.nbytes
        assert coord_value == coord[mp.time]


def test_spatiotemporal_cadence_mismatch_raises() -> None:
    stp = _stencil_patcher()
    stp.temporal = TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=2),
        sampler=temporal.sampler.RegularStride(step=2),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )
    mf = MatchedField(
        primary=_time_raster(8),
        secondaries={"s": _time_raster(4)},
        coreg={"s": lambda raw, prim: raw},
    )
    with pytest.raises(ValueError, match=r"'s' has 4 steps.*primary has 8"):
        list(MatchedSpatioTemporalPatcher(primary=stp).split(mf))


# ---------------------------------------------------------------------------
# The fork is gone (#203 / #175): the matched patcher drives the primary
# SpatioTemporalPatcher, so its read pipeline and runner knobs apply.
# ---------------------------------------------------------------------------


class _FlakyRaster:
    """A raster field whose reads fail at the given window origins."""

    def __init__(self, inner: Any, bad: set[tuple[int, int]]) -> None:
        self.inner = inner
        self.bad = bad
        self.reads: list[tuple[int, int]] = []

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, window: Any) -> Any:
        origin = (int(window.row_off), int(window.col_off))
        self.reads.append(origin)
        if origin in self.bad:
            raise OSError(f"tile {origin} unreadable")
        return self.inner.select(window)


def _real_stp(coupling: str = "product", anchors_: Any = None, **kw: Any) -> Any:
    from geopatcher import SpatialPatcher

    geometry = kw.pop("geometry", spatial.geometry.Rectangular(size=(8, 8)))
    sampler: Any = (
        spatial.sampler.RegularStride(step=8)
        if anchors_ is None
        else spatial.sampler.Explicit(anchors_=anchors_)
    )
    return SpatioTemporalPatcher(
        spatial=SpatialPatcher(
            geometry=geometry,
            sampler=sampler,
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
            **kw,
        ),
        temporal=TemporalPatcher(
            geometry=temporal.geometry.FixedLookback(length=2),
            sampler=temporal.sampler.RegularStride(step=2),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(),
        ),
        coupling=coupling,  # type: ignore[arg-type]
    )


def _flaky_mfield(bad: set[tuple[int, int]]) -> Any:
    return MatchedField(
        primary=_FlakyRaster(_time_raster(4), bad),
        secondaries={"s": _time_raster(4, scale=2.0)},
        coreg={"s": lambda raw, prim: raw},
    )


class _Hook:
    def __init__(self) -> None:
        self.errors: list[tuple[Any, Exception]] = []
        self.skipped: list[Any] = []
        self.done: list[Any] = []

    def on_error(self, anchor: Any, exc: Exception) -> None:
        self.errors.append((anchor, exc))

    def on_patch_skipped(self, anchor: Any) -> None:
        self.skipped.append(anchor)

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        self.done.append(anchor)


@pytest.mark.parametrize("coupling", ["product", "coupled"])
def test_on_error_skip_drops_failed_chips(coupling: str) -> None:
    anchors_ = None if coupling == "product" else [((0, 0), 3), ((8, 8), 3)]
    stp = _real_stp(coupling, anchors_, on_error="skip")
    hook = _Hook()

    matched = list(
        MatchedSpatioTemporalPatcher(primary=stp).split(
            _flaky_mfield({(8, 8)}), hooks=[hook]
        )
    )

    assert matched
    assert all(mp.space != (8, 8) for mp in matched)
    key = (8, 8) if coupling == "product" else ((8, 8), 3)
    assert [e.anchor for e in stp.spatial.errors] == [key]
    assert [(a, type(e)) for a, e in hook.errors] == [(key, OSError)]


@pytest.mark.parametrize("coupling", ["product", "coupled"])
def test_on_error_mask_yields_invalid_members(coupling: str) -> None:
    anchors_ = None if coupling == "product" else [((8, 8), 3)]
    stp = _real_stp(coupling, anchors_, on_error="mask")

    matched = [
        mp
        for mp in MatchedSpatioTemporalPatcher(primary=stp).split(
            _flaky_mfield({(8, 8)})
        )
        if mp.space == (8, 8)
    ]

    assert matched
    for mp in matched:
        assert set(mp.members) == {PRIMARY_KEY, "s"}
        assert mp.valid_mask is not None
        for name, member in mp.members.items():
            assert np.isnan(np.asarray(member.data)).all()
            assert not mp.valid_mask[name].any()


def test_journal_and_backpressure_reach_the_primary_walk() -> None:
    stp = _real_stp()
    msp = MatchedSpatioTemporalPatcher(primary=stp)
    mf = _flaky_mfield(set())

    class _Journal:
        def has(self, key: Any) -> bool:
            return key == ((0, 8), 2)

    hook = _Hook()
    matched = list(msp.split(mf, hooks=[hook], journal=_Journal(), max_in_flight=8))
    assert ((0, 8), 2) not in [(mp.space, mp.time) for mp in matched]
    assert hook.skipped == [((0, 8), 2)]
    assert all(mp._release is not None for mp in matched)
    for mp in matched:
        mp.close()
    with pytest.raises(ValueError, match="max_in_flight_bytes"):
        list(msp.split(mf, max_in_flight_bytes=1))


def test_split_cache_is_keyed_per_source(tmp_path: Any) -> None:
    from geopatcher._src.cache import PatchCache

    mf = _flaky_mfield(set())
    msp = MatchedSpatioTemporalPatcher(primary=_real_stp())
    cache = PatchCache(tmp_path / "cache", field_id="scene")
    first = list(msp.split(mf, cache=cache))
    mf.primary.reads.clear()
    second = list(msp.split(mf, cache=cache))
    assert mf.primary.reads == []
    for a, b in zip(first, second, strict=True):
        for name in (PRIMARY_KEY, "s"):
            np.testing.assert_array_equal(
                np.asarray(a.members[name].data), np.asarray(b.members[name].data)
            )


def test_coupled_chip_goes_through_the_spatial_pipeline() -> None:

    stp = _real_stp(
        "coupled",
        [((-4, -4), 3)],
        geometry=spatial.geometry.Rectangular(size=(8, 8), boundary="reflect"),
    )
    mf = MatchedField(
        primary=_time_raster(4),
        secondaries={"s": _time_raster(4, scale=2.0)},
        coreg={"s": lambda raw, prim: raw},
    )
    (mp,) = list(MatchedSpatioTemporalPatcher(primary=stp).split(mf))
    (ref,) = list(stp.split(mf.primary))
    np.testing.assert_array_equal(
        np.asarray(mp.members[PRIMARY_KEY].data), np.asarray(ref.data)
    )
    np.testing.assert_array_equal(
        np.asarray(mp.members["s"].data), 2 * np.asarray(ref.data)
    )

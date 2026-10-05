"""End-to-end tests for `MatchedField` + `MatchedTemporalPatcher`.

Exercises the split / merge pipeline against a stub primary
`TemporalPatcher` driven by a `MatchedField` whose ``select`` returns
a per-source dict of full-length numpy arrays. Mirrors the
`tests/test_matched_e2e.py` shape against the temporal axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _helpers import ArrField as _ArrField

from geopatcher._src.matched import (
    MatchedField,
    MatchedSpatioTemporalPatch,  # noqa: F401 — module-level export sanity
    MatchedTemporalPatch,
    MatchedTemporalPatcher,
)
from geopatcher._src.matched.patch import PRIMARY_KEY
from geopatcher._src.patch import TemporalPatch
from geopatcher._src.time.aggregation import TemporalAggregation, TemporalMean
from geopatcher._src.time.geometry import TemporalFixedLookback
from geopatcher._src.time.patcher import TemporalPatcher
from geopatcher._src.time.sampler import TemporalRegularStride
from geopatcher._src.time.window import TemporalCausalBoxcar


# ---------------------------------------------------------------------------
# Stub Field / Domain live in tests/_helpers.py. Each
# `_ArrField.select(slice(None))` returns the full underlying numpy
# series so the matched-temporal patcher can drive the primary
# `TemporalPatcher` on it.
# ---------------------------------------------------------------------------


class _RecordingTemporalAgg(TemporalAggregation):
    """TemporalAggregation stub — returns ``("merged", name, n_patches)``."""

    streaming_safe = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[int] = []

    def merge(self, patches: Any) -> Any:
        materialised = list(patches)
        self.calls.append(len(materialised))
        return ("merged", self.name, len(materialised))


def _make_patcher(
    aggregation: TemporalAggregation | None = None,
) -> TemporalPatcher:
    # "shrink" keeps anchor 0's short [0, 1) window, so every stride-10
    # anchor yields a patch and the counts below stay one per anchor.
    return TemporalPatcher(
        geometry=TemporalFixedLookback(length=5, boundary="shrink"),
        sampler=TemporalRegularStride(step=10),
        window=TemporalCausalBoxcar(),
        aggregation=aggregation if aggregation is not None else TemporalMean(),
    )


# ---------------------------------------------------------------------------
# MatchedTemporalPatcher.split — drive primary, unpack into MatchedTemporalPatch
# ---------------------------------------------------------------------------


class TestMatchedTemporalPatcherSplit:
    def _build(
        self, *, with_secondary: bool = True
    ) -> tuple[MatchedTemporalPatcher, MatchedField]:
        secondaries: dict[str, _ArrField] = {}
        coreg: dict[str, Any] = {}
        if with_secondary:
            secondaries["s2"] = _ArrField(np.arange(100, dtype=np.float64) * 2)
            coreg["s2"] = lambda raw, prim: raw
        mf = MatchedField(
            primary=_ArrField(np.arange(100, dtype=np.float64)),
            secondaries=secondaries,
            coreg=coreg,
        )
        mtp = MatchedTemporalPatcher(primary=_make_patcher())
        return mtp, mf

    def test_yields_matched_temporal_patches(self) -> None:
        mtp, mf = self._build()
        patches = list(mtp.split(mf))
        assert len(patches) == 10  # 100 time steps / stride 10
        for mp in patches:
            assert isinstance(mp, MatchedTemporalPatch)

    def test_matched_patch_members_keyed_by_source(self) -> None:
        mtp, mf = self._build()
        first = next(iter(mtp.split(mf)))
        assert set(first.members) == {PRIMARY_KEY, "s2"}
        assert isinstance(first.members[PRIMARY_KEY], TemporalPatch)
        assert isinstance(first.members["s2"], TemporalPatch)

    def test_inner_patches_carry_outer_metadata(self) -> None:
        mtp, mf = self._build()
        first = next(iter(mtp.split(mf)))
        for member_patch in first.members.values():
            assert member_patch.anchor == first.anchor
            assert member_patch.indices == first.members[PRIMARY_KEY].indices

    def test_secondary_data_sliced_in_lockstep(self) -> None:
        # secondary values are 2x primary values; per-anchor data
        # should match that relationship.
        mtp, mf = self._build()
        for mp in mtp.split(mf):
            prim = np.asarray(mp.members[PRIMARY_KEY].data)
            sec = np.asarray(mp.members["s2"].data)
            np.testing.assert_array_equal(sec, prim * 2)

    def test_primary_only_field(self) -> None:
        mtp, mf = self._build(with_secondary=False)
        first = next(iter(mtp.split(mf)))
        assert set(first.members) == {PRIMARY_KEY}

    def test_n_anchors_forwards(self) -> None:
        mtp, mf = self._build()
        assert mtp.n_anchors(mf) == 10

    def test_anchors_forwards(self) -> None:
        mtp, mf = self._build()
        anchors = mtp.anchors(mf)
        assert len(anchors) == 10
        assert all(isinstance(a, int) for a in anchors)

    def test_non_dict_select_raises(self) -> None:
        # A plain Field passed where a MatchedField is expected returns
        # a non-dict; we surface that loudly.
        plain = _ArrField(np.arange(20, dtype=np.float64))
        mtp = MatchedTemporalPatcher(primary=_make_patcher())
        with pytest.raises(TypeError, match=r"dict.*MatchedField"):
            list(mtp.split(plain))  # type: ignore[arg-type]

    def test_missing_primary_key_raises(self) -> None:
        class _BrokenMatchedField(MatchedField):
            def select(self, indexer: Any) -> dict[str, Any]:
                return {"only_secondary": np.arange(10, dtype=np.float64)}

        mf = _BrokenMatchedField(primary=_ArrField(np.arange(10, dtype=np.float64)))
        mtp = MatchedTemporalPatcher(primary=_make_patcher())
        with pytest.raises(ValueError, match="must include the primary key"):
            list(mtp.split(mf))


# ---------------------------------------------------------------------------
# MatchedTemporalPatcher.merge — per-source aggregation
# ---------------------------------------------------------------------------


class TestMatchedTemporalPatcherMerge:
    def _build(
        self,
        *,
        with_secondary_agg: bool = True,
    ) -> tuple[
        MatchedTemporalPatcher,
        MatchedField,
        _RecordingTemporalAgg,
        _RecordingTemporalAgg | None,
    ]:
        mf = MatchedField(
            primary=_ArrField(np.arange(100, dtype=np.float64)),
            secondaries={"s2": _ArrField(np.arange(100, dtype=np.float64) * 2)},
            coreg={"s2": lambda raw, prim: raw},
        )
        secondary_aggregators: dict[str, TemporalAggregation] = {}
        secondary_agg = _RecordingTemporalAgg("s2_agg") if with_secondary_agg else None
        if secondary_agg is not None:
            secondary_aggregators["s2"] = secondary_agg
        primary_agg = _RecordingTemporalAgg("primary_agg")
        mtp = MatchedTemporalPatcher(
            primary=_make_patcher(aggregation=primary_agg),
            secondary_aggregators=secondary_aggregators,
        )
        return mtp, mf, primary_agg, secondary_agg

    def test_merge_returns_dict_keyed_by_source(self) -> None:
        mtp, mf, _, _ = self._build()
        patches = list(mtp.split(mf))
        out = mtp.merge(patches, mf)
        assert set(out) == {PRIMARY_KEY, "s2"}

    def test_merge_dispatches_to_per_source_aggregators(self) -> None:
        mtp, mf, primary_agg, secondary_agg = self._build()
        patches = list(mtp.split(mf))
        mtp.merge(patches, mf)
        assert primary_agg.calls == [len(patches)]
        assert secondary_agg is not None
        assert secondary_agg.calls == [len(patches)]

    def test_merge_skips_secondary_without_aggregator(self) -> None:
        mtp, mf, _, _ = self._build(with_secondary_agg=False)
        patches = list(mtp.split(mf))
        out = mtp.merge(patches, mf)
        assert set(out) == {PRIMARY_KEY}

    def test_merge_consumes_iterator_once(self) -> None:
        mtp, mf, _, secondary_agg = self._build()
        patches_gen = mtp.split(mf)
        out = mtp.merge(patches_gen, mf)
        assert secondary_agg is not None
        # Both sources should have seen N patches in a single pass.
        assert out["s2"] == ("merged", "s2_agg", 10)
        assert out[PRIMARY_KEY] == ("merged", "primary_agg", 10)

    def test_round_trip_recognisable_results(self) -> None:
        # Round-trip: split → merge → recognisable per-source results.
        mtp, mf, _primary_agg, _secondary_agg = self._build()
        out = mtp.merge(list(mtp.split(mf)), mf)
        assert out[PRIMARY_KEY] == ("merged", "primary_agg", 10)
        assert out["s2"] == ("merged", "s2_agg", 10)


# ---------------------------------------------------------------------------
# Typo guard on `secondary_aggregators` names
# ---------------------------------------------------------------------------


class TestUnknownAggregatorNamesRejected:
    def _build(self, *, agg_name: str) -> tuple[MatchedTemporalPatcher, MatchedField]:
        mf = MatchedField(
            primary=_ArrField(np.arange(50, dtype=np.float64)),
            secondaries={"s2": _ArrField(np.arange(50, dtype=np.float64))},
            coreg={"s2": lambda raw, prim: raw},
        )
        mtp = MatchedTemporalPatcher(
            primary=_make_patcher(),
            secondary_aggregators={agg_name: _RecordingTemporalAgg("typo")},
        )
        return mtp, mf

    def test_split_rejects_unknown_aggregator_name(self) -> None:
        mtp, mf = self._build(agg_name="s22")
        with pytest.raises(ValueError, match=r"not in mfield\.secondaries"):
            list(mtp.split(mf))

    def test_merge_rejects_unknown_aggregator_name(self) -> None:
        mtp, mf = self._build(agg_name="s22")
        mp = MatchedTemporalPatch(
            anchor=0,
            members={
                PRIMARY_KEY: TemporalPatch(
                    data=np.zeros(3), anchor=0, indices=slice(0, 3)
                )
            },
        )
        with pytest.raises(ValueError, match=r"not in mfield\.secondaries"):
            mtp.merge([mp], mf)


# ---------------------------------------------------------------------------
# Construction / default state
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_secondary_aggregators_empty(self) -> None:
        mtp = MatchedTemporalPatcher(primary=_make_patcher())
        assert mtp.secondary_aggregators == {}

    def test_module_namespace_exposes_class(self) -> None:
        import geopatcher.matched as matched_ns

        assert matched_ns.MatchedTemporalPatcher is MatchedTemporalPatcher
        assert matched_ns.MatchedTemporalPatch is MatchedTemporalPatch


# ---------------------------------------------------------------------------
# Carrier-level invariants
# ---------------------------------------------------------------------------


class TestMatchedTemporalPatchInvariants:
    def test_missing_primary_rejected(self) -> None:
        with pytest.raises(ValueError, match="must contain the primary key"):
            MatchedTemporalPatch(
                anchor=0,
                members={
                    "s2": TemporalPatch(data=np.zeros(3), anchor=0, indices=slice(0, 3))
                },
            )

    def test_valid_mask_keys_must_subset_members(self) -> None:
        with pytest.raises(ValueError, match="valid_mask has keys not present"):
            MatchedTemporalPatch(
                anchor=0,
                members={
                    PRIMARY_KEY: TemporalPatch(
                        data=np.zeros(3), anchor=0, indices=slice(0, 3)
                    )
                },
                valid_mask={"ghost": np.zeros(3, dtype=bool)},
            )

    def test_secondary_names_excludes_primary(self) -> None:
        mp = MatchedTemporalPatch(
            anchor=0,
            members={
                PRIMARY_KEY: TemporalPatch(
                    data=np.zeros(3), anchor=0, indices=slice(0, 3)
                ),
                "s2": TemporalPatch(data=np.zeros(3), anchor=0, indices=slice(0, 3)),
            },
        )
        assert mp.secondary_names == ("s2",)
        assert mp.primary is mp.members[PRIMARY_KEY]


# ---------------------------------------------------------------------------
# Real Field adapters, cadence check, primary-only counting, hooks (#203)
# ---------------------------------------------------------------------------


def _raster_series(values: np.ndarray, fill: Any = np.nan) -> Any:
    import rasterio
    from georeader.geotensor import GeoTensor

    from geopatcher._src.fields.raster import RasterField

    return RasterField(
        GeoTensor(
            values=values,
            transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
            crs="EPSG:32629",
            fill_value_default=fill,
        )
    )


def _series(n: int, *, scale: float = 1.0) -> np.ndarray:
    return (np.arange(n * 2 * 2, dtype=np.float32).reshape(n, 2, 2)) * scale


def test_real_field_indexer() -> None:
    # A (time, H, W) RasterField and an XarrayField pair: the default
    # indexer is each domain's full extent, which both adapters accept.
    xr = pytest.importorskip("xarray")
    from geopatcher._src.fields.xarray import XarrayField

    raster = MatchedField(
        primary=_raster_series(_series(20)),
        secondaries={"s": _raster_series(_series(20, scale=2.0))},
        coreg={"s": lambda raw, prim: raw},
    )
    grid = MatchedField(
        primary=XarrayField(xr.DataArray(_series(20), dims=("time", "y", "x"))),
        secondaries={
            "s": XarrayField(
                xr.DataArray(_series(20, scale=2.0), dims=("time", "y", "x"))
            )
        },
        coreg={"s": lambda raw, prim: raw},
    )
    mtp = MatchedTemporalPatcher(primary=_make_patcher())
    for mf in (raster, grid):
        patches = list(mtp.split(mf))
        assert len(patches) == 2
        assert mtp.n_anchors(mf) == 2
        assert mtp.anchors(mf) == [mp.anchor for mp in patches]
        for mp in patches:
            prim = np.asarray(mp.members[PRIMARY_KEY].data)
            assert prim.shape[1:] == (2, 2)
            np.testing.assert_array_equal(np.asarray(mp.members["s"].data), prim * 2)


def test_cadence_mismatch_raises() -> None:
    # A 5-step secondary against a 10-step primary used to yield member
    # lengths (5, 0) silently.
    mf = MatchedField(
        primary=_ArrField(np.arange(10, dtype=np.float64)),
        secondaries={"s": _ArrField(np.arange(5, dtype=np.float64))},
        coreg={"s": lambda raw, prim: raw},
    )
    mtp = MatchedTemporalPatcher(primary=_make_patcher())
    with pytest.raises(ValueError, match=r"'s' has 5 steps.*primary has 10"):
        list(mtp.split(mf))


def test_n_anchors_and_anchors_read_the_primary_only() -> None:
    calls: list[str] = []

    def coreg(raw: Any, prim: Any) -> Any:
        calls.append("coreg")
        return raw

    class _CountingArrField(_ArrField):
        def select(self, indexer: Any) -> Any:
            calls.append("select")
            return super().select(indexer)

    mf = MatchedField(
        primary=_ArrField(np.arange(100, dtype=np.float64)),
        secondaries={"s": _CountingArrField(np.arange(100, dtype=np.float64))},
        coreg={"s": coreg},
    )
    mtp = MatchedTemporalPatcher(primary=_make_patcher())
    assert mtp.n_anchors(mf) == 10
    assert len(mtp.anchors(mf)) == 10
    assert calls == []
    list(mtp.split(mf))
    assert calls == ["select", "coreg"]


def test_valid_mask_uses_source_nodata() -> None:
    # int16 series with nodata 0: the members are bare ndarray slices,
    # but the mask still honours the source carrier's fill_value_default.
    values = np.ones((15, 2, 2), dtype=np.int16)
    values[10] = 0
    mf = MatchedField(primary=_raster_series(values, fill=0))
    mtp = MatchedTemporalPatcher(primary=_make_patcher())
    _, mp = list(mtp.split(mf))  # anchor 10: steps 6..10
    assert mp.valid_mask is not None
    mask = mp.valid_mask[PRIMARY_KEY]
    assert mask.shape == (5, 2, 2)
    assert not mask[-1].any()
    assert mask[:-1].all()


class _RecordingHook:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def on_split_start(self, n: int) -> None:
        self.events.append(("split_start", n))

    def on_patch_start(self, anchor: Any, coord_value: Any = None) -> None:
        self.events.append(("patch_start", anchor, coord_value))

    def on_patch_done(
        self, anchor: Any, elapsed: float, bytes_: int, coord_value: Any = None
    ) -> None:
        self.events.append(("patch_done", anchor, bytes_, coord_value))

    def on_split_end(self) -> None:
        self.events.append(("split_end",))

    def on_merge_start(self, n: int) -> None:
        self.events.append(("merge_start", n))

    def on_merge_end(self, output_bytes: int) -> None:
        self.events.append(("merge_end", output_bytes))


def test_hooks_sum_member_bytes_at_one_level() -> None:
    mf = MatchedField(
        primary=_ArrField(np.arange(100, dtype=np.float64)),
        secondaries={"s": _ArrField(np.arange(100, dtype=np.float64) * 2)},
        coreg={"s": lambda raw, prim: raw},
    )

    class _OnesAgg(TemporalAggregation):
        streaming_safe = True

        def merge(self, patches: Any) -> Any:
            return np.ones(len(list(patches)))

    mtp = MatchedTemporalPatcher(
        primary=_make_patcher(aggregation=_OnesAgg()),
        secondary_aggregators={"s": _OnesAgg()},
    )
    hook = _RecordingHook()
    patches = list(mtp.split(mf, hooks=[hook]))
    done = [e for e in hook.events if e[0] == "patch_done"]
    assert len(done) == len(patches) == 10
    for event, mp in zip(done, patches, strict=True):
        assert event[2] == sum(m.data.nbytes for m in mp.members.values()) > 0
    assert hook.events[0] == ("split_start", 10)
    assert hook.events[-1] == ("split_end",)

    hook = _RecordingHook()
    out = mtp.merge(patches, mf, hooks=[hook])
    assert [e[0] for e in hook.events] == ["merge_start", "merge_end"]
    assert hook.events[1][1] == sum(v.nbytes for v in out.values()) == 160


def test_merge_streams_every_source() -> None:
    # Each aggregation must see its members while the producer is still
    # running — not after every source's full patch list was collected.
    produced = 0
    lags: dict[str, int] = {}

    class _LagAgg(TemporalAggregation):
        streaming_safe = True

        def __init__(self, name: str) -> None:
            self.name = name

        def merge(self, patches: Any) -> Any:
            seen = 0
            for _ in patches:
                seen += 1
                lags[self.name] = max(lags.get(self.name, 0), produced - seen)
            return seen

    def stream() -> Any:
        nonlocal produced
        for i in range(50):
            produced += 1
            yield MatchedTemporalPatch(
                anchor=i,
                members={
                    name: TemporalPatch(data=np.zeros(3), anchor=i, indices=slice(0, 3))
                    for name in (PRIMARY_KEY, "s")
                },
            )

    mf = MatchedField(
        primary=_ArrField(np.arange(10, dtype=np.float64)),
        secondaries={"s": _ArrField(np.arange(10, dtype=np.float64))},
        coreg={"s": lambda raw, prim: raw},
    )
    mtp = MatchedTemporalPatcher(
        primary=_make_patcher(aggregation=_LagAgg(PRIMARY_KEY)),
        secondary_aggregators={"s": _LagAgg("s")},
    )
    assert mtp.merge(stream(), mf) == {PRIMARY_KEY: 50, "s": 50}
    assert max(lags.values()) <= 4

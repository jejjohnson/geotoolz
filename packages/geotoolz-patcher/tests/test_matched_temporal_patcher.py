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
    return TemporalPatcher(
        geometry=TemporalFixedLookback(length=5),
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

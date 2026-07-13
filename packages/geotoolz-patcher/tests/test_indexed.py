"""Tests for `IndexedPatchView` — random-access wrapper for ML loaders."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import (
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.indexed import IndexedPatchView


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestSequenceProtocol:
    def test_len_matches_anchor_count(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        assert len(view) == 16  # 4x4 tiles

    def test_indexed_access_matches_split(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        from_split = list(patcher.split(field))
        for i, expected in enumerate(from_split):
            np.testing.assert_array_equal(
                np.asarray(view[i].data), np.asarray(expected.data)
            )

    def test_iter_walks_all_patches(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        n = sum(1 for _ in view)
        assert n == len(view)

    def test_negative_index(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        np.testing.assert_array_equal(
            np.asarray(view[-1].data), np.asarray(view[len(view) - 1].data)
        )

    def test_out_of_range_raises_indexerror(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        with pytest.raises(IndexError):
            _ = view[len(view)]
        with pytest.raises(IndexError):
            _ = view[-len(view) - 1]

    def test_slice_returns_list(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        subset = view[2:5]
        assert isinstance(subset, list)
        assert len(subset) == 3


class TestCaching:
    def test_cache_off_returns_fresh_patch_each_call(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field, cache=False)
        a = view[0]
        b = view[0]
        # Same patch payload, different objects (no cache).
        assert a is not b
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))

    def test_cache_on_returns_identical_object(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field, cache=True)
        a = view[3]
        b = view[3]
        assert a is b

    def test_preload_requires_cache(self, patcher, field) -> None:
        with pytest.raises(ValueError, match="preload=True requires cache=True"):
            IndexedPatchView(patcher, field, preload=True)

    def test_clear_cache_drops_entries(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field, cache=True)
        a = view[0]
        view.clear_cache()
        b = view[0]
        assert a is not b  # fresh after clear


class TestErrors:
    def test_non_patcher_raises_typeerror(self, field) -> None:
        with pytest.raises(TypeError, match=r"anchors.*patch_at"):
            IndexedPatchView(object(), field)


class TestAnchorsAttr:
    def test_anchors_property_is_a_copy(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        view.anchors.clear()
        # Mutating the returned list must not affect the view's state.
        assert len(view) == 16


class TestPreloadMaterialises:
    """`preload=True` must actually materialise lazy data into RAM."""

    def test_load_method_called(self, patcher, field) -> None:
        # Stub a patcher whose patch_at returns a Patch with lazy data
        # implementing .load().
        from geopatcher._src.patch import Patch

        class _LazyData:
            def __init__(self) -> None:
                self.loaded = False

            def load(self) -> np.ndarray:
                self.loaded = True
                return np.array([42])

        class _StubPatcher:
            def anchors(self, _field) -> list[int]:
                return [0]

            def patch_at(self, _field, _anchor) -> Patch:
                return Patch(
                    data=_LazyData(),
                    anchor=_anchor,
                    indices=None,
                    weights=None,
                )

        view = IndexedPatchView(_StubPatcher(), field, cache=True, preload=True)
        patch = view[0]
        # Patch.data was replaced with the .load() return value.
        np.testing.assert_array_equal(np.asarray(patch.data), [42])

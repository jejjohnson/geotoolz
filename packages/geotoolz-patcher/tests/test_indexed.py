"""Tests for `IndexedPatchView` — random-access wrapper for ML loaders."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import SpatialPatcher, TemporalPatcher, spatial, temporal
from geopatcher._src.indexed import IndexedPatchView


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16)),
        sampler=spatial.sampler.RegularStride(step=16),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
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


class TestIndexValidation:
    def test_float_index_raises(self, patcher, field) -> None:
        # `int(1.9)` would silently read patch 1.
        view = IndexedPatchView(patcher, field)
        with pytest.raises(TypeError, match="integers or slices, not float"):
            _ = view[1.9]  # type: ignore[call-overload]

    def test_numpy_integer_index(self, patcher, field) -> None:
        view = IndexedPatchView(patcher, field)
        np.testing.assert_array_equal(
            np.asarray(view[np.int64(3)].data), np.asarray(view[3].data)
        )

    def test_preload_with_patch_cache_names_the_mode(
        self, patcher, field, tmp_path
    ) -> None:
        from geopatcher.run import PatchCache

        cache = PatchCache(tmp_path, field_id="scene")
        with pytest.raises(ValueError, match="only to the in-memory cache"):
            IndexedPatchView(patcher, field, cache=cache, preload=True)
        with pytest.raises(ValueError, match="only to the in-memory cache"):
            IndexedPatchView(patcher, field, cache=cache, cache_size=2)


def _temporal(geometry) -> TemporalPatcher:
    return TemporalPatcher(
        geometry=geometry,
        sampler=temporal.sampler.RegularStride(step=7),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )


def _assert_matches_split(view, expected) -> None:
    assert len(view) == len(expected)
    for got, ref in zip(view, expected, strict=True):
        assert got.anchor == ref.anchor
        assert got.indices == ref.indices
        np.testing.assert_array_equal(got.data, ref.data)
        np.testing.assert_array_equal(got.weights, ref.weights)


class TestTemporalPatcher:
    def test_temporal_patcher_supported(self) -> None:
        series = np.arange(60.0).reshape(30, 2)
        tp = _temporal(temporal.geometry.FixedLookback(length=5))
        view = IndexedPatchView(tp, series)
        _assert_matches_split(view, list(tp.split(series)))

    def test_multi_scale_matches_split(self) -> None:
        # Several patches per anchor: the view indexes per patch, not per
        # anchor, so `view[i]` is still the i-th patch of `split`.
        series = np.arange(40.0)
        tp = _temporal(temporal.geometry.MultiScale(scales=[3, 8]))
        view = IndexedPatchView(tp, series)
        assert len(view) == tp.n_anchors(series) == 2 * len(tp.anchors(series))
        _assert_matches_split(view, list(tp.split(series)))

    def test_patcher_kwargs_forwarded(self) -> None:
        series = np.arange(60.0).reshape(2, 30)
        tp = _temporal(temporal.geometry.LookbackHorizon(lookback=4, horizon=2))
        view = IndexedPatchView(tp, series, patcher_kwargs={"time_axis": 1})
        _assert_matches_split(view, list(tp.split(series, time_axis=1)))

    def test_lazy_series_matches_split(self) -> None:
        xr = pytest.importorskip("xarray")
        pytest.importorskip("dask")
        da = xr.DataArray(np.arange(30.0), dims="time").chunk({"time": 5})
        tp = _temporal(temporal.geometry.FixedLookback(length=5))
        view = IndexedPatchView(tp, da)
        _assert_matches_split(view, list(tp.split(da.values)))

    def test_bare_anchor_on_multi_scale_raises(self) -> None:
        tp = _temporal(temporal.geometry.MultiScale(scales=[3, 8]))
        with pytest.raises(ValueError, match=r"pass \(7, k\)"):
            tp.patch_at(np.arange(40.0), 7)
        with pytest.raises(ValueError, match="out of range"):
            tp.patch_at(np.arange(40.0), (7, 2))

    def test_patch_cache_serves_temporal_patches(self, tmp_path) -> None:
        # `TemporalPatcher.patch_at` takes ``cache=`` / ``field_id=`` (#190),
        # so the view's on-disk cache mode works for temporal patchers too.
        from geopatcher.run import PatchCache

        cache = PatchCache(tmp_path, field_id="series")
        series = np.arange(30.0)
        tp = _temporal(temporal.geometry.MultiScale(scales=[2, 5]))
        first = IndexedPatchView(tp, series, cache=cache)
        expected = list(tp.split(series))
        _assert_matches_split(first, expected)
        assert cache.stats()["misses"] == len(expected)
        again = IndexedPatchView(tp, series, cache=cache)
        _assert_matches_split(again, expected)
        assert cache.stats()["hits"] == len(expected)
        assert [again[i].window_index for i in range(len(again))] == [
            p.window_index for p in expected
        ]

    def test_in_memory_cache_works(self) -> None:
        tp = _temporal(temporal.geometry.FixedLookback(length=5))
        view = IndexedPatchView(tp, np.arange(30.0), cache=True)
        assert view[2] is view[2]

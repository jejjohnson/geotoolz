"""Concurrency + LRU-bound tests for `IndexedPatchView`'s cache."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from geopatcher._src.indexed import IndexedPatchView
from geopatcher._src.patch import Patch


class _CountingPatcher:
    """Stub patcher recording every `patch_at` call."""

    def __init__(self, n: int = 8) -> None:
        self.n = n
        self.calls: list[int] = []
        self._lock = threading.Lock()

    def anchors(self, _field) -> list[int]:
        return list(range(self.n))

    def patch_at(self, _field, anchor: int) -> Patch:
        with self._lock:
            self.calls.append(anchor)
        return Patch(
            data=np.full((2, 2), anchor, dtype=np.float32),
            anchor=anchor,
            indices=None,
            weights=None,
        )


class TestCacheSizeValidation:
    def test_cache_size_requires_cache(self) -> None:
        with pytest.raises(ValueError, match="cache_size requires cache=True"):
            IndexedPatchView(_CountingPatcher(), None, cache_size=2)

    def test_cache_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="cache_size must be >= 1"):
            IndexedPatchView(_CountingPatcher(), None, cache=True, cache_size=0)

    def test_none_means_unbounded(self) -> None:
        patcher = _CountingPatcher(n=6)
        view = IndexedPatchView(patcher, None, cache=True)
        for i in range(6):
            view[i]
        for i in range(6):
            view[i]
        # Every anchor built exactly once — nothing was evicted.
        assert sorted(patcher.calls) == list(range(6))


class TestLruBound:
    def test_eviction_is_least_recently_used(self) -> None:
        patcher = _CountingPatcher(n=4)
        view = IndexedPatchView(patcher, None, cache=True, cache_size=2)
        view[0]
        view[1]
        view[0]  # touch 0 → 1 becomes least recently used
        view[2]  # evicts 1
        assert patcher.calls == [0, 1, 2]
        view[0]  # still cached
        assert patcher.calls == [0, 1, 2]
        view[1]  # evicted — rebuilt
        assert patcher.calls == [0, 1, 2, 1]

    def test_cache_never_exceeds_bound(self) -> None:
        patcher = _CountingPatcher(n=8)
        view = IndexedPatchView(patcher, None, cache=True, cache_size=3)
        for i in range(8):
            view[i]
        assert len(view._cache) == 3

    def test_cache_hit_returns_same_object(self) -> None:
        view = IndexedPatchView(_CountingPatcher(), None, cache=True, cache_size=4)
        assert view[1] is view[1]


class TestThreadSafety:
    def test_concurrent_reads_of_same_index_converge_on_one_object(self) -> None:
        patcher = _CountingPatcher(n=4)
        view = IndexedPatchView(patcher, None, cache=True)
        results: list[Patch] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def _read() -> None:
            barrier.wait()
            patch = view[2]
            with results_lock:
                results.append(patch)

        threads = [threading.Thread(target=_read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert len(results) == 8
        # The race may build the patch more than once, but every caller
        # must observe the single cached object.
        assert len({id(p) for p in results}) == 1
        assert view[2] is results[0]

    def test_concurrent_mixed_indices_with_lru_bound(self) -> None:
        patcher = _CountingPatcher(n=8)
        view = IndexedPatchView(patcher, None, cache=True, cache_size=2)
        errors: list[BaseException] = []

        def _walk(seed: int) -> None:
            try:
                for i in range(50):
                    idx = (seed + i) % 8
                    patch = view[idx]
                    assert patch.anchor == idx
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_walk, args=(s,)) for s in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert errors == []
        assert len(view._cache) <= 2

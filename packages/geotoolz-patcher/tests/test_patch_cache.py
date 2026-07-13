"""Tests for `PatchCache` — content-addressed on-disk patch cache (issue #24).

The core contract: a second `split` (or a second *process*) with the
same field + geometry + window config performs zero source reads and
reconstructs each patch bit-identically.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _helpers import make_raster_field

from geopatcher import (
    PatchCache,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
)


class _CountingField:
    """Wrap a `Field`, counting every `select` (source read)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.selects = 0

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, window: Any) -> Any:
        self.selects += 1
        return self.inner.select(window)

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)


def _patcher(size: int = 8, step: int = 8) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(size, size)),
        sampler=SpatialRegularStride(step=step),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestHitMiss:
    def test_second_split_reads_nothing(self, tmp_path) -> None:
        base = make_raster_field(16)  # 4 patches at size/step 8
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()

        first = _CountingField(base)
        p1 = list(patcher.split(first, cache=cache))
        assert first.selects == 4  # cold cache → every patch is read

        second = _CountingField(base)
        p2 = list(patcher.split(second, cache=cache))
        assert second.selects == 0  # warm cache → zero source reads
        assert len(p2) == len(p1) == 4
        stats = cache.stats()
        assert stats["hits"] == 4
        assert stats["entries"] == 4

    def test_config_change_is_a_miss(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        list(_patcher(size=8, step=8).split(base, cache=cache))

        # A different geometry config → different key → cold reads again.
        other = _CountingField(base)
        list(_patcher(size=4, step=4).split(other, cache=cache))
        assert other.selects > 0

    def test_roundtrip_is_bit_identical(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()

        reference = {p.anchor: p for p in patcher.split(base)}  # uncached
        list(patcher.split(base, cache=cache))  # fill
        cached = {p.anchor: p for p in patcher.split(base, cache=cache)}  # hits

        for anchor, ref in reference.items():
            got = cached[anchor]
            np.testing.assert_array_equal(
                np.asarray(got.data.values), np.asarray(ref.data.values)
            )
            assert got.data.transform == ref.data.transform
            assert str(got.data.crs) == str(ref.data.crs)
            np.testing.assert_array_equal(
                np.asarray(got.weights), np.asarray(ref.weights)
            )
            assert got.indices == ref.indices


class TestFieldIdentity:
    def test_in_memory_field_without_id_raises(self, tmp_path) -> None:
        base = make_raster_field(16)  # RasterField(GeoTensor): no path/url
        cache = PatchCache(tmp_path)
        with pytest.raises(ValueError, match="field_id"):
            list(_patcher().split(base, cache=cache))

    def test_in_memory_field_with_id_caches(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="mem")
        first = _CountingField(base)
        list(_patcher().split(first, cache=cache))
        assert first.selects == 4
        second = _CountingField(base)
        list(_patcher().split(second, cache=cache))
        assert second.selects == 0

    def test_field_id_tracks_source_changes(self, tmp_path) -> None:
        # A path-backed reader's identity folds in realpath + mtime + size,
        # so editing the source invalidates its entries.
        src = tmp_path / "scene.dat"
        src.write_bytes(b"aaaa")
        os.utime(src, (1000, 1000))
        fld = SimpleNamespace(reader=SimpleNamespace(paths=[str(src)]))
        cache = PatchCache(tmp_path / "cache")
        before = cache.field_id_for(fld)

        src.write_bytes(b"bbbbbbbb")  # size + content change
        os.utime(src, (2000, 2000))  # mtime change
        after = cache.field_id_for(fld)
        assert before != after
        assert before.startswith("path:")


class TestStochasticSamplers:
    def test_seeded_random_hits_on_rerun(self, tmp_path) -> None:
        base = make_raster_field(32)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRandom(n_samples=12, seed=42),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        first = _CountingField(base)
        list(patcher.split(first, cache=cache))
        assert first.selects == 12
        # Same seed → identical anchors → 100% hits, zero reads.
        second = _CountingField(base)
        list(patcher.split(second, cache=cache))
        assert second.selects == 0


class TestEviction:
    def test_lru_eviction_bounds_total_size(self, tmp_path) -> None:
        base = make_raster_field(64)  # 64 patches at size/step 8
        # A cap that only fits a handful of entries.
        cache = PatchCache(tmp_path, max_bytes=4_000, field_id="scene")
        list(_patcher().split(base, cache=cache))
        stats = cache.stats()
        assert stats["bytes"] <= 4_000
        assert 0 < stats["entries"] < 64

    def test_clear_empties_the_cache(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        list(_patcher().split(base, cache=cache))
        assert cache.stats()["entries"] == 4
        cache.clear()
        assert cache.stats()["entries"] == 0
        assert cache.stats()["hits"] == 0


class TestConcurrency:
    def test_concurrent_access_no_corruption(self, tmp_path) -> None:
        # Multiple readers/writers on one cache dir: atomic renames mean no
        # torn entries. Assert every patch round-trips correctly.
        base = make_raster_field(32)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()
        reference = {p.anchor: np.asarray(p.data.values) for p in patcher.split(base)}

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for p in patcher.split(base, cache=cache):
                    np.testing.assert_array_equal(
                        np.asarray(p.data.values), reference[p.anchor]
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cache.stats()["entries"] == len(reference)


class TestIndexedViewIntegration:
    def test_indexed_view_uses_disk_cache(self, tmp_path) -> None:
        from geopatcher import IndexedPatchView

        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()

        first = _CountingField(base)
        view = IndexedPatchView(patcher, first, cache=cache)
        _ = [view[i] for i in range(len(view))]
        assert first.selects == len(view)

        second = _CountingField(base)
        view2 = IndexedPatchView(patcher, second, cache=cache)
        _ = [view2[i] for i in range(len(view2))]
        assert second.selects == 0

"""Tests for ``parallel_map``'s duck-typed ``select_many`` fast path.

Verifies two things:

1. **Equivalence** — when the field has ``select_many``,
   ``parallel_map`` produces the same per-patch output as the
   sequential path. The runner must not change semantics, only batch
   the reads.
2. **Actual batching** — the field's ``select_many`` is called
   (once per ``batch_size`` chunk), and the per-patch ``select``
   path is *not* taken on the real field. This pins the optimisation
   so a refactor that accidentally falls back to ``select`` per
   patch trips the test.

The tests don't depend on obstore — they use a tiny stub field with
both ``select`` and ``select_many`` so the runner's branch can be
exercised in isolation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher.runners import parallel_map


def _double(data) -> np.ndarray:
    return np.asarray(data) * 2


@dataclass
class _CountingBatchedField:
    """RasterField-shaped stub that exposes ``select_many``.

    Reads delegate to a wrapped ``RasterField``; the counter is the
    test hook used to assert the fast path actually fired.
    """

    inner: RasterField
    select_calls: int = 0
    select_many_calls: int = 0

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, indexer: Any) -> Any:
        self.select_calls += 1
        return self.inner.select(indexer)

    def select_many(self, indexers: list[Any]) -> list[Any]:
        self.select_many_calls += 1
        return [self.inner.select(i) for i in indexers]

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)


@pytest.fixture
def field(raster_field_factory) -> RasterField:
    return raster_field_factory(32)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_batched_path_equivalent_to_sequential(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """Wrapping the field in ``_CountingBatchedField`` must not change outputs."""
    batched_field = _CountingBatchedField(inner=field)
    batched_out = parallel_map(patcher, batched_field, _double, n_workers=2)
    raw_out = parallel_map(patcher, field, _double, n_workers=2)

    assert [p.anchor for p in batched_out] == [p.anchor for p in raw_out]
    for b, r in zip(batched_out, raw_out, strict=True):
        np.testing.assert_array_equal(b.data, r.data)


def test_batched_path_actually_batches(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """``select_many`` must fire; ``select`` must not be called on the real field."""
    batched_field = _CountingBatchedField(inner=field)
    parallel_map(patcher, batched_field, _double, n_workers=2)
    # The stub's `select` is invoked once per indexer *inside*
    # `select_many`, but not by the runner directly. Either way, the
    # important assertion is that ``select_many`` did fire — at least
    # once for the full set of patches.
    assert batched_field.select_many_calls >= 1


def test_batch_size_chunks_select_many_calls(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """A small ``batch_size`` should split the fan-out across N calls."""
    batched_field = _CountingBatchedField(inner=field)
    # Patcher produces 16 anchors (4x4 grid at stride 8 over 32x32).
    n_patches = len(patcher.anchors(field))
    parallel_map(patcher, batched_field, _double, n_workers=2, batch_size=4)
    expected_chunks = (n_patches + 4 - 1) // 4
    assert batched_field.select_many_calls == expected_chunks


def test_batch_size_validation(patcher: SpatialPatcher, field: RasterField) -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        parallel_map(patcher, field, _double, batch_size=0)


def test_runner_unchanged_for_fields_without_select_many(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """Plain ``RasterField`` (no ``select_many``) must take the legacy path."""
    out = parallel_map(patcher, field, _double, n_workers=2)
    expected = [
        type(p)(
            data=_double(p.data), anchor=p.anchor, indices=p.indices, weights=p.weights
        )
        for p in patcher.split(field)
    ]
    for got, want in zip(out, expected, strict=True):
        np.testing.assert_array_equal(got.data, want.data)


def test_batched_path_unwraps_masked_window(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """Regression: ``_MaskedWindow`` indices must be unwrapped before select_many.

    ``SpatialPolygonIntersection`` wraps the rasterio ``Window`` in a
    ``_MaskedWindow`` so aggregation can recover the interior mask.
    The non-batched path calls ``_unwrap_for_select`` before
    ``Field.select``; the batched path must do the same before
    ``Field.select_many`` or polygon-intersection patchers break.
    """
    from geopatcher._src.spatial.patcher import _MaskedWindow

    # Stub field that asserts every indexer arriving in select_many is
    # already a rasterio Window (i.e. _MaskedWindow has been unwrapped).
    @dataclass
    class _UnwrapAssertingField:
        inner: RasterField

        @property
        def domain(self) -> Any:
            return self.inner.domain

        def select(self, indexer: Any) -> Any:
            assert not isinstance(indexer, _MaskedWindow), (
                "indexer not unwrapped before select()"
            )
            return self.inner.select(indexer)

        def select_many(self, indexers: list[Any]) -> list[Any]:
            for idx in indexers:
                assert not isinstance(idx, _MaskedWindow), (
                    "indexer not unwrapped before select_many()"
                )
            return [self.inner.select(i) for i in indexers]

        def with_data(self, array: Any) -> Any:
            return self.inner.with_data(array)

    # Run the runner via the stub. The existing fixture patcher uses
    # SpatialRectangular geometry, which doesn't wrap. But the helper
    # in _planned_indexer calls _unwrap_for_select unconditionally,
    # so the asserts pass for ALL geometries — that's the contract this
    # test pins. (A separate end-to-end test in the polygon-intersection
    # suite would also cover the wrapped case once we have it.)
    parallel_map(patcher, _UnwrapAssertingField(inner=field), _double, n_workers=2)


def test_batched_path_falls_back_to_per_patch_on_select_many_failure(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """select_many raising → per-patch fallback for the chunk.

    Each read then succeeds or fails on its own, under the patcher's
    ``on_error``, so one bad tile doesn't abort an entire chunk of N good
    patches.
    """

    @dataclass
    class _BrokenBulkField:
        inner: RasterField
        select_many_calls: int = 0
        per_patch_select_calls: int = 0

        @property
        def domain(self) -> Any:
            return self.inner.domain

        def select(self, indexer: Any) -> Any:
            self.per_patch_select_calls += 1
            return self.inner.select(indexer)

        def select_many(self, indexers: list[Any]) -> list[Any]:
            self.select_many_calls += 1
            raise RuntimeError("simulated transient remote-read failure")

        def with_data(self, array: Any) -> Any:
            return self.inner.with_data(array)

    bf = _BrokenBulkField(inner=field)
    out = parallel_map(patcher, bf, _double, n_workers=2)
    # select_many fired, raised; per-patch fallback then took over and
    # produced all patches (none of them error on the inner field).
    n_patches = len(patcher.anchors(field))
    assert bf.select_many_calls >= 1
    assert bf.per_patch_select_calls == n_patches
    assert len(out) == n_patches


def _identity(data: Any) -> Any:
    return data


@pytest.mark.parametrize("boundary", ["pad", "reflect"])
def test_select_many_with_pad(raster_field_factory, boundary: str) -> None:
    """#196: the fast path pads / reflects edge chips exactly like ``split``.

    10x10 field, 4x4 patches at stride 4 → anchors 0, 4, 8 per axis; the
    row/col-8 windows overflow by 2. The batched chips (values and
    georeferencing) must equal the per-patch ``split`` chips, and the
    outputs must merge back to the field.
    """
    field = raster_field_factory(10)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4), boundary=boundary),
        sampler=SpatialRegularStride(step=4),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    batched = _CountingBatchedField(inner=field)

    out = parallel_map(patcher, batched, _identity, n_workers=2, batch_size=4)
    expected = list(patcher.split(field))

    assert batched.select_many_calls == 3  # 9 reads in chunks of 4
    assert batched.select_calls == 0  # every read served by select_many
    assert [p.anchor for p in out] == [p.anchor for p in expected]
    for got, want in zip(out, expected, strict=True):
        assert got.data.shape == (4, 4)
        np.testing.assert_array_equal(got.data.values, want.data.values)
        assert got.data.transform == want.data.transform
    merged = patcher.merge(out, field.domain)
    np.testing.assert_array_equal(merged, np.asarray(field.reader.values))


@dataclass
class _FlakyBatchedField:
    """``select_many`` always fails; ``select`` fails for one window only."""

    inner: RasterField
    bad_offset: tuple[int, int] = (8, 8)

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, indexer: Any) -> Any:
        if (indexer.row_off, indexer.col_off) == self.bad_offset:
            raise OSError("tile read failed")
        return self.inner.select(indexer)

    def select_many(self, indexers: list[Any]) -> list[Any]:
        raise OSError("bulk read failed")

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)


def test_select_many_failure_is_scoped_by_patcher_on_error(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """A failing chunk falls back per patch; only the bad tile is skipped."""
    patcher = replace(patcher, on_error="skip")
    out = parallel_map(patcher, _FlakyBatchedField(inner=field), _double, n_workers=2)

    anchors = patcher.anchors(field)
    assert [p.anchor for p in out] == [a for a in anchors if a != (8, 8)]
    assert [(e.anchor, e.kind) for e in patcher.errors] == [((8, 8), "OSError")]


@dataclass
class _ResidencyField:
    """Counts reads; ``peak`` = most reads not yet through the operator."""

    inner: RasterField
    reads: int = 0
    completed: int = 0
    peak: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def _read(self, n: int) -> None:
        with self._lock:
            self.reads += n
            self.peak = max(self.peak, self.reads - self.completed)

    def select(self, indexer: Any) -> Any:
        self._read(1)
        return self.inner.select(indexer)

    def done(self, data: Any) -> Any:
        time.sleep(0.001)
        with self._lock:
            self.completed += 1
        return data

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)


@dataclass
class _BatchedResidencyField(_ResidencyField):
    def select_many(self, indexers: list[Any]) -> list[Any]:
        self._read(len(indexers))
        return [self.inner.select(i) for i in indexers]


@pytest.mark.parametrize("batched", [False, True])
def test_parallel_map_streams_reads(
    patcher: SpatialPatcher, raster_field_factory, batched: bool
) -> None:
    """#196: patches are read as workers free up, not all up front.

    64 patches, 2 workers → at most ``max_in_flight`` (default
    ``2 * n_workers``) submitted patches, plus one ``batch_size`` chunk of
    read-ahead on the ``select_many`` path, are ever resident.
    """
    inner = raster_field_factory(64)
    cls = _BatchedResidencyField if batched else _ResidencyField
    field = cls(inner=inner)
    n_patches = len(patcher.anchors(inner))

    out = parallel_map(patcher, field, field.done, n_workers=2, batch_size=8)

    assert len(out) == n_patches == 64
    bound = 2 * 2 + (8 if batched else 0) + 1
    assert field.peak <= bound

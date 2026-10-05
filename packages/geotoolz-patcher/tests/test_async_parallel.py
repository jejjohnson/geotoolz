"""Tests for async and parallel patching helpers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import pytest
from _helpers import ArrayField

from geopatcher import (
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    TemporalCausalBoxcar,
    TemporalFixedLookback,
    TemporalMean,
    TemporalPatcher,
    TemporalRegularStride,
)
from geopatcher.jax import batch_split, unbatch


class AsyncArrayField(ArrayField):
    async def aselect(self, window: Any) -> np.ndarray:
        await asyncio.sleep(0)
        return self.select(window)


@pytest.fixture
def field() -> ArrayField:
    return ArrayField(np.arange(16 * 16, dtype=np.float32).reshape(16, 16))


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_spatial_asplit_matches_split(
    field: ArrayField, patcher: SpatialPatcher
) -> None:
    async def collect() -> list[Any]:
        return [patch async for patch in patcher.asplit(AsyncArrayField(field.array))]

    sync_patches = list(patcher.split(field))
    async_patches = asyncio.run(collect())
    assert [p.anchor for p in async_patches] == [p.anchor for p in sync_patches]
    for async_patch, sync_patch in zip(async_patches, sync_patches, strict=True):
        np.testing.assert_array_equal(async_patch.data, sync_patch.data)


@pytest.mark.parametrize(
    ("boundary", "n", "size"),
    [("pad", 10, 4), ("reflect", 11, 4)],
)
def test_spatial_asplit_matches_split_boundary(
    boundary: str, n: int, size: int
) -> None:
    # asplit must mirror split for the clip-and-pad edge modes (issue #19).
    array = np.arange(n * n, dtype=np.float32).reshape(n, n)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(size, size), boundary=boundary),
        sampler=SpatialRegularStride(step=size),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )

    async def collect() -> list[Any]:
        return [patch async for patch in patcher.asplit(AsyncArrayField(array))]

    sync_patches = list(patcher.split(ArrayField(array)))
    async_patches = asyncio.run(collect())
    assert [p.anchor for p in async_patches] == [p.anchor for p in sync_patches]
    for async_patch, sync_patch in zip(async_patches, sync_patches, strict=True):
        np.testing.assert_array_equal(
            np.asarray(async_patch.data), np.asarray(sync_patch.data)
        )


def test_spatial_split_prefetch_starts_background_read(patcher: SpatialPatcher) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingField(ArrayField):
        def select(self, window: Any) -> np.ndarray:
            started.set()
            assert release.wait(timeout=1)
            return super().select(window)

    iterator = patcher.split(
        BlockingField(np.arange(16 * 16, dtype=np.float32).reshape(16, 16)),
        prefetch=1,
    )
    assert started.wait(timeout=1)
    release.set()
    assert next(iterator).anchor == (0, 0)
    list(iterator)


def test_prefetch_replays_worker_exception(patcher: SpatialPatcher) -> None:
    class FailingField(ArrayField):
        def select(self, window: Any) -> np.ndarray:
            raise RuntimeError("read failed")

    with pytest.raises(RuntimeError, match="read failed"):
        next(
            patcher.split(
                FailingField(np.arange(16 * 16, dtype=np.float32).reshape(16, 16)),
                prefetch=1,
            )
        )


def test_temporal_asplit_matches_split() -> None:
    patcher = TemporalPatcher(
        geometry=TemporalFixedLookback(length=4),
        sampler=TemporalRegularStride(step=4),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalMean(),
    )
    series = np.arange(16)

    async def collect() -> list[Any]:
        return [patch async for patch in patcher.asplit(series)]

    sync_patches = list(patcher.split(series))
    async_patches = asyncio.run(collect())
    assert [p.anchor for p in async_patches] == [p.anchor for p in sync_patches]


def test_batch_split_pads_last_batch_and_unbatches(
    field: ArrayField, patcher: SpatialPatcher
) -> None:
    batches = list(batch_split(patcher, field, batch_size=3))
    assert [batch.data.shape[0] for batch in batches] == [3, 3]
    np.testing.assert_array_equal(batches[-1].valid, [True, False, False])

    patches = [patch for batch in batches for patch in unbatch(batch)]
    assert [p.anchor for p in patches] == [p.anchor for p in patcher.split(field)]


@pytest.mark.parametrize("policy", ["skip", "mask", "retry"])
@pytest.mark.parametrize("cls", ["SpatialPatcher", "AsyncSpatialPatcher"])
def test_asplit_on_error_parity(policy: str, cls: str) -> None:
    """asplit applies the on_error policy exactly as split does (#188)."""
    import geopatcher

    array = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)

    class FlakyField(AsyncArrayField):
        def select(self, window: Any) -> np.ndarray:
            if (window.row_off, window.col_off) == (8, 0):
                raise OSError("tile unreadable")
            return super().select(window)

    def make(name: str) -> Any:
        return getattr(geopatcher, name)(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRegularStride(step=8),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error=policy,
            max_retries=1,
        )

    async_patcher, sync_patcher = make(cls), make("SpatialPatcher")
    sync_patches = list(sync_patcher.split(FlakyField(array)))

    async def collect() -> list[Any]:
        return [p async for p in async_patcher.asplit(FlakyField(array))]

    async_patches = asyncio.run(collect())
    assert [p.anchor for p in async_patches] == [p.anchor for p in sync_patches]
    for a, s in zip(async_patches, sync_patches, strict=True):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(s.data))
    assert async_patcher.errors
    assert [(e.anchor, e.kind, e.retry_count) for e in async_patcher.errors] == [
        (e.anchor, e.kind, e.retry_count) for e in sync_patcher.errors
    ]

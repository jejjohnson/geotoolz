"""Tests for async and parallel patching helpers."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import ModuleType
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


def test_spatial_to_delayed_builds_one_task_per_anchor(
    monkeypatch: pytest.MonkeyPatch, field: ArrayField, patcher: SpatialPatcher
) -> None:
    class DelayedCall:
        def __init__(self, fn: Any, args: tuple[Any, ...]) -> None:
            self.fn = fn
            self.args = args

    def delayed(fn: Any) -> Any:
        def wrapper(*args: Any) -> DelayedCall:
            return DelayedCall(fn, args)

        return wrapper

    fake_dask = ModuleType("dask")
    fake_dask.delayed = delayed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dask", fake_dask)

    tasks = patcher.to_delayed(field, operator=lambda patch: patch)
    assert len(tasks) == patcher.n_anchors(field)
    assert all(isinstance(task.args[0], DelayedCall) for task in tasks)


def test_batch_split_pads_last_batch_and_unbatches(
    field: ArrayField, patcher: SpatialPatcher
) -> None:
    batches = list(batch_split(patcher, field, batch_size=3))
    assert [batch.data.shape[0] for batch in batches] == [3, 3]
    np.testing.assert_array_equal(batches[-1].valid, [True, False, False])

    patches = [patch for batch in batches for patch in unbatch(batch)]
    assert [p.anchor for p in patches] == [p.anchor for p in patcher.split(field)]

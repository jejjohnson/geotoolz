"""`asplit` backpressure: releases land on the owning event loop (#195).

`asyncio` primitives are not thread-safe, yet a patch's release can run on
any thread — `Patch.close` from a worker, or the garbage-collection
finalizer wherever the last reference is dropped (e.g. the ``to_thread``
worker of an async ``merge``). Every release must be routed back onto the
loop that acquired the slot, and must be harmless after that loop closed.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from typing import Any

import numpy as np
import pytest
from _helpers import ArrayField

from geopatcher import AsyncSpatialPatcher, SpatialPatcher, spatial
from geopatcher._src.spatial import patcher as patcher_module


_CHIP_BYTES = 8 * 8 * 4  # one float32 8x8 patch


class _AsyncArrayField(ArrayField):
    async def aselect(self, window: Any) -> np.ndarray:
        await asyncio.sleep(0)
        return self.select(window)


class _RecordingSemaphore(asyncio.BoundedSemaphore):
    """Records the thread every ``release`` runs on."""

    release_threads: list[int] = []  # noqa: RUF012 - reset per test

    def release(self) -> None:
        _RecordingSemaphore.release_threads.append(threading.get_ident())
        super().release()


@pytest.fixture
def recording_slots(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    _RecordingSemaphore.release_threads = []
    monkeypatch.setattr(patcher_module, "AsyncBoundedSemaphore", _RecordingSemaphore)
    return _RecordingSemaphore.release_threads


def _field() -> _AsyncArrayField:
    return _AsyncArrayField(np.arange(16 * 16, dtype=np.float32).reshape(16, 16))


_PATCHER_KW: dict[str, Any] = {
    "geometry": spatial.geometry.Rectangular(size=(8, 8)),
    "sampler": spatial.sampler.RegularStride(step=8),
    "window": spatial.window.Boxcar(),
    "aggregation": spatial.aggregation.OverlapAdd(),
}

PATCHERS = pytest.mark.parametrize(
    "patcher",
    [SpatialPatcher(**_PATCHER_KW), AsyncSpatialPatcher(**_PATCHER_KW)],
    ids=["SpatialPatcher.asplit", "AsyncSpatialPatcher.asplit"],
)


@PATCHERS
@pytest.mark.parametrize(
    "limits",
    [{"max_in_flight": 1}, {"max_in_flight_bytes": _CHIP_BYTES}],
    ids=["slots", "bytes"],
)
def test_finalizer_release_on_worker_thread_lands_on_loop(
    patcher: Any, limits: dict[str, int], recording_slots: list[int]
) -> None:
    async def run() -> int:
        loop_thread = threading.get_ident()
        stream = patcher.asplit(_field(), **limits)
        first = await anext(stream)
        # Fire the patch's GC finalizer on a worker thread, exactly as a
        # collection triggered there would (the suspended generator frame
        # still references the patch, so a plain ``del`` would only fire
        # it later, on the loop). The producer needs the slot / bytes the
        # first patch owns, so this must wake the loop-side waiter.
        await asyncio.to_thread(first._release_finalizer)
        del first
        second = await asyncio.wait_for(anext(stream), timeout=5.0)
        second.close()
        await stream.aclose()
        return loop_thread

    loop_thread = asyncio.run(run())
    if "max_in_flight" in limits:
        assert recording_slots, "no slot was released"
        assert set(recording_slots) == {loop_thread}, (
            "asyncio semaphore released off its event loop"
        )


@PATCHERS
def test_close_on_worker_thread_lands_on_loop(
    patcher: Any, recording_slots: list[int]
) -> None:
    async def run() -> int:
        stream = patcher.asplit(_field(), max_in_flight=1)
        first = await anext(stream)
        await asyncio.to_thread(first.close)
        second = await asyncio.wait_for(anext(stream), timeout=5.0)
        second.close()
        await stream.aclose()
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert recording_slots
    assert set(recording_slots) == {loop_thread}


@PATCHERS
def test_release_after_loop_closed_is_harmless(patcher: Any) -> None:
    held: list[Any] = []

    async def grab() -> None:
        stream = patcher.asplit(_field(), max_in_flight=1, max_in_flight_bytes=10**6)
        held.append(await anext(stream))
        await stream.aclose()

    asyncio.run(grab())
    held[0].close()  # the loop is closed: dropped quietly, no RuntimeError
    held.clear()
    gc.collect()


@PATCHERS
def test_no_release_callback_without_limits(patcher: Any) -> None:
    async def collect() -> list[Any]:
        return [patch async for patch in patcher.asplit(_field())]

    patches = asyncio.run(collect())
    assert len(patches) == 4
    assert all(patch._release is None for patch in patches)

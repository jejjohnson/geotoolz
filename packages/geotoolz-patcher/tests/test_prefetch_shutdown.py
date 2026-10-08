"""Producer-thread shutdown tests for `prefetch_iterable`.

A consumer that abandons iteration (break / dropped iterator) must not
leave the producer thread blocked forever in a bounded ``Queue.put``.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest
from _helpers import ArrayField

from geopatcher import SpatialPatcher, spatial
from geopatcher._src.prefetch import (
    _SENTINEL,
    _STOP_POLL_S,
    _PrefetchIterator,
    prefetch_iterable,
)
from geopatcher._src.spatial import patcher as patcher_module


def _endless() -> Iterator[int]:
    i = 0
    while True:
        yield i
        i += 1


def _wait_dead(thread, deadline_s: float = 5.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if not thread.is_alive():
            return True
        time.sleep(0.02)
    return not thread.is_alive()


def test_normal_exhaustion_unchanged() -> None:
    assert list(prefetch_iterable(range(5), prefetch=2)) == [0, 1, 2, 3, 4]


def test_break_then_close_stops_producer() -> None:
    it = prefetch_iterable(_endless(), prefetch=2)
    assert isinstance(it, _PrefetchIterator)
    for i, _ in enumerate(it):
        if i == 3:
            break
    it.close()
    assert _wait_dead(it._thread), "producer thread still blocked after close()"


def test_close_is_idempotent_and_ends_iteration() -> None:
    it = prefetch_iterable(_endless(), prefetch=1)
    assert isinstance(it, _PrefetchIterator)
    next(it)
    it.close()
    it.close()
    assert list(it) == []  # next() after close raises StopIteration


def test_dropped_iterator_stops_producer_via_finalizer() -> None:
    it = prefetch_iterable(_endless(), prefetch=1)
    assert isinstance(it, _PrefetchIterator)
    next(it)
    thread = it._thread
    del it
    gc.collect()
    assert _wait_dead(thread), "producer thread leaked after iterator was dropped"


def test_exception_replay_still_joins_thread() -> None:
    def _boom() -> Iterator[int]:
        yield 1
        raise RuntimeError("boom")

    it = prefetch_iterable(_boom(), prefetch=2)
    assert isinstance(it, _PrefetchIterator)
    assert next(it) == 1
    try:
        next(it)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:  # pragma: no cover - failure path
        raise AssertionError("expected RuntimeError")
    assert _wait_dead(it._thread)


def test_close_closes_buffered_items() -> None:
    closed: list[int] = []

    class Item:
        def __init__(self, i: int) -> None:
            self.i = i

        def close(self) -> None:
            closed.append(self.i)

    it = prefetch_iterable((Item(i) for i in range(10)), prefetch=3)
    assert isinstance(it, _PrefetchIterator)
    first = next(it)
    assert _wait_until(lambda: it._queue.full())
    it.close()
    assert _wait_dead(it._thread)
    assert first.i not in closed  # the consumer's item is its own business
    # The three buffered items are closed, plus (timing-dependent) the one
    # the producer was holding when it noticed the stop flag.
    assert {1, 2, 3} <= set(closed) <= {1, 2, 3, 4}
    assert it._queue.empty() or list(it._queue.queue) == [_SENTINEL]


# --- patcher backpressure + prefetch ------------------------------------


class _CountingSemaphore(threading.BoundedSemaphore):
    """`BoundedSemaphore` that tracks the number of slots held right now."""

    instances: ClassVar[list[_CountingSemaphore]] = []

    def __init__(self, value: int = 1) -> None:
        super().__init__(value)
        self.held = 0
        self._count_lock = threading.Lock()
        _CountingSemaphore.instances.append(self)

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        ok = super().acquire(blocking, timeout)
        if ok:
            with self._count_lock:
                self.held += 1
        return ok

    def release(self, n: int = 1) -> None:
        super().release(n)
        with self._count_lock:
            self.held -= n


@pytest.fixture
def counting_slots(monkeypatch: pytest.MonkeyPatch) -> list[_CountingSemaphore]:
    _CountingSemaphore.instances = []
    monkeypatch.setattr(patcher_module, "BoundedSemaphore", _CountingSemaphore)
    return _CountingSemaphore.instances


def _patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(4, 4)),
        sampler=spatial.sampler.RegularStride(step=4),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


def _field() -> ArrayField:
    return ArrayField(np.arange(16 * 16, dtype=np.float32).reshape(16, 16))


def _wait_until(predicate, deadline_s: float = 5.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _start_blocked_split(counting_slots: list[_CountingSemaphore]):
    """``split(prefetch=4, max_in_flight=2)`` with the producer slot-blocked.

    The consumer holds patch 1, patch 2 sits in the prefetch queue (owning
    the second slot) and the producer has read patch 3 and is blocked
    waiting for a slot.
    """
    field = _field()
    it = _patcher().split(field, prefetch=4, max_in_flight=2)
    assert isinstance(it, _PrefetchIterator)
    held = next(it)
    (slots,) = counting_slots
    assert _wait_until(lambda: slots.held == 2 and it._queue.qsize() == 1)
    time.sleep(3 * _STOP_POLL_S)  # producer is now parked on the third read
    assert slots.held == 2 and it._queue.qsize() == 1
    return weakref.ref(field), it, held, slots


def test_close_releases_queued_slots(
    counting_slots: list[_CountingSemaphore],
) -> None:
    """Early break + ``close()`` stops the producer and frees queued slots (#195).

    The producer used to stay blocked forever in the plain semaphore
    ``acquire()``, pinning the field, the generator and the queued patch
    (and its slot) until the consumer released its own patch.
    """
    field_ref, it, held, slots = _start_blocked_split(counting_slots)
    queued = list(it._queue.queue)
    thread = it._thread
    it.close()
    assert _wait_dead(thread), "producer thread still blocked after close()"
    # Only the consumer's patch still owns a slot; the buffered one was closed.
    assert slots.held == 1
    assert all(patch._release is None for patch in queued)
    del it, queued
    gc.collect()
    assert field_ref() is None, "abandoned split still pins the field"
    held.close()
    assert slots.held == 0


def test_dropped_split_iterator_stops_producer_and_frees_slots(
    counting_slots: list[_CountingSemaphore],
) -> None:
    """Generator finalisation (dropping the iterator) is enough too."""
    field_ref, it, held, slots = _start_blocked_split(counting_slots)
    thread = it._thread
    del it
    gc.collect()
    assert _wait_dead(thread), "producer thread leaked after iterator was dropped"
    assert _wait_until(lambda: slots.held == 1)
    gc.collect()
    assert field_ref() is None
    held.close()
    assert slots.held == 0


def test_close_interrupts_byte_budget_wait() -> None:
    """The byte-budget condition wait re-checks the stop flag as well."""
    chip = 4 * 4 * 4  # float32 4x4
    it = _patcher().split(_field(), prefetch=4, max_in_flight_bytes=2 * chip)
    assert isinstance(it, _PrefetchIterator)
    held = next(it)
    assert _wait_until(lambda: it._queue.qsize() == 1)
    time.sleep(3 * _STOP_POLL_S)
    thread = it._thread
    it.close()
    assert _wait_dead(thread), "producer stuck in the byte-budget wait"
    held.close()

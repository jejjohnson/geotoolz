"""Producer-thread shutdown tests for `prefetch_iterable`.

A consumer that abandons iteration (break / dropped iterator) must not
leave the producer thread blocked forever in a bounded ``Queue.put``.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator

from geopatcher._src.prefetch import _PrefetchIterator, prefetch_iterable


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

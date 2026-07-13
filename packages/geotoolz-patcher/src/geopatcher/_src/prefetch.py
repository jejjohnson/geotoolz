"""Thread-backed prefetching for synchronous patch iterators."""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread
from typing import Any


_SENTINEL = object()

#: How often (seconds) the producer re-checks the stop flag while the
#: queue is full. Bounds how long an abandoned producer thread lingers.
_STOP_POLL_S = 0.05


@dataclass(frozen=True)
class _Raised:
    exc: BaseException


def prefetch_iterable[T](iterable: Iterable[T], prefetch: int) -> Iterator[T]:
    """Return ``iterable`` with up to ``prefetch`` items read ahead."""
    if prefetch < 0:
        raise ValueError("prefetch must be >= 0")
    if prefetch == 0:
        return iter(iterable)
    return _PrefetchIterator(iterable, prefetch)


class _PrefetchIterator[T](Iterator[T]):
    """Bounded-queue read-ahead over ``iterable`` on a daemon thread.

    The producer never blocks forever on a full queue: every ``put`` is
    a short-timeout retry loop that checks a stop event. A consumer that
    abandons iteration can call `close` (deterministic), or simply drop
    the iterator — a `weakref.finalize` sets the stop event on
    collection — and the producer thread exits promptly instead of
    staying blocked in ``Queue.put`` forever.
    """

    def __init__(self, iterable: Iterable[T], prefetch: int) -> None:
        self._queue: Queue[T | _Raised | object] = Queue(maxsize=prefetch)
        self._stop = Event()
        self._thread = Thread(
            target=_produce,
            args=(iter(iterable), self._queue, self._stop),
            daemon=True,
        )
        # Safety net: dropping the iterator without exhausting it must
        # stop the producer. The finalizer holds no reference back to
        # ``self`` (only to the Event's bound method), so it cannot keep
        # the iterator alive.
        self._finalizer = weakref.finalize(self, self._stop.set)
        self._thread.start()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._stop.is_set():
            raise StopIteration
        item = self._queue.get()
        if item is _SENTINEL:
            self._thread.join(timeout=1.0)
            raise StopIteration
        if isinstance(item, _Raised):
            self._thread.join(timeout=1.0)
            raise item.exc
        return item

    def close(self) -> None:
        """Stop the producer thread and end iteration.

        Idempotent. After ``close`` the iterator raises ``StopIteration``
        on the next ``next()`` call. The join is best-effort (bounded):
        a producer blocked inside the *upstream* iterator's ``__next__``
        cannot be interrupted, but it is a daemon thread and will exit
        as soon as that read returns.
        """
        self._stop.set()
        # Unblock a consumer concurrently waiting in ``Queue.get``.
        with contextlib.suppress(Full):
            self._queue.put_nowait(_SENTINEL)
        self._thread.join(timeout=1.0)


def _produce(iterator: Iterator[Any], queue: Queue[Any], stop: Event) -> None:
    """Producer loop: forward items until exhaustion, error, or stop."""
    try:
        for item in iterator:
            if not _put_until_stopped(queue, item, stop):
                return
    except BaseException as exc:
        _put_until_stopped(queue, _Raised(exc), stop)
    finally:
        _put_until_stopped(queue, _SENTINEL, stop)


def _put_until_stopped(queue: Queue[Any], item: Any, stop: Event) -> bool:
    """``queue.put`` that gives up once ``stop`` is set.

    Returns:
        ``True`` if the item was enqueued, ``False`` if the stop event
        fired first (the consumer abandoned iteration).
    """
    while not stop.is_set():
        try:
            queue.put(item, timeout=_STOP_POLL_S)
            return True
        except Full:
            continue
    return False

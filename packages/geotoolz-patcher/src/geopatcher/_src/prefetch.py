"""Thread-backed prefetching for synchronous patch iterators."""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any


_SENTINEL = object()

#: How often (seconds) a producer re-checks the stop flag while it waits
#: (a full queue here, a backpressure slot / byte budget in the patcher).
#: Bounds how long an abandoned producer thread lingers.
_STOP_POLL_S = 0.05


@dataclass(frozen=True)
class _Raised:
    exc: BaseException


def prefetch_iterable[T](
    iterable: Iterable[T], prefetch: int, *, stop: Event | None = None
) -> Iterator[T]:
    """Return ``iterable`` with up to ``prefetch`` items read ahead.

    Args:
        iterable: The upstream items, read on a background thread.
        prefetch: Read-ahead depth; ``0`` returns ``iter(iterable)``.
        stop: Optional stop event shared with the upstream iterator. The
            prefetch iterator sets it on ``close()`` / collection, so an
            upstream generator blocked in its own interruptible wait (e.g.
            the patcher's backpressure) can notice and return.
    """
    if prefetch < 0:
        raise ValueError("prefetch must be >= 0")
    if prefetch == 0:
        return iter(iterable)
    return _PrefetchIterator(iterable, prefetch, stop=stop)


class _PrefetchIterator[T](Iterator[T]):
    """Bounded-queue read-ahead over ``iterable`` on a daemon thread.

    The producer never blocks forever on a full queue: every ``put`` is
    a short-timeout retry loop that checks a stop event. A consumer that
    abandons iteration can call `close` (deterministic), or simply drop
    the iterator — a `weakref.finalize` sets the stop event on
    collection — and the producer thread exits promptly instead of
    staying blocked in ``Queue.put`` forever.

    `close` also closes the items still buffered (when they have a
    ``close()``), so patches holding backpressure slots hand them back
    instead of waiting for the garbage collector; a dropped iterator's
    buffered items are released by their own finalizers once the producer
    thread has exited.
    """

    def __init__(
        self, iterable: Iterable[T], prefetch: int, *, stop: Event | None = None
    ) -> None:
        self._queue: Queue[T | _Raised | object] = Queue(maxsize=prefetch)
        self._stop = Event() if stop is None else stop
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
        """Stop the producer thread, close buffered items, end iteration.

        Idempotent. After ``close`` the iterator raises ``StopIteration``
        on the next ``next()`` call. Items already read ahead are drained
        from the queue and closed, releasing any backpressure slots they
        own. The join is best-effort (bounded): a producer blocked inside
        the *upstream* iterator's ``__next__`` cannot be interrupted
        unless that iterator watches the shared stop event, but it is a
        daemon thread and will exit as soon as that read returns (and it
        closes the item it was holding rather than enqueueing it).
        """
        self._stop.set()
        self._thread.join(timeout=1.0)
        while True:
            try:
                _close_item(self._queue.get_nowait())
            except Empty:
                break
        # Unblock a consumer concurrently waiting in ``Queue.get``.
        with contextlib.suppress(Full):
            self._queue.put_nowait(_SENTINEL)


def _produce(iterator: Iterator[Any], queue: Queue[Any], stop: Event) -> None:
    """Producer loop: forward items until exhaustion, error, or stop."""
    try:
        for item in iterator:
            if not _put_until_stopped(queue, item, stop):
                _close_item(item)
                return
    except BaseException as exc:
        _put_until_stopped(queue, _Raised(exc), stop)
    finally:
        if stop.is_set():
            # Abandoned: finalise the upstream generator here, on the
            # thread that owns it, so its ``finally`` blocks run now.
            close = getattr(iterator, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()
        _put_until_stopped(queue, _SENTINEL, stop)


def _close_item(item: Any) -> None:
    """Close a buffered item that will never reach the consumer."""
    if item is _SENTINEL or isinstance(item, _Raised):
        return
    close = getattr(item, "close", None)
    if close is not None:
        close()


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

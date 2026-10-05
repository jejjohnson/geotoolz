"""The anchor-walk core every patcher's ``split`` / ``asplit`` runs on.

One loop — `walk` (sync) and its twin `awalk` (async) — turns anchors into
patches for `SpatialPatcher`, `AsyncSpatialPatcher`, `TemporalPatcher`
and both `SpatioTemporalPatcher` couplings, so the read pipeline is the
same on every path:

- the ``on_error`` / ``max_retries`` / ``retry_on`` /
  ``capture_traceback`` policy (`ReadPolicy`), recording each failure as a
  `PatchErrorRecord`;
- a `PatchCache` consulted before every read and filled after a
  successful one (never with a ``"mask"`` placeholder);
- a `PatchJournal` whose keys are skipped (reported through the
  ``on_patch_skipped`` hook);
- the ``max_in_flight`` / ``max_in_flight_bytes`` backpressure, whose slot
  each yielded patch owns until ``patch.close()``;
- the hook lifecycle: ``on_split_start`` / ``on_split_end`` around the
  walk, ``on_patch_start`` / ``on_patch_done`` per patch, and
  ``on_error`` with the original exception, whether it is raised or
  swallowed by the policy.

A patcher describes each anchor as one or more `_Read` units; a unit
either yields its patch (a *leaf*) or, once read, expands into child units
(a *parent*: the spatio-temporal split reads one spatial chip, then slices
it into its time windows).
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Any, Literal

from geopatcher._src._serialize import qualified_name
from geopatcher._src.hooks import PatcherHook, _as_hooks, _dispatch, _nbytes


OnErrorPolicy = Literal["raise", "skip", "mask", "retry"]


@dataclass(eq=False)
class PatchErrorRecord:
    """A failed patch read recorded by a patcher's ``split``.

    Args:
        anchor: Key of the patch that failed to build — the anchor, or the
            patch key (``(anchor, k)``, a ``(space, time)`` pair) where a
            patcher keys its patches more finely.
        kind: Exception class name.
        message: Exception message.
        traceback: Formatted traceback for debugging.
        retry_count: Number of retries already attempted for this failure.
    """

    anchor: Any
    kind: str
    message: str
    traceback: str
    retry_count: int


def _validate_error_policy(on_error: str, max_retries: int) -> None:
    if on_error not in ("raise", "skip", "mask", "retry"):
        raise ValueError(
            "invalid on_error policy "
            f"{on_error!r}; expected 'raise', 'skip', 'mask', or 'retry'"
        )
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")


def _validate_retry_on(
    retry_on: Iterable[type[BaseException] | str],
) -> tuple[type[BaseException] | str, ...]:
    """Coerce ``retry_on`` to a tuple, rejecting entries that never match."""
    out = tuple(retry_on)
    for candidate in out:
        if isinstance(candidate, str) or (
            isinstance(candidate, type) and issubclass(candidate, BaseException)
        ):
            continue
        raise TypeError(
            "retry_on entries must be exception classes or their names, got "
            f"{candidate!r}."
        )
    return out


def _matches_retry_on(
    exc: BaseException, retry_on: tuple[type[BaseException] | str, ...]
) -> bool:
    """Whether ``exc`` should be retried.

    A class matches by ``isinstance``. A string matches any class in the
    exception's MRO by bare ``__name__`` (``"OSError"``) or by
    ``module.qualname`` (``"rasterio.errors.RasterioIOError"``) — so a
    name keeps the subclass semantics of the class it names, and the
    qualified names `get_config` emits reload with unchanged behaviour.
    """
    names: set[str] | None = None
    for candidate in retry_on:
        if isinstance(candidate, str):
            if names is None:
                names = {
                    name
                    for cls in type(exc).__mro__
                    for name in (cls.__name__, qualified_name(cls))
                }
            if candidate in names:
                return True
        elif isinstance(exc, candidate):
            return True
    return False


def _record_patch_error(
    errors: list[PatchErrorRecord],
    anchor: Any,
    exc: Exception,
    retry_count: int,
    capture_traceback: bool = True,
) -> None:
    tb = "".join(traceback.format_exception(exc)) if capture_traceback else ""
    errors.append(
        PatchErrorRecord(
            anchor=anchor,
            kind=type(exc).__name__,
            message=str(exc),
            traceback=tb,
            retry_count=retry_count,
        )
    )


_Action = Literal["raise", "mask", "retry", "skip"]


@dataclass(frozen=True, eq=False)
class ReadPolicy:
    """A patcher's ``on_error`` knobs, bound to the ``errors`` list of one split.

    Args:
        on_error: ``"raise"``, ``"skip"``, ``"mask"`` or ``"retry"`` (see
            `SpatialPatcher`).
        max_retries: Retries per read under ``"retry"``.
        retry_on: Exception classes / names worth retrying.
        capture_traceback: Format a traceback into each record.
        errors: Where failures are recorded.
    """

    on_error: OnErrorPolicy = "raise"
    max_retries: int = 0
    retry_on: tuple[type[BaseException] | str, ...] = (OSError, TimeoutError)
    capture_traceback: bool = True
    errors: list[PatchErrorRecord] | None = None

    @classmethod
    def of(cls, patcher: Any) -> ReadPolicy:
        """The policy of ``patcher``, recording into its current ``errors``."""
        return cls(
            on_error=patcher.on_error,
            max_retries=patcher.max_retries,
            retry_on=patcher.retry_on,
            capture_traceback=patcher.capture_traceback,
            errors=patcher.errors,
        )

    def _failed(self, exc: Exception, key: Any, retry_count: int) -> _Action:
        """Record a failed attempt (unless raising) and decide what follows."""
        # Preserve KeyboardInterrupt/SystemExit: callers handle only Exception.
        if isinstance(exc, StopIteration) or self.on_error == "raise":
            return "raise"
        if self.errors is not None:
            _record_patch_error(
                self.errors, key, exc, retry_count, self.capture_traceback
            )
        if self.on_error == "mask":
            return "mask"
        if self.on_error == "retry":
            if not _matches_retry_on(exc, self.retry_on):
                return "raise"
            if retry_count < self.max_retries:
                return "retry"
        return "skip"

    def read[T](
        self,
        read: Callable[[], T],
        *,
        mask: Callable[[], T] | None,
        key: Any,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> tuple[T | None, bool]:
        """Run one ``read`` under the policy.

        Returns ``(patch, read_ok)``: ``read_ok`` is ``True`` only when
        ``read`` itself succeeded, so a ``"mask"`` placeholder or a
        skipped ``None`` is never stored in a `PatchCache`. Every
        swallowed failure is recorded under ``key`` and handed to
        ``on_failure`` (the ``on_error`` hooks) with the original
        exception; a failure the policy re-raises is left to the caller.
        The mask placeholder is built outside the ``except`` block, so an
        error building it is not chained onto the read failure.
        """
        retries = self.max_retries if self.on_error == "retry" else 0
        for retry_count in range(retries + 1):
            try:
                return read(), True
            except Exception as exc:
                action = self._failed(exc, key, retry_count)
                if action == "raise":
                    raise
                failure = exc
            if on_failure is not None:
                on_failure(failure)
            if action == "mask":
                return (None if mask is None else mask()), False
            if action == "skip":
                return None, False
        return None, False

    async def aread[T](
        self,
        read: Callable[[], Any],
        *,
        mask: Callable[[], T] | None,
        key: Any,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> tuple[T | None, bool]:
        """Async `read`: ``read()`` may return an awaitable."""
        retries = self.max_retries if self.on_error == "retry" else 0
        for retry_count in range(retries + 1):
            try:
                return await _resolve(read()), True
            except Exception as exc:
                action = self._failed(exc, key, retry_count)
                if action == "raise":
                    raise
                failure = exc
            if on_failure is not None:
                on_failure(failure)
            if action == "mask":
                return (None if mask is None else mask()), False
            if action == "skip":
                return None, False
        return None, False


async def _resolve(value: Any) -> Any:
    """``await value`` when it is awaitable, else ``value``."""
    if inspect.isawaitable(value):
        return await value
    return value


# -- split control (the semaphores live in `spatial.patcher`) --------------


def _validate_backpressure(
    max_in_flight: int | None, max_in_flight_bytes: int | None
) -> None:
    if max_in_flight is not None and max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if max_in_flight_bytes is not None and max_in_flight_bytes < 1:
        raise ValueError("max_in_flight_bytes must be >= 1")


class _SplitCancelled(Exception):
    """Raised by `_Backpressure.acquire` once the consumer abandoned `split`."""


# -- the walk ----------------------------------------------------------------


@dataclass(eq=False)
class _Read:
    """One unit of a walk.

    A *leaf* (``expand`` is ``None``) yields the patch ``read`` builds. A
    *parent* is read the same way and then ``expand(patch)`` lists the
    child units it yields instead (each spatio-temporal chip expands
    into its time windows).

    Args:
        key: Journal / cache / ``errors`` key.
        anchor: The anchor hooks receive.
        read: Builds the patch (may return an awaitable under `awalk`).
        mask: Builds the ``on_error="mask"`` placeholder.
        coord_value: Trailing hook payload (`TemporalPatcher` coords).
        cached: Returns a `PatchCache` hit, or ``None`` on a miss.
        store: Stores a successfully read patch in the cache.
        expand: Makes this a parent — child units of the read patch.
        planned: For a parent: its children's ``(key, anchor,
            coord_value)`` when knowable without reading (``None``
            otherwise); a parent whose children are all journaled is
            skipped without being read.
        governed: Whether ``read`` runs under the `ReadPolicy`. Pure
            post-processing (slicing a chip already read) is not: its
            errors always raise.
    """

    key: Any
    anchor: Any
    read: Callable[[], Any]
    mask: Callable[[], Any] | None = None
    coord_value: Any = None
    cached: Callable[[], Any] | None = None
    store: Callable[[Any], None] | None = None
    expand: Callable[[Any], Iterable[_Read]] | None = None
    planned: Callable[[], list[tuple[Any, Any, Any]] | None] | None = None
    governed: bool = True


class _Walker:
    """The per-split state shared by `walk` / `awalk`'s units."""

    def __init__(
        self,
        policy: ReadPolicy,
        hooks: tuple[PatcherHook, ...],
        journal: Any | None,
    ) -> None:
        self.policy = policy
        self.hooks = hooks
        self.journal = journal

    def error(self, anchor: Any, exc: Exception) -> None:
        _dispatch(self.hooks, "on_error", anchor, exc)

    def journaled(self, r: _Read) -> bool:
        """Skip ``r`` when the journal holds it (all its children, for a parent)."""
        if self.journal is None:
            return False
        if r.expand is None:
            if not self.journal.has(r.key):
                return False
            _dispatch(self.hooks, "on_patch_skipped", r.anchor, r.coord_value)
            return True
        planned = None if r.planned is None else r.planned()
        if not planned or not all(self.journal.has(key) for key, _, _ in planned):
            return False
        for _, anchor, coord_value in planned:
            _dispatch(self.hooks, "on_patch_skipped", anchor, coord_value)
        return True

    def start(self, r: _Read) -> float:
        if not self.hooks:
            return 0.0
        _dispatch(self.hooks, "on_patch_start", r.anchor, r.coord_value)
        return perf_counter()

    def done(self, r: _Read, patch: Any, start: float) -> None:
        if self.hooks:
            _dispatch(
                self.hooks,
                "on_patch_done",
                r.anchor,
                perf_counter() - start,
                _nbytes(patch.data),
                r.coord_value,
            )

    def read(self, r: _Read) -> Any | None:
        """``r``'s patch: a cache hit, else a read under the policy."""
        try:
            patch = None if r.cached is None else r.cached()
            if patch is not None:
                return patch
            if r.governed and self.policy.on_error != "raise":
                patch, ok = self.policy.read(
                    r.read,
                    mask=r.mask,
                    key=r.key,
                    on_failure=partial(self.error, r.anchor),
                )
            else:
                patch, ok = r.read(), True
            if ok and r.store is not None:
                r.store(patch)
            return patch
        except Exception as exc:
            self.error(r.anchor, exc)
            raise

    async def aread(self, r: _Read) -> Any | None:
        """Async `read`; the cache is consulted in a worker thread."""
        try:
            patch = None if r.cached is None else await asyncio.to_thread(r.cached)
            if patch is not None:
                return patch
            if r.governed and self.policy.on_error != "raise":
                patch, ok = await self.policy.aread(
                    r.read,
                    mask=r.mask,
                    key=r.key,
                    on_failure=partial(self.error, r.anchor),
                )
            else:
                patch, ok = await _resolve(r.read()), True
            if ok and r.store is not None:
                await asyncio.to_thread(r.store, patch)
            return patch
        except Exception as exc:
            self.error(r.anchor, exc)
            raise

    def children(self, r: _Read, patch: Any) -> Iterable[_Read]:
        assert r.expand is not None
        try:
            return list(r.expand(patch))
        except Exception as exc:
            self.error(r.anchor, exc)
            raise

    def units(self, expand: Callable[[Any], Iterable[_Read]], anchor: Any) -> list:
        try:
            return list(expand(anchor))
        except Exception as exc:
            self.error(anchor, exc)
            raise

    def leaves(self, reads: Iterable[_Read], backpressure: Any) -> Iterator[Any]:
        for r in reads:
            if self.journaled(r):
                continue
            if r.expand is not None:
                parent = self.read(r)
                if parent is not None:
                    yield from self.leaves(self.children(r, parent), backpressure)
                continue
            start = self.start(r)
            patch = self.read(r)
            if patch is None:
                continue
            release = backpressure.acquire(patch)
            if release is not None:
                # Attach ownership in-place so the yielded patch releases
                # the exact slot acquired for this read.
                patch._release = release
            self.done(r, patch, start)
            yield patch

    async def aleaves(
        self, reads: Iterable[_Read], backpressure: Any
    ) -> AsyncIterator[Any]:
        for r in reads:
            if self.journaled(r):
                continue
            if r.expand is not None:
                parent = await self.aread(r)
                if parent is not None:
                    async for patch in self.aleaves(
                        self.children(r, parent), backpressure
                    ):
                        yield patch
                continue
            start = self.start(r)
            patch = await self.aread(r)
            if patch is None:
                continue
            release = await backpressure.acquire(patch)
            if release is not None:
                patch._release = release
            self.done(r, patch, start)
            yield patch


def walk(
    anchors: Iterable[Any],
    expand: Callable[[Any], Iterable[_Read]],
    *,
    policy: ReadPolicy,
    backpressure: Any,
    hooks: Iterable[PatcherHook] | None = None,
    journal: Any | None = None,
    total: Callable[[list[Any]], int] = len,
) -> Iterator[Any]:
    """Walk ``anchors``, yielding the patches of each one's `_Read` units.

    ``expand(anchor)`` lists an anchor's units; an error there reaches the
    ``on_error`` hooks with ``anchor`` and propagates. With hooks the
    anchors are materialised first so ``on_split_start`` gets
    ``total(anchors)``; without, they stay lazy. The walk ends quietly
    when the consumer abandons a backpressured (prefetched) split.
    """
    hook_list = _as_hooks(hooks)
    walker = _Walker(policy, hook_list, journal)
    if hook_list:
        anchors = list(anchors)
        _dispatch(hook_list, "on_split_start", total(anchors))
    try:
        for anchor in anchors:
            yield from walker.leaves(walker.units(expand, anchor), backpressure)
    except _SplitCancelled:
        return
    finally:
        if hook_list:
            _dispatch(hook_list, "on_split_end")


async def awalk(
    anchors: Iterable[Any],
    expand: Callable[[Any], Iterable[_Read]],
    *,
    policy: ReadPolicy,
    backpressure: Any,
    hooks: Iterable[PatcherHook] | None = None,
    journal: Any | None = None,
    total: Callable[[list[Any]], int] = len,
) -> AsyncIterator[Any]:
    """Async `walk`: units' ``read`` may return awaitables."""
    hook_list = _as_hooks(hooks)
    walker = _Walker(policy, hook_list, journal)
    if hook_list:
        anchors = list(anchors)
        _dispatch(hook_list, "on_split_start", total(anchors))
    try:
        for anchor in anchors:
            async for patch in walker.aleaves(
                walker.units(expand, anchor), backpressure
            ):
                yield patch
    finally:
        if hook_list:
            _dispatch(hook_list, "on_split_end")

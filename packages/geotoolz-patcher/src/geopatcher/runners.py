"""Reference runners for applying operators over patch streams."""

from __future__ import annotations

import multiprocessing
import pickle
import sys
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from multiprocessing.context import BaseContext
from time import perf_counter
from typing import Any, Literal

import numpy as np

from geopatcher._src.hooks import PatcherHook
from geopatcher._src.patch import Patch
from geopatcher._src.protocols import Field
from geopatcher._src.spatial.patcher import (
    SpatialPatcher,
    _pad_request,
    _raise_if_overflows,
    _unwrap_for_select,
)


Backend = Literal["thread", "process"]
ErrorPolicy = Literal["raise", "skip"]


def parallel_map(
    patcher: SpatialPatcher,
    field: Field,
    operator: Callable[[Any], Any],
    *,
    n_workers: int = 8,
    backend: Backend = "thread",
    mp_context: str | BaseContext | None = None,
    show_progress: bool = False,
    journal: Any | None = None,
    on_error: ErrorPolicy = "raise",
    batch_size: int = 64,
    hooks: Iterable[PatcherHook] | None = None,
    max_in_flight: int | None = None,
) -> list[Patch]:
    """Apply ``operator`` to each spatial patch with a reference executor.

    Reads happen in the calling process, through ``patcher.split`` — so
    the patcher's boundary mode (``pad`` / ``reflect`` / …), its
    ``on_error`` read policy (``patcher.errors`` is populated as usual),
    the ``journal`` and the ``hooks`` all apply exactly as they do to a
    plain ``split`` loop. Patches are submitted to the executor as they
    are read, never more than ``max_in_flight`` at a time, so at most
    ``max_in_flight`` patches (plus up to ``batch_size`` read-ahead chips
    on the ``select_many`` path) are resident at once. The returned
    outputs are, of course, all kept.

    Process backend: workers are started with an explicit
    `multiprocessing` start method — ``"forkserver"`` where available
    (Linux / macOS), ``"spawn"`` elsewhere — never the platform ``fork``
    default, which is unsafe in a process running GDAL / obstore / dask
    threads. The operator, every patch and every output are pickled to
    and from the workers, so ``operator`` must be a picklable,
    module-level callable. The field itself is never sent. `GeoTensor`
    chips and outputs travel in a wire form that keeps their transform /
    CRS / nodata / attrs (a bare GeoTensor pickle drops them).

    Args:
        patcher: Spatial patcher that defines the anchor schedule, the
            boundary mode and the read-error policy.
        field: Field to split into patches.
        operator: Callable applied to each patch's ``data``.
        n_workers: Number of worker threads or processes.
        backend: ``"thread"`` for `ThreadPoolExecutor` or ``"process"`` for
            `ProcessPoolExecutor`.
        mp_context: Start method for the process backend: a name
            (``"forkserver"``, ``"spawn"``, ``"fork"``) or a
            `multiprocessing` context. ``None`` (default) picks
            ``"forkserver"`` where available, else ``"spawn"``. Ignored by
            the thread backend.
        show_progress: If ``True``, print a lightweight completion counter
            to stderr.
        journal: Optional `PatchJournal`. Anchors it already holds an
            ``"ok"`` row for are skipped (no read, no operator call); every
            processed patch is committed with ``status="ok"`` or
            ``status="error"`` (plus the operator runtime and error text),
            so a crashed or interrupted run resumes where it stopped.
        on_error: Operator-error policy: ``"raise"`` to fail fast (queued
            patches are cancelled), or ``"skip"`` to warn and omit failed
            patches from the returned list. Read errors are governed by
            ``patcher.on_error``, not by this argument.
        batch_size: Maximum number of reads coalesced into one
            ``field.select_many`` call when the field supports it (e.g.
            `ObstoreCogField`); reads are fetched one chunk ahead of the
            patcher, so at most ``batch_size`` chips wait in the read-ahead
            buffer. No effect on fields without ``select_many``.
        hooks: `PatcherHook` instances forwarded to ``patcher.split`` —
            they observe the read path (``on_patch_done`` fires when a
            patch is read, before the operator runs).
        max_in_flight: Maximum number of patches submitted to the
            executor and not yet finished. Defaults to ``2 * n_workers``,
            enough to keep every worker busy while one batch of results is
            collected.

    Returns:
        Patches with ``data`` replaced by ``operator(patch.data)``, ordered by
        the patcher's anchor schedule.

    Examples:
        Thread pool over a raster, failing fast on any error::

            outputs = parallel_map(patcher, field, model, n_workers=8)

        Restartable process-pool job that tolerates bad tiles::

            patcher = SpatialPatcher(..., on_error="skip")
            journal = PatchJournal("out/run.jsonl")
            outputs = parallel_map(
                patcher, field, model,
                backend="process", journal=journal, on_error="skip",
            )

        Explicit start method::

            parallel_map(patcher, field, model, backend="process",
                         mp_context="spawn")
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if backend not in {"thread", "process"}:
        raise ValueError("backend must be 'thread' or 'process'")
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_in_flight is None:
        max_in_flight = 2 * n_workers
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if backend == "process":
        _ensure_picklable_operator(operator)

    # Materialise the schedule once: the batched path plans its reads from
    # it, the progress counter needs its length, and an unseeded sampler
    # must not place the patches differently from the plan.
    anchors = patcher.anchors(field)
    if journal is not None:
        anchors = [anchor for anchor in anchors if not journal.has(anchor)]

    # Duck-type for the batched-read fast path. Fields that implement
    # ``select_many`` (e.g. ``ObstoreCogField``) fetch a chunk of reads in
    # one coalesced request. The patcher still drives every read through
    # its own pipeline (boundary padding, ``on_error``), against a wrapper
    # whose ``select`` serves the prefetched chips.
    read_field: Any = field
    if callable(getattr(field, "select_many", None)):
        read_field = _BatchedReadField(field, patcher, anchors, batch_size)

    stream = patcher._split_anchors(read_field, anchors, hooks=hooks, journal=journal)
    with _make_executor(backend, n_workers, mp_context) as executor:
        results = _run(
            executor,
            stream,
            operator,
            to_process=backend == "process",
            max_in_flight=max_in_flight,
            journal=journal,
            on_error=on_error,
            total=len(anchors) if show_progress else None,
        )
    if show_progress:
        print(file=sys.stderr)
    return [patch for _, patch in sorted(results, key=lambda item: item[0])]


def _make_executor(
    backend: Backend, n_workers: int, mp_context: str | BaseContext | None
) -> Executor:
    if backend == "thread":
        return ThreadPoolExecutor(max_workers=n_workers)
    return ProcessPoolExecutor(
        max_workers=n_workers, mp_context=_resolve_mp_context(mp_context)
    )


def _resolve_mp_context(mp_context: str | BaseContext | None) -> BaseContext:
    """Explicit start method; ``None`` → forkserver where available, else spawn."""
    if isinstance(mp_context, BaseContext):
        return mp_context
    if mp_context is None:
        methods = multiprocessing.get_all_start_methods()
        mp_context = "forkserver" if "forkserver" in methods else "spawn"
    return multiprocessing.get_context(mp_context)


def _run(
    executor: Executor,
    stream: Iterator[Patch],
    operator: Callable[[Any], Any],
    *,
    to_process: bool,
    max_in_flight: int,
    journal: Any | None,
    on_error: ErrorPolicy,
    total: int | None,
) -> list[tuple[int, Patch]]:
    """Stream ``stream`` through ``executor`` with at most ``max_in_flight`` pending."""
    pending: dict[Future, tuple[int, Patch]] = {}
    results: list[tuple[int, Patch]] = []
    done = 0

    def collect(futures: Iterable[Future]) -> None:
        nonlocal done
        for future in futures:
            index, original = pending.pop(future)
            try:
                _, output, runtime_s = future.result()
                if to_process:
                    output = output.with_data(_from_wire(output.data))
            except Exception as exc:
                if journal is not None:
                    journal.commit(
                        original.anchor,
                        status="error",
                        runtime_s=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if on_error == "raise":
                    raise
                warnings.warn(
                    f"parallel_map skipped patch {index} after operator error: {exc}",
                    RuntimeWarning,
                    stacklevel=4,
                )
            else:
                if journal is not None:
                    journal.commit(original.anchor, status="ok", runtime_s=runtime_s)
                results.append((index, output))
            finally:
                # `Patch.close` runs any backpressure release exactly once,
                # whether the worker raised or returned.
                original.close()
            done += 1
            if total is not None:
                print(f"\r{done}/{total}", end="", file=sys.stderr)

    try:
        for index, patch in enumerate(stream):
            while len(pending) >= max_in_flight:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                collect(finished)
            if to_process:
                # `with_data` drops the (unpicklable) release closure — the
                # parent keeps it on `patch` and runs it in `collect` — and
                # the wire form keeps a GeoTensor's georeferencing.
                payload = patch.with_data(_to_wire(patch.data))
                future = executor.submit(_apply_operator_wire, index, payload, operator)
            else:
                future = executor.submit(_apply_operator, index, patch, operator)
            pending[future] = (index, patch)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect(finished)
    except BaseException:
        # Fail fast: drop queued work instead of running every submitted
        # patch to completion on the way out.
        executor.shutdown(wait=True, cancel_futures=True)
        for _, patch in pending.values():
            patch.close()
        raise
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return results


def _apply_operator(
    index: int, patch: Patch, operator: Callable[[Any], Any]
) -> tuple[int, Patch, float]:
    start = perf_counter()
    data = operator(patch.data)
    # `with_data` drops the release closure, so the output never owns (and
    # can never double-release) the input's backpressure slot.
    return index, patch.with_data(data), perf_counter() - start


def _apply_operator_wire(
    index: int, patch: Patch, operator: Callable[[Any], Any]
) -> tuple[int, Patch, float]:
    """Process-pool twin of `_apply_operator`: unwraps / rewraps `_WireGeoTensor`."""
    index, output, runtime_s = _apply_operator(
        index, patch.with_data(_from_wire(patch.data)), operator
    )
    return index, output.with_data(_to_wire(output.data)), runtime_s


class _WireGeoTensor:
    """Picklable stand-in for a `GeoTensor`.

    georeader's `GeoTensor` is an ndarray subclass whose pickle drops the
    transform / CRS / nodata / attrs, so a chip (or output) sent through a
    process pool would come back ungeoreferenced. This carries the plain
    values plus the metadata, and `_from_wire` rebuilds the `GeoTensor`.
    """

    __slots__ = ("attrs", "crs", "fill_value_default", "transform", "values")

    def __init__(self, data: Any) -> None:
        self.values = np.asarray(data)
        self.transform = data.transform
        self.crs = data.crs
        self.fill_value_default = data.fill_value_default
        self.attrs = dict(getattr(data, "attrs", None) or {})

    def __getstate__(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__slots__)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        for name, value in zip(self.__slots__, state, strict=True):
            setattr(self, name, value)

    def rebuild(self) -> Any:
        from georeader.geotensor import GeoTensor

        return GeoTensor(
            values=self.values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )


def _to_wire(data: Any) -> Any:
    try:
        from georeader.geotensor import GeoTensor
    except ImportError:  # pragma: no cover - georeader is a core dependency
        return data
    return _WireGeoTensor(data) if isinstance(data, GeoTensor) else data


def _from_wire(data: Any) -> Any:
    return data.rebuild() if isinstance(data, _WireGeoTensor) else data


def _ensure_picklable_operator(operator: Callable[[Any], Any]) -> None:
    try:
        pickle.dumps(operator)
    except Exception as exc:
        raise TypeError(
            "parallel_map(..., backend='process') requires a picklable operator; "
            "use a top-level function, use backend='thread', or wrap your "
            "operator with a cloudpickle-based runner."
        ) from exc


class _BatchedReadField:
    """Serve ``select`` from chunked ``select_many`` reads, one chunk ahead.

    Planned from the anchor schedule: for every anchor the exact indexer
    the patcher will hand to ``select`` (the clipped / inward-grown source
    window under ``boundary="pad"`` / ``"reflect"``, the unwrapped window
    otherwise). When the patcher asks for a planned indexer that has not
    been fetched yet, the next ``batch_size`` planned reads are fetched in
    one ``select_many`` call. Anything not served from the buffer — an
    unplanned indexer, a retry of an already-served read, a chunk whose
    ``select_many`` raised — falls back to the real ``field.select``, so
    every failure surfaces per patch, inside the patcher's ``on_error``
    policy.

    Single-threaded: driven by the one ``split`` generator in the parent.
    """

    def __init__(
        self,
        real: Any,
        patcher: SpatialPatcher,
        anchors: list[Any],
        batch_size: int,
    ) -> None:
        self._real = real
        self._batch_size = batch_size
        domain = real.domain
        self._plan = [_planned_indexer(patcher, domain, anchor) for anchor in anchors]
        self._keys = [None if ix is None else _indexer_key(ix) for ix in self._plan]
        # key → planned positions not yet fetched (in schedule order).
        self._unfetched: dict[Any, deque[int]] = {}
        for pos, key in enumerate(self._keys):
            if key is not None:
                self._unfetched.setdefault(key, deque()).append(pos)
        self._cursor = 0
        self._buffer: dict[Any, deque[Any]] = {}

    @property
    def domain(self) -> Any:
        return self._real.domain

    @property
    def buffered(self) -> int:
        """Number of fetched chips waiting to be served."""
        return sum(len(chips) for chips in self._buffer.values())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def select(self, indexer: Any) -> Any:
        key = _indexer_key(indexer)
        if key is not None:
            positions = self._unfetched.get(key)
            while positions and not self._buffer.get(key):
                self._fetch_next_chunk()
            chips = self._buffer.get(key)
            if chips:
                chip = chips.popleft()
                if not chips:
                    del self._buffer[key]
                return chip
        return self._real.select(indexer)

    def _fetch_next_chunk(self) -> None:
        stop = min(self._cursor + self._batch_size, len(self._plan))
        positions = [p for p in range(self._cursor, stop) if self._keys[p] is not None]
        self._cursor = stop
        for pos in positions:
            self._unfetched[self._keys[pos]].popleft()
        if not positions:
            return
        try:
            chips = self._real.select_many([self._plan[p] for p in positions])
        except Exception:
            # Leave the chunk unbuffered: each read falls back to a
            # per-patch `select`, where the patcher's policy scopes the
            # failure to the patch that actually fails.
            return
        if len(chips) != len(positions):
            raise RuntimeError(
                f"field.select_many returned {len(chips)} arrays for "
                f"{len(positions)} indices; expected one array per indexer."
            )
        for pos, chip in zip(positions, chips, strict=True):
            self._buffer.setdefault(self._keys[pos], deque()).append(chip)


def _planned_indexer(patcher: SpatialPatcher, domain: Any, anchor: Any) -> Any:
    """The indexer ``patcher.split`` will pass to ``field.select`` for ``anchor``.

    ``None`` when the patcher will not read (``boundary="raise"`` overflow,
    a window with no in-domain cell, an un-mirrorable reflect axis) — the
    patcher raises on its own path, under its own ``on_error``.
    """
    boundary = getattr(patcher.geometry, "boundary", "drop")
    try:
        indices = patcher.geometry.neighborhood(domain, anchor)
        if boundary == "raise":
            _raise_if_overflows(indices, domain)
        window = _unwrap_for_select(indices)
        if boundary not in ("pad", "reflect"):
            return window
        request = _pad_request(window, domain)
        if request is None:
            return window
        return request.source_indexer(request.sources(boundary == "reflect"))
    except Exception:
        return None


def _indexer_key(indexer: Any) -> Any:
    """Hashable, value-based key for an indexer; ``None`` if it has none.

    Two indexers with equal keys select the same data. Unknown shapes get
    ``None`` (never batched) rather than a lossy ``repr``-based key.
    """
    if all(hasattr(indexer, a) for a in ("row_off", "col_off", "height", "width")):
        fields = (indexer.row_off, indexer.col_off, indexer.height, indexer.width)
        return ("window", *(_indexer_key(value) for value in fields))
    if isinstance(indexer, slice):
        parts = (_indexer_key(indexer.start), _indexer_key(indexer.stop))
        return ("slice", *parts, _indexer_key(indexer.step))
    if isinstance(indexer, dict):
        items = []
        for name, value in indexer.items():
            key = _indexer_key(value)
            if key is None:
                return None
            items.append((name, key))
        return ("dict", tuple(items))
    if isinstance(indexer, (list, tuple)):
        keys = tuple(_indexer_key(value) for value in indexer)
        return None if None in keys else ("seq", keys)
    if isinstance(indexer, np.ndarray):
        return ("array", indexer.dtype.str, indexer.shape, indexer.tobytes())
    if indexer is None or isinstance(indexer, (bool, int, float, str, np.generic)):
        return ("scalar", type(indexer).__name__, indexer)
    return None

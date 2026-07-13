"""`SpatialPatcher` — composes the four spatial axes.

The Patcher is intentionally tiny — it just orchestrates
``SpatialSampler.anchors → Geometry.neighborhood → SpatialWindow.weights →
Field.select`` and hands the result to `SpatialAggregation.merge` when the
caller asks. Split returns an `Iterator[Patch]` so streaming is the
default; ``list(patcher.split(field))`` materialises eagerly when that's
what's wanted.

See ``docs/concepts.md`` ("The four-axis abstraction") for the
four-axis framework.
"""

from __future__ import annotations

import sys
import traceback
from asyncio import BoundedSemaphore as AsyncBoundedSemaphore, to_thread
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from threading import BoundedSemaphore, Condition
from time import perf_counter
from typing import Any, Literal

import numpy as np

from geopatcher._src.hooks import (
    PatcherHook,
    _as_hooks,
    _dispatch,
    _len_or_unknown,
    _nbytes,
)
from geopatcher._src.patch import Patch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.protocols import AsyncField, Field
from geopatcher._src.spatial.aggregation import (
    SpatialAggregation,
    _warn_if_unsafe_streaming,
)
from geopatcher._src.spatial.geometry import (
    SpatialGeometry,
    _is_raster_domain,
    _MaskedWindow,
)
from geopatcher._src.spatial.sampler import SpatialSampler
from geopatcher._src.spatial.window import SpatialWindow


OnErrorPolicy = Literal["raise", "skip", "mask", "retry"]


@dataclass(eq=False)
class PatchErrorRecord:
    """A failed patch read recorded by `SpatialPatcher.split`.

    Args:
        anchor: Anchor whose patch failed to build.
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


@dataclass(eq=False)
class SpatialPatcher:
    """The four-axis spatial Patcher.

    Args:
        geometry: How a neighborhood is shaped around an anchor.
        sampler: Where anchors go.
        window: Boundary treatment / per-pixel weights.
        aggregation: Local → global merge strategy.
        on_error: Patch-read error policy. ``"raise"`` preserves the
            historical fail-fast behavior, ``"skip"`` logs and omits the
            failed anchor, ``"mask"`` emits a NaN-valued patch for the
            failed anchor, and ``"retry"`` retries matching exceptions up to
            `max_retries` before logging and skipping.
        max_retries: Number of retries when `on_error` is ``"retry"``.
        retry_on: Exception classes or class names that should be retried.
            Defaults to I/O-shaped failures (`OSError`, `TimeoutError`) so
            programmer errors are not retried unless explicitly requested.
        capture_traceback: If ``True`` (default), each `PatchErrorRecord`
            includes a formatted traceback. Set to ``False`` to skip
            formatting — useful for high-volume ``"skip"`` workloads
            where thousands of expected failures would otherwise inflate
            ``errors`` with megabytes of formatted frames.

    Examples:
        Sliding-window inference over a raster::

            patcher = SpatialPatcher(
                geometry    = SpatialRectangular(size=(256, 256)),
                sampler     = SpatialRegularStride(step=(192, 192)),
                window      = SpatialHann(),
                aggregation = SpatialOverlapAdd(),
            )
            patches = list(patcher.split(field))
            outs    = [run_operator(p) for p in patches]
            stitched = patcher.merge(outs, field.domain)
    """

    geometry: SpatialGeometry
    sampler: SpatialSampler
    window: SpatialWindow
    aggregation: SpatialAggregation
    on_error: OnErrorPolicy = "raise"
    max_retries: int = 0
    retry_on: tuple[type[BaseException] | str, ...] = (OSError, TimeoutError)
    capture_traceback: bool = True
    errors: list[PatchErrorRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _validate_error_policy(self.on_error, self.max_retries)

    def split(
        self,
        field: Field,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[Patch]:
        """Yield patches lazily — one per anchor placed by the sampler.

        When ``max_in_flight`` / ``max_in_flight_bytes`` bound the number
        of outstanding patches, each yielded patch owns one backpressure
        slot until it is released. Consumers must release promptly by
        calling ``patch.close()`` (or using each patch as a context
        manager: ``with patch: ...``). A garbage-collection finalizer
        returns leaked slots eventually, but it is a safety net, not the
        mechanism — relying on it can stall this iterator until the
        collector runs.

        A `PatchCache` passed as ``cache`` is consulted before every
        read: on a hit the source is never touched (only ``field.domain``
        metadata is), on a miss the patch is read then stored. Composes
        with ``journal`` (which records completion) and ``prefetch``
        (the cache check runs in the producer thread).
        """
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        return prefetch_iterable(
            self._split(
                field,
                hooks=hooks,
                journal=journal,
                cache=cache,
                max_in_flight=max_in_flight,
                max_in_flight_bytes=max_in_flight_bytes,
            ),
            prefetch,
        )

    def _split(
        self,
        field: Field,
        *,
        hooks: Iterable[PatcherHook] | None = None,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[Patch]:
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        cache_ctx = self._cache_context(cache, field)
        hook_list = _as_hooks(hooks)
        slots = (
            BoundedSemaphore(value=max_in_flight) if max_in_flight is not None else None
        )
        byte_budget = _ByteBudget(max_in_flight_bytes)
        if not hook_list:
            for anchor in self.sampler.anchors(domain, self.geometry):
                if journal is not None and journal.has(anchor):
                    continue
                patch = self._cached_patch(cache_ctx, domain, anchor)
                if patch is None:
                    patch = _build_patch_with_policy(
                        field=field,
                        domain=domain,
                        anchor=anchor,
                        geometry=self.geometry,
                        base_weights=base_weights,
                        boundary=boundary,
                        on_error=self.on_error,
                        max_retries=self.max_retries,
                        retry_on=self.retry_on,
                        errors=self.errors,
                        capture_traceback=self.capture_traceback,
                    )
                    self._store_patch(cache_ctx, anchor, patch)
                if patch is not None:
                    release = _acquire_backpressure(patch, slots, byte_budget)
                    if release is not None:
                        # Attach ownership in-place so the yielded patch
                        # releases the exact slot acquired for this read.
                        patch._release = release
                    yield patch
            return
        anchors = list(self.sampler.anchors(domain, self.geometry))
        _dispatch(hook_list, "on_split_start", len(anchors))
        try:
            for anchor in anchors:
                if journal is not None and journal.has(anchor):
                    continue
                _dispatch(hook_list, "on_patch_start", anchor)
                start = perf_counter()
                errors_before = len(self.errors)
                cached = self._cached_patch(cache_ctx, domain, anchor)
                try:
                    patch = cached
                    if patch is None:
                        patch = _build_patch_with_policy(
                            field=field,
                            domain=domain,
                            anchor=anchor,
                            geometry=self.geometry,
                            base_weights=base_weights,
                            boundary=boundary,
                            on_error=self.on_error,
                            max_retries=self.max_retries,
                            retry_on=self.retry_on,
                            errors=self.errors,
                            capture_traceback=self.capture_traceback,
                        )
                        self._store_patch(cache_ctx, anchor, patch)
                except Exception as exc:
                    _dispatch(hook_list, "on_error", anchor, exc)
                    raise
                for record in self.errors[errors_before:]:
                    _dispatch(
                        hook_list, "on_error", anchor, _exception_from_record(record)
                    )
                if patch is None:
                    continue
                release = _acquire_backpressure(patch, slots, byte_budget)
                if release is not None:
                    patch._release = release
                _dispatch(
                    hook_list,
                    "on_patch_done",
                    anchor,
                    perf_counter() - start,
                    _nbytes(patch.data),
                )
                yield patch
        finally:
            _dispatch(hook_list, "on_split_end")

    async def asplit(
        self,
        field: AsyncField,
        *,
        hooks: Iterable[PatcherHook] | None = None,
        journal: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> AsyncIterator[Patch]:
        """Async mirror of `split` over an `AsyncField`.

        The ``max_in_flight`` / ``max_in_flight_bytes`` slot-ownership
        contract matches `split`: close each yielded patch promptly
        (``patch.close()`` or ``with patch: ...``); the finalizer-based
        release on garbage collection is a safety net, not the mechanism.
        """
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        hook_list = _as_hooks(hooks)
        slots = (
            AsyncBoundedSemaphore(value=max_in_flight)
            if max_in_flight is not None
            else None
        )
        byte_budget = _ByteBudget(max_in_flight_bytes)
        if not hook_list:
            for anchor in self.sampler.anchors(domain, self.geometry):
                if journal is not None and journal.has(anchor):
                    continue
                patch = await _build_patch_async(
                    field, domain, anchor, self.geometry, base_weights, boundary
                )
                release = await _acquire_backpressure_async(patch, slots, byte_budget)
                if release is not None:
                    patch._release = release
                yield patch
            return
        anchors = list(self.sampler.anchors(domain, self.geometry))
        _dispatch(hook_list, "on_split_start", len(anchors))
        try:
            for anchor in anchors:
                if journal is not None and journal.has(anchor):
                    continue
                _dispatch(hook_list, "on_patch_start", anchor)
                start = perf_counter()
                try:
                    patch = await _build_patch_async(
                        field, domain, anchor, self.geometry, base_weights, boundary
                    )
                except Exception as exc:
                    _dispatch(hook_list, "on_error", anchor, exc)
                    raise
                release = await _acquire_backpressure_async(patch, slots, byte_budget)
                if release is not None:
                    patch._release = release
                _dispatch(
                    hook_list,
                    "on_patch_done",
                    anchor,
                    perf_counter() - start,
                    _nbytes(patch.data),
                )
                yield patch
        finally:
            _dispatch(hook_list, "on_split_end")

    def patch_at(self, field: Field, anchor: Any, *, cache: Any | None = None) -> Patch:
        """Read a single `Patch` at a specific anchor.

        The same geometry → ``field.select`` → window-weights pipeline
        as `split`, but driven by one explicit anchor instead of
        walking the sampler. Designed for random-access ML datasets
        (torch `Dataset.__getitem__`, Grain `RandomAccessDataSource`)
        that need lazy single-patch reads without materialising the
        whole iterator first.

        Args:
            field: The `Field` to read from.
            anchor: An anchor in the same format the sampler emits
                (e.g. ``(row, col)`` for raster, ``dict`` for grid).
                Typically obtained from
                ``patcher.anchors(field)[index]``.
            cache: Optional `PatchCache`. When set, a cache hit returns
                the stored patch without touching the source; a miss
                reads then stores it.

        Returns:
            A single `Patch` bit-identical to the one ``split`` would
            yield for the same anchor.
        """
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        cache_ctx = self._cache_context(cache, field)
        cached = self._cached_patch(cache_ctx, domain, anchor)
        if cached is not None:
            return cached
        patch = _build_patch(
            field, domain, anchor, self.geometry, base_weights, boundary
        )
        self._store_patch(cache_ctx, anchor, patch)
        return patch

    def _cache_context(self, cache: Any | None, field: Field) -> Any | None:
        """Bind ``cache`` to this field + config, or ``None`` when disabled."""
        if cache is None:
            return None
        field_id = cache.field_id_for(field)
        config_id = cache.config_id_for(self.geometry, self.window)
        return (cache, field_id, config_id)

    def _cached_patch(self, ctx: Any | None, domain: Any, anchor: Any) -> Patch | None:
        """Return a cache-hit patch for ``anchor``, or ``None`` on a miss."""
        if ctx is None:
            return None
        cache, field_id, config_id = ctx
        payload = cache.get(field_id, config_id, anchor)
        if payload is None:
            return None
        indices = self.geometry.neighborhood(domain, anchor)
        return cache.build_patch(payload, anchor, indices)

    def _store_patch(self, ctx: Any | None, anchor: Any, patch: Patch | None) -> None:
        """Store a freshly-built ``patch`` under ``anchor`` when caching is on."""
        if ctx is None or patch is None:
            return
        cache, field_id, config_id = ctx
        cache.put(field_id, config_id, anchor, patch)

    def anchors(self, field: Field) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``field``.

        Returns the same sequence ``split(field)`` walks, as a list
        the caller can ``len()`` and index. Same determinism contract
        as `n_anchors` (deterministic given an int sampler seed,
        re-drawn when seed is ``None``).
        """
        return list(self.sampler.anchors(field.domain, self.geometry))

    def n_anchors(self, field: Field) -> int:
        """Number of patches `split(field)` will yield.

        Enumerates the sampler's anchors without touching the field —
        only the domain is consulted.

        Determinism contract: holds exactly for samplers that return the
        same anchor set on every call given the same ``(domain,
        geometry)``. That covers all five samplers when a seed is set;
        for unseeded `SpatialRandom` / `SpatialJitteredStride` /
        `SpatialPoissonDisk` the count is still well-defined
        (``n_samples`` for the first two; a probabilistic estimate for
        the third), but the anchors materialised here are different
        draws from the ones a subsequent `split` will see. See
        ``docs/decisions.md`` (ADR-001) for why `split` returns an
        iterator and this helper exists as the ``len`` substitute.
        """
        return sum(1 for _ in self.sampler.anchors(field.domain, self.geometry))

    def merge(
        self,
        patches: Iterable[Any],
        domain: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """Hand off to the aggregation; warn on streaming-unsafe types."""
        hook_list = _as_hooks(hooks)
        _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
        _warn_if_unsafe_streaming(self.aggregation)
        try:
            output = self.aggregation.merge(patches, domain)
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(output))
        return output

    async def amerge(
        self,
        patches: AsyncIterable[Any] | Iterable[Any],
        domain: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """Async-friendly merge that accepts async or sync patch iterables."""
        if isinstance(patches, AsyncIterable):
            materialized = []
            async for patch in patches:
                materialized.append(patch)
            return self.merge(materialized, domain, hooks=hooks)
        return self.merge(patches, domain, hooks=hooks)

    def merge_to_xarray(
        self,
        patches: Iterable[Any],
        field: Field,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """`merge` + rewrap as `xarray.DataArray`, restoring the original coords.

        Convenience wrapper for the xrpatcher-style migration story: the
        bare `merge` returns an `np.ndarray` against the field's domain
        shape; this calls `field.with_data(...)` to put it back inside a
        DataArray with the field's coord metadata intact, and unwraps the
        resulting `XarrayField` to return the underlying `xarray.DataArray`.

        Args:
            patches: Iterable of patches to merge.
            field: The `Field` the patches came from. Must expose
                `with_data(array) -> Field` returning a wrapper that
                exposes the rebuilt array via a `.da` attribute — i.e.
                an `XarrayField` (or equivalent).

        Returns:
            ``xarray.DataArray`` carrying the merged values and the
            original coords.

        Raises:
            TypeError: If ``field`` does not expose `with_data`, or if the
                wrapper returned by `with_data` has no `.da` attribute.
        """
        with_data = getattr(field, "with_data", None)
        if with_data is None:
            raise TypeError(
                "merge_to_xarray needs a field with `with_data` "
                f"(e.g. XarrayField); got {type(field).__name__}."
            )
        merged = self.merge(patches, field.domain, hooks=hooks)
        rewrapped = with_data(merged)
        if not hasattr(rewrapped, "da"):
            raise TypeError(
                f"{type(field).__name__}.with_data must return a wrapper "
                "exposing the rebuilt array via `.da` (got "
                f"{type(rewrapped).__name__}). XarrayField is the canonical "
                "implementation."
            )
        return rewrapped.da

    def to_delayed(self, field: Field, operator: Any | None = None) -> list[Any]:
        """Build a Dask delayed graph for patches, optionally mapped by an operator."""
        from geopatcher.dask import to_delayed

        return to_delayed(self, field, operator)

    def to_dask_bag(self, field: Field) -> Any:
        """Build a Dask bag containing one item per patch."""
        from geopatcher.dask import to_dask_bag

        return to_dask_bag(self, field)

    def reduce(self, field: Field, agg: SpatialAggregation) -> Any:
        """Run a streaming pass over patches and return ``agg``'s result."""
        anchors = self.anchors(field)
        return agg.merge(
            (self.patch_at(field, anchor) for anchor in anchors), field.domain
        )

    def two_pass(
        self,
        field: Field,
        *,
        reduce_with: SpatialAggregation,
        apply: Callable[[Any, Any], Any],
        aggregation: SpatialAggregation | None = None,
    ) -> Any:
        """Run a global-statistics pass, then apply an operator with the result."""
        anchors = self.anchors(field)
        stats = reduce_with.merge(
            (self.patch_at(field, anchor) for anchor in anchors), field.domain
        )
        merge_with = aggregation or self.aggregation
        _warn_if_unsafe_streaming(merge_with)

        def _applied_patches() -> Iterator[Patch]:
            for anchor in anchors:
                patch = self.patch_at(field, anchor)
                yield replace(patch, data=apply(patch.data, stats))

        return merge_with.merge(_applied_patches(), field.domain)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": {
                "class": type(self.geometry).__name__,
                "config": self.geometry.get_config(),
            },
            "sampler": {
                "class": type(self.sampler).__name__,
                "config": self.sampler.get_config(),
            },
            "window": {
                "class": type(self.window).__name__,
                "config": self.window.get_config(),
            },
            "aggregation": {
                "class": type(self.aggregation).__name__,
                "config": self.aggregation.get_config(),
            },
            "on_error": self.on_error,
            "max_retries": self.max_retries,
            "retry_on": [
                exc if isinstance(exc, str) else exc.__name__ for exc in self.retry_on
            ],
            "capture_traceback": self.capture_traceback,
        }


@dataclass(eq=False)
class AsyncSpatialPatcher:
    """Async mirror of `SpatialPatcher` over an `AsyncField`.

    `split` is an ``async for``-able iterator. Useful with
    `AsyncGeoTIFFReader` for high-concurrency per-tile fan-out.

    The `on_error` / `max_retries` / `retry_on` / `capture_traceback`
    knobs mirror `SpatialPatcher`. Iteration is serialized (one
    ``await`` per anchor), so the `errors` accumulator is safe to read
    from the same coroutine without external locking.
    """

    geometry: SpatialGeometry
    sampler: SpatialSampler
    window: SpatialWindow
    aggregation: SpatialAggregation
    on_error: OnErrorPolicy = "raise"
    max_retries: int = 0
    retry_on: tuple[type[BaseException] | str, ...] = (OSError, TimeoutError)
    capture_traceback: bool = True
    errors: list[PatchErrorRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _validate_error_policy(self.on_error, self.max_retries)

    async def split(
        self,
        field: AsyncField,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        journal: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> AsyncIterator[Patch]:
        """Backward-compatible alias for `asplit`."""
        async for patch in self.asplit(
            field,
            hooks=hooks,
            journal=journal,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        ):
            yield patch

    async def asplit(
        self,
        field: AsyncField,
        *,
        hooks: Iterable[PatcherHook] | None = None,
        journal: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> AsyncIterator[Patch]:
        """Yield patches lazily over an `AsyncField`.

        The ``max_in_flight`` / ``max_in_flight_bytes`` slot-ownership
        contract matches `SpatialPatcher.split`: close each yielded
        patch promptly (``patch.close()`` or ``with patch: ...``); the
        finalizer-based release on garbage collection is a safety net,
        not the mechanism.
        """
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        hook_list = _as_hooks(hooks)
        slots = (
            AsyncBoundedSemaphore(value=max_in_flight)
            if max_in_flight is not None
            else None
        )
        byte_budget = _ByteBudget(max_in_flight_bytes)
        if not hook_list:
            for anchor in self.sampler.anchors(domain, self.geometry):
                if journal is not None and journal.has(anchor):
                    continue
                patch = await _build_patch_async_with_policy(
                    field=field,
                    domain=domain,
                    anchor=anchor,
                    geometry=self.geometry,
                    base_weights=base_weights,
                    boundary=boundary,
                    on_error=self.on_error,
                    max_retries=self.max_retries,
                    retry_on=self.retry_on,
                    errors=self.errors,
                    capture_traceback=self.capture_traceback,
                )
                if patch is not None:
                    release = await _acquire_backpressure_async(
                        patch, slots, byte_budget
                    )
                    if release is not None:
                        # Attach ownership in-place so the yielded patch
                        # releases the exact slot acquired for this read.
                        patch._release = release
                    yield patch
            return
        anchors = list(self.sampler.anchors(domain, self.geometry))
        _dispatch(hook_list, "on_split_start", len(anchors))
        try:
            for anchor in anchors:
                if journal is not None and journal.has(anchor):
                    continue
                _dispatch(hook_list, "on_patch_start", anchor)
                start = perf_counter()
                errors_before = len(self.errors)
                try:
                    patch = await _build_patch_async_with_policy(
                        field=field,
                        domain=domain,
                        anchor=anchor,
                        geometry=self.geometry,
                        base_weights=base_weights,
                        boundary=boundary,
                        on_error=self.on_error,
                        max_retries=self.max_retries,
                        retry_on=self.retry_on,
                        errors=self.errors,
                        capture_traceback=self.capture_traceback,
                    )
                except Exception as exc:
                    _dispatch(hook_list, "on_error", anchor, exc)
                    raise
                for record in self.errors[errors_before:]:
                    _dispatch(
                        hook_list, "on_error", anchor, _exception_from_record(record)
                    )
                if patch is None:
                    continue
                release = await _acquire_backpressure_async(patch, slots, byte_budget)
                if release is not None:
                    patch._release = release
                _dispatch(
                    hook_list,
                    "on_patch_done",
                    anchor,
                    perf_counter() - start,
                    _nbytes(patch.data),
                )
                yield patch
        finally:
            _dispatch(hook_list, "on_split_end")

    async def patch_at(self, field: AsyncField, anchor: Any) -> Patch:
        """Read a single `Patch` at a specific anchor.

        Async mirror of `SpatialPatcher.patch_at` — the read goes
        through ``await field.select(...)``. Designed for random-access
        cloud-tile readers driving a Grain / torch `Dataset` with
        per-item HTTP fan-out.
        """
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        return await _build_patch_async(
            field, domain, anchor, self.geometry, base_weights, boundary
        )

    def anchors(self, field: AsyncField) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``field``.

        Anchors are placed without touching the field, so this is sync
        even on the async patcher. See `SpatialPatcher.anchors`.
        """
        return list(self.sampler.anchors(field.domain, self.geometry))

    def n_anchors(self, field: AsyncField) -> int:
        """Number of patches `split(field)` will yield.

        See `SpatialPatcher.n_anchors`.
        """
        return sum(1 for _ in self.sampler.anchors(field.domain, self.geometry))

    def merge(
        self,
        patches: Iterable[Any],
        domain: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        hook_list = _as_hooks(hooks)
        _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
        _warn_if_unsafe_streaming(self.aggregation)
        try:
            output = self.aggregation.merge(patches, domain)
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(output))
        return output

    async def amerge(
        self,
        patches: AsyncIterable[Any] | Iterable[Any],
        domain: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        if isinstance(patches, AsyncIterable):
            materialized = []
            async for patch in patches:
                materialized.append(patch)
            return self.merge(materialized, domain, hooks=hooks)
        return self.merge(patches, domain, hooks=hooks)


def _safe_base_weights(
    window: SpatialWindow, geometry: SpatialGeometry
) -> np.ndarray | None:
    """Compute the geometry-shaped base weights, or `None` for windows
    that don't expose a static weight grid (e.g. graph-based geometries
    where weights are anchor-dependent)."""
    try:
        return window.weights(geometry)
    except TypeError:
        return None


def _validate_error_policy(on_error: str, max_retries: int) -> None:
    if on_error not in ("raise", "skip", "mask", "retry"):
        raise ValueError(
            "invalid on_error policy "
            f"{on_error!r}; expected 'raise', 'skip', 'mask', or 'retry'"
        )
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")


def _build_patch_with_policy(
    *,
    field: Field,
    domain: Any,
    anchor: Any,
    geometry: SpatialGeometry,
    base_weights: np.ndarray | None,
    boundary: str,
    on_error: OnErrorPolicy,
    max_retries: int,
    retry_on: tuple[type[BaseException] | str, ...],
    errors: list[PatchErrorRecord],
    capture_traceback: bool = True,
) -> Patch | None:
    retries = max_retries if on_error == "retry" else 0
    indices = geometry.neighborhood(domain, anchor)
    pad_value = getattr(geometry, "pad_value", None)
    for retry_count in range(retries + 1):
        try:
            return _build_patch_from_indices(
                field, domain, anchor, indices, base_weights, boundary, pad_value
            )
        except Exception as exc:
            # Preserve KeyboardInterrupt/SystemExit by handling only Exception.
            if isinstance(exc, StopIteration):
                raise
            if on_error == "raise":
                raise
            _record_patch_error(errors, anchor, exc, retry_count, capture_traceback)
            if on_error == "mask":
                return _build_mask_patch(
                    domain, anchor, indices, base_weights, boundary
                )
            if on_error == "retry":
                if not _matches_retry_on(exc, retry_on):
                    raise
                if retry_count < retries:
                    continue
                return None
            return None


async def _build_patch_async_with_policy(
    *,
    field: AsyncField,
    domain: Any,
    anchor: Any,
    geometry: SpatialGeometry,
    base_weights: np.ndarray | None,
    boundary: str,
    on_error: OnErrorPolicy,
    max_retries: int,
    retry_on: tuple[type[BaseException] | str, ...],
    errors: list[PatchErrorRecord],
    capture_traceback: bool = True,
) -> Patch | None:
    retries = max_retries if on_error == "retry" else 0
    indices = geometry.neighborhood(domain, anchor)
    pad_value = getattr(geometry, "pad_value", None)
    for retry_count in range(retries + 1):
        try:
            return await _build_patch_async_from_indices(
                field, domain, anchor, indices, base_weights, boundary, pad_value
            )
        except Exception as exc:
            # Preserve KeyboardInterrupt/SystemExit by handling only Exception.
            if isinstance(exc, StopIteration):
                raise
            if on_error == "raise":
                raise
            _record_patch_error(errors, anchor, exc, retry_count, capture_traceback)
            if on_error == "mask":
                return _build_mask_patch(
                    domain, anchor, indices, base_weights, boundary
                )
            if on_error == "retry":
                if not _matches_retry_on(exc, retry_on):
                    raise
                if retry_count < retries:
                    continue
                return None
            return None


def _exception_from_record(record: PatchErrorRecord) -> Exception:
    """Synthesize an Exception for hook dispatch from a recorded patch failure.

    Used when the patcher swallows an exception under a non-``raise`` policy
    but still wants to notify observability hooks. The reconstructed instance
    carries only the message — frames have already been formatted into
    ``record.traceback``.
    """
    return RuntimeError(f"{record.kind}: {record.message}")


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


def _matches_retry_on(
    exc: BaseException, retry_on: tuple[type[BaseException] | str, ...]
) -> bool:
    for candidate in retry_on:
        if isinstance(candidate, str):
            if type(exc).__name__ == candidate:
                return True
        elif isinstance(exc, candidate):
            return True
    return False


def _validate_backpressure(
    max_in_flight: int | None, max_in_flight_bytes: int | None
) -> None:
    if max_in_flight is not None and max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if max_in_flight_bytes is not None and max_in_flight_bytes < 1:
        raise ValueError("max_in_flight_bytes must be >= 1")


class _ByteBudget:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.used = 0
        self._condition = Condition()

    def acquire(self, patch: Patch) -> int:
        nbytes = int(getattr(np.asarray(patch.data), "nbytes", 0))
        if self.limit is not None and nbytes > self.limit:
            raise ValueError(
                f"patch uses {nbytes} bytes, exceeding max_in_flight_bytes={self.limit}"
            )
        if self.limit is None:
            # Still take the lock: `release` runs on consumer threads,
            # so unbounded budgets must not mutate `used` unlocked.
            with self._condition:
                self.used += nbytes
            return nbytes
        with self._condition:
            while self.used + nbytes > self.limit:
                self._condition.wait()
            self.used += nbytes
        return nbytes

    def release(self, nbytes: int) -> None:
        with self._condition:
            self.used = max(0, self.used - nbytes)
            self._condition.notify()


def _acquire_backpressure(
    patch: Patch, slots: BoundedSemaphore | None, byte_budget: _ByteBudget
) -> Any | None:
    nbytes = byte_budget.acquire(patch)
    if slots is not None:
        slots.acquire()
    if slots is None and nbytes == 0:
        return None

    def release() -> None:
        try:
            if slots is not None:
                slots.release()
            byte_budget.release(nbytes)
        except Exception:
            # A finalizer-driven release during interpreter shutdown can
            # hit already-torn-down synchronisation primitives; swallow
            # only in that case so real bugs still surface.
            if not sys.is_finalizing():
                raise

    return release


async def _acquire_backpressure_async(
    patch: Patch,
    slots: AsyncBoundedSemaphore | None,
    byte_budget: _ByteBudget,
) -> Any | None:
    nbytes = await to_thread(byte_budget.acquire, patch)
    if slots is not None:
        await slots.acquire()
    if slots is None and nbytes == 0:
        return None

    def release() -> None:
        try:
            if slots is not None:
                slots.release()
            byte_budget.release(nbytes)
        except Exception:
            # See the sync twin: swallow only shutdown-time teardown
            # failures from a finalizer-driven release.
            if not sys.is_finalizing():
                raise

    return release


def _build_patch(
    field: Field,
    domain: Any,
    anchor: Any,
    geometry: SpatialGeometry,
    base_weights: np.ndarray | None,
    boundary: str,
) -> Patch:
    """Single-anchor read pipeline shared by `split` and `patch_at`."""
    indices = geometry.neighborhood(domain, anchor)
    return _build_patch_from_indices(
        field,
        domain,
        anchor,
        indices,
        base_weights,
        boundary,
        getattr(geometry, "pad_value", None),
    )


def _build_patch_from_indices(
    field: Field,
    domain: Any,
    anchor: Any,
    indices: Any,
    base_weights: np.ndarray | None,
    boundary: str,
    pad_value: float | None = None,
) -> Patch:
    if boundary == "raise":
        _raise_if_overflows(indices, domain)
    window = _unwrap_for_select(indices)
    if boundary in ("pad", "reflect"):
        data = _select_padded(field, domain, window, boundary, pad_value)
    else:
        data = field.select(window)
    weights = _build_weights(indices, base_weights, boundary=boundary)
    return Patch(data=data, anchor=anchor, indices=indices, weights=weights)


async def _build_patch_async(
    field: AsyncField,
    domain: Any,
    anchor: Any,
    geometry: SpatialGeometry,
    base_weights: np.ndarray | None,
    boundary: str,
) -> Patch:
    """Async mirror of `_build_patch` — awaits `field.select`."""
    indices = geometry.neighborhood(domain, anchor)
    return await _build_patch_async_from_indices(
        field,
        domain,
        anchor,
        indices,
        base_weights,
        boundary,
        getattr(geometry, "pad_value", None),
    )


async def _build_patch_async_from_indices(
    field: AsyncField,
    domain: Any,
    anchor: Any,
    indices: Any,
    base_weights: np.ndarray | None,
    boundary: str,
    pad_value: float | None = None,
) -> Patch:
    if boundary == "raise":
        _raise_if_overflows(indices, domain)
    window = _unwrap_for_select(indices)
    if boundary in ("pad", "reflect"):
        data = await _select_padded_async(field, domain, window, boundary, pad_value)
    else:
        data = await _select_async(field, window)
    weights = _build_weights(indices, base_weights, boundary=boundary)
    return Patch(data=data, anchor=anchor, indices=indices, weights=weights)


def _build_mask_patch(
    domain: Any,
    anchor: Any,
    indices: Any,
    base_weights: np.ndarray | None,
    boundary: str,
) -> Patch:
    if boundary == "raise":
        _raise_if_overflows(indices, domain)
    weights = _build_weights(indices, base_weights, boundary=boundary)
    h, w = _indices_hw(indices)
    prefix = tuple(getattr(domain, "shape", ())[:-2])
    if prefix:
        shape = (*prefix, h, w)
    elif weights is not None:
        shape = tuple(np.shape(weights))
    else:
        shape = (h, w)
    data = np.full(shape, np.nan, dtype=float)
    return Patch(data=data, anchor=anchor, indices=indices, weights=weights)


def _indices_hw(indices: Any) -> tuple[int, int]:
    """Infer raster/grid mask dimensions for known patch index structures."""
    if isinstance(indices, _MaskedWindow):
        indices = indices.window
    h = getattr(indices, "height", None)
    w = getattr(indices, "width", None)
    if h is not None and w is not None:
        return int(h), int(w)
    if isinstance(indices, dict):
        sizes = []
        for index in indices.values():
            if (
                isinstance(index, slice)
                and index.start is not None
                and index.stop is not None
            ):
                sizes.append(int(index.stop) - int(index.start))
        if len(sizes) >= 2:
            return sizes[-2], sizes[-1]
    raise ValueError(f"cannot infer mask shape for indices {indices!r}")


async def _select_async(field: AsyncField, indexer: Any) -> Any:
    aselect = getattr(field, "aselect", None)
    if aselect is not None:
        return await aselect(indexer)
    return await field.select(indexer)


def _unwrap_for_select(indices: Any) -> Any:
    """Unwrap a `_MaskedWindow` to the underlying rasterio `Window` for `Field.select`.

    `SpatialPolygonIntersection.neighborhood` returns a `_MaskedWindow`
    so `_build_weights` can recover the interior mask. But `Field.select`
    expects a plain `Window` (or dict / index list) — the wrapper would
    confuse downstream readers like `RasterField.read_from_window`. Strip
    it here at the call boundary; keep the wrapper on `Patch.indices` so
    aggregation still sees the mask via `_resolve_indices`.
    """
    if isinstance(indices, _MaskedWindow):
        return indices.window
    return indices


def _build_weights(
    indices: Any,
    base_weights: np.ndarray | None,
    *,
    boundary: str = "drop",
) -> Any:
    """Resolve a patch's weight array.

    If the indices is a `_MaskedWindow` (SpatialPolygonIntersection on a raster),
    return the interior mask — the window controls *which pixels count*,
    not how heavily they're tapered. Otherwise return the geometry-shaped
    base weights from `SpatialWindow.weights`, cropped to the actual window
    size when boundary == "shrink" (because the window was clipped).
    """
    if isinstance(indices, _MaskedWindow):
        return indices.mask
    if boundary == "shrink" and base_weights is not None:
        h = getattr(indices, "height", None)
        w = getattr(indices, "width", None)
        if h is not None and w is not None:
            bh, bw = base_weights.shape[-2:]
            if (h, w) != (bh, bw):
                return base_weights[..., : int(h), : int(w)]
    return base_weights


def _raise_if_overflows(indices: Any, domain: Any) -> None:
    """Raise ``ValueError`` if ``indices`` extends past ``domain``.

    Used by `SpatialPatcher.split` when the geometry's ``boundary``
    policy is ``"raise"``. Only meaningful for raster-shaped indices
    (rasterio `Window`); non-raster indices return early.
    """
    if not (hasattr(indices, "row_off") and hasattr(indices, "col_off")):
        return
    if not (hasattr(domain, "shape") and len(domain.shape) >= 2):
        return
    dh, dw = int(domain.shape[-2]), int(domain.shape[-1])
    r0, c0 = int(indices.row_off), int(indices.col_off)
    rh, cw = int(indices.height), int(indices.width)
    if r0 < 0 or c0 < 0 or r0 + rh > dh or c0 + cw > dw:
        raise ValueError(
            f"patch window {indices!r} overflows the domain shape "
            f"({dh}, {dw}); set boundary='pad' or 'shrink' to allow."
        )


def _overflows_window(window: Any, domain: Any) -> bool:
    """True if a raster ``window`` extends past the ``domain`` edge."""
    if not (hasattr(window, "row_off") and hasattr(window, "col_off")):
        return False
    if not (hasattr(domain, "shape") and len(domain.shape) >= 2):
        return False
    dh, dw = int(domain.shape[-2]), int(domain.shape[-1])
    r0, c0 = int(window.row_off), int(window.col_off)
    rh, cw = int(window.height), int(window.width)
    return r0 < 0 or c0 < 0 or r0 + rh > dh or c0 + cw > dw


def _clip_pads(window: Any, domain: Any) -> tuple[Any, tuple[int, int, int, int]]:
    """Clip ``window`` to the domain, returning ``(clipped, (t, b, l, r))``.

    ``(t, b, l, r)`` are the pad widths that grow the clipped read back up
    to the original window size on the top / bottom / left / right edges.
    """
    from rasterio.windows import Window

    dh, dw = int(domain.shape[-2]), int(domain.shape[-1])
    r0, c0 = int(window.row_off), int(window.col_off)
    r1, c1 = r0 + int(window.height), c0 + int(window.width)
    cr0, cc0 = max(r0, 0), max(c0, 0)
    cr1, cc1 = min(r1, dh), min(c1, dw)
    clipped = Window(
        col_off=cc0,
        row_off=cr0,
        width=max(cc1 - cc0, 0),
        height=max(cr1 - cr0, 0),
    )
    pads = (cr0 - r0, r1 - cr1, cc0 - c0, c1 - cc1)
    return clipped, pads


def _carrier_nodata(data: Any) -> Any:
    """Best-effort nodata / fill value for a selected patch carrier."""
    fill = getattr(data, "fill_value_default", None)
    if fill is not None:
        return fill
    rio = getattr(getattr(data, "da", None), "rio", None)
    if rio is not None and getattr(rio, "nodata", None) is not None:
        return rio.nodata
    return 0


def _pad_carrier(
    data: Any, pads: tuple[int, int, int, int], mode: str, fill: Any
) -> Any:
    """Pad a selected carrier up to full size, preserving georeferencing.

    Handles a georeader `GeoTensor` (whose ``pad`` shifts the transform),
    an xarray-backed field exposing ``.da``, and a plain ndarray.
    """
    pt, pb, pl, pr = pads
    if pt == pb == pl == pr == 0:
        return data
    const = {"constant_values": fill} if mode == "constant" else {}
    if hasattr(data, "pad") and hasattr(data, "transform"):
        return data.pad({"y": (pt, pb), "x": (pl, pr)}, mode=mode, **const)
    da = getattr(data, "da", None)
    if da is not None:
        y_dim, x_dim = da.rio.y_dim, da.rio.x_dim
        padded = da.pad({y_dim: (pt, pb), x_dim: (pl, pr)}, mode=mode, **const)
        return type(data)(padded)
    arr = np.asarray(data)
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(pt, pb), (pl, pr)]
    if mode == "constant":
        return np.pad(arr, pad_width, mode="constant", constant_values=fill)
    return np.pad(arr, pad_width, mode="reflect")


def _reflect_guard(window: Any, clipped: Any, pads: tuple[int, int, int, int]) -> None:
    """Raise a clear error when a reflect pad exceeds the in-domain extent."""
    pt, pb, pl, pr = pads
    ch, cw = int(clipped.height), int(clipped.width)
    if pt >= ch or pb >= ch or pl >= cw or pr >= cw:
        raise ValueError(
            f"boundary='reflect' needs the in-domain extent to exceed the "
            f"overflow on every side; window {window!r} clips to ({ch}, {cw}) "
            f"but the pads are (top={pt}, bottom={pb}, left={pl}, right={pr}). "
            f"Use boundary='pad' for overflows this large."
        )


def _select_padded(
    field: Field, domain: Any, window: Any, boundary: str, pad_value: float | None
) -> Any:
    """Read ``window`` under ``pad`` / ``reflect``, padding overflow to full size.

    Interior (non-overflowing) windows take the plain read path — the
    padding machinery only engages at the domain edge.
    """
    if not _overflows_window(window, domain):
        return field.select(window)
    clipped, pads = _clip_pads(window, domain)
    if boundary == "reflect":
        _reflect_guard(window, clipped, pads)
    data = field.select(clipped)
    mode = "reflect" if boundary == "reflect" else "constant"
    fill = pad_value if pad_value is not None else _carrier_nodata(data)
    return _pad_carrier(data, pads, mode, fill)


async def _select_padded_async(
    field: AsyncField,
    domain: Any,
    window: Any,
    boundary: str,
    pad_value: float | None,
) -> Any:
    """Async mirror of `_select_padded`."""
    if not _overflows_window(window, domain):
        return await _select_async(field, window)
    clipped, pads = _clip_pads(window, domain)
    if boundary == "reflect":
        _reflect_guard(window, clipped, pads)
    data = await _select_async(field, clipped)
    mode = "reflect" if boundary == "reflect" else "constant"
    fill = pad_value if pad_value is not None else _carrier_nodata(data)
    return _pad_carrier(data, pads, mode, fill)


# Re-export `_is_raster_domain` to discourage cross-imports from geometry.py.
__all__ = [
    "AsyncSpatialPatcher",
    "PatchErrorRecord",
    "SpatialPatcher",
    "_is_raster_domain",
]

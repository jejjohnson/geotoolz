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

import asyncio
import sys
import traceback
from asyncio import BoundedSemaphore as AsyncBoundedSemaphore, to_thread
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Condition
from time import perf_counter
from typing import Any, Literal

import numpy as np

from geopatcher._src.domains import GridDomain
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
        return self._split_anchors(
            field,
            None,
            hooks=hooks,
            prefetch=prefetch,
            journal=journal,
            cache=cache,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )

    def _split_anchors(
        self,
        field: Field,
        anchors: Iterable[Any] | None,
        *,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[Patch]:
        """`split` over explicit ``anchors`` (``None`` walks the sampler).

        `two_pass` hands both passes the same materialised anchor list so
        an unseeded sampler cannot place the second pass differently.
        """
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        return prefetch_iterable(
            self._split(
                field,
                anchors=anchors,
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
        anchors: Iterable[Any] | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[Patch]:
        domain = field.domain
        if anchors is None:
            anchors = self.sampler.anchors(domain, self.geometry)
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        cache_ctx = self._cache_context(cache, field)
        hook_list = _as_hooks(hooks)
        slots = (
            BoundedSemaphore(value=max_in_flight) if max_in_flight is not None else None
        )
        byte_budget = _ByteBudget(max_in_flight_bytes)
        if not hook_list:
            for anchor in anchors:
                if journal is not None and journal.has(anchor):
                    continue
                patch = self._cached_patch(cache_ctx, domain, anchor)
                if patch is None:
                    patch, read_ok = _build_patch_with_policy(
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
                    if read_ok:
                        self._store_patch(cache_ctx, anchor, patch)
                if patch is not None:
                    release = _acquire_backpressure(patch, slots, byte_budget)
                    if release is not None:
                        # Attach ownership in-place so the yielded patch
                        # releases the exact slot acquired for this read.
                        patch._release = release
                    yield patch
            return
        anchors = list(anchors)
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
                        patch, read_ok = _build_patch_with_policy(
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
                        if read_ok:
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
        """Hand the patches to the aggregation and return its raw output.

        The result is whatever ``self.aggregation.merge`` produces — a bare
        ``np.ndarray`` on the domain grid for the dense aggregations, a
        ``dict`` for `SpatialMeanStd` / `SpatialInvVarWeightedMean` /
        `SpatialByIndex`, a zarr array for streaming `SpatialOverlapAdd`.
        Use `merge_to_field` (or `merge_to_xarray`) to get a georeferenced
        carrier back. A ``streaming_safe = False`` aggregation warns (or
        raises under `set_strict`) at the caller's line.
        """
        return _merge_with_hooks(self.aggregation, patches, domain, hooks, stacklevel=2)

    async def amerge(
        self,
        patches: AsyncIterable[Any] | Iterable[Any],
        domain: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """Async-friendly `merge` that accepts async or sync patch iterables.

        An async stream is not materialised: the aggregation runs in a
        worker thread and pulls one patch at a time from the event loop,
        so a streaming-safe aggregation holds one patch, not the stream.
        """
        return await _amerge_with_hooks(
            self.aggregation, patches, domain, hooks, stacklevel=2
        )

    def merge_to_field(
        self,
        patches: Iterable[Any],
        field: Field,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """`merge` + rebuild a georeferenced carrier via ``field.with_data``.

        `merge` returns the aggregation's raw output (see ADR-007 in
        ``docs/patcher/decisions.md``); this wraps it back onto the
        field's grid so the transform, CRS, nodata
        (``fill_value_default`` / ``rio.nodata``) and attrs of the source
        survive. What comes back is the adapter's ``with_data`` result: a
        `GeoTensor` for `RasterField`, a `RioXarrayField` / `XarrayField`
        wrapping the rebuilt ``DataArray`` (``.da``) for the xarray
        adapters.

        The merged values keep the source dtype when they fit it (the
        rule in `merge_to_xarray`); otherwise the aggregation's dtype is
        kept — e.g. float64 carrying a NaN fill on an integer source.

        Args:
            patches: Iterable of patches to merge.
            field: The `Field` the patches came from; must expose
                ``with_data``.
            hooks: Optional observability hooks forwarded to `merge`.

        Returns:
            ``field.with_data(merged)``.

        Raises:
            TypeError: If ``field`` has no ``with_data``, if the aggregation
                returns a ``dict`` (`SpatialMeanStd`,
                `SpatialInvVarWeightedMean`, `SpatialByIndex`, …) or
                anything without an array ``shape``, or if the output shape
                differs from ``field.domain.shape``.

        Examples:
            Stitch Hann-weighted chips back into a `GeoTensor`::

                out = patcher.merge_to_field(patcher.split(field), field)
                out.transform == field.reader.transform  # True

            Keep the raw array instead::

                arr = patcher.merge(patcher.split(field), field.domain)
        """
        with_data = _require_with_data(field, "merge_to_field")
        merged = _merge_with_hooks(
            self.aggregation, patches, field.domain, hooks, stacklevel=2
        )
        return with_data(
            _field_values(
                merged,
                self.aggregation,
                shape=_domain_shape(field.domain),
                dtype=_source_dtype(field),
                caller="merge_to_field",
            )
        )

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

        Dtype rule (shared with `merge_to_field`): the merged values are
        cast back to the source dtype when every value is representable
        in it. An integer / bool source needs every value finite, integral
        and in range — otherwise (a fractional mean, a NaN fill) the
        aggregation's dtype is kept. A floating source is cast whenever no
        finite value overflows it (precision is rounded, as for the
        source). Other dtypes are left untouched.

        Args:
            patches: Iterable of patches to merge.
            field: The `Field` the patches came from. Must expose
                `with_data(array) -> Field` returning a wrapper that
                exposes the rebuilt array via a `.da` attribute — i.e.
                an `XarrayField` (or equivalent).
            hooks: Optional observability hooks forwarded to `merge`.

        Returns:
            ``xarray.DataArray`` carrying the merged values and the
            original coords.

        Raises:
            TypeError: If ``field`` does not expose `with_data`, if the
                aggregation returns a ``dict`` (or anything that is not an
                array on the domain grid), or if the wrapper returned by
                `with_data` has no `.da` attribute.
        """
        with_data = _require_with_data(field, "merge_to_xarray")
        merged = _merge_with_hooks(
            self.aggregation, patches, field.domain, hooks, stacklevel=2
        )
        rewrapped = with_data(
            _field_values(
                merged,
                self.aggregation,
                shape=_domain_shape(field.domain),
                dtype=_source_dtype(field),
                caller="merge_to_xarray",
            )
        )
        if not hasattr(rewrapped, "da"):
            raise TypeError(
                f"{type(field).__name__}.with_data must return a wrapper "
                "exposing the rebuilt array via `.da` (got "
                f"{type(rewrapped).__name__}). XarrayField is the canonical "
                "implementation."
            )
        return rewrapped.da

    def to_delayed(self, field: Field, operator: Any | None = None) -> list[Any]:
        """Build a Dask delayed graph for patches, optionally mapped by an operator.

        One ``patch_at`` task per anchor; the field enters the graph once.
        Runner-level policies (``on_error``, hooks, journal, cache,
        prefetch) do not apply — see `geopatcher.dask`.
        """
        from geopatcher.dask import to_delayed

        return to_delayed(self, field, operator)

    def to_dask_bag(self, field: Field) -> Any:
        """Build a Dask bag with one item (and one partition) per patch.

        Same per-patch tasks and policy caveats as `to_delayed`.
        """
        from geopatcher.dask import to_dask_bag

        return to_dask_bag(self, field)

    def reduce(
        self,
        field: Field,
        agg: SpatialAggregation,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Any:
        """Run one streaming pass of `split` into ``agg`` and return its result.

        The patches come from `split`, so the patcher's ``on_error`` /
        retry policy and the ``hooks`` / ``prefetch`` / ``journal`` /
        ``cache`` / backpressure knobs apply exactly as they do there;
        each patch is closed (its backpressure slot released) once ``agg``
        moves on to the next. ``agg`` gets the same streaming-safety check
        as `merge` (a warning, or `RuntimeError` under `set_strict`).

        Examples:
            Global statistics for a normalisation pass::

                stats = patcher.reduce(field, SpatialMeanStd())

            Tolerate unreadable tiles while reducing::

                patcher = replace(patcher, on_error="skip")
                bounds = patcher.reduce(field, SpatialMinMax(), prefetch=2)
        """
        patches = self._split_anchors(
            field,
            None,
            hooks=hooks,
            prefetch=prefetch,
            journal=journal,
            cache=cache,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )
        return _merge_with_hooks(
            agg, _closing(patches), field.domain, hooks, stacklevel=2
        )

    def two_pass(
        self,
        field: Field,
        *,
        reduce_with: SpatialAggregation,
        apply: Callable[[Any, Any], Any],
        aggregation: SpatialAggregation | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Any:
        """Run a global-statistics pass, then apply an operator with the result.

        Pass one is `reduce` with ``reduce_with``; pass two splits again,
        maps every patch through ``apply(data, stats)`` (via
        `Patch.with_data`) and merges with ``aggregation`` (default: the
        patcher's own). Both passes go through `split` — ``on_error``,
        hooks, journal, cache, prefetch and backpressure apply to each —
        over one anchor list materialised up front, so an unseeded random
        sampler places both passes identically. Each patch is read twice;
        pass a ``cache`` to serve the second pass without re-reading.

        Examples:
            Standardise a scene with its global mean / std::

                out = patcher.two_pass(
                    field,
                    reduce_with=SpatialMeanStd(),
                    apply=lambda x, s: (np.asarray(x) - s["mean"]) / s["std"],
                )
        """
        anchors = self.anchors(field)
        split_kwargs: dict[str, Any] = {
            "hooks": hooks,
            "prefetch": prefetch,
            "journal": journal,
            "cache": cache,
            "max_in_flight": max_in_flight,
            "max_in_flight_bytes": max_in_flight_bytes,
        }
        stats = _merge_with_hooks(
            reduce_with,
            _closing(self._split_anchors(field, anchors, **split_kwargs)),
            field.domain,
            hooks,
            stacklevel=2,
        )
        applied = (
            patch.with_data(apply(patch.data, stats))
            for patch in _closing(self._split_anchors(field, anchors, **split_kwargs))
        )
        return _merge_with_hooks(
            aggregation or self.aggregation,
            applied,
            field.domain,
            hooks,
            stacklevel=2,
        )

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
        """Hand the patches to the aggregation; see `SpatialPatcher.merge`."""
        return _merge_with_hooks(self.aggregation, patches, domain, hooks, stacklevel=2)

    async def amerge(
        self,
        patches: AsyncIterable[Any] | Iterable[Any],
        domain: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """Merge an async (or sync) patch stream; see `SpatialPatcher.amerge`."""
        return await _amerge_with_hooks(
            self.aggregation, patches, domain, hooks, stacklevel=2
        )


_STREAM_END = object()


def _check_streaming(aggregation: SpatialAggregation, *, stacklevel: int) -> None:
    """Run the strict / streaming-safety check, warning at the caller's caller.

    ``stacklevel`` counts like `warnings.warn`'s, from this helper's
    caller (``1`` = the caller itself), so the warning lands on the
    user's ``merge`` / ``reduce`` line instead of inside geopatcher.
    Under `set_strict` the `RuntimeError` propagates unchanged.
    """
    # +2: one frame for `_warn_if_unsafe_streaming`, one for this helper.
    _warn_if_unsafe_streaming(aggregation, stacklevel=stacklevel + 2)


def _merge_with_hooks(
    aggregation: SpatialAggregation,
    patches: Iterable[Any],
    domain: Any,
    hooks: Iterable[PatcherHook] | None,
    *,
    stacklevel: int | None,
) -> Any:
    """``aggregation.merge`` inside the merge hook events and the streaming check.

    ``stacklevel`` is relative to this helper's caller (see
    `_check_streaming`); ``None`` skips the check because the caller
    already ran it on the user's frame.
    """
    hook_list = _as_hooks(hooks)
    _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
    if stacklevel is not None:
        _check_streaming(aggregation, stacklevel=stacklevel + 1)
    try:
        output = aggregation.merge(patches, domain)
    except Exception as exc:
        _dispatch(hook_list, "on_error", None, exc)
        raise
    _dispatch(hook_list, "on_merge_end", _nbytes(output))
    return output


async def _amerge_with_hooks(
    aggregation: SpatialAggregation,
    patches: AsyncIterable[Any] | Iterable[Any],
    domain: Any,
    hooks: Iterable[PatcherHook] | None,
    *,
    stacklevel: int,
) -> Any:
    """Async adapter: feed an async patch stream into a sync ``merge``.

    `SpatialAggregation.merge` takes a plain iterable, so the merge runs
    in a worker thread over a generator that fetches each patch from the
    event loop on demand (``run_coroutine_threadsafe(anext(...))``). The
    stream is consumed as fast as the aggregation folds it — nothing is
    buffered — and the loop stays free to serve the producer meanwhile.
    """
    if not isinstance(patches, AsyncIterable):
        return _merge_with_hooks(
            aggregation, patches, domain, hooks, stacklevel=stacklevel + 1
        )
    # Check on the event-loop thread: from the worker thread there is no
    # user frame to attribute the warning to.
    _check_streaming(aggregation, stacklevel=stacklevel + 1)
    loop = asyncio.get_running_loop()
    stream = aiter(patches)

    async def pull_one() -> Any:
        try:
            return await anext(stream)
        except StopAsyncIteration:
            return _STREAM_END

    def pulled() -> Iterator[Any]:
        while True:
            item = asyncio.run_coroutine_threadsafe(pull_one(), loop).result()
            if item is _STREAM_END:
                return
            yield item

    try:
        result = await to_thread(
            _merge_with_hooks, aggregation, pulled(), domain, hooks, stacklevel=None
        )
    except Exception:
        await _aclose(stream)
        raise
    await _aclose(stream)
    return result


async def _aclose(stream: Any) -> None:
    """Close an async iterator early (a no-op for exhausted / plain ones)."""
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


def _closing(patches: Iterable[Patch]) -> Iterator[Patch]:
    """Yield each patch and close it once the consumer asks for the next.

    Aggregations do not call `Patch.close`, so under ``max_in_flight`` /
    ``max_in_flight_bytes`` the split producer would block on the slot the
    previous patch still holds. Closing on advance releases it first.
    """
    for patch in patches:
        with patch:
            yield patch


def _require_with_data(field: Any, caller: str) -> Callable[[Any], Any]:
    """``field.with_data``, or a `TypeError` naming ``caller``."""
    with_data = getattr(field, "with_data", None)
    if with_data is None:
        raise TypeError(
            f"{caller} needs a field with `with_data` (e.g. RasterField, "
            f"XarrayField); got {type(field).__name__}."
        )
    return with_data


def _domain_shape(domain: Any) -> tuple[int, ...] | None:
    """The domain's dense array shape, or ``None`` when it has none."""
    shape = getattr(domain, "shape", None)
    return None if shape is None else tuple(int(n) for n in shape)


def _source_dtype(field: Any) -> np.dtype | None:
    """The dtype of the field's backing array (field, ``.da`` or ``.reader``)."""
    for owner in (field, getattr(field, "da", None), getattr(field, "reader", None)):
        dtype = getattr(owner, "dtype", None)
        if dtype is None:
            continue
        try:
            return np.dtype(dtype)
        except TypeError:
            continue
    return None


def _field_values(
    output: Any,
    aggregation: SpatialAggregation,
    *,
    shape: tuple[int, ...] | None,
    dtype: np.dtype | None,
    caller: str,
) -> np.ndarray:
    """Validate a merge output for ``with_data`` and restore the source dtype.

    Raises `TypeError` for a ``dict`` output, a non-array output, or an
    array whose shape is not the domain's — ``with_data`` can only put
    values back on the field's own grid.
    """
    name = type(aggregation).__name__
    if isinstance(output, Mapping):
        keys = ", ".join(repr(k) for k in list(output)[:3])
        more = ", …" if len(output) > 3 else ""
        raise TypeError(
            f"{caller} needs an aggregation that returns one array on the "
            f"field's grid, but {name} returned a dict (keys: {keys}{more}). "
            "Call `merge` for the raw output instead."
        )
    out_shape = getattr(output, "shape", None)
    if out_shape is None:
        raise TypeError(
            f"{caller} needs an aggregation that returns an array; {name} "
            f"returned {type(output).__name__}. Call `merge` for the raw "
            "output instead."
        )
    if shape is not None and tuple(out_shape) != shape:
        raise TypeError(
            f"{caller}: {name} returned shape {tuple(out_shape)} but the "
            f"field's domain has shape {shape}; with_data can only rebuild "
            "a field on the domain grid. Call `merge` for the raw output."
        )
    values = np.asarray(output)
    if dtype is None or values.dtype == dtype or not _fits_dtype(values, dtype):
        return values
    return values.astype(dtype)


def _fits_dtype(values: np.ndarray, dtype: np.dtype) -> bool:
    """Whether every value of ``values`` is representable in ``dtype``.

    Integer / bool targets need finite, integral, in-range values (so a
    NaN fill or a fractional mean keeps the float output); float targets
    only need no finite value to overflow. Non-numeric dtypes never fit.
    """
    if values.dtype.kind not in "biuf" or dtype.kind not in "biuf":
        return False
    if values.size == 0:
        return True
    if dtype.kind == "f":
        finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
        return finite.size == 0 or float(np.abs(finite).max()) <= np.finfo(dtype).max
    if values.dtype.kind == "f" and not (
        np.isfinite(values).all() and (values == np.trunc(values)).all()
    ):
        return False
    lo, hi = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
    return int(values.min()) >= lo and int(values.max()) <= hi


def _safe_base_weights(
    window: SpatialWindow, geometry: SpatialGeometry
) -> np.ndarray | None:
    """Compute the geometry-shaped base weights, or `None` for a ragged
    geometry (graph, polygon, spherical-cap) whose patch shape — and so
    any weight grid — is anchor-dependent.

    Only the "no fixed size" case is detected, and it is detected up
    front: a ``TypeError`` raised by the window itself (say, a bug in a
    `SpatialCustom` ``fn``) propagates instead of silently dropping the
    weights.
    """
    if getattr(geometry, "size", None) is None:
        return None
    return window.weights(geometry)


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
) -> tuple[Patch | None, bool]:
    """Read one anchor under the ``on_error`` policy.

    Returns ``(patch, read_ok)``: ``read_ok`` is ``True`` only when the
    patch came from a successful ``field.select``. A ``"mask"``
    placeholder (or a skipped ``None``) is ``False`` so callers never
    persist it in a `PatchCache`.
    """
    retries = max_retries if on_error == "retry" else 0
    indices = geometry.neighborhood(domain, anchor)
    pad_value = getattr(geometry, "pad_value", None)
    for retry_count in range(retries + 1):
        try:
            patch = _build_patch_from_indices(
                field, domain, anchor, indices, base_weights, boundary, pad_value
            )
            return patch, True
        except Exception as exc:
            # Preserve KeyboardInterrupt/SystemExit by handling only Exception.
            if isinstance(exc, StopIteration):
                raise
            if on_error == "raise":
                raise
            _record_patch_error(errors, anchor, exc, retry_count, capture_traceback)
            if on_error == "mask":
                placeholder = _build_mask_patch(
                    domain, anchor, indices, base_weights, boundary
                )
                return placeholder, False
            if on_error == "retry":
                if not _matches_retry_on(exc, retry_on):
                    raise
                if retry_count < retries:
                    continue
                return None, False
            return None, False
    return None, False


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
    weights = _build_weights(indices, base_weights, boundary=boundary, anchor=anchor)
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
    weights = _build_weights(indices, base_weights, boundary=boundary, anchor=anchor)
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
    weights = _build_weights(indices, base_weights, boundary=boundary, anchor=anchor)
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
    it here at the call boundary; the wrapper stays on `Patch.indices`,
    where every dense aggregation's `_resolve_indices` carries the mask
    into its valid-cell test (cells outside it are never counted).
    """
    if isinstance(indices, _MaskedWindow):
        return indices.window
    return indices


def _build_weights(
    indices: Any,
    base_weights: np.ndarray | None,
    *,
    boundary: str = "drop",
    anchor: Any = None,
) -> Any:
    """Resolve a patch's weight array.

    If the indices is a `_MaskedWindow` (SpatialPolygonIntersection on a raster),
    return the interior mask — the window controls *which pixels count*,
    not how heavily they're tapered. Otherwise return the geometry-shaped
    base weights from `SpatialWindow.weights`. Under ``boundary="shrink"``
    the geometry clipped the window to the domain on every side (top/left
    too, for a negative anchor), so the weights are cropped to the same
    in-domain part of the full patch: rows
    ``[row_off - anchor_row, row_off - anchor_row + height)``, and
    likewise for columns or `GridDomain` dims.
    """
    if isinstance(indices, _MaskedWindow):
        return indices.mask
    if boundary != "shrink" or base_weights is None:
        return base_weights
    crop = _shrink_crop(indices, anchor, base_weights.ndim)
    return base_weights if crop is None else base_weights[crop]


def _shrink_crop(indices: Any, anchor: Any, ndim: int) -> tuple[Any, ...] | None:
    """Slicer cropping full-patch weights to a shrunk window, or ``None``."""
    if hasattr(indices, "row_off") and hasattr(indices, "col_off"):
        r0, c0 = int(indices.row_off), int(indices.col_off)
        ar, ac = (r0, c0) if anchor is None else (int(anchor[-2]), int(anchor[-1]))
        dr, dc = r0 - ar, c0 - ac
        return (
            Ellipsis,
            slice(dr, dr + int(indices.height)),
            slice(dc, dc + int(indices.width)),
        )
    if not (isinstance(indices, dict) and isinstance(anchor, dict)):
        return None
    crop = []
    for dim, index in indices.items():
        if not isinstance(index, slice) or dim not in anchor:
            return None
        offset = int(index.start) - int(anchor[dim])
        crop.append(slice(offset, offset + int(index.stop) - int(index.start)))
    return tuple(crop) if len(crop) == ndim else None


def _raise_if_overflows(indices: Any, domain: Any) -> None:
    """Raise ``ValueError`` if ``indices`` extends past ``domain``.

    Used by `SpatialPatcher.split` when the geometry's ``boundary``
    policy is ``"raise"``. Meaningful for raster windows and `GridDomain`
    slice dicts; any other indices return early.
    """
    if _pad_request(indices, domain) is not None:
        raise ValueError(
            f"patch window {indices!r} overflows the domain shape "
            f"{tuple(getattr(domain, 'shape', ()))}; set boundary='pad', "
            "'reflect' or 'shrink' to allow."
        )


@dataclass(frozen=True)
class _Span:
    """One axis of a padded read: ``[start, stop)`` requested on ``[0, length)``."""

    start: int
    stop: int
    length: int

    @property
    def clipped(self) -> tuple[int, int]:
        """The in-domain part ``[lo, hi)`` of the requested range."""
        return max(self.start, 0), min(self.stop, self.length)

    @property
    def pads(self) -> tuple[int, int]:
        """``(before, after)`` widths that grow the clipped read back to full size."""
        lo, hi = self.clipped
        return lo - self.start, self.stop - hi

    def source(self, reflect: bool) -> tuple[int, int]:
        """The in-domain range to read so the pad can be filled.

        ``"pad"`` reads just the clipped range. ``"reflect"`` mirrors
        about the edge pixel (numpy's ``mode="reflect"``), so a pad of
        ``p`` before needs rows ``1 … p`` and a pad after needs
        ``L - 1 - p … L - 2``; the read grows inward to cover them, capped
        at the domain. When the pad exceeds ``L - 1`` the whole axis is
        read and numpy reflects repeatedly (period ``2(L - 1)``) — the same
        values a reflect-extended domain would hold.
        """
        lo, hi = self.clipped
        if not reflect:
            return lo, hi
        before, after = self.pads
        if before:
            hi = max(hi, min(before + 1, self.length))
        if after:
            lo = min(lo, max(self.length - 1 - after, 0))
        return lo, hi


@dataclass(frozen=True)
class _PadRequest:
    """An overflowing raster window or `GridDomain` slice dict, per axis.

    ``spans`` is keyed ``"y"`` / ``"x"`` for a raster `Window` and by dim
    name (in indexer order) for a `GridDomain` dict; ``indexer`` is the
    original indices, kept for the non-slice entries of a grid dict.
    """

    spans: dict[str, _Span]
    raster: bool
    indexer: Any

    def sources(self, reflect: bool) -> dict[str, tuple[int, int]]:
        if reflect:
            for dim, span in self.spans.items():
                if any(span.pads) and span.length < 2:
                    raise ValueError(
                        f"boundary='reflect' cannot mirror axis {dim!r} of "
                        f"length {span.length}; use boundary='pad'."
                    )
        return {dim: span.source(reflect) for dim, span in self.spans.items()}

    def source_indexer(self, sources: dict[str, tuple[int, int]]) -> Any:
        """The in-domain indexer handed to ``field.select``."""
        if self.raster:
            from rasterio.windows import Window

            (r0, r1), (c0, c1) = sources["y"], sources["x"]
            return Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
        return {**self.indexer, **{d: slice(lo, hi) for d, (lo, hi) in sources.items()}}

    def crops(self, sources: dict[str, tuple[int, int]]) -> dict[str, slice]:
        """Per-axis slice taking the padded source read down to the request.

        The padded read spans ``[src_lo - before, src_hi + after)``; the
        request starts ``clipped_lo - src_lo`` cells into it.
        """
        out = {}
        for dim, span in self.spans.items():
            offset = span.clipped[0] - sources[dim][0]
            out[dim] = slice(offset, offset + span.stop - span.start)
        return out


def _pad_request(indices: Any, domain: Any) -> _PadRequest | None:
    """Describe how ``indices`` overflows ``domain``, or ``None`` if it fits.

    Raises ``ValueError`` for a window with no in-domain cell at all —
    nothing can be read, and there is no edge to pad or mirror from.
    """
    if hasattr(indices, "row_off") and hasattr(indices, "col_off"):
        shape: tuple[int, ...] = tuple(getattr(domain, "shape", ()))
        if len(shape) < 2:
            return None
        r0, c0 = int(indices.row_off), int(indices.col_off)
        spans = {
            "y": _Span(r0, r0 + int(indices.height), int(shape[-2])),
            "x": _Span(c0, c0 + int(indices.width), int(shape[-1])),
        }
        raster = True
    elif isinstance(indices, dict) and isinstance(domain, GridDomain):
        spans = {}
        for dim, index in indices.items():
            if isinstance(index, slice) and index.step in (None, 1):
                length = len(domain.coords[dim])
                start = 0 if index.start is None else int(index.start)
                stop = length if index.stop is None else int(index.stop)
                spans[dim] = _Span(start, stop, length)
        raster = False
    else:
        return None
    if not any(any(span.pads) for span in spans.values()):
        return None
    for dim, span in spans.items():
        lo, hi = span.clipped
        if hi <= lo:
            raise ValueError(
                f"patch window {indices!r} does not intersect the domain on "
                f"axis {dim!r} (length {span.length})."
            )
    return _PadRequest(spans=spans, raster=raster, indexer=indices)


def _carrier_nodata(data: Any) -> Any:
    """Best-effort nodata / fill value for a selected patch carrier."""
    fill = getattr(data, "fill_value_default", None)
    if fill is not None:
        return fill
    rio = getattr(data, "rio", None) if _is_rio_dataarray(data) else None
    if rio is not None and getattr(rio, "nodata", None) is not None:
        return rio.nodata
    return 0


def _check_pad_value(pad_value: float, data: Any) -> None:
    """Raise if ``pad_value`` cannot be stored exactly in ``data``'s dtype.

    ``np.pad`` casts the constant silently: ``-999.0`` into ``uint16``
    wraps to ``64537``, ``NaN`` into an integer raster becomes an
    arbitrary integer. The check is ``np.can_cast`` of the value's
    minimal scalar type: an integer dtype needs a finite, integral value
    in range; a float dtype needs one that does not overflow it.
    """
    dtype = getattr(data, "dtype", None)
    dtype = np.asarray(data).dtype if dtype is None else np.dtype(dtype)
    value = float(pad_value)
    if dtype == np.bool_:
        ok = value in (0.0, 1.0)
    elif np.issubdtype(dtype, np.integer):
        ok = (
            np.isfinite(value)
            and value.is_integer()
            and bool(np.can_cast(np.min_scalar_type(int(value)), dtype))
        )
    else:
        ok = bool(np.can_cast(np.min_scalar_type(value), dtype))
    if not ok:
        raise ValueError(
            f"pad_value={pad_value!r} cannot be represented in the field's "
            f"{dtype} dtype; choose a value that fits it (or leave pad_value "
            "unset to pad with the reader's nodata)."
        )


def _is_rio_dataarray(data: Any) -> bool:
    """True for an `xarray.DataArray` carrying a rioxarray ``.rio`` accessor.

    Checked structurally (``dims`` + ``rio``) so xarray stays an optional
    import. Tested before the `GeoTensor` branch because xarray exposes
    ``attrs`` keys as attributes — a DataArray with a ``transform`` attr
    would otherwise look like a GeoTensor.
    """
    return hasattr(data, "dims") and hasattr(data, "coords") and hasattr(data, "rio")


def _pad_carrier(
    data: Any, pads: tuple[int, int, int, int], mode: str, fill: Any
) -> Any:
    """Pad a selected carrier up to full size, preserving georeferencing.

    Handles a rioxarray `DataArray` (coords rebuilt from the shifted
    affine, transform written), a georeader `GeoTensor` (whose ``pad``
    shifts the transform), and a plain ndarray.
    """
    pt, pb, pl, pr = pads
    if pt == pb == pl == pr == 0:
        return data
    if _is_rio_dataarray(data):
        from geopatcher._src.fields.rio_xarray import pad_dataarray

        return pad_dataarray(data, pads, mode=mode, fill=fill)
    const = {"constant_values": fill} if mode == "constant" else {}
    if hasattr(data, "pad") and hasattr(data, "transform"):
        return data.pad({"y": (pt, pb), "x": (pl, pr)}, mode=mode, **const)
    arr = np.asarray(data)
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(pt, pb), (pl, pr)]
    if mode == "constant":
        return np.pad(arr, pad_width, mode="constant", constant_values=fill)
    return np.pad(arr, pad_width, mode="reflect")


def _crop_carrier(data: Any, rows: slice, cols: slice) -> Any:
    """Crop a raster carrier's spatial axes, keeping its georeferencing exact."""
    if _is_rio_dataarray(data):
        from rasterio.windows import Window, transform as window_transform

        from geopatcher._src.fields.rio_xarray import _georeference

        y_dim, x_dim = data.rio.y_dim, data.rio.x_dim
        sub = data.isel({y_dim: rows, x_dim: cols})
        window = Window(
            col_off=cols.start,
            row_off=rows.start,
            width=sub.sizes[x_dim],
            height=sub.sizes[y_dim],
        )
        return _georeference(sub, window_transform(window, data.rio.transform()))
    if hasattr(data, "isel") and hasattr(data, "transform"):
        return data.isel({"y": rows, "x": cols})
    return np.asarray(data)[..., rows, cols]


def _finish_raster(
    data: Any,
    request: _PadRequest,
    sources: dict[str, tuple[int, int]],
    mode: str,
    fill: Any,
) -> Any:
    """Pad the in-domain raster read to the requested window."""
    (pt, pb), (pl, pr) = request.spans["y"].pads, request.spans["x"].pads
    padded = _pad_carrier(data, (pt, pb, pl, pr), mode, fill)
    crops = request.crops(sources)
    rows, cols = crops["y"], crops["x"]
    height, width = np.shape(padded)[-2:]
    if (rows.start, cols.start, rows.stop, cols.stop) == (0, 0, height, width):
        return padded
    return _crop_carrier(padded, rows, cols)


def _finish_grid(
    data: Any,
    request: _PadRequest,
    sources: dict[str, tuple[int, int]],
    domain: GridDomain,
    mode: str,
    fill: Any,
) -> Any:
    """Pad the in-domain `GridDomain` read to the requested slices.

    A `DataArray` chip is padded by dim name; its 1-D coordinates on the
    padded dims are rebuilt from the domain's (extrapolated past the edge
    at the edge spacing), and a rioxarray-georeferenced chip gets its
    transform shifted to the request origin. A plain array is padded
    positionally, in indexer order.
    """
    pads = {d: span.pads for d, span in request.spans.items()}
    crops = request.crops(sources)
    const = {"constant_values": fill} if mode == "constant" else {}
    if hasattr(data, "dims") and hasattr(data, "pad"):
        padded = data.pad(pads, mode=mode, **const).isel(crops)
        return _regrid_chip(data, padded, request, sources, domain)
    arr = np.asarray(data)
    dims = list(request.indexer)
    width = [pads.get(d, (0, 0)) for d in dims] + [(0, 0)] * (arr.ndim - len(dims))
    if mode == "constant":
        padded_arr = np.pad(arr, width, mode="constant", constant_values=fill)
    else:
        padded_arr = np.pad(arr, width, mode="reflect")
    return padded_arr[tuple(crops.get(d, slice(None)) for d in dims)]


def _regrid_chip(
    source: Any,
    padded: Any,
    request: _PadRequest,
    sources: dict[str, tuple[int, int]],
    domain: GridDomain,
) -> Any:
    """Rebuild a padded grid chip's coordinates (and rio transform)."""
    updates = {}
    for dim, span in request.spans.items():
        if dim in padded.coords and padded[dim].dims == (dim,):
            values = _extended_coord(domain.coords[dim], span.start, span.stop)
            if values is not None:
                updates[dim] = (dim, values, padded[dim].attrs)
    if updates:
        padded = padded.assign_coords(updates)
    if not _is_rio_dataarray(source):
        return padded
    try:
        rio = source.rio
        y_dim, x_dim = rio.y_dim, rio.x_dim
        georeferenced = rio.grid_mapping in source.coords
    except Exception:  # rioxarray's MissingSpatialDimensionError & co.
        return padded
    if not georeferenced:
        return padded
    from rasterio import Affine

    dy = request.spans[y_dim].start - sources[y_dim][0] if y_dim in sources else 0
    dx = request.spans[x_dim].start - sources[x_dim][0] if x_dim in sources else 0
    return padded.rio.write_transform(rio.transform() * Affine.translation(dx, dy))


def _extended_coord(coord: np.ndarray, start: int, stop: int) -> np.ndarray | None:
    """``coord[start:stop]``, linearly extrapolated past either end.

    Out-of-domain positions continue at the edge spacing (``c[1] - c[0]``
    before, ``c[-1] - c[-2]`` after). ``None`` for a coordinate shorter
    than two entries (no spacing to extrapolate with).
    """
    coord = np.asarray(coord)
    if len(coord) < 2:
        return None
    idx = np.arange(start, stop)
    inside = np.clip(idx, 0, len(coord) - 1)
    values = coord[inside]
    below, above = idx < 0, idx >= len(coord)
    if below.any():
        values[below] = coord[0] + idx[below] * (coord[1] - coord[0])
    if above.any():
        steps = idx[above] - (len(coord) - 1)
        values[above] = coord[-1] + steps * (coord[-1] - coord[-2])
    return values


def _finish_padded(
    data: Any,
    request: _PadRequest,
    sources: dict[str, tuple[int, int]],
    domain: Any,
    boundary: str,
    pad_value: float | None,
) -> Any:
    """Grow the in-domain read back to the requested window / slices."""
    mode = "reflect" if boundary == "reflect" else "constant"
    if mode == "constant" and pad_value is not None:
        _check_pad_value(pad_value, data)
    fill = pad_value if pad_value is not None else _carrier_nodata(data)
    if request.raster:
        return _finish_raster(data, request, sources, mode, fill)
    return _finish_grid(data, request, sources, domain, mode, fill)


def _select_padded(
    field: Field, domain: Any, window: Any, boundary: str, pad_value: float | None
) -> Any:
    """Read ``window`` under ``pad`` / ``reflect``, padding overflow to full size.

    Interior (non-overflowing) windows take the plain read path — the
    padding machinery only engages at the domain edge. An overflowing
    window is clipped to the domain (grown inward under ``reflect`` so
    the mirror source is in hand), read once, padded, and cropped back
    to the requested extent.
    """
    request = _pad_request(window, domain)
    if request is None:
        return field.select(window)
    sources = request.sources(boundary == "reflect")
    data = field.select(request.source_indexer(sources))
    return _finish_padded(data, request, sources, domain, boundary, pad_value)


async def _select_padded_async(
    field: AsyncField,
    domain: Any,
    window: Any,
    boundary: str,
    pad_value: float | None,
) -> Any:
    """Async mirror of `_select_padded`."""
    request = _pad_request(window, domain)
    if request is None:
        return await _select_async(field, window)
    sources = request.sources(boundary == "reflect")
    data = await _select_async(field, request.source_indexer(sources))
    return _finish_padded(data, request, sources, domain, boundary, pad_value)


# Re-export `_is_raster_domain` to discourage cross-imports from geometry.py.
__all__ = [
    "AsyncSpatialPatcher",
    "PatchErrorRecord",
    "SpatialPatcher",
    "_is_raster_domain",
]

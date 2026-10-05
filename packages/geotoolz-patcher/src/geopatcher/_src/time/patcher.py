"""`TemporalPatcher` — composes the four time axes.

Mirror of `SpatialPatcher` over a 1-D time axis. The Patcher splits a
series along its time dimension; for each anchor it produces a
`TemporalPatch` of data sliced by `TemporalGeometry.window`.

The series is read lazily, one window at a time: a `Field` with a
`GridDomain` (`XarrayField`, `DaskField`) through ``select({time_dim: s})``,
any other shape-bearing array (numpy, dask, an `xarray.DataArray`, zarr)
by slicing it, so a lazy series is never loaded whole. The runner-level
knobs are `SpatialPatcher`'s — ``on_error`` / ``max_retries`` /
``retry_on`` (with the same `PatchErrorRecord` log in ``errors``),
``journal``, ``cache``, ``prefetch``, ``max_in_flight`` /
``max_in_flight_bytes`` and the streaming / strict check on every merge —
and run through the same helpers.

Coordinate-aware components (`TemporalStencilGeometry`,
`TemporalStencilSampler`) opt in via the ``needs_coord = True`` ClassVar.
When either component sets it, every public method that takes ``series``
also requires a ``coord=`` 1-D coordinate vector along ``time_axis``
(defaulted from a `GridDomain` field's time coordinate). The integer path
is unchanged when no component is coord-aware. See ADR-004 in
``docs/decisions.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np

from geopatcher._src._serialize import patcher_config
from geopatcher._src.exceptions import IncompleteScanConfiguration
from geopatcher._src.hooks import (
    PatcherHook,
    _as_hooks,
    _dispatch,
    _nbytes,
)
from geopatcher._src.patch import TemporalPatch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.spatial.patcher import (
    _NO_DOMAIN,
    OnErrorPolicy,
    PatchErrorRecord,
    _amerge_with_hooks,
    _Backpressure,
    _closing,
    _exception_from_record,
    _merge_with_hooks,
    _read_with_policy,
    _SplitCancelled,
    _validate_backpressure,
    _validate_error_policy,
    _validate_retry_on,
)
from geopatcher._src.time.aggregation import TemporalAggregation
from geopatcher._src.time.geometry import TemporalGeometry
from geopatcher._src.time.sampler import TemporalSampler
from geopatcher._src.time.stencils import coord_step
from geopatcher._src.time.window import TemporalWindow


@dataclass(eq=False)
class TemporalPatcher:
    """Four-axis temporal Patcher.

    Args:
        geometry: How a temporal window is shaped around an anchor.
        sampler: Where time anchors are placed.
        window: Temporal boundary treatment (recency / taper / periodic).
        aggregation: Time → time merge strategy.
        on_error: Patch-read error policy, as `SpatialPatcher.on_error`:
            ``"raise"`` (default), ``"skip"``, ``"mask"`` (a NaN patch of
            the window's shape) or ``"retry"``. Failures are recorded in
            ``errors`` keyed by the patch key (see `patch_anchors`).
        max_retries: Number of retries when `on_error` is ``"retry"``.
        retry_on: Exception classes or class names that should be retried.
        capture_traceback: If ``False``, `PatchErrorRecord.traceback` is
            left empty.

    Examples:
        Lookback + horizon forecasting on a ``(time, feature)`` array::

            tp = TemporalPatcher(
                geometry    = TemporalLookbackHorizon(lookback=12, horizon=6),
                sampler     = TemporalRegularStride(step=1),
                window      = TemporalCausalBoxcar(),
                aggregation = TemporalForecast(horizon=6),
            )
            patches = list(tp.split(series))
            preds   = [p.with_data(model(p.data)) for p in patches]
            aligned = tp.merge(preds)   # {anchor: horizon block}
    """

    geometry: TemporalGeometry
    sampler: TemporalSampler
    window: TemporalWindow
    aggregation: TemporalAggregation
    on_error: OnErrorPolicy = "raise"
    max_retries: int = 0
    retry_on: tuple[type[BaseException] | str, ...] = (OSError, TimeoutError)
    capture_traceback: bool = True
    errors: list[PatchErrorRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _validate_error_policy(self.on_error, self.max_retries)
        self.retry_on = _validate_retry_on(self.retry_on)

    # -- coordinate plumbing ---------------------------------------------

    def _needs_coord(self) -> bool:
        return bool(
            getattr(self.geometry, "needs_coord", False)
            or getattr(self.sampler, "needs_coord", False)
        )

    def _require_coord(
        self, coord: Any | None, time_len: int | None = None
    ) -> np.ndarray | None:
        """Validate ``coord`` once and return it as an ndarray (or ``None``).

        1-D, ``time_len`` long and strictly increasing; evenly spaced too
        when a component is coordinate-aware (the stencil path resolves
        windows arithmetically from the step).
        """
        if self._needs_coord() and coord is None:
            raise ValueError(
                "Coordinate-aware geometry/sampler requires coord= "
                "(a 1-D monotonic-ascending coordinate along time_axis)."
            )
        if coord is None:
            return None
        coord = np.asarray(coord)
        if coord.ndim != 1:
            raise ValueError(f"coord must be 1-D; got shape {coord.shape}.")
        if time_len is not None and coord.shape[0] != int(time_len):
            # Catches mixed pipelines (e.g. integer sampler + stencil
            # geometry) where the sampler would yield indices past the
            # coord without this check.
            raise ValueError(
                "coord length must equal series.shape[time_axis]: "
                f"got coord.shape={coord.shape} vs time_len={time_len}."
            )
        if self._needs_coord():
            coord_step(coord)
        elif coord.shape[0] > 1:
            steps = np.diff(coord)
            if not np.all(steps > steps.dtype.type(0)):
                raise ValueError(
                    "coord must be strictly increasing (sorted ascending, no "
                    "duplicates)."
                )
        return coord

    def _resolve_coord(self, src: _TimeSource, coord: Any | None) -> np.ndarray | None:
        if coord is None and self._needs_coord() and src.dim is not None:
            coord = src.time_coord()
        return self._require_coord(coord, src.time_len)

    def _sampler_anchors(
        self, time_len: int, coord: np.ndarray | None
    ) -> Iterable[int]:
        if getattr(self.sampler, "needs_coord", False):
            return self.sampler.anchors(time_len, coord=coord)  # type: ignore[call-arg]
        return self.sampler.anchors(time_len)

    # -- window resolution -----------------------------------------------

    def _window(
        self, time_len: int, anchor: int, coord: np.ndarray | None
    ) -> slice | list[slice] | None:
        if getattr(self.geometry, "needs_coord", False):
            return self.geometry.window_coord(coord, anchor)  # type: ignore[attr-defined]
        return self.geometry.window(time_len, anchor)

    def _keyed_windows(
        self, time_len: int, anchor: int, coord: np.ndarray | None
    ) -> list[tuple[Any, int, slice]]:
        """``(patch key, window index, slice)`` for each window of ``anchor``.

        The key is the bare anchor for a single-window geometry and
        ``(anchor, k)`` for each window of a multi-window one — the keys
        `patch_anchors` lists, `patch_at` accepts, and the journal /
        cache / ``errors`` record.
        """
        window = self._window(time_len, anchor, coord)
        if window is None:
            return []
        if isinstance(window, list):
            return [((anchor, k), k, s) for k, s in enumerate(window)]
        return [(anchor, 0, window)]

    def _window_slices(
        self, time_len: int, anchor: int, coord: np.ndarray | None
    ) -> list[slice]:
        """The windows ``anchor`` yields, as a list (empty when dropped).

        The one place every temporal family (`TemporalPatcher`,
        `SpatioTemporalPatcher`, the matched patchers) resolves a temporal
        window: coord-aware geometries go through ``window_coord``, the
        rest through ``window``, and a ``None`` (dropped by the geometry's
        ``boundary``) becomes ``[]``.
        """
        return [s for _, _, s in self._keyed_windows(time_len, anchor, coord)]

    def _check_full_scan(
        self, anchors: list[int], time_len: int, coord: np.ndarray | None
    ) -> None:
        covered = np.zeros(int(time_len), dtype=bool)
        for anchor in anchors:
            for s in self._window_slices(time_len, anchor, coord):
                covered[s] = True
        if not covered.all():
            missing = np.flatnonzero(~covered)
            raise IncompleteScanConfiguration(
                f"{type(self.sampler).__name__}(check_full_scan=True): the "
                f"{type(self.geometry).__name__} windows leave {missing.size} of "
                f"{time_len} time steps uncovered (first: {missing[:5].tolist()}). "
                "Use a step no longer than the window, a start that reaches the "
                "head, or boundary='shrink'."
            )

    # -- reading ---------------------------------------------------------

    def _read_patch(
        self,
        src: _TimeSource,
        anchor: int,
        k: int,
        s: slice,
        weights: np.ndarray | None = None,
    ) -> TemporalPatch:
        if weights is None:
            weights = self.window.weights(self.geometry, s.stop - s.start)
        return TemporalPatch(
            data=src.read(s), anchor=anchor, indices=s, weights=weights, window_index=k
        )

    def _mask_patch(
        self, src: _TimeSource, anchor: int, k: int, s: slice, weights: np.ndarray
    ) -> TemporalPatch:
        return TemporalPatch(
            data=src.mask(s), anchor=anchor, indices=s, weights=weights, window_index=k
        )

    def _cache_context(
        self, cache: Any | None, src: _TimeSource, *, field_id: str | None = None
    ) -> Any | None:
        """Bind ``cache`` to this series + config, or ``None`` when disabled."""
        if cache is None:
            return None
        if field_id is None:
            field_id = cache.field_id_for(src.series)
        config_id = (
            f"{cache.config_id_for(self.geometry, self.window)}"
            f"|time_axis={src.time_axis}"
        )
        return (cache, field_id, config_id)

    def _cached_patch(
        self, ctx: Any | None, key: Any, anchor: int, k: int, s: slice
    ) -> TemporalPatch | None:
        if ctx is None:
            return None
        cache, field_id, config_id = ctx
        payload = cache.get(field_id, config_id, key)
        if payload is None:
            return None
        hit = cache.build_patch(payload, anchor, s)
        return TemporalPatch(
            data=hit.data, anchor=anchor, indices=s, weights=hit.weights, window_index=k
        )

    def _store_patch(self, ctx: Any | None, key: Any, patch: TemporalPatch) -> None:
        if ctx is not None:
            cache, field_id, config_id = ctx
            cache.put(field_id, config_id, key, patch)

    def _build_patch(
        self,
        src: _TimeSource,
        cache_ctx: Any | None,
        key: Any,
        anchor: int,
        k: int,
        s: slice,
        weights: np.ndarray,
    ) -> TemporalPatch | None:
        """One window: a cache hit, else a read under the ``on_error`` policy."""
        if cache_ctx is not None:
            hit = self._cached_patch(cache_ctx, key, anchor, k, s)
            if hit is not None:
                return hit
        if self.on_error == "raise":  # the policy loop would only re-raise
            patch: TemporalPatch | None = self._read_patch(src, anchor, k, s, weights)
            read_ok = True
        else:
            patch, read_ok = _read_with_policy(
                partial(self._read_patch, src, anchor, k, s, weights),
                mask=partial(self._mask_patch, src, anchor, k, s, weights),
                anchor=key,
                on_error=self.on_error,
                max_retries=self.max_retries,
                retry_on=self.retry_on,
                errors=self.errors,
                capture_traceback=self.capture_traceback,
            )
        if read_ok and patch is not None:
            self._store_patch(cache_ctx, key, patch)
        return patch

    # -- split -----------------------------------------------------------

    def split(
        self,
        series: Any,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[TemporalPatch]:
        """Yield temporal patches lazily.

        Same runner contract as `SpatialPatcher.split`: the ``on_error``
        policy, ``journal`` (skips patch keys it ``has``), ``cache`` (a
        `PatchCache` consulted before every read), ``prefetch`` and the
        ``max_in_flight`` / ``max_in_flight_bytes`` backpressure, whose
        slots each yielded patch owns until ``patch.close()`` (or
        ``with patch: ...``).

        Args:
            series: What to patch along ``time_axis``: a `Field` with a
                `GridDomain` (read through ``select``), or any array-like
                with ``shape`` that slices (numpy, dask, xarray, zarr) —
                only each window is read. Anything else goes through
                ``np.asarray``.
            time_axis: Which axis is the time axis. Default 0.
            coord: 1-D, strictly increasing coordinate along ``time_axis``.
                Required (and must be evenly spaced) when the geometry or
                sampler is coordinate-aware — a `GridDomain` field supplies
                its time coordinate by default; otherwise only reported to
                hooks as ``coord_value``.
            hooks: Optional observability hooks for split callbacks.
            prefetch: If positive, eagerly buffer up to ``prefetch`` patches
                in a background thread for I/O overlap.
            journal: Optional `PatchJournal`; patch keys it has are skipped.
            cache: Optional `PatchCache`.
            max_in_flight: Maximum number of unreleased patches.
            max_in_flight_bytes: Maximum total bytes of unreleased patches.

        Raises:
            IncompleteScanConfiguration: The sampler sets
                ``check_full_scan=True`` and the windows leave time steps
                uncovered (raised before the first patch).
        """
        return self._split_anchors(
            series,
            None,
            time_axis=time_axis,
            coord=coord,
            hooks=hooks,
            prefetch=prefetch,
            journal=journal,
            cache=cache,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )

    def _split_anchors(
        self,
        series: Any,
        anchors: Iterable[int] | None,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[TemporalPatch]:
        """`split` over explicit ``anchors`` (``None`` walks the sampler)."""
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        src = _source(series, time_axis)
        coord = self._resolve_coord(src, coord)
        stop = Event()
        return prefetch_iterable(
            self._split(
                src,
                anchors=anchors,
                coord=coord,
                hooks=hooks,
                journal=journal,
                cache=cache,
                backpressure=_Backpressure(max_in_flight, max_in_flight_bytes, stop),
            ),
            prefetch,
            stop=stop,
        )

    def _split(
        self,
        src: _TimeSource,
        *,
        anchors: Iterable[int] | None,
        coord: np.ndarray | None,
        hooks: Iterable[PatcherHook] | None,
        journal: Any | None,
        cache: Any | None,
        backpressure: _Backpressure,
    ) -> Iterator[TemporalPatch]:
        time_len = src.time_len
        if anchors is None:
            anchors = self._sampler_anchors(time_len, coord)
        hook_list = _as_hooks(hooks)
        full_scan = bool(getattr(self.sampler, "check_full_scan", False))
        if hook_list or full_scan:
            anchors = [int(a) for a in anchors]
            if full_scan:
                self._check_full_scan(anchors, time_len, coord)
        cache_ctx = self._cache_context(cache, src)
        # Window weights depend only on the window length: compute each
        # length once per split (every patch of that length shares it).
        weights_by_len: dict[int, np.ndarray] = {}
        start = 0.0  # timed only when hooks listen
        if hook_list:
            _dispatch(hook_list, "on_split_start", len(anchors))
        try:
            for raw_anchor in anchors:
                anchor = int(raw_anchor)
                try:
                    windows = self._keyed_windows(time_len, anchor, coord)
                except Exception as exc:
                    _dispatch(hook_list, "on_error", anchor, exc)
                    raise
                coord_value = coord[anchor] if hook_list and coord is not None else None
                for key, k, s in windows:
                    if journal is not None and journal.has(key):
                        continue
                    if hook_list:
                        _dispatch(hook_list, "on_patch_start", anchor, coord_value)
                        start = perf_counter()
                    errors_before = len(self.errors)
                    try:
                        n = s.stop - s.start
                        weights = weights_by_len.get(n)
                        if weights is None:
                            weights = self.window.weights(self.geometry, n)
                            weights_by_len[n] = weights
                        patch = self._build_patch(
                            src, cache_ctx, key, anchor, k, s, weights
                        )
                    except Exception as exc:
                        _dispatch(hook_list, "on_error", anchor, exc)
                        raise
                    for record in self.errors[errors_before:] if hook_list else ():
                        _dispatch(
                            hook_list,
                            "on_error",
                            anchor,
                            _exception_from_record(record),
                        )
                    if patch is None:
                        continue
                    try:
                        release = backpressure.acquire(patch)
                    except _SplitCancelled:
                        return
                    if release is not None:
                        patch._release = release
                    if hook_list:
                        _dispatch(
                            hook_list,
                            "on_patch_done",
                            anchor,
                            perf_counter() - start,
                            _nbytes(patch.data),
                            coord_value,
                        )
                    yield patch
        finally:
            if hook_list:
                _dispatch(hook_list, "on_split_end")

    async def asplit(
        self,
        series: Any,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> AsyncIterator[TemporalPatch]:
        """Async iterator over `split` for async pipeline composition.

        Takes `split`'s arguments; each patch is pulled in a worker thread
        (``asyncio.to_thread``), so reads never block the event loop and
        ``prefetch`` overlaps them with the consumer.
        """
        patches = self.split(
            series,
            time_axis=time_axis,
            coord=coord,
            hooks=hooks,
            prefetch=prefetch,
            journal=journal,
            cache=cache,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )
        try:
            while True:
                patch: TemporalPatch | None = await asyncio.to_thread(
                    next, patches, None
                )
                if patch is None:
                    return
                yield patch
        finally:
            close = getattr(patches, "close", None)
            if close is not None:
                # A cancelled pull may still be running in its thread.
                with contextlib.suppress(ValueError):
                    close()

    # -- random access ---------------------------------------------------

    def patches_at(
        self,
        series: Any,
        anchor: int,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
    ) -> list[TemporalPatch]:
        """Return the patches `split` would yield for a single anchor.

        Always a list — length 1 for the common single-slice
        geometries, length N for `TemporalMultiScale` /
        `TemporalPhaseWindow` (one entry per window), and empty when the
        geometry's ``boundary="drop"`` drops the anchor.
        The spatial counterpart returns a single `Patch`; the temporal
        side has to flatten the multi-scale case, so the return type
        is a list either way for callers to handle uniformly.

        Args:
            series: Same input as `split`; only the anchor's windows are
                read.
            anchor: A single anchor value (typically from
                ``patcher.anchors(series)[index]``).
            time_axis: Which axis is the time axis. Default 0.
            coord: See `split`.
        """
        src = _source(series, time_axis)
        coord = self._resolve_coord(src, coord)
        return [
            self._read_patch(src, int(anchor), k, s)
            for _, k, s in self._keyed_windows(src.time_len, int(anchor), coord)
        ]

    def patch_at(
        self,
        series: Any,
        anchor: int | tuple[int, int],
        *,
        time_axis: int = 0,
        coord: Any | None = None,
        cache: Any | None = None,
        field_id: str | None = None,
    ) -> TemporalPatch:
        """Return the single patch `split` yields for one patch key.

        The temporal counterpart of `SpatialPatcher.patch_at`, so a
        `TemporalPatcher` drops into `IndexedPatchView` (including its
        ``cache=PatchCache(...)`` mode). Only the selected window is read
        from ``series``.

        Args:
            series: Same input as `split`.
            anchor: A key from `patch_anchors` — a bare anchor for a
                single-window geometry, or ``(anchor, k)`` for the
                ``k``-th window of a multi-window geometry such as
                `TemporalMultiScale` (``(anchor, 0)`` also works for a
                single-window geometry).
            time_axis: Which axis is the time axis. Default 0.
            coord: See `split`.
            cache: Optional `PatchCache`, as in `SpatialPatcher.patch_at`.
            field_id: ``cache.field_id_for(series)`` resolved once by the
                caller and reused for every key.

        Raises:
            ValueError: If a bare anchor is given for a geometry that
                yields several windows for it, or ``k`` is out of range.
        """
        if isinstance(anchor, tuple):
            base, k = int(anchor[0]), int(anchor[1])
        else:
            base, k = int(anchor), None
        src = _source(series, time_axis)
        coord = self._resolve_coord(src, coord)
        windows = self._keyed_windows(src.time_len, base, coord)
        if k is None:
            if len(windows) != 1:
                raise ValueError(
                    f"anchor {base} yields {len(windows)} patches with "
                    f"{type(self.geometry).__name__}; pass ({base}, k) to pick "
                    "one (see `patch_anchors`) or use `patches_at`."
                )
            k = 0
        if not 0 <= k < len(windows):
            raise ValueError(
                f"anchor {base} yields {len(windows)} patches; window index "
                f"{k} is out of range."
            )
        key, k, s = windows[k]
        ctx = self._cache_context(cache, src, field_id=field_id)
        cached = self._cached_patch(ctx, key, base, k, s)
        if cached is not None:
            return cached
        patch = self._read_patch(src, base, k, s)
        self._store_patch(ctx, key, patch)
        return patch

    def patch_anchors(
        self,
        series: Any,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
    ) -> list[int | tuple[int, int]]:
        """One `patch_at` key per patch `split` yields, in `split` order.

        A bare anchor where the geometry yields one window for it, and
        ``(anchor, k)`` for each of the windows a multi-window geometry
        (`TemporalMultiScale`) yields, so
        ``len(patch_anchors(series)) == n_anchors(series)``.
        `IndexedPatchView` indexes by these keys; the journal, cache and
        ``errors`` record the same keys.
        """
        src = _source(series, time_axis)
        coord = self._resolve_coord(src, coord)
        return [
            key
            for anchor in self._sampler_anchors(src.time_len, coord)
            for key, _, _ in self._keyed_windows(src.time_len, int(anchor), coord)
        ]

    def anchors(
        self,
        series: Any,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
    ) -> list[int]:
        """Materialise the anchors `split` yields patches for.

        The sampler's sequence minus the anchors the geometry's
        ``boundary="drop"`` drops, so every returned anchor yields at
        least one patch. ``len(anchors) <= n_anchors`` — multi-window
        geometries emit several patches per anchor. Same determinism
        contract as `n_anchors`. See `SpatialPatcher.anchors`.
        """
        src = _source(series, time_axis)
        coord = self._resolve_coord(src, coord)
        return [
            int(a)
            for a in self._sampler_anchors(src.time_len, coord)
            if self._window_slices(src.time_len, int(a), coord)
        ]

    def n_anchors(
        self,
        series: Any,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
    ) -> int:
        """Number of patches `split(series)` will yield.

        Walks the sampler **and** the geometry's per-anchor window — a
        single geometry call may return a ``list[slice]`` (e.g.
        `TemporalMultiScale`), in which case `split` yields one patch
        per slice. Only ``series.shape`` (or a field's domain) is read,
        so lazy series don't get materialised here. See
        ``docs/decisions.md`` (ADR-001).
        """
        return len(self.patch_anchors(series, time_axis=time_axis, coord=coord))

    # -- merge -----------------------------------------------------------

    def merge(
        self, patches: Iterable[Any], hooks: Iterable[PatcherHook] | None = None
    ) -> Any:
        """Hand the patches to the aggregation and return its output.

        A ``streaming_safe = False`` aggregation warns (or raises under
        `set_strict`) at the caller's line, as `SpatialPatcher.merge` does.
        """
        return _merge_with_hooks(
            self.aggregation, patches, _NO_DOMAIN, hooks, stacklevel=2
        )

    async def amerge(
        self,
        patches: AsyncIterable[Any] | Iterable[Any],
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Any:
        """Async-friendly `merge` over an async or sync patch iterable.

        An async stream is not materialised: the aggregation runs in a
        worker thread and pulls one patch at a time from the event loop.
        """
        return await _amerge_with_hooks(
            self.aggregation, patches, _NO_DOMAIN, hooks, stacklevel=2
        )

    def reduce(
        self,
        series: Any,
        agg: TemporalAggregation,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Any:
        """Run one streaming pass of `split` into ``agg`` and return its result.

        The `split` knobs (``on_error``, hooks, prefetch, journal, cache,
        backpressure) apply as there; each patch is closed (its
        backpressure slot released) once ``agg`` moves on, and ``agg``
        gets `merge`'s streaming-safety check.
        """
        patches = self._split_anchors(
            series,
            None,
            time_axis=time_axis,
            coord=coord,
            hooks=hooks,
            prefetch=prefetch,
            journal=journal,
            cache=cache,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )
        return _merge_with_hooks(
            agg,
            _closing(patches),
            _NO_DOMAIN,
            hooks,
            stacklevel=2,
        )

    def two_pass(
        self,
        series: Any,
        *,
        reduce_with: TemporalAggregation,
        apply: Callable[[Any, Any], Any],
        aggregation: TemporalAggregation | None = None,
        time_axis: int = 0,
        coord: Any | None = None,
        hooks: Iterable[PatcherHook] | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Any:
        """A statistics pass, then ``apply(data, stats)`` and a merge.

        Mirrors `SpatialPatcher.two_pass`: pass one is `reduce` with
        ``reduce_with``; pass two maps every patch through
        ``apply(data, stats)`` and merges with ``aggregation`` (default:
        the patcher's own). Both passes walk one anchor list materialised
        up front, so an unseeded `TemporalRandom` places them identically.
        """
        src = _source(series, time_axis)
        anchors = [
            int(a)
            for a in self._sampler_anchors(
                src.time_len, self._resolve_coord(src, coord)
            )
        ]
        split_kwargs: dict[str, Any] = {
            "time_axis": time_axis,
            "coord": coord,
            "hooks": hooks,
            "prefetch": prefetch,
            "journal": journal,
            "cache": cache,
            "max_in_flight": max_in_flight,
            "max_in_flight_bytes": max_in_flight_bytes,
        }
        stats = _merge_with_hooks(
            reduce_with,
            _closing(self._split_anchors(series, anchors, **split_kwargs)),
            _NO_DOMAIN,
            hooks,
            stacklevel=2,
        )
        applied = (
            patch.with_data(apply(patch.data, stats))
            for patch in _closing(self._split_anchors(series, anchors, **split_kwargs))
        )
        return _merge_with_hooks(
            aggregation or self.aggregation,
            applied,
            _NO_DOMAIN,
            hooks,
            stacklevel=2,
        )

    def to_delayed(
        self,
        series: Any,
        operator: Any | None = None,
        *,
        time_axis: int = 0,
        coord: Any | None = None,
    ) -> list[Any]:
        """One Dask delayed `patch_at` task per patch key (see `geopatcher.dask`).

        Runner-level policies (``on_error``, hooks, journal, cache,
        prefetch) do not apply.
        """
        from geopatcher.dask import to_delayed

        return to_delayed(self, series, operator, time_axis=time_axis, coord=coord)

    def to_dask_bag(
        self, series: Any, *, time_axis: int = 0, coord: Any | None = None
    ) -> Any:
        """A Dask bag with one item (and one partition) per patch key."""
        from geopatcher.dask import to_dask_bag

        return to_dask_bag(self, series, time_axis=time_axis, coord=coord)

    def get_config(self) -> dict[str, Any]:
        """Axes as ``{"class", "config"}`` envelopes plus the runner knobs.

        Same shape as `SpatialPatcher.get_config` (`patcher_config`):
        `geopatcher.from_config` rebuilds the patcher, ``on_error`` /
        ``retry_on`` included.
        """
        return patcher_config(self)


@dataclass(frozen=True, eq=False)
class _TimeSource:
    """The series a `TemporalPatcher` reads, one time window at a time.

    ``dim`` is set for a `GridDomain` field (read through
    ``select({dim: s})``); otherwise ``series`` is an array-like sliced
    directly, so a lazy array materialises only the window.
    """

    series: Any
    time_axis: int
    shape: tuple[int, ...]
    dim: str | None = None

    @property
    def time_len(self) -> int:
        return self.shape[self.time_axis]

    def read(self, s: slice) -> Any:
        if self.dim is not None:
            return self.series.select({self.dim: s})
        if self.time_axis == 0:
            return np.asarray(self.series[s])
        idx: list[Any] = [slice(None)] * len(self.shape)
        idx[self.time_axis] = s
        return np.asarray(self.series[tuple(idx)])

    def mask(self, s: slice) -> np.ndarray:
        shape = list(self.shape)
        shape[self.time_axis] = s.stop - s.start
        return np.full(shape, np.nan)

    def time_coord(self) -> np.ndarray:
        return np.asarray(self.series.domain.coords[self.dim])


def _source(series: Any, time_axis: int) -> _TimeSource:
    """Wrap ``series`` for windowed reads without loading it.

    A `Field` whose domain carries ``coords`` (`GridDomain`) is read via
    ``select``; any object with ``shape`` is sliced as-is (``np.asarray``
    runs on each window only); anything else is converted once.
    """
    domain = getattr(series, "domain", None)
    if domain is not None and callable(getattr(series, "select", None)):
        coords = getattr(domain, "coords", None)
        if not isinstance(coords, dict):
            raise TypeError(
                "TemporalPatcher reads a Field through its GridDomain "
                "(XarrayField, DaskField); "
                f"{type(series).__name__} has a {type(domain).__name__}. Pass "
                "the underlying array instead."
            )
        dims = list(coords)
        shape = tuple(len(coords[d]) for d in dims)
        axis = _normalize_axis(time_axis, len(shape))
        return _TimeSource(series, axis, shape, dims[axis])
    if not hasattr(series, "shape"):
        series = np.asarray(series)
    shape = tuple(int(n) for n in series.shape)
    return _TimeSource(series, _normalize_axis(time_axis, len(shape)), shape)


def _normalize_axis(time_axis: int, ndim: int) -> int:
    if not -ndim <= int(time_axis) < ndim:
        raise ValueError(f"time_axis {time_axis} is out of range for {ndim}-D data")
    return int(time_axis) % ndim

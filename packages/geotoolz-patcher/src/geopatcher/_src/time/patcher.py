"""`TemporalPatcher` — composes the four time axes.

Mirror of `SpatialPatcher` over a 1-D time axis. The Patcher splits a
field along its time dimension; for each anchor it produces a
`TemporalPatch` of data sliced by `TemporalGeometry.window`.

Coordinate-aware components (`TemporalStencilGeometry`,
`TemporalStencilSampler`) opt in via the ``needs_coord = True`` ClassVar.
When either component sets it, every public method that takes ``series``
also requires a ``coord=`` 1-D coordinate vector along ``time_axis``. The
integer path is unchanged when no component is coord-aware. See ADR-004 in
``docs/decisions.md``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from geopatcher._src._serialize import axis_envelope
from geopatcher._src.hooks import (
    PatcherHook,
    _as_hooks,
    _dispatch,
    _len_or_unknown,
    _nbytes,
)
from geopatcher._src.patch import TemporalPatch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.spatial.aggregation import _warn_if_unsafe_streaming
from geopatcher._src.time.aggregation import TemporalAggregation
from geopatcher._src.time.geometry import TemporalGeometry
from geopatcher._src.time.sampler import TemporalSampler
from geopatcher._src.time.window import TemporalWindow


@dataclass(eq=False)
class TemporalPatcher:
    """Four-axis temporal Patcher.

    Args:
        geometry: How a temporal window is shaped around an anchor.
        sampler: Where time anchors are placed.
        window: Temporal boundary treatment (recency / taper / periodic).
        aggregation: Time → time merge strategy.

    Examples:
        Lookback + horizon forecasting on a ``(time, feature)`` array::

            tp = TemporalPatcher(
                geometry    = TemporalLookbackHorizon(lookback=12, horizon=6),
                sampler     = TemporalRegularStride(step=1),
                window      = TemporalCausalBoxcar(),
                aggregation = TemporalForecast(horizon=6),
            )
            patches = list(tp.split(series))
            preds   = [model(p.data) for p in patches]
            aligned = tp.merge(preds_as_patches)
    """

    geometry: TemporalGeometry
    sampler: TemporalSampler
    window: TemporalWindow
    aggregation: TemporalAggregation

    def _needs_coord(self) -> bool:
        return bool(
            getattr(self.geometry, "needs_coord", False)
            or getattr(self.sampler, "needs_coord", False)
        )

    def _require_coord(
        self, coord: np.ndarray | None, time_len: int | None = None
    ) -> None:
        if self._needs_coord() and coord is None:
            raise ValueError(
                "Coordinate-aware geometry/sampler requires coord= "
                "(a 1-D monotonic-ascending coordinate along time_axis)."
            )
        if coord is not None:
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

    def _sampler_anchors(
        self, time_len: int, coord: np.ndarray | None
    ) -> Iterable[int]:
        if getattr(self.sampler, "needs_coord", False):
            return self.sampler.anchors(time_len, coord=coord)  # type: ignore[call-arg]
        return self.sampler.anchors(time_len)

    def split(
        self,
        series: Any,
        time_axis: int = 0,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        coord: np.ndarray | None = None,
        prefetch: int = 0,
    ) -> Iterator[TemporalPatch]:
        """Yield temporal patches lazily.

        Args:
            series: Numpy array (or anything with ``shape`` + slicing) to
                slice along ``time_axis``.
            time_axis: Which axis is the time axis. Default 0.
            hooks: Optional observability hooks for split callbacks.
            coord: 1-D coordinate vector along ``time_axis``. Required when
                the geometry or sampler is coordinate-aware; ignored
                otherwise.
            prefetch: If positive, eagerly buffer up to ``prefetch`` patches
                in a background thread for I/O overlap.
        """
        return prefetch_iterable(
            self._split(series, time_axis, coord=coord, hooks=hooks), prefetch
        )

    def _split(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Iterator[TemporalPatch]:
        arr = np.asarray(series)
        time_len = int(arr.shape[time_axis])
        self._require_coord(coord, time_len)
        hook_list = _as_hooks(hooks)
        if not hook_list:
            for anchor in self._sampler_anchors(time_len, coord):
                yield from self._patches_for_anchor(
                    arr, time_len, int(anchor), time_axis, coord=coord
                )
            return
        anchors = [int(a) for a in self._sampler_anchors(time_len, coord)]
        _dispatch(hook_list, "on_split_start", len(anchors))
        try:
            for anchor in anchors:
                yield from self._patches_for_anchor(
                    arr, time_len, anchor, time_axis, hook_list, coord=coord
                )
        finally:
            _dispatch(hook_list, "on_split_end")

    async def asplit(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> AsyncIterator[TemporalPatch]:
        """Async iterator mirror of `split` for async pipeline composition."""
        for patch in self._split(series, time_axis, coord=coord, hooks=hooks):
            yield patch

    def patches_at(
        self,
        series: Any,
        anchor: int,
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
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
            series: Same input shape as `split`.
            anchor: A single anchor value (typically from
                ``patcher.anchors(series)[index]``).
            time_axis: Which axis is the time axis. Default 0.
            coord: See `split`.
        """
        arr = _sliceable(series)
        time_len = int(arr.shape[time_axis])
        self._require_coord(coord, time_len)
        return list(
            self._patches_for_anchor(arr, time_len, int(anchor), time_axis, coord=coord)
        )

    def patch_at(
        self,
        series: Any,
        anchor: int | tuple[int, int],
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
    ) -> TemporalPatch:
        """Return the single patch `split` yields for one patch key.

        The temporal counterpart of `SpatialPatcher.patch_at`, so a
        `TemporalPatcher` drops into `IndexedPatchView`. Only the selected
        window is sliced out of ``series`` (then converted with
        ``np.asarray``), so a lazy series is not materialised whole.

        Args:
            series: Same input shape as `split`.
            anchor: A key from `patch_anchors` — a bare anchor for a
                single-window geometry, or ``(anchor, k)`` for the
                ``k``-th window of a multi-window geometry such as
                `TemporalMultiScale` (``(anchor, 0)`` also works for a
                single-window geometry).
            time_axis: Which axis is the time axis. Default 0.
            coord: See `split`.

        Raises:
            ValueError: If a bare anchor is given for a geometry that
                yields several windows for it, or ``k`` is out of range.
        """
        if isinstance(anchor, tuple):
            base, k = int(anchor[0]), int(anchor[1])
        else:
            base, k = int(anchor), None
        patches = self.patches_at(series, base, time_axis, coord=coord)
        if k is None:
            if len(patches) != 1:
                raise ValueError(
                    f"anchor {base} yields {len(patches)} patches with "
                    f"{type(self.geometry).__name__}; pass ({base}, k) to pick "
                    "one (see `patch_anchors`) or use `patches_at`."
                )
            return patches[0]
        if not 0 <= k < len(patches):
            raise ValueError(
                f"anchor {base} yields {len(patches)} patches; window index "
                f"{k} is out of range."
            )
        return patches[k]

    def patch_anchors(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
    ) -> list[int | tuple[int, int]]:
        """One `patch_at` key per patch `split` yields, in `split` order.

        A bare anchor where the geometry yields one window for it, and
        ``(anchor, k)`` for each of the windows a multi-window geometry
        (`TemporalMultiScale`) yields, so
        ``len(patch_anchors(series)) == n_anchors(series)``.
        `IndexedPatchView` indexes by these keys.
        """
        shape = getattr(series, "shape", None) or np.shape(series)
        time_len = int(shape[time_axis])
        self._require_coord(coord, time_len)
        keys: list[int | tuple[int, int]] = []
        for anchor in self._sampler_anchors(time_len, coord):
            a = int(anchor)
            window = self._window(time_len, a, coord)
            if isinstance(window, list):
                keys.extend((a, k) for k in range(len(window)))
            elif window is not None:
                keys.append(a)
        return keys

    def _window(
        self, time_len: int, anchor: int, coord: np.ndarray | None
    ) -> slice | list[slice] | None:
        if getattr(self.geometry, "needs_coord", False):
            return self.geometry.window_coord(coord, anchor)  # type: ignore[attr-defined]
        return self.geometry.window(time_len, anchor)

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
        window = self._window(time_len, anchor, coord)
        if window is None:
            return []
        return window if isinstance(window, list) else [window]

    def anchors(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
    ) -> list[int]:
        """Materialise the anchors `split` yields patches for.

        The sampler's sequence minus the anchors the geometry's
        ``boundary="drop"`` drops, so every returned anchor yields at
        least one patch. ``len(anchors) <= n_anchors`` — multi-window
        geometries emit several patches per anchor. Same determinism
        contract as `n_anchors`. See `SpatialPatcher.anchors`.
        """
        shape = getattr(series, "shape", None) or np.shape(series)
        time_len = int(shape[time_axis])
        self._require_coord(coord, time_len)
        return [
            int(a)
            for a in self._sampler_anchors(time_len, coord)
            if self._window_slices(time_len, int(a), coord)
        ]

    def _patches_for_anchor(
        self,
        arr: np.ndarray,
        time_len: int,
        anchor: int,
        time_axis: int,
        hooks: Iterable[PatcherHook] = (),
        *,
        coord: np.ndarray | None = None,
    ) -> Iterator[TemporalPatch]:
        try:
            slices = self._window_slices(time_len, anchor, coord)
        except Exception as exc:
            _dispatch(hooks, "on_error", anchor, exc)
            raise
        coord_value = coord[int(anchor)] if coord is not None else None
        for k, s in enumerate(slices):
            _dispatch(hooks, "on_patch_start", anchor, coord_value)
            start = perf_counter()
            try:
                idx = [slice(None)] * arr.ndim
                idx[time_axis] = s
                # A no-op for the ndarray `split` passes; slices a lazy
                # series (`patch_at`) before converting only the window.
                data = np.asarray(arr[tuple(idx)])
                weights = self.window.weights(self.geometry, s.stop - s.start)
                patch = TemporalPatch(
                    data=data,
                    anchor=anchor,
                    indices=s,
                    weights=weights,
                    window_index=k,
                )
            except Exception as exc:
                _dispatch(hooks, "on_error", anchor, exc)
                raise
            _dispatch(
                hooks,
                "on_patch_done",
                anchor,
                perf_counter() - start,
                _nbytes(patch.data),
                coord_value,
            )
            yield patch

    def n_anchors(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        coord: np.ndarray | None = None,
    ) -> int:
        """Number of patches `split(series)` will yield.

        Walks the sampler **and** the geometry's per-anchor window — a
        single geometry call may return a ``list[slice]`` (e.g.
        `TemporalMultiScale`), in which case `split` yields one patch
        per slice. We read ``series.shape`` rather than calling
        ``np.asarray(series)`` so generic / lazy series don't get
        materialised here. See ``docs/decisions.md`` (ADR-001).
        """
        return len(self.patch_anchors(series, time_axis, coord=coord))

    def merge(
        self, patches: Iterable[Any], hooks: Iterable[PatcherHook] | None = None
    ) -> Any:
        """Hand the patches to the aggregation and return its output.

        A ``streaming_safe = False`` aggregation warns (or raises under
        `set_strict`) at the caller's line, as `SpatialPatcher.merge` does.
        """
        hook_list = _as_hooks(hooks)
        _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
        _warn_if_unsafe_streaming(self.aggregation, stacklevel=3)
        try:
            output = self.aggregation.merge(patches)
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(output))
        return output

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": axis_envelope(self.geometry),
            "sampler": axis_envelope(self.sampler),
            "window": axis_envelope(self.window),
            "aggregation": axis_envelope(self.aggregation),
        }


def _sliceable(series: Any) -> Any:
    """``series`` itself when it slices like an array, else ``np.asarray``.

    Lets `patches_at` / `patch_at` slice a lazy (dask / xarray) series
    before converting, instead of materialising the whole of it per call.
    """
    if hasattr(series, "shape") and hasattr(series, "ndim"):
        return series
    return np.asarray(series)

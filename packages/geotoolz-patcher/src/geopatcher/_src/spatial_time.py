"""`SpatioTemporalPatcher` — composes a `SpatialPatcher` and a `TemporalPatcher`.

Two coupling modes:

- ``"product"`` (default) - every spatial anchor crossed with every time anchor.
  The right default for dense gridded data where space and time are
  independent grids (climate model output, regular satellite revisits).
- ``"coupled"`` — explicit ``(space, time)`` anchor pairs. The right
  shape for event-triggered patches (methane plume detections, Argo
  profile (lat, lon, t) records, storm tracks).

The Field is expected to expose a temporal axis as either an integer
``time_len`` attribute or a ``time`` coordinate. The patcher reads the
spatial slice, then the temporal slice, then yields a `SpatioTemporalPatch`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from geopatcher._src.hooks import (
    UNKNOWN_TOTAL,
    PatcherHook,
    _as_hooks,
    _dispatch,
    _len_or_unknown,
    _nbytes,
)
from geopatcher._src.patch import SpatioTemporalPatch, TemporalPatch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.spatial import SpatialPatcher
from geopatcher._src.spatial.patcher import _select_async
from geopatcher._src.time.patcher import TemporalPatcher


@dataclass(eq=False)
class SpatioTemporalPatcher:
    """Composition of a spatial and a temporal Patcher.

    Args:
        spatial: A `SpatialPatcher`.
        temporal: A `TemporalPatcher`.
        coupling: ``"product"`` (Cartesian product of anchors) or
            ``"coupled"`` (explicit ``(space, time)`` tuples from the
            spatial sampler's anchors_).
        time_axis: Which axis of the spatial patch's data is the time
            axis after the spatial slice has been read. Default ``0``.
    """

    spatial: SpatialPatcher
    temporal: TemporalPatcher
    coupling: Literal["product", "coupled"] = "product"
    time_axis: int = 0

    def split(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        coord: np.ndarray | None = None,
        prefetch: int = 0,
    ) -> Iterator[SpatioTemporalPatch]:
        """Yield `SpatioTemporalPatch`es lazily.

        The coupled mode expects ``self.spatial.sampler.anchors_`` to be
        an iterable of ``(space_anchor, time_anchor)`` tuples and is
        only valid with `SpatialExplicit` spatial / time samplers.

        Args:
            field: The field to split.
            hooks: Optional observability hooks for split callbacks.
            coord: 1-D coordinate vector along ``time_axis`` of the
                spatial patch's data. Required when the temporal
                geometry or sampler is coordinate-aware
                (``needs_coord = True``); ignored otherwise.
            prefetch: If positive, eagerly buffer up to ``prefetch``
                patches in a background thread for I/O overlap.
        """
        return prefetch_iterable(self._split(field, coord=coord, hooks=hooks), prefetch)

    def _split(
        self,
        field: Any,
        *,
        coord: np.ndarray | None = None,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Iterator[SpatioTemporalPatch]:
        coupling = self._checked_coupling()
        self.temporal._require_coord(coord)
        hook_list = _as_hooks(hooks)
        if not hook_list:
            if coupling == "product":
                yield from self._split_product(field, coord=coord)
            else:
                yield from self._split_coupled(field, coord=coord)
            return
        _dispatch(hook_list, "on_split_start", self._split_total_hint(field))
        try:
            if coupling == "product":
                yield from self._split_product(field, hook_list, coord=coord)
            else:
                yield from self._split_coupled(field, hook_list, coord=coord)
        finally:
            _dispatch(hook_list, "on_split_end")

    async def asplit(
        self,
        field: Any,
        *,
        coord: np.ndarray | None = None,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> AsyncIterator[SpatioTemporalPatch]:
        """Async iterator mirror of `split` for async spatial fields."""
        coupling = self._checked_coupling()
        self.temporal._require_coord(coord)
        hook_list = _as_hooks(hooks)
        if not hook_list:
            if coupling == "product":
                async for patch in self._asplit_product(field, coord=coord):
                    yield patch
            else:
                async for patch in self._asplit_coupled(field, coord=coord):
                    yield patch
            return
        _dispatch(hook_list, "on_split_start", self._split_total_hint(field))
        try:
            if coupling == "product":
                async for patch in self._asplit_product(field, hook_list, coord=coord):
                    yield patch
            else:
                async for patch in self._asplit_coupled(field, hook_list, coord=coord):
                    yield patch
        finally:
            _dispatch(hook_list, "on_split_end")

    def _checked_coupling(self) -> Literal["product", "coupled"]:
        if self.coupling not in {"product", "coupled"}:
            raise ValueError(f"unknown coupling: {self.coupling!r}")
        return self.coupling

    def _temporal_window(
        self, time_len: int, t_anchor: int, coord: np.ndarray | None
    ) -> Any:
        """Resolve the temporal geometry's window, coord-aware when needed.

        Mirrors `TemporalPatcher._patches_for_anchor`'s dispatch: a
        geometry with ``needs_coord = True`` resolves through
        ``window_coord(coord, anchor)``, everything else through the
        integer ``window(time_len, anchor)`` path.
        """
        if getattr(self.temporal.geometry, "needs_coord", False):
            return self.temporal.geometry.window_coord(coord, t_anchor)  # type: ignore[attr-defined]
        return self.temporal.geometry.window(time_len, t_anchor)

    def _split_product(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] = (),
        *,
        coord: np.ndarray | None = None,
    ) -> Iterator[SpatioTemporalPatch]:
        for sp in self.spatial.split(field):
            arr = np.asarray(sp.data)
            time_len = int(arr.shape[self.time_axis])
            self.temporal._require_coord(coord, time_len)
            for t_anchor in self.temporal._sampler_anchors(time_len, coord):
                t_window = self._temporal_window(time_len, int(t_anchor), coord)
                slices = t_window if isinstance(t_window, list) else [t_window]
                coord_value = coord[int(t_anchor)] if coord is not None else None
                for s in slices:
                    anchor = (sp.anchor, int(t_anchor))
                    _dispatch(hooks, "on_patch_start", anchor, coord_value)
                    start = perf_counter()
                    try:
                        idx = [slice(None)] * arr.ndim
                        idx[self.time_axis] = s
                        sub = arr[tuple(idx)]
                        patch = SpatioTemporalPatch(
                            data=sub,
                            space=sp.anchor,
                            time=int(t_anchor),
                            spatial_indices=sp.indices,
                            temporal_indices=s,
                            weights=sp.weights,
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

    def _split_coupled(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] = (),
        *,
        coord: np.ndarray | None = None,
    ) -> Iterator[SpatioTemporalPatch]:
        anchors = getattr(self.spatial.sampler, "anchors_", None)
        if anchors is None:
            raise TypeError(
                "coupled coupling requires the spatial sampler to expose an "
                "`anchors_` list of (space_anchor, time_anchor) tuples — i.e. "
                "use SpatialExplicit(anchors_=[...])."
            )
        # We can't know the time_len without reading a patch; assume the
        # operator-defined temporal geometry can clip indices itself, and
        # treat negative-time anchors as the caller's responsibility.
        for pair in anchors:
            space_anchor, time_anchor = pair
            anchor = (space_anchor, int(time_anchor))
            # Bounded lookup: coord length isn't validated until the patch
            # is read (time_len is unknown before the read in coupled
            # mode), so an out-of-range anchor must not raise IndexError
            # here — _require_coord below reports the documented
            # ValueError through the on_error dispatch instead.
            coord_value = (
                coord[int(time_anchor)]
                if coord is not None and 0 <= int(time_anchor) < len(coord)
                else None
            )
            _dispatch(hooks, "on_patch_start", anchor, coord_value)
            start = perf_counter()
            try:
                indices = self.spatial.geometry.neighborhood(field.domain, space_anchor)
                data = field.select(indices)
                arr = np.asarray(data)
                time_len = int(arr.shape[self.time_axis])
                self.temporal._require_coord(coord, time_len)
                t_window = self._temporal_window(time_len, int(time_anchor), coord)
                slices = t_window if isinstance(t_window, list) else [t_window]
                try:
                    base_weights = self.spatial.window.weights(self.spatial.geometry)
                except TypeError:
                    base_weights = None
            except Exception as exc:
                _dispatch(hooks, "on_error", anchor, exc)
                raise
            for s in slices:
                try:
                    idx = [slice(None)] * arr.ndim
                    idx[self.time_axis] = s
                    sub = arr[tuple(idx)]
                    patch = SpatioTemporalPatch(
                        data=sub,
                        space=space_anchor,
                        time=int(time_anchor),
                        spatial_indices=indices,
                        temporal_indices=s,
                        weights=base_weights,
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

    async def _asplit_product(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] = (),
        *,
        coord: np.ndarray | None = None,
    ) -> AsyncIterator[SpatioTemporalPatch]:
        async for sp in self.spatial.asplit(field):
            arr = np.asarray(sp.data)
            time_len = int(arr.shape[self.time_axis])
            self.temporal._require_coord(coord, time_len)
            for t_anchor in self.temporal._sampler_anchors(time_len, coord):
                t_window = self._temporal_window(time_len, int(t_anchor), coord)
                slices = t_window if isinstance(t_window, list) else [t_window]
                coord_value = coord[int(t_anchor)] if coord is not None else None
                for s in slices:
                    anchor = (sp.anchor, int(t_anchor))
                    _dispatch(hooks, "on_patch_start", anchor, coord_value)
                    start = perf_counter()
                    try:
                        idx = [slice(None)] * arr.ndim
                        idx[self.time_axis] = s
                        sub = arr[tuple(idx)]
                        patch = SpatioTemporalPatch(
                            data=sub,
                            space=sp.anchor,
                            time=int(t_anchor),
                            spatial_indices=sp.indices,
                            temporal_indices=s,
                            weights=sp.weights,
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

    async def _asplit_coupled(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] = (),
        *,
        coord: np.ndarray | None = None,
    ) -> AsyncIterator[SpatioTemporalPatch]:
        anchors = getattr(self.spatial.sampler, "anchors_", None)
        if anchors is None:
            raise TypeError(
                "coupled coupling requires the spatial sampler to expose an "
                "`anchors_` list of (space_anchor, time_anchor) tuples — i.e. "
                "use SpatialExplicit(anchors_=[...])."
            )
        for pair in anchors:
            space_anchor, time_anchor = pair
            anchor = (space_anchor, int(time_anchor))
            # Bounded lookup: coord length isn't validated until the patch
            # is read (time_len is unknown before the read in coupled
            # mode), so an out-of-range anchor must not raise IndexError
            # here — _require_coord below reports the documented
            # ValueError through the on_error dispatch instead.
            coord_value = (
                coord[int(time_anchor)]
                if coord is not None and 0 <= int(time_anchor) < len(coord)
                else None
            )
            _dispatch(hooks, "on_patch_start", anchor, coord_value)
            start = perf_counter()
            try:
                indices = self.spatial.geometry.neighborhood(field.domain, space_anchor)
                data = await _select_async(field, indices)
                arr = np.asarray(data)
                time_len = int(arr.shape[self.time_axis])
                self.temporal._require_coord(coord, time_len)
                t_window = self._temporal_window(time_len, int(time_anchor), coord)
                slices = t_window if isinstance(t_window, list) else [t_window]
                try:
                    base_weights = self.spatial.window.weights(self.spatial.geometry)
                except TypeError:
                    base_weights = None
            except Exception as exc:
                _dispatch(hooks, "on_error", anchor, exc)
                raise
            for s in slices:
                try:
                    idx = [slice(None)] * arr.ndim
                    idx[self.time_axis] = s
                    sub = arr[tuple(idx)]
                    patch = SpatioTemporalPatch(
                        data=sub,
                        space=space_anchor,
                        time=int(time_anchor),
                        spatial_indices=indices,
                        temporal_indices=s,
                        weights=base_weights,
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

    def merge(
        self,
        patches: Iterable[Any],
        field: Any,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> list[tuple[Any, Any]]:
        """Group patches by spatial anchor and apply the temporal aggregation.

        Returns ``[(spatial_anchor, temporal_aggregation_result), …]`` — a
        list of pairs rather than a ``dict`` because GridDomain anchors are
        ``dict[str, …]`` (unhashable), KNN-graph anchors are numpy arrays
        (also unhashable), and we want to preserve the original anchor
        object on the result. The per-anchor temporal merge runs through
        `self.temporal.aggregation`, but the spatial aggregation is
        intentionally **not** applied — the returned list is the
        by-anchor view callers typically want for spatiotemporal
        workflows (e.g. event-triggered patching, where the anchor *is*
        the unit of interest). Users who need a full spatial merge
        across the temporal results can pass the values through
        ``self.spatial.aggregation.merge`` themselves.

        Args:
            patches: Iterable of `SpatioTemporalPatch` instances.
            field: The original field — currently unused, kept for the
                symmetry with `SpatialPatcher.merge(patches, domain)` so
                callers can wire the two interchangeably.

        Returns:
            ``[(anchor, merged), …]`` in first-seen anchor order.
        """
        hook_list = _as_hooks(hooks)
        _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
        try:
            # Group on a hashable surrogate (dict anchors → sorted-item tuples,
            # arrays → bytes) but keep the original anchor object alongside
            # the per-group patch list for downstream consumers.
            by_space: dict[Any, tuple[Any, list[Any]]] = {}
            for p in patches:
                key = _hashable(p.space)
                by_space.setdefault(key, (p.space, []))[1].append(p)
            # Temporal aggregations read `anchor` + `indices`, but
            # SpatioTemporalPatch stores them as `time` + `temporal_indices`;
            # rebox each group as TemporalPatch so TemporalForecast /
            # TemporalHierarchicalCombine / etc. don't crash on AttributeError.
            output = [
                (
                    anchor,
                    self.temporal.aggregation.merge(
                        [
                            TemporalPatch(
                                data=p.data,
                                anchor=p.time,
                                indices=p.temporal_indices,
                                weights=p.weights,
                            )
                            for p in group
                        ]
                    ),
                )
                for anchor, group in by_space.values()
            ]
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(output))
        return output

    def _split_total_hint(self, field: Any) -> int:
        if self.coupling == "coupled":
            anchors = getattr(self.spatial.sampler, "anchors_", None)
            return UNKNOWN_TOTAL if anchors is None else _len_or_unknown(anchors)
        if self.coupling != "product":
            return UNKNOWN_TOTAL
        try:
            return self.spatial.n_anchors(field)
        except (
            AttributeError,
            NotImplementedError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            # Progress totals are best-effort only; if asking the spatial
            # sampler for a count touches a backend that fails, keep splitting.
            # These cases cover missing shape metadata, unsupported sampler
            # counts, and invalid geometry/backend state.
            return UNKNOWN_TOTAL

    def get_config(self) -> dict[str, Any]:
        return {
            "spatial": self.spatial.get_config(),
            "temporal": self.temporal.get_config(),
            "coupling": self.coupling,
            "time_axis": self.time_axis,
        }


def _hashable(anchor: Any) -> Any:
    """Coerce an anchor into a hashable surrogate for use as a dict key.

    GridDomain samplers emit ``dict[str, …]`` anchors; numpy arrays / lists
    of pixel coords show up for KNN-graph anchors. None of these are
    hashable. Tuplise dicts in sorted-key order so the surrogate is stable
    across iteration order; arrays go via ``.tobytes()``; sequences go via
    ``tuple()``. Anything already hashable passes through unchanged.
    """
    try:
        hash(anchor)
        return anchor
    except TypeError:
        pass
    if isinstance(anchor, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in anchor.items()))
    if isinstance(anchor, np.ndarray):
        return (anchor.shape, anchor.dtype.str, anchor.tobytes())
    if isinstance(anchor, (list, tuple)):
        return tuple(_hashable(v) for v in anchor)
    # Last resort — stringify; this loses identity but keeps merge() from
    # crashing on exotic anchor types.
    return repr(anchor)

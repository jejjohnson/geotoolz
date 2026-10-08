"""`SpatioTemporalPatcher` — composes a `SpatialPatcher` and a `TemporalPatcher`.

Two coupling modes:

- ``"product"`` (default) - every spatial anchor crossed with every time anchor.
  The right default for dense gridded data where space and time are
  independent grids (climate model output, regular satellite revisits).
- ``"coupled"`` — explicit ``(space, time)`` anchor pairs. The right
  shape for event-triggered patches (methane plume detections, Argo
  profile (lat, lon, t) records, storm tracks).

Both run on the anchor-walk core (`geopatcher._src.walk`): each spatial
chip is read exactly as `SpatialPatcher.split` reads it — the spatial
patcher's geometry / boundary / padding / weights, its ``on_error`` policy
(failures land in ``spatial.errors``) and an optional `PatchCache` — and
is then sliced along ``time_axis`` into the temporal patcher's windows,
keeping the chip's carrier (`GeoTensor`, `DataArray`, …).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from threading import Event
from typing import Any, Literal

import numpy as np

from geopatcher._src._serialize import patcher_config
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
from geopatcher._src.spatial.patcher import (
    SpatialPatcher,
    _AsyncBackpressure,
    _Backpressure,
    _chip_reader,
)
from geopatcher._src.temporal.patcher import TemporalPatcher
from geopatcher._src.walk import (
    ReadPolicy,
    _Read,
    _validate_backpressure,
    awalk,
    walk,
)


@dataclass(eq=False)
class SpatioTemporalPatcher:
    """Composition of a spatial and a temporal Patcher.

    Args:
        spatial: A `SpatialPatcher`. Its geometry, window and read policy
            (``on_error`` / ``max_retries`` / ``retry_on``) read every
            spatial chip; failures are recorded in ``spatial.errors``.
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
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[SpatioTemporalPatch]:
        """Yield `SpatioTemporalPatch`es lazily.

        The coupled mode expects ``self.spatial.sampler.anchors_`` to be
        an iterable of ``(space_anchor, time_anchor)`` tuples and is
        only valid with `spatial.sampler.Explicit` spatial / time samplers.

        Each spatial chip is read once — through the spatial patcher's
        read pipeline and ``on_error`` policy — then sliced into its time
        windows. A patch is keyed (``journal``) by ``(space_anchor,
        time_key)``, ``time_key`` being the temporal `patch_anchors` key
        (the time anchor, or ``(anchor, k)`` for a multi-window geometry);
        hooks receive ``(space_anchor, time_anchor)`` and the time
        anchor's ``coord_value``. A failed chip read reaches the
        ``on_error`` hooks with the spatial anchor (product) or the pair
        (coupled). Product mode reports ``UNKNOWN_TOTAL`` to
        ``on_split_start`` (the window count needs the chips' time
        length); coupled mode the number of pairs.

        Args:
            field: The field to split.
            hooks: Optional observability hooks for split callbacks.
            coord: 1-D coordinate vector along ``time_axis`` of the
                spatial patch's data. Required when the temporal
                geometry or sampler is coordinate-aware
                (``needs_coord = True``); otherwise only reported to
                hooks as ``coord_value``.
            prefetch: If positive, eagerly buffer up to ``prefetch``
                patches in a background thread for I/O overlap.
            journal: Optional `PatchJournal`; patch keys it has are
                skipped (a chip whose every window is journaled is not
                read once the time length is known).
            cache: Optional `PatchCache` for the spatial chips, keyed as
                `SpatialPatcher.split` keys them.
            max_in_flight: Maximum number of unreleased patches.
            max_in_flight_bytes: Maximum total bytes of unreleased patches.
        """
        return self._split(
            field,
            hooks,
            coord=coord,
            prefetch=prefetch,
            journal=journal,
            cache=cache,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )

    def _split(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] | None,
        *,
        coord: np.ndarray | None,
        prefetch: int,
        journal: Any | None,
        cache: Any | None,
        max_in_flight: int | None,
        max_in_flight_bytes: int | None,
        unpack: Callable[[Any], Any] | None = None,
    ) -> Iterator[SpatioTemporalPatch]:
        """`split`; ``unpack`` maps each chip's data before it is sliced.

        `MatchedSpatioTemporalPatcher` passes one that turns a
        `MatchedField` read into its checked ``{source: data}`` dict, each
        member of which is then sliced in lockstep.
        """
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        policy = _start_split(self.spatial)
        stop = Event()
        return prefetch_iterable(
            self._walk(
                field,
                coord=coord,
                policy=policy,
                hooks=hooks,
                journal=journal,
                cache=cache,
                backpressure=_Backpressure(max_in_flight, max_in_flight_bytes, stop),
                unpack=unpack,
            ),
            prefetch,
            stop=stop,
        )

    def _walk(
        self,
        field: Any,
        *,
        coord: np.ndarray | None,
        policy: ReadPolicy,
        hooks: Iterable[PatcherHook] | None,
        journal: Any | None,
        cache: Any | None,
        backpressure: _Backpressure,
        unpack: Callable[[Any], Any] | None,
    ) -> Iterator[Any]:
        plan = self._plan(field, coord, cache, aio=False, unpack=unpack)
        yield from walk(
            plan.anchors,
            plan.units,
            policy=policy,
            backpressure=backpressure,
            hooks=hooks,
            journal=journal,
            total=plan.total,
        )

    async def asplit(
        self,
        field: Any,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        coord: np.ndarray | None = None,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> AsyncIterator[SpatioTemporalPatch]:
        """Async mirror of `split` for async spatial fields.

        The same walk as `split` (policy, hooks, journal, cache,
        backpressure); each chip is awaited (``aselect`` / ``select``).
        ``prefetch`` has no async counterpart.
        """
        _validate_backpressure(max_in_flight, max_in_flight_bytes)
        policy = _start_split(self.spatial)
        plan = self._plan(field, coord, cache, aio=True)
        async for patch in awalk(
            plan.anchors,
            plan.units,
            policy=policy,
            backpressure=_AsyncBackpressure(max_in_flight, max_in_flight_bytes),
            hooks=hooks,
            journal=journal,
            total=plan.total,
        ):
            yield patch

    def _checked_coupling(self) -> Literal["product", "coupled"]:
        if self.coupling not in {"product", "coupled"}:
            raise ValueError(f"unknown coupling: {self.coupling!r}")
        return self.coupling

    def _plan(
        self,
        field: Any,
        coord: Any | None,
        cache: Any | None,
        *,
        aio: bool,
        unpack: Callable[[Any], Any] | None = None,
    ) -> _STPlan:
        coupling = self._checked_coupling()
        coord = self.temporal._require_coord(coord)
        domain = field.domain
        if coupling == "product":
            anchors: Iterable[Any] = self.spatial.sampler.anchors(
                domain, self.spatial.geometry
            )
        else:
            anchors = _coupled_pairs(self.spatial)
        return _STPlan(
            stp=self,
            reader=_chip_reader(self.spatial, field, domain, cache, aio=aio),
            coord=coord,
            coupled=coupling == "coupled",
            anchors=anchors,
            unpack=unpack,
        )

    def _time_len(self, data: Any) -> int:
        """Length of a chip's ``data`` along ``time_axis``.

        A ``{source: data}`` mapping (a matched read, already checked to
        share one time axis) answers with its first member.
        """
        member: Any = next(iter(data.values())) if isinstance(data, Mapping) else data
        return int(np.shape(member)[self.time_axis])

    def _slice_time(self, data: Any, s: slice) -> Any:
        """``data`` sliced to ``s`` along ``time_axis``, keeping its carrier.

        Positional indexing keeps a `GeoTensor` (``__getitem__`` →
        ``isel``), an `xarray.DataArray`, a dask or a numpy array as it is;
        a mapping is sliced member by member (keeping its type).
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._slice_time(v, s) for k, v in data.items()})
        idx: list[Any] = [slice(None)] * len(np.shape(data))
        idx[self.time_axis] = s
        return data[tuple(idx)]

    def _patch(self, chip: Any, data: Any, time: int, s: slice) -> Any:
        """The `SpatioTemporalPatch` of window ``s`` of ``chip`` (``data``)."""
        return SpatioTemporalPatch(
            data=self._slice_time(data, s),
            space=chip.anchor,
            time=time,
            spatial_indices=chip.indices,
            temporal_indices=s,
            weights=chip.weights,
        )

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
            output = _merge_by_space(patches, self.temporal.aggregation)
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(output))
        return output

    def get_config(self) -> dict[str, Any]:
        """Inner patchers as ``{"class", "config"}`` envelopes (`patcher_config`)."""
        return patcher_config(self)


def _start_split(spatial: Any) -> ReadPolicy:
    """The spatial patcher's read policy over a fresh ``errors`` list.

    A duck-typed spatial stand-in without the runner knobs reads under
    the default (``"raise"``) policy.
    """
    start = getattr(spatial, "_start_split", None)
    return ReadPolicy() if start is None else start()


def _merge_by_space(patches: Iterable[Any], aggregation: Any) -> list[tuple[Any, Any]]:
    """``[(spatial_anchor, aggregation.merge(group)), …]`` in first-seen order.

    Groups on a hashable surrogate of each patch's ``space`` (dict anchors
    → sorted-item tuples, arrays → bytes) but keeps the original anchor
    object. Temporal aggregations read ``anchor`` + ``indices``, while a
    `SpatioTemporalPatch` stores ``time`` + ``temporal_indices``, so each
    member is reboxed as a `TemporalPatch`.
    """
    by_space: dict[Any, tuple[Any, list[Any]]] = {}
    for p in patches:
        by_space.setdefault(_hashable(p.space), (p.space, []))[1].append(p)
    return [
        (
            anchor,
            aggregation.merge(
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


def _coupled_pairs(spatial: SpatialPatcher) -> list[Any]:
    """The ``(space_anchor, time_anchor)`` pairs of a coupled split."""
    anchors = getattr(spatial.sampler, "anchors_", None)
    if anchors is None:
        raise TypeError(
            "coupled coupling requires the spatial sampler to expose an "
            "`anchors_` list of (space_anchor, time_anchor) tuples — i.e. "
            "use spatial.sampler.Explicit(anchors_=[...])."
        )
    return list(anchors)


class _STPlan:
    """One spatio-temporal split as walk units.

    Each anchor (a spatial anchor, or a coupled ``(space, time)`` pair)
    is one *parent* unit: the spatial chip read by the spatial patcher's
    `_ChipReader`, which then expands into one leaf per time window.
    The chips' time length is learned from the first one read, after
    which a chip whose every window is journaled is skipped unread.
    """

    def __init__(
        self,
        *,
        stp: SpatioTemporalPatcher,
        reader: Any,
        coord: np.ndarray | None,
        coupled: bool,
        anchors: Iterable[Any],
        unpack: Callable[[Any], Any] | None = None,
    ) -> None:
        self.stp = stp
        self.unpack = unpack
        self.reader = reader
        self.coord = coord
        self.coupled = coupled
        self.anchors = anchors
        self.time_len: int | None = None

    def total(self, anchors: list[Any]) -> int:
        """``on_split_start``'s total: the pairs, or unknown for a product."""
        return len(anchors) if self.coupled else UNKNOWN_TOTAL

    def units(self, anchor: Any) -> list[_Read]:
        if not self.coupled:
            return [
                self.reader.unit(
                    anchor,
                    expand=partial(self._windows, anchor, None),
                    planned=partial(self._planned, anchor, None),
                )
            ]
        space, time = anchor
        pair = (space, int(time))
        return [
            self.reader.unit(
                space,
                key=pair,
                anchor=pair,
                coord_value=self._coord_value(int(time)),
                expand=partial(self._windows, space, int(time)),
                planned=partial(self._planned, space, int(time)),
            )
        ]

    def _coord_value(self, time: int) -> Any:
        # Bounded lookup: in coupled mode the time anchor is only checked
        # against the chip's time length once the chip is read, so an
        # out-of-range anchor must not raise IndexError here.
        coord = self.coord
        return coord[time] if coord is not None and 0 <= time < len(coord) else None

    def _slots(
        self, space: Any, time: int | None, time_len: int
    ) -> list[tuple[Any, Any, Any, int, slice]]:
        """``(key, hook anchor, coord_value, time, window)`` per window."""
        temporal = self.stp.temporal
        times = (
            [int(t) for t in temporal._sampler_anchors(time_len, self.coord)]
            if time is None
            else [time]
        )
        return [
            ((space, key), (space, t), self._coord_value(t), t, s)
            for t in times
            for key, _, s in temporal._keyed_windows(time_len, t, self.coord)
        ]

    def _windows(self, space: Any, time: int | None, chip: Any) -> list[_Read]:
        data = chip.data if self.unpack is None else self.unpack(chip.data)
        time_len = self.stp._time_len(data)
        self.stp.temporal._require_coord(self.coord, time_len)
        self.time_len = time_len
        return [
            _Read(
                key=key,
                anchor=anchor,
                coord_value=coord_value,
                read=partial(self.stp._patch, chip, data, t, s),
                governed=False,
            )
            for key, anchor, coord_value, t, s in self._slots(space, time, time_len)
        ]

    def _planned(
        self, space: Any, time: int | None
    ) -> list[tuple[Any, Any, Any]] | None:
        if self.time_len is None:
            return None
        try:
            slots = self._slots(space, time, self.time_len)
        except Exception:  # surfaces (with hooks) once the chip is read
            return None
        return [(key, anchor, coord_value) for key, anchor, coord_value, _, _ in slots]


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

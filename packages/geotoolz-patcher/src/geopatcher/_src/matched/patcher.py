"""Matched-axis patchers — orchestrate split / merge across sources.

Thin wrappers around the single-source patchers (`SpatialPatcher`,
`TemporalPatcher`, `SpatioTemporalPatcher`) that:

* split a `MatchedField` into per-axis matched patch carriers
  (`MatchedPatch`, `MatchedTemporalPatch`,
  `MatchedSpatioTemporalPatch`),
* on ``merge``, dispatch to per-source aggregators and return a
  ``dict[str, …]`` of per-source reconstructions instead of a single
  field.

The single-source patcher is reused as the primary; secondary
aggregators live in a parallel mapping. The four-axis decomposition
is untouched — these classes only add the per-source fan-out on the
merge side. See ADR-003.

Hooks are dispatched once, at the matched level: one ``on_patch_done``
per matched patch whose byte count sums every member, and one
``on_merge_start`` / ``on_merge_end`` pair per ``merge`` whose output
bytes sum every source's result.

Every ``merge`` streams: each source's aggregation runs on its own
worker thread, fed through a small bounded queue, so a single pass over
the matched patches drives every aggregation at once and only a few
patches per source are resident — never the whole patch list of every
source (`MatchedSpatioTemporalPatcher.merge` still groups each source's
patches by spatial anchor, as `SpatioTemporalPatcher.merge` does).
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from rasterio.windows import Window

from geopatcher._src._serialize import patcher_config
from geopatcher._src.domains import GridDomain, PointDomain, VectorDomain
from geopatcher._src.hooks import (
    PatcherHook,
    _as_hooks,
    _dispatch,
    _len_or_unknown,
    _nbytes,
)
from geopatcher._src.matched.field import MatchedField
from geopatcher._src.matched.patch import (
    PRIMARY_KEY,
    MatchedPatch,
    MatchedSpatioTemporalPatch,
    MatchedTemporalPatch,
)
from geopatcher._src.patch import Patch, SpatioTemporalPatch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.spatial.geometry import _is_raster_domain
from geopatcher._src.spatial_time import _merge_by_space


if TYPE_CHECKING:
    from geopatcher._src.spatial.aggregation import (
        Aggregation as SpatialAggregation,
    )
    from geopatcher._src.spatial.patcher import SpatialPatcher
    from geopatcher._src.spatial_time import SpatioTemporalPatcher
    from geopatcher._src.temporal.aggregation import (
        Aggregation as TemporalAggregation,
    )
    from geopatcher._src.temporal.patcher import TemporalPatcher


def _compute_valid_mask(data: Any, nodata: Any = None) -> np.ndarray | None:
    """Best-effort validity mask for ``data`` (True = valid).

    A cell is invalid when it equals the carrier's declared nodata —
    a `GeoTensor`'s ``fill_value_default`` (exactly what its
    ``validmask()`` tests) or a rioxarray `DataArray`'s ``rio.nodata``
    — and, for floating / complex data, when it is NaN or ±inf. Integer
    nodata (the norm for L1/L2 products, and what a reproject-like coreg
    pads with) is therefore masked; a bare ndarray carries no nodata, so
    only the float NaN / inf test applies to it. For non-array data or
    non-numeric arrays, returns None so the caller can simply omit that
    source's mask entry rather than emit a meaningless all-True /
    all-False array.

    Args:
        data: The member's data.
        nodata: Nodata of the carrier ``data`` was sliced from, used when
            ``data`` itself is a bare array (the temporal paths slice
            ndarrays out of the per-source carriers).
    """
    try:
        arr = np.asarray(data)
    except (TypeError, ValueError):
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    mask = np.ones(arr.shape, dtype=bool)
    if np.issubdtype(arr.dtype, np.inexact):
        mask &= np.isfinite(arr)
    declared = _declared_nodata(data)
    nodata = nodata if declared is None else declared
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask &= arr != nodata
    return mask


def _declared_nodata(data: Any) -> Any:
    """The nodata value a patch carrier declares, or None.

    Unlike the padding helper's ``_carrier_nodata`` this never defaults
    to ``0``: a carrier that declares nothing masks nothing.
    """
    from geopatcher._src.spatial.patcher import _is_rio_dataarray

    if _is_rio_dataarray(data):
        return getattr(data.rio, "nodata", None)
    return getattr(data, "fill_value_default", None)


def _validate_aggregator_names(
    secondary_aggregators: Mapping[str, Any],
    mfield: MatchedField,
    cls_name: str,
) -> None:
    """Reject typoed ``secondary_aggregators`` keys up front.

    Without this guard, a typo like
    ``secondary_aggregators={"s22": ...}`` would silently drop
    every real ``"s2"`` patch and still call the typoed
    aggregator with an empty list — producing a bogus
    reconstructed field with no error.

    Best-effort: if ``mfield`` doesn't expose ``secondaries``
    (i.e. caller mistakenly passed a plain Field), the
    type-error path in ``split`` / the empty-merge path will
    surface that misuse — we don't double-fault here.

    Args:
        secondary_aggregators: The patcher's ``{name: aggregator}``
            mapping to check.
        mfield: The `MatchedField` whose ``secondaries`` names are
            authoritative.
        cls_name: Patcher class name interpolated into the error
            message (``type(self).__name__`` at call sites).
    """
    secondaries = getattr(mfield, "secondaries", None)
    if secondaries is None:
        return
    unknown = set(secondary_aggregators) - set(secondaries)
    if unknown:
        raise ValueError(
            f"{cls_name}.secondary_aggregators has names "
            "not in mfield.secondaries: "
            f"{sorted(unknown)!r}. "
            f"Known secondaries: {sorted(secondaries)!r}."
        )


def _check_matched_dict(data_by_name: Any, cls_name: str, expects: str) -> None:
    """Validate the per-source dict shape produced by ``MatchedField.select``.

    A plain `Field` fed to a matched patcher yields non-dict patch
    data; surface that misuse here rather than an obscure `KeyError`
    later.

    Args:
        data_by_name: Value expected to be a ``dict[str, data]``.
        cls_name: Patcher class name interpolated into the error
            message (``type(self).__name__`` at call sites).
        expects: Phrase naming what was expected to carry the dict,
            e.g. ``"each Patch.data to be"`` — interpolated verbatim
            so each patcher keeps its historical message text.
    """
    if not isinstance(data_by_name, dict):
        raise TypeError(
            f"{cls_name}.split expects {expects} a dict[str, data] "
            "(as produced by "
            "MatchedField.select); got "
            f"{type(data_by_name).__name__}. "
            "Did you pass a plain Field instead of a MatchedField?"
        )
    if PRIMARY_KEY not in data_by_name:
        raise ValueError(
            f"MatchedField.select must include the primary key "
            f"{PRIMARY_KEY!r}; got keys {sorted(data_by_name)!r}."
        )


def _is_mask_placeholder(data: Any, mfield: Any, spatial: Any) -> bool:
    """True for the all-NaN patch `SpatialPatcher` emits under ``on_error="mask"``.

    A failed read (or coregistration) of a `MatchedField` never produces
    the per-source dict, so the inner `SpatialPatcher` substitutes its
    NaN placeholder ndarray. Recognised only for a real `MatchedField`
    under a ``"mask"`` policy, so a plain `Field` passed by mistake
    still gets the "Did you pass a plain Field" error.
    """
    return (
        isinstance(mfield, MatchedField)
        and getattr(spatial, "on_error", None) == "mask"
        and isinstance(data, np.ndarray)
        and np.issubdtype(data.dtype, np.floating)
        and bool(np.isnan(data).all())
    )


def _unpack_matched(
    data: Any, mfield: Any, spatial: Any, cls_name: str, expects: str
) -> tuple[dict[str, Any], bool]:
    """``(data_by_name, masked)`` for one spatial read of ``mfield``.

    A mask placeholder fans out to one all-NaN copy per source, each on
    the primary's grid (where every member lives after coregistration).
    """
    if _is_mask_placeholder(data, mfield, spatial):
        names = (PRIMARY_KEY, *mfield.secondaries)
        return {name: data.copy() for name in names}, True
    _check_matched_dict(data, cls_name, expects)
    return data, False


def _compute_member_masks(
    members: Mapping[str, Any],
    mfield: MatchedField,
    nodata: Mapping[str, Any] | None = None,
    *,
    masked: bool = False,
) -> dict[str, np.ndarray] | None:
    """Per-source validity masks for a matched patch's ``members``.

    Args:
        members: ``{name: patch}`` whose ``data`` attributes are masked.
        mfield: The originating `MatchedField`; masks are only computed
            when its ``valid_mask`` flag is truthy.
        nodata: Optional ``{name: nodata}`` from the carriers the
            members' bare-array data was sliced from.
        masked: The members are ``on_error="mask"`` placeholders: every
            mask is all-False, whatever ``mfield.valid_mask`` says — it
            is the only signal that the patch holds no data.

    Returns:
        ``{name: mask}`` for members whose data is numeric and
        array-coercible, or None when masking is disabled or no member
        produced a mask.
    """
    if masked:
        return {
            name: np.zeros(np.shape(patch.data), dtype=bool)
            for name, patch in members.items()
        }
    if not getattr(mfield, "valid_mask", False):
        return None
    mask_dict = {
        name: mask
        for name, patch in members.items()
        if (
            mask := _compute_valid_mask(
                patch.data, None if nodata is None else nodata.get(name)
            )
        )
        is not None
    }
    return mask_dict or None


def _member_weights(members: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """``{name: weights}`` of the members that carry window weights."""
    weights = {
        name: patch.weights
        for name, patch in members.items()
        if patch.weights is not None
    }
    return weights or None


def _members_nbytes(members: Mapping[str, Any]) -> int:
    """Sum of every member's data bytes — the matched hook payload."""
    return sum(_nbytes(patch.data) for patch in members.values())


def _hand_over_release(outer: Any, matched: Any) -> None:
    """Move ``outer``'s ``max_in_flight`` slot onto the matched carrier.

    The matched patch replaces the patch the inner `SpatialPatcher`
    yielded; leaving the slot on the dropped outer patch would release
    it as soon as the outer is collected, defeating the bound.
    """
    release = getattr(outer, "_release", None)
    if release is None:
        return
    outer._release = None
    matched._release = release


def _full_indexer(domain: Any) -> Any:
    """An indexer reading a field's whole extent, from its domain.

    A raster domain gets the full pixel ``Window`` (what `RasterField`,
    `RioXarrayField` and `CogField` read), a `GridDomain` the empty
    ``isel`` dict (`XarrayField`, `DaskField`). Vector / point domains
    hold rows, not a time series, and raise. Anything else — duck-typed
    array fields — gets ``slice(None)``.

    Raises:
        TypeError: For a `VectorDomain` / `PointDomain`.
    """
    if _is_raster_domain(domain):
        height, width = (int(n) for n in tuple(domain.shape)[-2:])
        return Window(col_off=0, row_off=0, width=width, height=height)
    if isinstance(domain, GridDomain):
        return {}
    if isinstance(domain, (VectorDomain, PointDomain)):
        raise TypeError(
            f"cannot derive a full-series indexer for a "
            f"{type(domain).__name__}; pass indexer= explicitly."
        )
    return slice(None)


def _check_time_lengths(
    arrays: Mapping[str, Any], time_axis: int, cls_name: str
) -> int:
    """Every source's length along ``time_axis``; must equal the primary's.

    The matched temporal patchers slice every source with the primary's
    time indices, so a secondary on another cadence would silently yield
    short or empty members.

    Returns:
        The primary's time length.

    Raises:
        ValueError: A secondary's time length differs from the primary's.
    """
    time_len = int(np.shape(arrays[PRIMARY_KEY])[time_axis])
    for name, arr in arrays.items():
        if name == PRIMARY_KEY:
            continue
        shape = np.shape(arr)
        n = int(shape[time_axis]) if len(shape) > time_axis else None
        if n != time_len:
            raise ValueError(
                f"{cls_name}: secondary {name!r} has {n} steps along "
                f"time_axis={time_axis}, the primary has {time_len}. Every "
                "source is sliced with the primary's time indices, so its "
                "coreg callable must return it on the primary's time axis "
                "(resample a different cadence there)."
            )
    return time_len


@dataclass(frozen=True)
class _Shaped:
    """Shape-only series: `TemporalPatcher.anchors` reads nothing else."""

    shape: tuple[int, ...]


# -- per-source streaming merge ------------------------------------------

# Members buffered per source between the producer and each aggregation.
_FAN_OUT_DEPTH = 2
_END = object()
_ABORT = object()


class _UpstreamFailedError(Exception):
    """Ends a per-source aggregation whose patch producer failed."""


def _close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if close is not None:
        close()


def _members_of(patches: Iterable[Any], name: str) -> Iterator[Any]:
    """``name``'s member of each matched patch, closing each on advance."""
    for mp in patches:
        try:
            member = mp.members.get(name)
            if member is not None:
                yield member
        finally:
            _close(mp)


def _fan_out_merge(
    patches: Iterable[Any],
    consumers: Mapping[str, Callable[[Iterator[Any]], Any]],
) -> dict[str, Any]:
    """Feed one pass over ``patches`` to a per-source consumer each.

    Every consumer (``aggregation.merge`` over that source's members)
    runs on its own worker thread and pulls from a bounded queue the
    caller's thread fills, so the aggregations run concurrently over a
    single pass: at most ``_FAN_OUT_DEPTH`` members per source wait in
    memory, whatever the patch count. A lone consumer runs inline.

    A consumer that stops early is drained so the producer never blocks
    on it; a consumer failure stops the pass and is re-raised (the first
    in ``consumers`` order); a failure of ``patches`` itself aborts every
    consumer and is re-raised.

    Returns:
        ``{name: consumer result}`` in ``consumers`` order.
    """
    names = list(consumers)
    if len(names) == 1:
        (name,) = names
        return {name: consumers[name](_members_of(patches, name))}

    queues: dict[str, queue.Queue[Any]] = {
        name: queue.Queue(maxsize=_FAN_OUT_DEPTH) for name in names
    }
    results: dict[str, Any] = {}
    failures: dict[str, BaseException] = {}

    def consume(name: str) -> None:
        q = queues[name]
        ended = False

        def items() -> Iterator[Any]:
            nonlocal ended
            while True:
                item = q.get()
                if item is _END or item is _ABORT:
                    ended = True
                    if item is _ABORT:
                        raise _UpstreamFailedError
                    return
                yield item

        try:
            results[name] = consumers[name](items())
        except BaseException as exc:
            failures[name] = exc
        finally:
            while not ended:  # drain: the producer must never block on us
                item = q.get()
                ended = item is _END or item is _ABORT

    workers = [
        threading.Thread(
            target=consume,
            args=(name,),
            name=f"geopatcher-matched-merge-{name}",
            daemon=True,
        )
        for name in names
    ]
    for worker in workers:
        worker.start()
    end = _END
    stopped_early = False
    try:
        for mp in patches:
            try:
                if failures:
                    stopped_early = True
                    break
                for name in names:
                    member = mp.members.get(name)
                    if member is not None:
                        queues[name].put(member)
            finally:
                _close(mp)
    except BaseException:
        end = _ABORT
        raise
    finally:
        for name in names:
            queues[name].put(end)
        for worker in workers:
            worker.join()
        if stopped_early:
            _close(patches)
    for name in names:
        if name in failures:
            raise failures[name]
    return {name: results[name] for name in names}


def _merge_per_source(
    patches: Iterable[Any],
    consumers: Mapping[str, Callable[[Iterator[Any]], Any]],
    hooks: Iterable[PatcherHook] | None,
) -> dict[str, Any]:
    """`_fan_out_merge` inside one matched-level merge hook lifecycle."""
    hook_list = _as_hooks(hooks)
    _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
    try:
        result = _fan_out_merge(patches, consumers)
    except Exception as exc:
        _dispatch(hook_list, "on_error", None, exc)
        raise
    _dispatch(hook_list, "on_merge_end", _nbytes(result))
    return result


# -- per-source PatchCache -----------------------------------------------

# Config part of every matched cache key: entries hold one source's raw
# read at one indexer, independent of the geometry / window.
_MATCHED_CONFIG_ID = json.dumps({"matched": "source-read"})


def _indexer_key(indexer: Any) -> Any:
    """A JSON-able cache key for a source indexer.

    Raises:
        TypeError: For an indexer with no exact JSON form.
    """
    if isinstance(indexer, Window):
        return {"window": [float(v) for v in indexer.flatten()]}
    if isinstance(indexer, dict):
        return {
            "isel": {
                str(k): [v.start, v.stop, v.step] if isinstance(v, slice) else v
                for k, v in indexer.items()
            }
        }
    if isinstance(indexer, slice):
        return {"slice": [indexer.start, indexer.stop, indexer.step]}
    return indexer


class _CachedSourceReads:
    """`Field` view of a `MatchedField` whose source reads hit a `PatchCache`.

    Each source's raw read is cached under ``("<name>:" + field_id,
    indexer)`` — its own identity (path / URL / ``cache_id()`` and domain,
    via `PatchCache.field_id_for`) qualified by its role name, so an
    explicit ``PatchCache(field_id=...)`` shared by the matched set still
    keys every source apart. The coregistration is not cached: it runs on
    every read (a coreg callable has no stable identity to key on).
    """

    def __init__(self, mfield: MatchedField, cache: Any) -> None:
        self._mfield = mfield
        self._cache = cache
        self._ids = {PRIMARY_KEY: f"{PRIMARY_KEY}:{cache.field_id_for(mfield.primary)}"}
        for name, source in mfield.secondaries.items():
            self._ids[name] = f"{name}:{cache.field_id_for(source)}"

    @property
    def domain(self) -> Any:
        return self._mfield.domain

    def select(self, indexer: Any) -> dict[str, Any]:
        return self._mfield._select(indexer, self._read)

    def with_data(self, array: Any) -> Any:
        return self._mfield.with_data(array)

    def _read(self, name: str, source: Any, indexer: Any) -> Any:
        key = _indexer_key(indexer)
        payload = self._cache.get(self._ids[name], _MATCHED_CONFIG_ID, key)
        if payload is not None:
            return self._cache.build_patch(payload, key, indexer).data
        data = source.select(indexer)
        self._cache.put(
            self._ids[name],
            _MATCHED_CONFIG_ID,
            key,
            Patch(data=data, anchor=key, indices=indexer),
        )
        return data


class _MatchedConfigMixin:
    """The ``get_config`` shared by the matched patcher family.

    Follows the patcher-family envelope convention: nested components
    serialize as ``{"class": type(x).__name__, "config": x.get_config()}``
    via the shared `patcher_config`.
    """

    primary: Any
    secondary_aggregators: Mapping[str, Any]

    def get_config(self) -> dict[str, Any]:
        """Serialize the inner primary patcher + per-secondary aggregators."""
        return patcher_config(self)


@dataclass(eq=False)
class MatchedSpatialPatcher(_MatchedConfigMixin):
    """Spatial patcher that yields `MatchedPatch`es and merges per-source.

    Args:
        primary: A regular `SpatialPatcher` configured for the
            primary `Field`. Drives anchor placement, geometry,
            window, primary aggregation and the ``on_error`` policy
            (which covers every source's read and coregistration).
        secondary_aggregators: ``{name: spatial.aggregation.Aggregation}`` — one
            aggregator per secondary. Names that don't match any
            entry in ``mfield.secondaries`` raise on ``split`` /
            ``merge`` rather than silently skipping (catches config
            typos like ``"s22"`` instead of ``"s2"``). Omitting a
            secondary from this mapping is fine — that source is
            simply not merged back, which is the documented opt-out.
    """

    primary: SpatialPatcher
    secondary_aggregators: Mapping[str, SpatialAggregation] = field(
        default_factory=dict
    )

    def split(
        self,
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[MatchedPatch]:
        """Yield `MatchedPatch`es by walking ``mfield`` with the primary's sampler.

        Internally drives ``self.primary.split(mfield)`` — since
        `MatchedField` already satisfies the `Field` Protocol, the
        existing sampler / geometry / window machinery works
        unchanged. Each outer ``Patch`` carries the per-source
        ``dict`` returned by `MatchedField.select` in its ``data``
        field; this method unpacks that dict into a `MatchedPatch`
        whose ``members`` is ``{name: Patch}``, whose ``anchor`` /
        ``indices`` / ``weights`` mirror the outer patch, and whose
        ``weights`` maps each member to its window weights.

        Per-source ``valid_mask`` arrays are computed when
        ``mfield.valid_mask`` is True (the default): for numeric
        array-coercible data a cell is invalid where it equals the
        carrier's declared nodata (``fill_value_default`` /
        ``rio.nodata``) or, for float data, is NaN / ±inf. Non-array
        members are simply omitted from the mask dict (and the dict
        drops to ``None`` if no member produced a mask).

        The primary's ``on_error`` policy covers every source: a failed
        read or coregistration raises (``"raise"``), drops the anchor
        (``"skip"`` / exhausted ``"retry"``), or under ``"mask"`` yields a
        `MatchedPatch` whose every member is the all-NaN placeholder on the
        primary's grid with an all-False ``valid_mask``.

        Args:
            mfield: A `MatchedField` to drive the primary sampler over.
            hooks: Optional observability hooks, dispatched by the primary
                `SpatialPatcher.split` once per matched patch; the
                ``on_patch_done`` byte count sums every member.
            prefetch: Forwarded to `SpatialPatcher.split` — read ahead up
                to this many matched patches on a background thread.
            journal: Forwarded to `SpatialPatcher.split` — anchors the
                `PatchJournal` already holds are skipped.
            cache: A `PatchCache` consulted per source: each source's raw
                read is keyed on its own identity and role name, so a hit
                skips that source's I/O (the coregistration still runs).
                In-memory sources need ``PatchCache(field_id=...)``.
            max_in_flight: Forwarded to `SpatialPatcher.split`; each
                yielded `MatchedPatch` owns the slot — release it with
                ``mp.close()`` or ``with mp: ...``.
            max_in_flight_bytes: Forwarded to `SpatialPatcher.split`; a
                matched patch is sized as the sum of its members' bytes.
        """
        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )
        source: Any = mfield if cache is None else _CachedSourceReads(mfield, cache)
        outers = self.primary.split(
            source,
            hooks=hooks,
            prefetch=prefetch,
            journal=journal,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
        )
        cls_name = type(self).__name__
        for outer in outers:
            data_by_name, masked = _unpack_matched(
                outer.data, mfield, self.primary, cls_name, "each Patch.data to be"
            )
            members = {
                name: outer.with_data(data) for name, data in data_by_name.items()
            }
            matched = MatchedPatch(
                anchor=outer.anchor,
                members=members,
                valid_mask=_compute_member_masks(members, mfield, masked=masked),
                weights=_member_weights(members),
            )
            _hand_over_release(outer, matched)
            yield matched

    def n_anchors(self, mfield: MatchedField) -> int:
        """Number of `MatchedPatch`es ``split`` will yield (no reads)."""
        return self.primary.n_anchors(mfield)

    def anchors(self, mfield: MatchedField) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``mfield`` (no reads)."""
        return self.primary.anchors(mfield)

    def merge(
        self,
        patches: Iterable[MatchedPatch],
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> dict[str, Any]:
        """Per-source merge: dict of ``name -> aggregation result``.

        Returns the primary under ``MatchedPatch.PRIMARY_KEY``;
        secondaries appear under the names supplied to
        ``MatchedField.secondaries``. Names whose
        ``secondary_aggregators`` entry is missing are skipped (you
        can choose to only reconstruct a subset). Names that *are*
        in ``secondary_aggregators`` but not in
        ``mfield.secondaries`` raise — typo guard.

        Each value is the aggregation's raw output, exactly as
        `SpatialPatcher.merge` returns it: a bare ``np.ndarray`` on the
        primary's grid for the dense aggregations (no transform, CRS,
        nodata or attrs), a ``dict`` for `spatial.aggregation.MeanStd` /
        `spatial.aggregation.InvVarWeightedMean` / `spatial.aggregation.ByIndex`. Use
        `merge_to_field` to get georeferenced carriers back.

        Every source is aggregated against the primary's domain
        because the coregistration callable mapped each secondary
        onto the primary's grid at split time. Reconstructing a
        secondary back into its own original grid would require
        re-inverting the coregistration, which is the user's
        problem if they need it.

        Streaming: ``patches`` is consumed once and each source's
        aggregation runs on its own thread over a bounded queue, so only
        a few patches per source are resident at a time — a streaming
        aggregation (e.g. zarr-backed `spatial.aggregation.OverlapAdd`) keeps its
        bound on every source. Every aggregation (primary and secondary)
        gets the strict-mode streaming-safety check, which warns (or
        raises under `set_strict`) for a ``streaming_safe = False`` one.

        Args:
            patches: Iterable of `MatchedPatch` instances; each is closed
                once its members are handed on.
            mfield: Original `MatchedField` (used for typo-guard and
                to recover the primary domain for aggregation).
            hooks: Optional observability hooks, dispatched once at the
                matched level (one ``on_merge_start`` / ``on_merge_end``
                per call; the output bytes sum every source).
        """
        return self._merge_sources(patches, mfield, hooks, stacklevel=2)

    def merge_to_field(
        self,
        patches: Iterable[MatchedPatch],
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> dict[str, Any]:
        """`merge` + rebuild each source on the primary's grid via ``with_data``.

        Every source was aggregated on the primary's domain, so each
        value is wrapped with ``mfield.primary.with_data`` — it carries
        the primary's transform and CRS, and also the primary's nodata
        and attrs (a secondary's own nodata / attrs are not carried; its
        values keep that secondary's dtype when they fit, by the
        `SpatialPatcher.merge_to_xarray` rule).

        Raises:
            TypeError: As `SpatialPatcher.merge_to_field` — for a ``dict``
                output, a non-array output, or a shape that is not the
                primary domain's.
        """
        from geopatcher._src.spatial.patcher import (
            _declare_gap_fill,
            _domain_shape,
            _field_values,
            _is_raster_field,
            _require_with_data,
            _source_dtype,
        )

        with_data = _require_with_data(mfield.primary, "merge_to_field")
        merged = self._merge_sources(patches, mfield, hooks, stacklevel=2)
        shape = _domain_shape(mfield.domain)
        out: dict[str, Any] = {}
        for name, value in merged.items():
            if name == PRIMARY_KEY:
                agg, source = self.primary.aggregation, mfield.primary
            else:
                agg, source = self.secondary_aggregators[name], mfield.secondaries[name]
            values = _field_values(
                value,
                agg,
                shape=shape,
                dtype=_source_dtype(source),
                caller="merge_to_field",
                bands_may_change=_is_raster_field(mfield.primary),
            )
            out[name] = _declare_gap_fill(with_data(values), values, agg)
        return out

    def _merge_sources(
        self,
        patches: Iterable[MatchedPatch],
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None,
        *,
        stacklevel: int,
    ) -> dict[str, Any]:
        """`merge` body; ``stacklevel`` (from the caller) places the warning."""
        from geopatcher._src.spatial.patcher import _check_streaming

        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )
        aggregations = {
            PRIMARY_KEY: self.primary.aggregation,
            **self.secondary_aggregators,
        }
        for agg in aggregations.values():
            _check_streaming(agg, stacklevel=stacklevel + 1)
        domain = mfield.domain

        def consumer(agg: Any) -> Callable[[Iterator[Any]], Any]:
            return lambda members: agg.merge(members, domain)

        return _merge_per_source(
            patches,
            {name: consumer(agg) for name, agg in aggregations.items()},
            hooks,
        )


@dataclass(eq=False)
class MatchedTemporalPatcher(_MatchedConfigMixin):
    """Temporal patcher that yields `MatchedTemporalPatch`es and merges per-source.

    Mirror of `MatchedSpatialPatcher` over the time axis. The
    `TemporalPatcher` slices a numpy array directly (it does not call
    `Field.select` per anchor the way `SpatialPatcher` does), so the
    matched-temporal split first reads every source's full series once
    — ``mfield.select(indexer)``, with an indexer covering the primary's
    whole extent (see `split`) — then drives the primary patcher on the
    primary's series and slices each secondary's array in lockstep.

    Args:
        primary: A regular `TemporalPatcher` configured for the
            primary series. Drives anchor placement, geometry,
            window, and primary aggregation.
        secondary_aggregators: ``{name: temporal.aggregation.Aggregation}`` — one
            aggregator per secondary. Names that don't match any
            entry in ``mfield.secondaries`` raise on ``split`` /
            ``merge`` rather than silently skipping (typo guard).
            Omitting a secondary from this mapping is the documented
            opt-out for "don't reconstruct this source".
    """

    primary: TemporalPatcher
    secondary_aggregators: Mapping[str, TemporalAggregation] = field(
        default_factory=dict
    )

    def split(
        self,
        mfield: MatchedField,
        time_axis: int = 0,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        indexer: Any | None = None,
        coord: np.ndarray | None = None,
        prefetch: int = 0,
    ) -> Iterator[MatchedTemporalPatch]:
        """Yield `MatchedTemporalPatch`es by driving the primary on each anchor.

        Reads every source's series once via ``mfield.select(indexer)``,
        then drives the primary `TemporalPatcher` on the primary's array.
        For each yielded `TemporalPatch`, each secondary's array is sliced
        with the same ``indices`` and packaged into the matched carrier.

        Per-source ``valid_mask`` arrays are computed when
        ``mfield.valid_mask`` is True (the default): a cell is invalid
        where it equals the source carrier's declared nodata or, for
        float data, is NaN / ±inf.

        Args:
            mfield: A `MatchedField` whose ``select`` returns the
                per-source series as a dict keyed by source name.
            time_axis: Which axis of each source's array is the time
                axis. Default 0. Must be the same across sources.
            hooks: Optional observability hooks, dispatched once per
                matched patch (``on_patch_done`` bytes sum every member,
                ``coord_value`` as `TemporalPatcher` reports it).
            indexer: The indexer that reads the whole series. ``None``
                (default) derives it from the primary's domain: the full
                pixel ``Window`` of a raster domain (`RasterField`,
                `RioXarrayField`, …), ``{}`` for a `GridDomain`
                (`XarrayField`, `DaskField`), ``slice(None)`` for other
                duck-typed fields.
            coord: 1-D coordinate vector along ``time_axis``, forwarded to
                the primary — required by coordinate-aware (stencil)
                geometries / samplers.
            prefetch: If positive, read ahead up to ``prefetch`` matched
                patches on a background thread.

        Raises:
            ValueError: A secondary's length along ``time_axis`` differs
                from the primary's (a cadence the coreg did not resample).
        """
        return prefetch_iterable(
            self._split(mfield, time_axis, hooks, indexer=indexer, coord=coord),
            prefetch,
        )

    def _split(
        self,
        mfield: MatchedField,
        time_axis: int,
        hooks: Iterable[PatcherHook] | None,
        *,
        indexer: Any | None,
        coord: np.ndarray | None,
    ) -> Iterator[MatchedTemporalPatch]:
        cls_name = type(self).__name__
        _validate_aggregator_names(self.secondary_aggregators, mfield, cls_name)
        if indexer is None:
            indexer = _full_indexer(mfield.domain)
        data_by_name = mfield.select(indexer)
        _check_matched_dict(data_by_name, cls_name, "mfield.select to return")

        arrays = {name: np.asarray(data) for name, data in data_by_name.items()}
        nodata = {name: _declared_nodata(d) for name, d in data_by_name.items()}
        primary_arr = arrays[PRIMARY_KEY]
        _check_time_lengths(arrays, time_axis, cls_name)

        hook_list = _as_hooks(hooks)
        if hook_list:
            n = len(self.primary.anchors(primary_arr, time_axis=time_axis, coord=coord))
            _dispatch(hook_list, "on_split_start", n)
        try:
            for primary_patch in self.primary.split(
                primary_arr, time_axis=time_axis, coord=coord
            ):
                anchor = primary_patch.anchor
                coord_value = coord[int(anchor)] if coord is not None else None
                _dispatch(hook_list, "on_patch_start", anchor, coord_value)
                start = perf_counter()
                try:
                    idx: list[Any] = [slice(None)] * primary_arr.ndim
                    idx[time_axis] = primary_patch.indices
                    tup = tuple(idx)
                    members = {
                        name: primary_patch.with_data(arr[tup])
                        for name, arr in arrays.items()
                    }
                    matched = MatchedTemporalPatch(
                        anchor=anchor,
                        members=members,
                        valid_mask=_compute_member_masks(members, mfield, nodata),
                        weights=_member_weights(members),
                    )
                except Exception as exc:
                    _dispatch(hook_list, "on_error", anchor, exc)
                    raise
                _dispatch(
                    hook_list,
                    "on_patch_done",
                    anchor,
                    perf_counter() - start,
                    _members_nbytes(members),
                    coord_value,
                )
                yield matched
        finally:
            if hook_list:
                _dispatch(hook_list, "on_split_end")

    def _primary_series(self, mfield: MatchedField, indexer: Any | None) -> Any:
        """A shape-carrying stand-in for the primary's series.

        Neither a secondary read nor a coregistration runs — counting
        anchors needs only the primary's time length. With the default
        indexer the primary's domain shape answers without any I/O (a
        raster / grid domain's shape is the full read's shape); otherwise
        the primary alone is read.
        """
        domain_shape = getattr(mfield.domain, "shape", None)
        if indexer is None and domain_shape is not None:
            return _Shaped(tuple(int(n) for n in domain_shape))
        if indexer is None:
            indexer = _full_indexer(mfield.domain)
        return _Shaped(tuple(np.shape(mfield.primary.select(indexer))))

    def n_anchors(
        self,
        mfield: MatchedField,
        time_axis: int = 0,
        *,
        indexer: Any | None = None,
        coord: np.ndarray | None = None,
    ) -> int:
        """Number of `MatchedTemporalPatch`es ``split`` will yield.

        Reads the primary only (no secondary read, no coregistration).
        """
        series = self._primary_series(mfield, indexer)
        return self.primary.n_anchors(series, time_axis=time_axis, coord=coord)

    def anchors(
        self,
        mfield: MatchedField,
        time_axis: int = 0,
        *,
        indexer: Any | None = None,
        coord: np.ndarray | None = None,
    ) -> list[int]:
        """Materialise the sampler's anchor sequence for ``mfield``.

        Reads the primary only (no secondary read, no coregistration).
        """
        series = self._primary_series(mfield, indexer)
        return self.primary.anchors(series, time_axis=time_axis, coord=coord)

    def merge(
        self,
        patches: Iterable[MatchedTemporalPatch],
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> dict[str, Any]:
        """Per-source merge: dict of ``name -> aggregation result``.

        Returns the primary under ``MatchedTemporalPatch.PRIMARY_KEY``;
        secondaries appear under the names supplied to
        ``MatchedField.secondaries``. Names whose
        ``secondary_aggregators`` entry is missing are skipped (the
        user opted out for that source). Names that *are* in
        ``secondary_aggregators`` but not in ``mfield.secondaries``
        raise — typo guard.

        Unlike the spatial path, `temporal.aggregation.Aggregation.merge` takes
        only the patches (no domain argument), so ``mfield`` is used
        solely for the typo-guard check. ``patches`` is consumed once,
        streamed to every source's aggregation as in
        `MatchedSpatialPatcher.merge`.

        Args:
            patches: Iterable of `MatchedTemporalPatch` instances.
            mfield: Original `MatchedField` (used for the typo-guard).
            hooks: Optional observability hooks, dispatched once at the
                matched level (one ``on_merge_start`` / ``on_merge_end``
                per call; the output bytes sum every source).
        """
        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )
        aggregations = {
            PRIMARY_KEY: self.primary.aggregation,
            **self.secondary_aggregators,
        }

        def consumer(agg: Any) -> Callable[[Iterator[Any]], Any]:
            return lambda members: agg.merge(members)

        return _merge_per_source(
            patches,
            {name: consumer(agg) for name, agg in aggregations.items()},
            hooks,
        )


@dataclass(eq=False)
class MatchedSpatioTemporalPatcher(_MatchedConfigMixin):
    """Spatio-temporal matched patcher — yields `MatchedSpatioTemporalPatch`es.

    Mirror of `MatchedSpatialPatcher` over the spatio-temporal axis: it
    drives the primary `SpatioTemporalPatcher` over the `MatchedField`
    and turns each yielded patch into a matched carrier, so the coupling
    mode (``"product"`` or ``"coupled"``, from ``primary.coupling``), the
    chip reads, the ``on_error`` policy, hooks, journal and backpressure
    are the primary's own.

    Args:
        primary: A regular `SpatioTemporalPatcher` configured for the
            primary field. Drives both spatial anchor placement and
            temporal windowing.
        secondary_aggregators: ``{name: temporal.aggregation.Aggregation}`` — one
            temporal aggregator per secondary, matching the per-anchor
            temporal merge shape of `SpatioTemporalPatcher.merge`.
            Names that don't match any entry in ``mfield.secondaries``
            raise on ``split`` / ``merge`` (typo guard).
    """

    primary: SpatioTemporalPatcher
    secondary_aggregators: Mapping[str, TemporalAggregation] = field(
        default_factory=dict
    )

    def split(
        self,
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        coord: np.ndarray | None = None,
        prefetch: int = 0,
        journal: Any | None = None,
        cache: Any | None = None,
        max_in_flight: int | None = None,
        max_in_flight_bytes: int | None = None,
    ) -> Iterator[MatchedSpatioTemporalPatch]:
        """Yield `MatchedSpatioTemporalPatch`es lazily.

        Runs ``self.primary.split`` over ``mfield``: each spatial chip is
        one `MatchedField` read (every source over the primary chip's
        footprint, coregistered onto it) checked to hold the primary and
        to share its time length, and every source is sliced to the same
        time window. Each yielded patch is unpacked into a matched carrier
        whose ``members`` are per-source `SpatioTemporalPatch`es.

        The spatial patcher's ``on_error`` policy covers every source's
        read and coregistration; under ``"mask"`` a failed chip yields
        all-NaN members with an all-False ``valid_mask``. Per-source
        ``valid_mask`` arrays are otherwise computed when
        ``mfield.valid_mask`` is True (the default).

        Args:
            mfield: A `MatchedField` to walk with the primary
                spatio-temporal patcher.
            hooks: Optional observability hooks, dispatched by the
                primary's split once per matched patch — ``(space, time)``
                anchors, ``coord_value``, and ``on_patch_done`` bytes summed
                over every member.
            coord: 1-D coordinate vector along the time axis, as
                `SpatioTemporalPatcher.split` takes it — required by
                coordinate-aware (stencil) temporal geometries / samplers.
            prefetch: If positive, read ahead up to ``prefetch`` matched
                patches on a background thread.
            journal: Forwarded to `SpatioTemporalPatcher.split` — the
                ``(space, time_key)`` keys it holds are skipped.
            cache: A `PatchCache` consulted per source, as in
                `MatchedSpatialPatcher.split` (the coregistration still
                runs).
            max_in_flight: Forwarded to `SpatioTemporalPatcher.split`;
                each yielded matched patch owns the slot — release it with
                ``mp.close()`` or ``with mp: ...``.
            max_in_flight_bytes: Forwarded; a matched patch is sized as
                the sum of its members' bytes.

        Raises:
            ValueError: A secondary's length along the time axis differs
                from the primary's.
        """
        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )
        source: Any = mfield if cache is None else _CachedSourceReads(mfield, cache)
        outers = self.primary._split(
            source,
            hooks,
            coord=coord,
            prefetch=prefetch,
            journal=journal,
            cache=None,
            max_in_flight=max_in_flight,
            max_in_flight_bytes=max_in_flight_bytes,
            unpack=partial(self._unpack, mfield),
        )
        return self._matched(outers, mfield)

    def _unpack(self, mfield: MatchedField, data: Any) -> dict[str, Any]:
        """One chip read → its checked ``{source: data}`` dict.

        A mask placeholder fans out to one all-NaN copy per source, marked
        `_MaskedMembers` so `_matched` gives it an all-False mask.
        """
        cls_name = type(self).__name__
        data_by_name, masked = _unpack_matched(
            data,
            mfield,
            self.primary.spatial,
            cls_name,
            "each spatial Patch.data to be",
        )
        _check_time_lengths(data_by_name, self.primary.time_axis, cls_name)
        return _MaskedMembers(data_by_name) if masked else data_by_name

    def _matched(
        self, outers: Iterator[SpatioTemporalPatch], mfield: MatchedField
    ) -> Iterator[MatchedSpatioTemporalPatch]:
        try:
            for outer in outers:
                members = {
                    name: outer.with_data(data) for name, data in outer.data.items()
                }
                matched = MatchedSpatioTemporalPatch(
                    space=outer.space,
                    time=outer.time,
                    members=members,
                    valid_mask=_compute_member_masks(
                        members, mfield, masked=isinstance(outer.data, _MaskedMembers)
                    ),
                    weights=_member_weights(members),
                )
                _hand_over_release(outer, matched)
                yield matched
        finally:
            _close(outers)

    def merge(
        self,
        patches: Iterable[MatchedSpatioTemporalPatch],
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> dict[str, list[tuple[Any, Any]]]:
        """Per-source merge: dict of ``name -> [(spatial_anchor, temporal_merge), …]``.

        Each source's value is what `SpatioTemporalPatcher.merge` returns
        for that source's members: ``(spatial_anchor,
        temporal_aggregation_result)`` pairs grouped by spatial anchor
        (first-seen order). Secondaries use their `temporal.aggregation.Aggregation`;
        the primary uses ``self.primary.temporal.aggregation``.
        ``patches`` is consumed once and fanned out to every source as in
        `MatchedSpatialPatcher.merge`; each source's grouping still holds
        its groups until the pass ends.

        Args:
            patches: Iterable of `MatchedSpatioTemporalPatch` instances.
            mfield: Original `MatchedField` (used for the typo-guard).
            hooks: Optional observability hooks. Dispatched at the
                matched layer (one ``merge_start`` / ``merge_end`` per
                call; the output bytes sum every source).
        """
        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )
        aggregations = {
            PRIMARY_KEY: self.primary.temporal.aggregation,
            **self.secondary_aggregators,
        }

        def consumer(agg: Any) -> Callable[[Iterator[Any]], Any]:
            return lambda members: _merge_by_space(members, agg)

        return _merge_per_source(
            patches,
            {name: consumer(agg) for name, agg in aggregations.items()},
            hooks,
        )


class _MaskedMembers(dict):
    """``{source: data}`` of an ``on_error="mask"`` chip (all-NaN members)."""

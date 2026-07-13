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
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from geopatcher._src.hooks import (
    PatcherHook,
    _as_hooks,
    _dispatch,
    _len_or_unknown,
    _nbytes,
)


if TYPE_CHECKING:
    from geopatcher._src.matched.field import MatchedField
    from geopatcher._src.matched.patch import (
        MatchedPatch,
        MatchedSpatioTemporalPatch,
        MatchedTemporalPatch,
    )
    from geopatcher._src.spatial.aggregation import SpatialAggregation
    from geopatcher._src.spatial.patcher import SpatialPatcher
    from geopatcher._src.spatial_time import SpatioTemporalPatcher
    from geopatcher._src.time.aggregation import TemporalAggregation
    from geopatcher._src.time.patcher import TemporalPatcher


def _compute_valid_mask(data: Any) -> np.ndarray | None:
    """Best-effort validity mask for ``data``.

    For array-coercible numeric data (the common raster case),
    returns ``np.isfinite(data)`` — True where the value is real
    and finite, False on NaN / +-inf (the conventional nodata
    sentinel for float rasters). For non-array data or non-numeric
    arrays, returns None so the caller can simply omit that source's
    mask entry rather than emit a meaningless all-True / all-False
    array.
    """
    try:
        arr = np.asarray(data)
    except (TypeError, ValueError):
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.isfinite(arr)


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
    from geopatcher._src.matched.patch import PRIMARY_KEY

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


def _compute_member_masks(
    members: Mapping[str, Any], mfield: MatchedField
) -> dict[str, np.ndarray] | None:
    """Per-source validity masks for a matched patch's ``members``.

    Args:
        members: ``{name: patch}`` whose ``data`` attributes are masked.
        mfield: The originating `MatchedField`; masks are only computed
            when its ``valid_mask`` flag is truthy.

    Returns:
        ``{name: mask}`` for members whose data is numeric and
        array-coercible, or None when masking is disabled or no member
        produced a mask.
    """
    if not getattr(mfield, "valid_mask", False):
        return None
    mask_dict = {
        name: mask
        for name, patch in members.items()
        if (mask := _compute_valid_mask(patch.data)) is not None
    }
    return mask_dict or None


def _collect_per_source(
    patches: Iterable[Any], secondary_names: Iterable[str]
) -> dict[str, list[Any]]:
    """Fan matched patches out into per-source patch lists in one pass.

    Patches stream lazily — materialising ``{PRIMARY_KEY: [...],
    name: [...]}`` in a single pass avoids iterating ``patches``
    N+1 times. Member names not in ``secondary_names`` are dropped
    (the documented opt-out for "don't reconstruct this source").

    Args:
        patches: Iterable of matched patch carriers (anything with a
            per-source ``members`` mapping).
        secondary_names: Secondary names to collect, typically the
            patcher's ``secondary_aggregators`` keys.
    """
    from geopatcher._src.matched.patch import PRIMARY_KEY

    per_source: dict[str, list[Any]] = {PRIMARY_KEY: []}
    for name in secondary_names:
        per_source[name] = []
    for mp in patches:
        for name, patch in mp.members.items():
            if name in per_source:
                per_source[name].append(patch)
    return per_source


def _matched_patcher_config(
    primary: Any, secondary_aggregators: Mapping[str, Any]
) -> dict[str, Any]:
    """Shared ``get_config`` body for the matched patcher family.

    Follows the patcher-family envelope convention: nested components
    serialize as ``{"class": type(x).__name__, "config": x.get_config()}``.
    """
    return {
        "primary": {
            "class": type(primary).__name__,
            "config": primary.get_config(),
        },
        "secondary_aggregators": {
            name: {
                "class": type(agg).__name__,
                "config": agg.get_config(),
            }
            for name, agg in secondary_aggregators.items()
        },
    }


@dataclass(eq=False)
class MatchedSpatialPatcher:
    """Spatial patcher that yields `MatchedPatch`es and merges per-source.

    Args:
        primary: A regular `SpatialPatcher` configured for the
            primary `Field`. Drives anchor placement, geometry,
            window, and primary aggregation.
        secondary_aggregators: ``{name: SpatialAggregation}`` — one
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

    def get_config(self) -> dict[str, Any]:
        """Serialize the inner primary patcher + per-secondary aggregators."""
        return _matched_patcher_config(self.primary, self.secondary_aggregators)

    def split(
        self,
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Iterator[MatchedPatch]:
        """Yield `MatchedPatch`es by walking ``mfield`` with the primary's sampler.

        Internally drives ``self.primary.split(mfield)`` — since
        `MatchedField` already satisfies the `Field` Protocol, the
        existing sampler / geometry / window machinery works
        unchanged. Each outer ``Patch`` carries the per-source
        ``dict`` returned by `MatchedField.select` in its ``data``
        field; this method unpacks that dict into a `MatchedPatch`
        whose ``members`` is ``{name: Patch}`` and whose
        ``anchor`` / ``indices`` / ``weights`` mirror the outer
        patch so downstream aggregations see consistent metadata.

        Per-source ``valid_mask`` arrays are computed when
        ``mfield.valid_mask`` is True (the default): for numeric
        array-coercible data, ``np.isfinite(data)`` marks the
        positions of NaN / inf nodata sentinels. Non-array members
        are simply omitted from the mask dict (and the dict drops
        to ``None`` if no member produced a mask).

        Args:
            mfield: A `MatchedField` to drive the primary sampler over.
            hooks: Optional observability hooks forwarded to the
                underlying primary `SpatialPatcher.split`. Per-anchor
                callbacks fire on the outer single-source patch
                lifecycle; matched-specific bookkeeping does not emit
                additional events.
        """
        from geopatcher._src.matched.patch import MatchedPatch
        from geopatcher._src.patch import Patch

        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )

        for outer in self.primary.split(mfield, hooks=hooks):
            data_by_name = outer.data
            # Belt-and-braces: a SpatialPatcher fed a plain Field
            # would give us a non-dict here.
            _check_matched_dict(
                data_by_name, type(self).__name__, "each Patch.data to be"
            )
            members = {
                name: Patch(
                    data=data,
                    anchor=outer.anchor,
                    indices=outer.indices,
                    weights=outer.weights,
                )
                for name, data in data_by_name.items()
            }
            valid_mask = _compute_member_masks(members, mfield)
            yield MatchedPatch(
                anchor=outer.anchor,
                members=members,
                valid_mask=valid_mask,
            )

    def n_anchors(self, mfield: MatchedField) -> int:
        """Number of `MatchedPatch`es ``split`` will yield."""
        return self.primary.n_anchors(mfield)

    def anchors(self, mfield: MatchedField) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``mfield``."""
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

        The value type is intentionally ``Any`` because the
        underlying `SpatialAggregation.merge` returns whatever the
        aggregator produces — typically a `GeoTensor` for stitched
        rasters, but for ``Sum`` / ``Mean`` / ``Max`` it may be a
        plain numpy array. Callers that need a `Field` shape can
        wrap with the source's ``Field.with_data``.

        Every source is aggregated against the primary's domain
        because the coregistration callable mapped each secondary
        onto the primary's grid at split time. Reconstructing a
        secondary back into its own original grid would require
        re-inverting the coregistration, which is the user's
        problem if they need it.

        Strict-mode streaming-safety: each secondary aggregator is
        checked via the same ``_warn_if_unsafe_streaming`` helper
        the primary ``SpatialPatcher`` uses, so a non-streaming
        secondary aggregation surfaces the same warning/error in
        strict mode as the primary path.

        Args:
            patches: Iterable of `MatchedPatch` instances.
            mfield: Original `MatchedField` (used for typo-guard and
                to recover the primary domain for aggregation).
            hooks: Optional observability hooks forwarded to the
                primary ``SpatialPatcher.merge``; secondary aggregations
                are intentionally not double-dispatched so the hook event
                stream stays linear (one merge_start / merge_end per call).
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY
        from geopatcher._src.spatial.aggregation import _warn_if_unsafe_streaming

        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )

        per_source = _collect_per_source(patches, self.secondary_aggregators)

        primary_domain = mfield.domain
        result: dict[str, Any] = {
            PRIMARY_KEY: self.primary.merge(
                per_source[PRIMARY_KEY], primary_domain, hooks=hooks
            ),
        }
        for name, agg in self.secondary_aggregators.items():
            # Mirror the primary path's strict-mode streaming check
            # so a non-streaming secondary aggregator doesn't slip
            # through.
            _warn_if_unsafe_streaming(agg)
            result[name] = agg.merge(per_source[name], primary_domain)
        return result


@dataclass(eq=False)
class MatchedTemporalPatcher:
    """Temporal patcher that yields `MatchedTemporalPatch`es and merges per-source.

    Mirror of `MatchedSpatialPatcher` over the time axis. The
    `TemporalPatcher` slices a numpy array directly (it does not call
    `Field.select` per anchor the way `SpatialPatcher` does), so the
    matched-temporal split first materialises the per-source series
    via ``mfield.select(slice(None))`` (the full-range temporal
    indexer), then drives the primary patcher on the primary's series
    and slices each secondary's array in lockstep.

    Args:
        primary: A regular `TemporalPatcher` configured for the
            primary series. Drives anchor placement, geometry,
            window, and primary aggregation.
        secondary_aggregators: ``{name: TemporalAggregation}`` — one
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

    def get_config(self) -> dict[str, Any]:
        """Serialize the inner primary patcher + per-secondary aggregators."""
        return _matched_patcher_config(self.primary, self.secondary_aggregators)

    def split(
        self,
        mfield: MatchedField,
        time_axis: int = 0,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Iterator[MatchedTemporalPatch]:
        """Yield `MatchedTemporalPatch`es by driving the primary on each anchor.

        Materialises per-source full series once via
        ``mfield.select(slice(None))`` — the natural full-range
        temporal indexer — then drives the primary `TemporalPatcher`
        on the primary's array. For each yielded `TemporalPatch`,
        each secondary's array is sliced with the same
        ``indices`` and packaged into the matched carrier.

        Per-source ``valid_mask`` arrays are computed when
        ``mfield.valid_mask`` is True (the default): for numeric
        array-coercible data, ``np.isfinite(data)`` marks the
        positions of NaN / inf nodata sentinels.

        Args:
            mfield: A `MatchedField` whose ``select`` returns the
                per-source full series as a dict keyed by source name.
            time_axis: Which axis of each source's array is the time
                axis. Default 0. Must be the same across sources.
            hooks: Optional observability hooks forwarded to the
                underlying primary `TemporalPatcher.split`. Per-anchor
                callbacks fire on the primary single-source patch
                lifecycle; matched-specific bookkeeping does not emit
                additional events.
        """
        from geopatcher._src.matched.patch import (
            PRIMARY_KEY,
            MatchedTemporalPatch,
        )
        from geopatcher._src.patch import TemporalPatch

        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )

        data_by_name = mfield.select(slice(None))
        _check_matched_dict(
            data_by_name, type(self).__name__, "mfield.select to return"
        )

        arrays = {name: np.asarray(data) for name, data in data_by_name.items()}
        primary_arr = arrays[PRIMARY_KEY]

        for primary_patch in self.primary.split(primary_arr, time_axis, hooks=hooks):
            idx: list[Any] = [slice(None)] * primary_arr.ndim
            idx[time_axis] = primary_patch.indices
            tup = tuple(idx)
            members = {
                name: TemporalPatch(
                    data=arr[tup],
                    anchor=primary_patch.anchor,
                    indices=primary_patch.indices,
                    weights=primary_patch.weights,
                )
                for name, arr in arrays.items()
            }
            valid_mask = _compute_member_masks(members, mfield)
            yield MatchedTemporalPatch(
                anchor=primary_patch.anchor,
                members=members,
                valid_mask=valid_mask,
            )

    def n_anchors(self, mfield: MatchedField, time_axis: int = 0) -> int:
        """Number of `MatchedTemporalPatch`es ``split`` will yield."""
        primary_arr = np.asarray(mfield.select(slice(None))[_primary_key()])
        return self.primary.n_anchors(primary_arr, time_axis)

    def anchors(self, mfield: MatchedField, time_axis: int = 0) -> list[int]:
        """Materialise the sampler's anchor sequence for ``mfield``."""
        primary_arr = np.asarray(mfield.select(slice(None))[_primary_key()])
        return self.primary.anchors(primary_arr, time_axis)

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

        Unlike the spatial path, `TemporalAggregation.merge` takes
        only the patches (no domain argument), so ``mfield`` is used
        solely for the typo-guard check.

        Args:
            patches: Iterable of `MatchedTemporalPatch` instances.
            mfield: Original `MatchedField` (used for the typo-guard).
            hooks: Optional observability hooks forwarded to the
                primary ``TemporalPatcher.merge``; secondary aggregations
                are not double-dispatched so the hook event stream stays
                linear (one merge_start / merge_end per call).
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY

        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )

        per_source = _collect_per_source(patches, self.secondary_aggregators)

        result: dict[str, Any] = {
            PRIMARY_KEY: self.primary.merge(per_source[PRIMARY_KEY], hooks=hooks),
        }
        for name, agg in self.secondary_aggregators.items():
            result[name] = agg.merge(per_source[name])
        return result


@dataclass(eq=False)
class MatchedSpatioTemporalPatcher:
    """Spatio-temporal matched patcher — yields `MatchedSpatioTemporalPatch`es.

    Mirror of `MatchedSpatialPatcher` over the spatio-temporal axis.
    Inherits the coupling mode (``"product"`` or ``"coupled"``) from
    ``primary.coupling`` — the matched version does not add its own
    coupling argument so the primary stays the single source of truth.

    Args:
        primary: A regular `SpatioTemporalPatcher` configured for the
            primary field. Drives both spatial anchor placement and
            temporal windowing.
        secondary_aggregators: ``{name: TemporalAggregation}`` — one
            temporal aggregator per secondary, matching the per-anchor
            temporal merge shape of `SpatioTemporalPatcher.merge`.
            Names that don't match any entry in ``mfield.secondaries``
            raise on ``split`` / ``merge`` (typo guard).
    """

    primary: SpatioTemporalPatcher
    secondary_aggregators: Mapping[str, TemporalAggregation] = field(
        default_factory=dict
    )

    def get_config(self) -> dict[str, Any]:
        """Serialize the inner primary patcher + per-secondary aggregators."""
        return _matched_patcher_config(self.primary, self.secondary_aggregators)

    def split(
        self,
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Iterator[MatchedSpatioTemporalPatch]:
        """Yield `MatchedSpatioTemporalPatch`es lazily.

        Delegates to one of two private methods depending on
        ``self.primary.coupling`` — ``"product"`` for the Cartesian
        product of spatial / time anchors, ``"coupled"`` for explicit
        ``(space, time)`` pairs.

        Args:
            mfield: A `MatchedField` to walk with the primary
                spatio-temporal patcher.
            hooks: Optional observability hooks. The matched layer
                emits its own ``on_patch_start`` / ``on_patch_done`` /
                ``on_error`` for each spatio-temporal anchor pair so
                callers see the matched-level lifecycle rather than the
                interleaved single-source dispatch.
        """
        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )
        coupling = self.primary.coupling
        hook_list = _as_hooks(hooks)
        if not hook_list:
            if coupling == "product":
                yield from self._split_product(mfield)
            elif coupling == "coupled":
                yield from self._split_coupled(mfield)
            else:
                raise ValueError(f"unknown coupling: {coupling!r}")
            return
        _dispatch(hook_list, "on_split_start", self.primary._split_total_hint(mfield))
        try:
            if coupling == "product":
                yield from self._split_product(mfield, hook_list)
            elif coupling == "coupled":
                yield from self._split_coupled(mfield, hook_list)
            else:
                raise ValueError(f"unknown coupling: {coupling!r}")
        finally:
            _dispatch(hook_list, "on_split_end")

    def _split_product(
        self,
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] = (),
    ) -> Iterator[MatchedSpatioTemporalPatch]:
        from time import perf_counter

        from geopatcher._src.matched.patch import (
            PRIMARY_KEY,
            MatchedSpatioTemporalPatch,
        )
        from geopatcher._src.patch import SpatioTemporalPatch

        spatial = self.primary.spatial
        temporal = self.primary.temporal
        time_axis = self.primary.time_axis

        for sp in spatial.split(mfield):
            data_by_name = sp.data
            _check_matched_dict(
                data_by_name, type(self).__name__, "each spatial Patch.data to be"
            )
            arrays = {name: np.asarray(d) for name, d in data_by_name.items()}
            primary_arr = arrays[PRIMARY_KEY]
            time_len = int(primary_arr.shape[time_axis])
            for t_anchor in temporal.sampler.anchors(time_len):
                t_window = temporal.geometry.window(time_len, int(t_anchor))
                slices = t_window if isinstance(t_window, list) else [t_window]
                for s in slices:
                    anchor = (sp.anchor, int(t_anchor))
                    _dispatch(hooks, "on_patch_start", anchor)
                    start = perf_counter()
                    try:
                        idx: list[Any] = [slice(None)] * primary_arr.ndim
                        idx[time_axis] = s
                        tup = tuple(idx)
                        members = {
                            name: SpatioTemporalPatch(
                                data=arr[tup],
                                space=sp.anchor,
                                time=int(t_anchor),
                                spatial_indices=sp.indices,
                                temporal_indices=s,
                                weights=sp.weights,
                            )
                            for name, arr in arrays.items()
                        }
                        valid_mask = _compute_member_masks(members, mfield)
                        matched = MatchedSpatioTemporalPatch(
                            space=sp.anchor,
                            time=int(t_anchor),
                            members=members,
                            valid_mask=valid_mask,
                        )
                    except Exception as exc:
                        _dispatch(hooks, "on_error", anchor, exc)
                        raise
                    _dispatch(
                        hooks,
                        "on_patch_done",
                        anchor,
                        perf_counter() - start,
                        _nbytes(matched.members[PRIMARY_KEY].data),
                    )
                    yield matched

    def _split_coupled(
        self,
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] = (),
    ) -> Iterator[MatchedSpatioTemporalPatch]:
        from time import perf_counter

        from geopatcher._src.matched.patch import (
            PRIMARY_KEY,
            MatchedSpatioTemporalPatch,
        )
        from geopatcher._src.patch import SpatioTemporalPatch

        spatial = self.primary.spatial
        temporal = self.primary.temporal
        time_axis = self.primary.time_axis

        anchors = getattr(spatial.sampler, "anchors_", None)
        if anchors is None:
            raise TypeError(
                "coupled coupling requires the spatial sampler to expose an "
                "`anchors_` list of (space_anchor, time_anchor) tuples — i.e. "
                "use SpatialExplicit(anchors_=[...])."
            )
        for pair in anchors:
            space_anchor, time_anchor = pair
            anchor = (space_anchor, int(time_anchor))
            _dispatch(hooks, "on_patch_start", anchor)
            start = perf_counter()
            try:
                indices = spatial.geometry.neighborhood(mfield.domain, space_anchor)
                data_by_name = mfield.select(indices)
                _check_matched_dict(
                    data_by_name, type(self).__name__, "each spatial Patch.data to be"
                )
                arrays = {name: np.asarray(d) for name, d in data_by_name.items()}
                primary_arr = arrays[PRIMARY_KEY]
                time_len = int(primary_arr.shape[time_axis])
                t_window = temporal.geometry.window(time_len, int(time_anchor))
                slices = t_window if isinstance(t_window, list) else [t_window]
                try:
                    base_weights = spatial.window.weights(spatial.geometry)
                except TypeError:
                    base_weights = None
            except Exception as exc:
                _dispatch(hooks, "on_error", anchor, exc)
                raise
            for s in slices:
                try:
                    idx: list[Any] = [slice(None)] * primary_arr.ndim
                    idx[time_axis] = s
                    tup = tuple(idx)
                    members = {
                        name: SpatioTemporalPatch(
                            data=arr[tup],
                            space=space_anchor,
                            time=int(time_anchor),
                            spatial_indices=indices,
                            temporal_indices=s,
                            weights=base_weights,
                        )
                        for name, arr in arrays.items()
                    }
                    valid_mask = _compute_member_masks(members, mfield)
                    matched = MatchedSpatioTemporalPatch(
                        space=space_anchor,
                        time=int(time_anchor),
                        members=members,
                        valid_mask=valid_mask,
                    )
                except Exception as exc:
                    _dispatch(hooks, "on_error", anchor, exc)
                    raise
                _dispatch(
                    hooks,
                    "on_patch_done",
                    anchor,
                    perf_counter() - start,
                    _nbytes(matched.members[PRIMARY_KEY].data),
                )
                yield matched

    def merge(
        self,
        patches: Iterable[MatchedSpatioTemporalPatch],
        mfield: MatchedField,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> dict[str, list[tuple[Any, Any]]]:
        """Per-source merge: dict of ``name -> [(spatial_anchor, temporal_merge), …]``.

        Mirrors `SpatioTemporalPatcher.merge`: each source's value is
        a list of ``(spatial_anchor, temporal_aggregation_result)``
        pairs grouped by spatial anchor (first-seen order).
        Aggregations on each secondary use that secondary's
        `TemporalAggregation`; the primary uses
        ``self.primary.temporal.aggregation``.

        Args:
            patches: Iterable of `MatchedSpatioTemporalPatch` instances.
            mfield: Original `MatchedField` (used for the typo-guard).
            hooks: Optional observability hooks. Dispatched at the
                matched layer (one ``merge_start`` / ``merge_end`` per
                call). Per-source aggregations are not double-dispatched.
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY
        from geopatcher._src.patch import TemporalPatch
        from geopatcher._src.spatial_time import _hashable

        _validate_aggregator_names(
            self.secondary_aggregators, mfield, type(self).__name__
        )

        hook_list = _as_hooks(hooks)
        _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
        try:
            per_source_groups: dict[str, dict[Any, tuple[Any, list[Any]]]] = {
                PRIMARY_KEY: {}
            }
            for name in self.secondary_aggregators:
                per_source_groups[name] = {}

            for mp in patches:
                for name, patch in mp.members.items():
                    groups = per_source_groups.get(name)
                    if groups is None:
                        continue
                    key = _hashable(patch.space)
                    groups.setdefault(key, (patch.space, []))[1].append(patch)

            def _aggregate(
                groups: dict[Any, tuple[Any, list[Any]]],
                agg: TemporalAggregation,
            ) -> list[tuple[Any, Any]]:
                return [
                    (
                        anchor,
                        agg.merge(
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
                    for anchor, group in groups.values()
                ]

            result: dict[str, list[tuple[Any, Any]]] = {
                PRIMARY_KEY: _aggregate(
                    per_source_groups[PRIMARY_KEY],
                    self.primary.temporal.aggregation,
                ),
            }
            for name, agg in self.secondary_aggregators.items():
                result[name] = _aggregate(per_source_groups[name], agg)
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(result))
        return result


def _primary_key() -> str:
    # Late-import shim to avoid pulling `patch` into module-import time
    # for the lightweight helpers above. ``PRIMARY_KEY`` is a constant
    # but lives behind the matched.patch module to keep the carrier
    # the single source of truth.
    from geopatcher._src.matched.patch import PRIMARY_KEY

    return PRIMARY_KEY

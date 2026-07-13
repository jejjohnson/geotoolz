"""Implementation of `matchup()` — kept separate from the public
shim in ``engine.py`` so the user-facing module stays small.

The algorithm:

1. Materialise every secondary role into a list. Build a
   `shapely.strtree.STRtree` per role, so spatial pre-filtering is
   O(log M) per primary instead of O(M).
2. Walk the primary stream once. For each primary row:
   a. Query each role's STRtree for envelope-overlap candidates.
   b. Pull the candidates' intervals into a parallel
      `pd.IntervalIndex` and apply the temporal strategy → narrow
      candidate set.
   c. Apply the spatial strategy as the truth gate
      (STRtree only checks envelopes; intersection / IoU / contains
      need the precise geometry).
   d. Combine per-role survivors into MatchupRows. ``join="all"``
      drops primaries with any empty role; ``join="any"`` allows
      partial coverage.
3. Yield `MatchupRow`s as they're produced.

Tolerance metadata stored on each row is enough to re-run the
matchup deterministically: the strategy class name, its dataclass
fields, and the `join` policy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd


if TYPE_CHECKING:
    from geocatalog._src.matchup.engine import MatchupRow
    from geocatalog._src.matchup.spatial import SpatialStrategy
    from geocatalog._src.matchup.temporal import TemporalStrategy
    from geocatalog._src.sources._base import SourceRow


def run_matchup(
    primary: Iterable[SourceRow],
    secondary: Iterable[SourceRow] | Mapping[str, Iterable[SourceRow]],
    *,
    spatial: SpatialStrategy,
    temporal: TemporalStrategy,
    join: Literal["all", "any"] = "all",
    tag: str | None = None,
) -> Iterator[MatchupRow]:
    """See `geocatalog._src.matchup.engine.matchup` for the public docstring."""
    from shapely.strtree import STRtree

    from geocatalog._src.matchup.engine import (
        MatchupRow,
        _midpoint_seconds,
        _new_matchup_id,
        _utcnow,
    )

    # Normalise the secondary argument into a `dict[role, list[row]]`.
    # Pairwise → single role "secondary".
    roles: dict[str, list[SourceRow]] = (
        {"secondary": list(secondary)}
        if not isinstance(secondary, Mapping)
        else {role: list(rows) for role, rows in secondary.items()}
    )

    # Per-role STRtree. shapely's STRtree returns positional indices
    # since shapely 2.0, which lets us reuse the parallel row list.
    trees: dict[str, STRtree] = {
        role: STRtree([row.geometry for row in rows]) for role, rows in roles.items()
    }

    strategy_label = _strategy_label(spatial, temporal)
    tolerance_meta = {
        "spatial": _dataclass_summary(spatial),
        "temporal": _dataclass_summary(temporal),
        "join": join,
    }

    for primary_row in primary:
        per_role_survivors: dict[str, list[SourceRow]] = {}
        for role, rows in roles.items():
            tree = trees[role]
            # 1) Envelope pre-filter.
            cand_indices = list(tree.query(primary_row.geometry))
            if not cand_indices:
                per_role_survivors[role] = []
                continue
            cand_rows = [rows[i] for i in cand_indices]

            # 2) Temporal narrow.
            cand_intervals = pd.IntervalIndex.from_tuples(
                [(row.interval.left, row.interval.right) for row in cand_rows],
                closed="both",
            )
            keep = temporal.filter(primary_row.interval, cand_intervals)
            if len(keep) == 0:
                per_role_survivors[role] = []
                continue

            # `temporal.filter` returns a subset of `cand_intervals`
            # **in input position order**. Recover positions by a
            # parallel sequential walk — preserves both multiplicity
            # (two candidates with identical intervals) and selector
            # semantics (NearestInTime returns exactly one position,
            # not all rows that happen to share the same timestamp).
            keep_intervals = list(keep)
            keep_idx = 0
            narrowed_rows: list[SourceRow] = []
            for i, cand_iv in enumerate(cand_intervals):
                if keep_idx >= len(keep_intervals):
                    break
                target = keep_intervals[keep_idx]
                if cand_iv.left == target.left and cand_iv.right == target.right:
                    narrowed_rows.append(cand_rows[i])
                    keep_idx += 1

            # 3) Spatial truth gate.
            confirmed = [
                row
                for row in narrowed_rows
                if spatial.match(primary_row.geometry, row.geometry)
            ]
            per_role_survivors[role] = confirmed

        # 4) Combine per-role survivors.
        if join == "all" and any(
            len(rows) == 0 for rows in per_role_survivors.values()
        ):
            continue
        # Roles with 0 survivors under "any" contribute a single
        # "no match" slot (None). We model that by injecting [None]
        # so the cartesian product is well-defined; downstream code
        # filters None members when constructing the row.
        per_role_for_product = {
            role: rows if rows else [None] for role, rows in per_role_survivors.items()
        }
        role_names = list(per_role_for_product.keys())
        role_lists = [per_role_for_product[role] for role in role_names]

        for combo in product(*role_lists):
            members: list[SourceRow] = [primary_row]
            roles_out: list[str] = ["primary"]
            for role_name, sec_row in zip(role_names, combo, strict=True):
                if sec_row is None:
                    # Only possible under "any" join.
                    continue
                members.append(sec_row)
                roles_out.append(role_name)
            if len(members) == 1:
                # No secondary made it through under `any` (or
                # defensively, under `all` — though we already
                # dropped those above). A primary-only "matchup"
                # is meaningless; skip it regardless of join.
                continue

            ref_time = _interval_midpoint(primary_row.interval)
            offsets = [_midpoint_seconds(row.interval, ref_time) for row in members]
            common_geom = _common_intersection([row.geometry for row in members])
            yield MatchupRow(
                matchup_id=_new_matchup_id(),
                strategy=strategy_label,
                member_ids=tuple(row.id for row in members),
                member_sources=tuple(row.source for row in members),
                member_roles=tuple(roles_out),
                geometry_intersect=common_geom,
                time_reference=ref_time,
                time_offset_sec=tuple(offsets),
                tolerance=tolerance_meta,
                query_set=tag,
            )

    # Reference _utcnow so it doesn't get pruned as unused. The
    # helper is hoisted in `engine.py` for tests that want to pin
    # the clock; the engine itself derives ref_time from each
    # primary's interval, but having `_utcnow` available is part
    # of the public test surface.
    _ = _utcnow


def _interval_midpoint(interval: pd.Interval) -> Any:
    """Return ``primary.interval`` midpoint as a tz-aware datetime.

    Used as the per-row reference for `time_offset_sec` arithmetic.
    """
    left = pd.Timestamp(interval.left)
    right = pd.Timestamp(interval.right)
    mid = left + (right - left) / 2
    mid = mid.tz_localize("UTC") if mid.tzinfo is None else mid.tz_convert("UTC")
    return mid.to_pydatetime()


def _common_intersection(geometries: list[Any]) -> Any:
    """N-way intersection of geometries; primary first.

    Returns the geometric intersection of all member footprints.
    For the pairwise case this is just ``primary ∩ secondary``;
    for N-way it's the iterated intersection in member order.
    """
    if not geometries:
        raise ValueError("Cannot intersect zero geometries.")
    result = geometries[0]
    for geom in geometries[1:]:
        result = result.intersection(geom)
    return result


def _strategy_label(spatial: SpatialStrategy, temporal: TemporalStrategy) -> str:
    """Concise label like ``"IouAtLeast(0.2) & NearestInTime(6h)"``."""
    return f"{_dataclass_summary(spatial)} & {_dataclass_summary(temporal)}"


def _dataclass_summary(strategy: Any) -> str:
    """``ClassName(field=value, ...)`` for any dataclass-shaped strategy.

    Falls back to ``repr(strategy)`` for non-dataclass instances so
    custom strategies still get their state recorded in the
    persisted tolerance metadata.
    """
    if not dataclasses.is_dataclass(strategy):
        return repr(strategy)
    cls_name = type(strategy).__name__
    parts = []
    for field in dataclasses.fields(strategy):
        val = getattr(strategy, field.name)
        parts.append(f"{field.name}={val!r}")
    return f"{cls_name}({', '.join(parts)})"

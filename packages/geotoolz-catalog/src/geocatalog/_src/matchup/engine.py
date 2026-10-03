"""Matchup engine and `MatchupRow` carrier.

Joins two (or more) collections of observations on space + time:

1. Every input — a `GeoCatalog`, a `CatalogBundle` or an iterable of
   `SourceRow` — is normalised to members with a ``(source, collection, id)`` key,
   a footprint in one working CRS and a UTC-aware interval.
2. Each secondary role is sorted by ``(source, collection, id)`` and indexed in an
   STRtree, so candidate order — and hence every tie-break — does not
   depend on input order or tree traversal.
3. For each primary, the index returns candidates overlapping the
   spatial strategy's search envelope; the temporal strategy narrows
   them by time and the spatial strategy confirms the match (the index
   only checks envelopes).
4. Per-role survivors are combined into `MatchupRow`s, each with a
   content-hashed ``matchup_id``, so re-running on the same inputs
   yields the same rows.

The engine runs in memory with shapely; persist its output with
`CatalogBundle.write_matchups` (``matchups.parquet``). See
``docs/design/query-matchup.md`` §4.4 / §4.6.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from geocatalog._src._timeutil import to_utc_ts
from geocatalog._src.matchup.temporal import _midpoint, _to_timedelta


if TYPE_CHECKING:
    import pyproj
    import shapely.geometry.base

    from geocatalog._src.base import CatalogRow, GeoCatalog
    from geocatalog._src.bundle import CatalogBundle
    from geocatalog._src.matchup.spatial import SpatialStrategy
    from geocatalog._src.matchup.temporal import TemporalStrategy
    from geocatalog._src.sources._base import SourceRow


#: Role reserved for the primary member of every `MatchupRow`.
PRIMARY_ROLE = "primary"


@dataclasses.dataclass(frozen=True)
class MatchupRow:
    """A single matched tuple, persisted to ``matchups.parquet``.

    Attributes:
        matchup_id: Content hash (32 hex characters) of the strategy
            parameters and the members' ``(role, source, id)``; the
            same inputs and strategies always give the same id.
        strategy: Human-readable label naming the spatial and
            temporal strategies used
            (``"IouAtLeast(threshold=0.2) & NearestInTime(dt='6h')"``).
        member_ids: Parallel arrays with ``member_sources`` and
            ``member_roles``. ``member_ids[0]`` is always the primary.
        member_sources: Source of each member — ``SourceRow.source``,
            or a catalog row's ``source`` column (``"catalog"`` when
            it has none).
        member_roles: Role tags — ``"primary"``, ``"secondary"``, or
            user-defined names for N-way matchups.
        geometry_intersect: Common footprint of the members, in the
            matchup's working CRS (``tolerance["crs"]``).
        time_reference: Reference instant the offsets are measured
            from — the primary's interval midpoint, UTC-aware.
        time_offset_sec: Parallel to ``member_ids``; offset of each
            member's interval midpoint from ``time_reference``.
        tolerance: Strategy parameters as plain JSON values —
            ``{"spatial": {"type": ..., ...}, "temporal": {"type": ...,
            "<field>_sec": ...}, "join": ..., "crs": ...}``; durations
            are seconds, distances are in units of ``crs``.
        member_collections: Parallel to ``member_ids``; each member's
            collection (``SourceRow.collection``, or a catalog row's
            ``collection`` column, ``""`` when it has none). Together
            with ``member_sources`` and ``member_ids`` it identifies the
            bundle item a member refers to. Empty for rows written
            before it was recorded.
        query_set: Optional user label, persisted as the
            ``query_set`` column in ``matchups.parquet`` so
            ``geocatalog stage --matchup-tag <name>`` can select a
            named set. Mirrors the ``tag`` argument of `matchup()`.
    """

    matchup_id: str
    strategy: str
    member_ids: tuple[str, ...]
    member_sources: tuple[str, ...]
    member_roles: tuple[str, ...]
    geometry_intersect: shapely.geometry.base.BaseGeometry
    time_reference: datetime
    time_offset_sec: tuple[float, ...]
    tolerance: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    query_set: str | None = None
    member_collections: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class _Member:
    """An input row normalised for the join."""

    id: str
    source: str
    collection: str
    geometry: shapely.geometry.base.BaseGeometry  # in the working CRS
    interval: pd.Interval  # UTC-aware, closed="both"

    @property
    def key(self) -> tuple[str, str, str]:
        # The bundle's item identity: ids are unique only per collection.
        return (self.source, self.collection, self.id)


def matchup(
    primary: GeoCatalog | CatalogBundle | Iterable[SourceRow],
    secondary: GeoCatalog
    | CatalogBundle
    | Iterable[SourceRow]
    | Mapping[str, GeoCatalog | CatalogBundle | Iterable[SourceRow]],
    *,
    spatial: SpatialStrategy,
    temporal: TemporalStrategy,
    join: Literal["all", "any"] = "all",
    tag: str | None = None,
    crs: Any = None,
    include_self: bool = False,
) -> Iterator[MatchupRow]:
    """Find matching tuples of observations.

    Args:
        primary: Primary observations — a `GeoCatalog`, a
            `CatalogBundle` (its items) or an iterable of `SourceRow`.
            Read lazily; each row is processed once, in input order.
        secondary: One input of the same kinds (pairwise — every
            member gets the role ``"secondary"``) or a mapping of
            role → input (N-way — each role indexed independently).
            Secondaries are read in full before the first row is
            yielded. ``"primary"`` is reserved and cannot be a role.
        spatial: Strategy deciding spatial matches
            (e.g. ``Intersects()``, ``IouAtLeast(0.2)``).
        temporal: Strategy deciding temporal matches
            (e.g. ``NearestInTime(dt="6h")``). Shipped strategies
            select candidates by position; a custom strategy that
            only implements ``filter`` must return intervals from the
            candidates it was given.
        join: ``"all"`` (default) requires every secondary role to
            contribute a member; primaries with any empty role are
            skipped. ``"any"`` emits matchups missing some roles —
            handy for opportunistic fusion.
        tag: Optional user label persisted as ``query_set`` so a
            CLI user can ``--matchup-tag foo`` later.
        crs: Working CRS for the join; every footprint is reprojected
            into it and ``geometry_intersect`` is expressed in it.
            Defaults to the primary catalog's CRS, or EPSG:4326 when
            the primary is an iterable of `SourceRow` (whose
            footprints are lon/lat). Distances such as
            ``CentroidWithin.buffer`` are in its units, so pass a
            projected CRS (metres) for a distance in metres — or for
            footprints split at the antimeridian, which a local
            projection rejoins.
        include_self: Keep tuples that pair a row with itself (same
            ``(source, collection, id)``), e.g. when matching a catalog against
            itself. Off by default.

    Yields:
        `MatchupRow` instances, one per matched tuple, in primary
        input order and then by ``(source, collection, id)`` of the secondaries.
        Duplicate tuples are emitted once. Rows whose interval is
        missing (``NaT``) never match.

    Raises:
        ValueError: If a secondary role is named ``"primary"``, the
            role mapping is empty, or ``join`` is unknown.
    """
    import pyproj
    from shapely.strtree import STRtree

    from geocatalog._src.matchup.spatial import CentroidWithin, search_envelope

    if join not in ("all", "any"):
        raise ValueError(f"join must be 'all' or 'any'; got {join!r}")
    secondaries = (
        dict(secondary) if isinstance(secondary, Mapping) else {"secondary": secondary}
    )
    if not secondaries:
        raise ValueError("secondary mapping has no roles")
    if PRIMARY_ROLE in secondaries:
        raise ValueError(
            f"{PRIMARY_ROLE!r} is the primary's role; name the secondary role "
            "something else"
        )

    primary = _unwrap_bundle(primary)
    work_crs = pyproj.CRS.from_user_input(
        crs if crs is not None else _native_crs(primary) or "EPSG:4326"
    )
    if (
        isinstance(spatial, CentroidWithin)
        and isinstance(spatial.buffer, int | float)
        and spatial.buffer > 0
        and work_crs.is_geographic
    ):
        warnings.warn(
            f"CentroidWithin(buffer={spatial.buffer}) is in degrees: the "
            f"working CRS {work_crs.to_string()} is geographic. Pass `crs=` "
            "a projected CRS for a buffer in metres.",
            UserWarning,
            stacklevel=2,
        )

    roles: dict[str, list[_Member]] = {}
    for role, rows in secondaries.items():
        members = _dedup(_members(rows, work_crs))
        roles[role] = sorted(members, key=lambda m: m.key)
    trees = {
        role: STRtree([m.geometry for m in members]) for role, members in roles.items()
    }

    strategy_label = f"{_summary(spatial)} & {_summary(temporal)}"
    tolerance = {
        "spatial": _params(spatial),
        "temporal": _params(temporal),
        "join": join,
        "crs": work_crs.to_string(),
    }
    strategy_key = json.dumps(tolerance, sort_keys=True)

    seen: set[str] = set()
    for p in _members(primary, work_crs):
        survivors: dict[str, list[_Member]] = {}
        envelope = search_envelope(spatial, p.geometry)
        for role, members in roles.items():
            candidates = [
                members[i]
                for i in sorted(int(i) for i in trees[role].query(envelope))
                if include_self or members[i].key != p.key
            ]
            survivors[role] = [
                c
                for c in _select_in_time(temporal, p, candidates)
                if spatial.match(p.geometry, c.geometry)
            ]

        if join == "all" and not all(survivors.values()):
            continue
        # Under "any", an empty role contributes a "no match" slot.
        role_names = list(survivors)
        slots = [survivors[r] or [None] for r in role_names]
        for combo in product(*slots):
            members_out = [p]
            roles_out = [PRIMARY_ROLE]
            for role, member in zip(role_names, combo, strict=True):
                if member is not None:
                    members_out.append(member)
                    roles_out.append(role)
            if len(members_out) == 1:
                continue  # a primary-only "matchup" is meaningless
            keys = [m.key for m in members_out]
            if not include_self and len(set(keys)) < len(keys):
                continue  # one row in two roles
            matchup_id = _matchup_id(strategy_key, roles_out, keys)
            if matchup_id in seen:
                continue
            seen.add(matchup_id)
            reference = _midpoint(p.interval)
            yield MatchupRow(
                matchup_id=matchup_id,
                strategy=strategy_label,
                member_ids=tuple(m.id for m in members_out),
                member_sources=tuple(m.source for m in members_out),
                member_collections=tuple(m.collection for m in members_out),
                member_roles=tuple(roles_out),
                geometry_intersect=_common_intersection(
                    [m.geometry for m in members_out]
                ),
                time_reference=reference.to_pydatetime(),
                time_offset_sec=tuple(
                    (_midpoint(m.interval) - reference).total_seconds()
                    for m in members_out
                ),
                tolerance=tolerance,
                query_set=tag,
            )


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _unwrap_bundle(obj: Any) -> Any:
    """A `CatalogBundle` stands for its items catalog."""
    from geocatalog._src.bundle import CatalogBundle

    return obj.catalog if isinstance(obj, CatalogBundle) else obj


def _is_catalog(obj: Any) -> bool:
    return hasattr(obj, "iter_rows") and hasattr(obj, "crs")


def _native_crs(obj: Any) -> pyproj.CRS | None:
    return obj.crs if _is_catalog(obj) else None


def _members(obj: Any, work_crs: pyproj.CRS) -> Iterator[_Member]:
    """Normalise an input to `_Member`s, skipping rows with no interval."""
    import pyproj

    obj = _unwrap_bundle(obj)
    if _is_catalog(obj):
        to_work = _reprojector(obj.crs, work_crs)
        for row in obj.iter_rows():
            member = _from_catalog_row(row, to_work)
            if member is not None:
                yield member
        return
    to_work = _reprojector(pyproj.CRS.from_epsg(4326), work_crs)
    for row in obj:
        interval = _utc_interval(row.interval)
        if interval is not None:
            yield _Member(
                id=str(row.id),
                source=str(row.source),
                collection=str(row.collection),
                geometry=to_work(row.geometry),
                interval=interval,
            )


def _from_catalog_row(row: CatalogRow, to_work: Any) -> _Member | None:
    interval = _utc_interval(row.interval)
    if interval is None:
        return None
    # Bundle items carry the source row's `id` / `source`; a plain
    # catalog is keyed by its file paths.
    row_id = row.extras.get("id")
    source = row.extras.get("source")
    collection = row.extras.get("collection")
    return _Member(
        id=str(row_id) if _present(row_id) else str(row.filepath),
        source=str(source) if _present(source) else "catalog",
        collection=str(collection) if _present(collection) else "",
        geometry=to_work(row.geometry),
        interval=interval,
    )


def _present(value: Any) -> bool:
    """False for ``None`` and missing scalars (NaN, ``pd.NA``, ``NaT``)."""
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):  # list-likes: present
        return True


def _utc_interval(interval: Any) -> pd.Interval | None:
    """``interval`` with UTC-aware endpoints, or ``None`` if it has no times."""
    if not isinstance(interval, pd.Interval):
        return None  # a missing interval reads back as NaN
    left, right = interval.left, interval.right
    if pd.isna(left) or pd.isna(right):
        return None
    return pd.Interval(to_utc_ts(left), to_utc_ts(right), closed="both")


def _reprojector(src: pyproj.CRS, dst: pyproj.CRS) -> Any:
    """``geometry -> geometry`` from ``src`` to ``dst`` (identity if equal)."""
    import numpy as np
    import pyproj
    import shapely

    if src.equals(dst):
        return lambda geom: geom
    transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)

    def reproject(geom: Any) -> Any:
        return shapely.transform(
            geom, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1]))
        )

    return reproject


def _dedup(members: Iterable[_Member]) -> list[_Member]:
    """Drop repeats of a ``(source, collection, id)``; the first read wins."""
    out: dict[tuple[str, str], _Member] = {}
    for m in members:
        out.setdefault(m.key, m)
    return list(out.values())


# ---------------------------------------------------------------------------
# Temporal selection
# ---------------------------------------------------------------------------


def _select_in_time(
    temporal: TemporalStrategy, primary: _Member, candidates: list[_Member]
) -> list[_Member]:
    """The candidates ``temporal`` keeps, identified by position."""
    if not candidates:
        return []
    intervals = pd.IntervalIndex.from_tuples(
        [(c.interval.left, c.interval.right) for c in candidates], closed="both"
    )
    select = getattr(temporal, "select", None)
    if callable(select):
        return [candidates[i] for i in select(primary.interval, intervals)]
    # A custom strategy with only `filter`: map each interval it
    # returns to the first unused candidate holding that interval.
    unused: dict[tuple[Any, Any], list[int]] = {}
    for i, iv in enumerate(intervals):
        unused.setdefault((iv.left, iv.right), []).append(i)
    positions = []
    for iv in temporal.filter(primary.interval, intervals):
        slot = unused.get((to_utc_ts(iv.left), to_utc_ts(iv.right)))
        if not slot:
            raise ValueError(
                f"{type(temporal).__name__}.filter returned {iv!r}, which is "
                "not one of the candidate intervals"
            )
        positions.append(slot.pop(0))
    return [candidates[i] for i in sorted(positions)]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _matchup_id(
    strategy_key: str, roles: list[str], keys: list[tuple[str, str, str]]
) -> str:
    """Content hash of the strategy parameters and the members."""
    payload = json.dumps(
        [strategy_key, [[r, *k] for r, k in zip(roles, keys, strict=True)]]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _common_intersection(
    geometries: list[shapely.geometry.base.BaseGeometry],
) -> shapely.geometry.base.BaseGeometry:
    """Iterated intersection of the member footprints, primary first."""
    result = geometries[0]
    for geom in geometries[1:]:
        result = result.intersection(geom)
    return result


def _stable_repr(strategy: Any) -> str:
    """``repr`` that is the same in every process.

    A class without its own ``__repr__`` would print its memory address,
    which would make ``matchup_id`` differ between runs; it is described
    by its qualified name and instance attributes instead.
    """
    cls = type(strategy)
    if cls.__repr__ is not object.__repr__:
        return repr(strategy)
    state = ", ".join(
        f"{k}={v!r}" for k, v in sorted(getattr(strategy, "__dict__", {}).items())
    )
    return f"{cls.__module__}.{cls.__qualname__}({state})"


def _summary(strategy: Any) -> str:
    """``ClassName(field=value, ...)``; a stable repr for non-dataclasses."""
    if not dataclasses.is_dataclass(strategy):
        return _stable_repr(strategy)
    fields = ", ".join(
        f"{f.name}={getattr(strategy, f.name)!r}" for f in dataclasses.fields(strategy)
    )
    return f"{type(strategy).__name__}({fields})"


def _params(strategy: Any) -> dict[str, Any]:
    """Strategy parameters as JSON values; durations become seconds."""
    from geocatalog._src.matchup.temporal import (
        NearestInTime,
        Synchronous,
        WithinWindow,
    )

    out: dict[str, Any] = {"type": type(strategy).__name__}
    if not dataclasses.is_dataclass(strategy):
        out["repr"] = _stable_repr(strategy)
        return out
    durations = isinstance(strategy, NearestInTime | WithinWindow | Synchronous)
    for f in dataclasses.fields(strategy):
        value = getattr(strategy, f.name)
        if durations:
            out[f"{f.name}_sec"] = _to_timedelta(value).total_seconds()
        elif isinstance(value, bool | int | float | str) or value is None:
            out[f.name] = value
        else:
            out[f.name] = repr(value)
    return out

"""`MatchedPatch` — the carrier for a co-located patch across N sources.

Sibling carrier to `Patch` rather than a subclass. Two reasons,
captured as ADR-003:

1. `Patch` is parameterized over ``[AnchorT, IndicesT, DataT]`` — a
   single concrete shape per (Geometry x Domain) pairing. A
   `MatchedPatch` cannot satisfy that contract because it holds
   ``dict[str, Patch]`` with heterogeneous data types across keys.
2. Consumers that don't care about matchups continue to type
   against plain `Patch`; consumers that do care explicitly type
   against `MatchedPatch`. No LSP surprises in either direction.

`MatchedPatch.members["primary"]` always carries the patch read from
the primary field (the anchor space); secondaries are keyed by the
names supplied to `MatchedField`.

`MatchedTemporalPatch` and `MatchedSpatioTemporalPatch` are sibling
carriers for the temporal and spatio-temporal axes — same per-source
``members`` mapping, but the inner patches are `TemporalPatch` /
`SpatioTemporalPatch` so the temporal anchor / indices surface
correctly to downstream temporal aggregations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np

    from geopatcher._src.patch import Patch, SpatioTemporalPatch, TemporalPatch


# Convention: the primary patch is always stored under this key in
# ``members``. Public so callers can write
# ``mp.members[MatchedPatch.PRIMARY_KEY]`` instead of stringly typing.
PRIMARY_KEY = "primary"


class _MatchedMembersBase[MemberT]:
    """Shared invariants and accessors for the matched patch carriers.

    Not a dataclass itself — each concrete carrier declares its own
    dataclass fields; this base only contributes the ``__post_init__``
    validation and the ``primary`` / ``secondary_names`` accessors.
    """

    members: dict[str, MemberT]
    valid_mask: dict[str, np.ndarray] | None
    weights: dict[str, np.ndarray] | None

    PRIMARY_KEY = PRIMARY_KEY

    def __post_init__(self) -> None:
        # Enforce the invariant the carrier docstrings promise: every
        # matched patch carries a primary member under `PRIMARY_KEY`.
        # Without this guard, a malformed construction only fails later
        # at `mp.primary` access — much harder to debug.
        cls_name = type(self).__name__
        if PRIMARY_KEY not in self.members:
            raise ValueError(
                f"{cls_name}.members must contain the primary key "
                f"{PRIMARY_KEY!r}; got keys {sorted(self.members)!r}."
            )
        # `valid_mask` / `weights` are per-member auxiliaries; their
        # keys must be a subset of `members`, otherwise a stale mask
        # silently rides along after a member is dropped.
        member_keys = set(self.members)
        for attr_name, attr in (
            ("valid_mask", self.valid_mask),
            ("weights", self.weights),
        ):
            if attr is None:
                continue
            extra = set(attr) - member_keys
            if extra:
                raise ValueError(
                    f"{cls_name}.{attr_name} has keys not present in "
                    f"members: {sorted(extra)!r}."
                )

    @property
    def primary(self) -> MemberT:
        """Convenience accessor for ``members[PRIMARY_KEY]``."""
        return self.members[PRIMARY_KEY]

    @property
    def secondary_names(self) -> tuple[str, ...]:
        """The keys of ``members`` other than the primary."""
        return tuple(k for k in self.members if k != PRIMARY_KEY)


@dataclass(eq=False)
class MatchedPatch(_MatchedMembersBase["Patch"]):
    """A co-located patch read from N sources at a single anchor.

    Args:
        anchor: Where the patch lives in the primary's coordinate
            system. Same shape the primary's `Sampler` emits.
        members: ``{name: Patch}``. ``members["primary"]`` is the
            primary; secondary keys are the names given to
            `MatchedField.secondaries`.
        valid_mask: Optional ``{name: ndarray}`` of per-source masks
            indicating which pixels of each member contain valid
            data (False = nodata / out-of-swath / off-edge). When a
            secondary's coregistration produces partial coverage —
            e.g. the LEO swath only crosses half the GEO patch —
            the mask is the workhorse the downstream operator uses
            to decide what to ignore.
        weights: Optional ``{name: ndarray}`` of per-source window
            weights. Most callers leave this `None` and rely on the
            primary's `Window` axis.
    """

    anchor: Any
    members: dict[str, Patch]
    valid_mask: dict[str, np.ndarray] | None = None
    weights: dict[str, np.ndarray] | None = field(default=None)


@dataclass(eq=False)
class MatchedTemporalPatch(_MatchedMembersBase["TemporalPatch"]):
    """A co-located temporal patch read from N sources at a single anchor.

    Sibling of `MatchedPatch` for the temporal axis. ``members`` holds
    `TemporalPatch` values rather than `Patch` so the temporal
    ``anchor`` (an ``int`` time index) and ``indices`` (a ``slice``
    along the time axis) flow through to downstream
    `TemporalAggregation` consumers.

    Args:
        anchor: The time-index anchor in the primary's time axis.
        members: ``{name: TemporalPatch}``. ``members["primary"]`` is
            the primary; secondary keys are the names given to
            `MatchedField.secondaries`.
        valid_mask: Optional ``{name: ndarray}`` of per-source masks
            indicating which time steps of each member contain valid
            data (False = nodata / gap / off-edge).
        weights: Optional ``{name: ndarray}`` of per-source time
            weights. Most callers leave this `None` and rely on the
            primary's `TemporalWindow` axis.
    """

    anchor: Any
    members: dict[str, TemporalPatch]
    valid_mask: dict[str, np.ndarray] | None = None
    weights: dict[str, np.ndarray] | None = field(default=None)


@dataclass(eq=False)
class MatchedSpatioTemporalPatch(_MatchedMembersBase["SpatioTemporalPatch"]):
    """A co-located spatio-temporal patch read from N sources.

    Sibling of `MatchedPatch` for the spatio-temporal axis. ``members``
    holds `SpatioTemporalPatch` values so both spatial and temporal
    anchors flow through to downstream aggregations.

    Args:
        space: Spatial anchor in the primary's coordinate system.
        time: Time anchor in the primary's time axis.
        members: ``{name: SpatioTemporalPatch}``.
            ``members["primary"]`` is the primary; secondary keys are
            the names given to `MatchedField.secondaries`.
        valid_mask: Optional ``{name: ndarray}`` of per-source masks
            indicating which pixels/time steps of each member contain
            valid data.
        weights: Optional ``{name: ndarray}`` of per-source weights.
    """

    space: Any
    time: Any
    members: dict[str, SpatioTemporalPatch]
    valid_mask: dict[str, np.ndarray] | None = None
    weights: dict[str, np.ndarray] | None = field(default=None)

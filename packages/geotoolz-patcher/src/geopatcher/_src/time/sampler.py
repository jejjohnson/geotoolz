"""`TemporalSampler` — where to place time anchors.

Each sampler yields integer indices along the time axis. Five samplers
cover the common cases:

- `TemporalRegularStride` — uniform stride along the axis.
- `TemporalCausalRolling`     — past-only relative to a reference (defaults to the end).
- `TemporalEventTriggered`    — explicit anchor times (storm tracks, plume detections).
- `TemporalRandom`        — N random anchors (training augmentation).
- `TemporalExplicit`     — caller-supplied indices.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.time.stencils import Stencil, valid_origin_points


class TemporalSampler:
    """Base for temporal anchor placement.

    Subclasses implement `anchors(time_len) -> Iterable[int]`. The base
    signature is integer-only; coordinate-aware subclasses (e.g.
    `TemporalStencilSampler`) set ``needs_coord = True`` and accept a
    ``coord=`` keyword in `anchors`. `TemporalPatcher` passes the coord
    vector through when the flag is `True`. See ADR-004 in
    ``docs/decisions.md``.
    """

    forbid_in_yaml: ClassVar[bool] = False
    needs_coord: ClassVar[bool] = False

    def anchors(self, time_len: int) -> Iterable[int]:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class TemporalRegularStride(TemporalSampler):
    """Regular stride along the time axis."""

    step: int = 1

    def anchors(self, time_len: int) -> Iterator[int]:
        yield from range(0, int(time_len), int(self.step))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalCausalRolling(TemporalSampler):
    """Past-only rolling window, walking forward from index ``start``.

    Useful for training-time data loaders that only present the model
    with information available at each anchor.
    """

    step: int = 1
    start: int = 0

    def anchors(self, time_len: int) -> Iterator[int]:
        yield from range(int(self.start), int(time_len), int(self.step))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalEventTriggered(TemporalSampler):
    """TemporalExplicit event times — anchors come from a fixed list of indices.

    Useful for storm tracks, plume detections, and other event-aligned
    patching (the natural partner of `coupled` coupling in
    `SpatioTemporalPatcher`).

    Args:
        event_times: Iterable of integer time indices. Coerced to a list at
            construction so `get_config()` doesn't accidentally exhaust a
            user-supplied generator.
    """

    event_times: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Materialise eagerly — guards against passing in a one-shot iterator
        # whose contents would otherwise be consumed by `get_config()`.
        self.event_times = [int(t) for t in self.event_times]

    def anchors(self, time_len: int) -> Iterator[int]:
        for t in self.event_times:
            if 0 <= t < int(time_len):
                yield t

    def get_config(self) -> dict[str, Any]:
        return {"n_events": len(self.event_times)}


@dataclass(eq=False)
class TemporalRandom(TemporalSampler):
    """``n`` uniform-random anchors along the time axis.

    Args:
        n: Number of anchors to draw.
        seed: Integer seed for reproducible draws — same contract as
            `SpatialRandom.seed`. ``None`` re-seeds each call.
    """

    n: int = 1
    seed: int | None = None

    def anchors(self, time_len: int) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, int(time_len), size=int(self.n))
        for t in idx:
            yield int(t)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalStencilSampler(TemporalSampler):
    """Anchor sampler whose valid set is defined by a `Stencil`.

    Yields integer indices into the ``coord`` vector — same return type as
    every other `TemporalSampler`. The novelty is *which* integers count:
    only those whose stencil, when resolved against ``coord``, fits entirely
    in-record (no truncation at either end).

    Coordinate plumbing follows the `needs_coord = True` contract:
    `TemporalPatcher` forwards its ``coord=`` kwarg to `anchors`.

    Args:
        stencil: The `Stencil` (or `TimeStencil`) used to compute the
            valid-origin set. Typically the same stencil as the paired
            `TemporalStencilGeometry`.
        every: Thin the valid-origin set in *valid-anchor* space. ``every=2``
            keeps every other valid origin. Distinct from ``stencil.step``,
            which is the within-window cadence.
        shuffle: If true, shuffle the kept anchors before emitting. Useful
            for training-time data loaders.
        seed: Reproducibility seed for ``shuffle``. Same contract as
            `TemporalRandom.seed`.
    """

    stencil: Stencil
    every: int = 1
    shuffle: bool = False
    seed: int | None = None
    needs_coord: ClassVar[bool] = True

    def anchors(
        self,
        time_len: int,
        coord: np.ndarray | None = None,
    ) -> Iterator[int]:
        if coord is None:
            raise ValueError(
                "TemporalStencilSampler requires coord=; supply it via "
                "TemporalPatcher.split(..., coord=time_coord)."
            )
        if int(time_len) != coord.shape[0]:
            raise ValueError(
                "coord length must equal time_len: "
                f"got coord.shape={coord.shape} vs time_len={time_len}."
            )
        valid = valid_origin_points(coord, self.stencil)
        # coord is monotonic-ascending (precondition of build_sampling_slices),
        # so searchsorted is O(log n) per origin.
        idx = np.searchsorted(coord, valid)[:: self.every]
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(idx)
        for i in idx:
            yield int(i)

    def get_config(self) -> dict[str, Any]:
        return {
            "stencil": self.stencil.get_config(),
            **config_from_fields(self, exclude=("stencil",)),
        }


@dataclass(eq=False)
class TemporalExplicit(TemporalSampler):
    """Caller-supplied anchor indices — the universal escape hatch.

    Args:
        times: Iterable of integer time indices. Coerced to a list at
            construction so `get_config()` doesn't accidentally exhaust a
            user-supplied generator.
    """

    times: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.times = [int(t) for t in self.times]

    def anchors(self, time_len: int) -> Iterator[int]:
        for t in self.times:
            if 0 <= t < int(time_len):
                yield t

    def get_config(self) -> dict[str, Any]:
        return {"n_times": len(self.times)}

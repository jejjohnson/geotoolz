"""`TemporalSampler` — where to place time anchors.

Each sampler yields integer indices along the time axis:

- `TemporalRegularStride`  — every ``step`` steps from ``start``.
  `TemporalCausalRolling` is the same class (a rolling walk forward
  from ``start``; causality comes from the geometry, e.g. a lookback).
- `TemporalExplicit`       — caller-supplied indices (in-range ones, in
  the given order). `TemporalEventTriggered` is the same class, named
  for event-aligned patching (storm tracks, plume detections).
- `TemporalRandom`         — ``n_samples`` uniform-random anchors.
- `TemporalStencilSampler` — every origin whose `Stencil` fits in-record.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import axis_envelope, config_from_fields
from geopatcher._src.time.stencils import Stencil, coord_step, valid_origin_points


class TemporalSampler:
    """Base for temporal anchor placement.

    Subclasses implement `anchors(time_len) -> Iterable[int]`. The base
    signature is integer-only; coordinate-aware subclasses (e.g.
    `TemporalStencilSampler`) set ``needs_coord = True`` and accept a
    ``coord=`` keyword in `anchors`. `TemporalPatcher` passes the coord
    vector through when the flag is `True`. See ADR-004 in
    ``docs/patcher/decisions.md``.
    """

    forbid_in_yaml: ClassVar[bool] = False
    needs_coord: ClassVar[bool] = False

    def anchors(self, time_len: int) -> Iterable[int]:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class TemporalRegularStride(TemporalSampler):
    """Every ``step`` steps along the time axis, starting at ``start``.

    Args:
        step: Stride between anchors (``>= 1``).
        start: First anchor index (``>= 0``). With a causal lookback of
            ``L`` steps, ``start=L - 1`` is the first anchor whose window
            fits under ``boundary="drop"``.
        check_full_scan: If ``True``, `TemporalPatcher` raises
            `IncompleteScanConfiguration` before the first patch when the
            windows of the placed anchors leave any time step uncovered
            (a stride longer than the window, a dropped head / tail) —
            the temporal twin of `SpatialRegularStride.check_full_scan`.
    """

    step: int = 1
    start: int = 0
    check_full_scan: bool = False

    def __post_init__(self) -> None:
        if int(self.step) != self.step or self.step < 1:
            raise ValueError(f"step must be an integer >= 1, got {self.step!r}")
        if int(self.start) != self.start or self.start < 0:
            raise ValueError(f"start must be an integer >= 0, got {self.start!r}")

    def anchors(self, time_len: int) -> Iterator[int]:
        yield from range(int(self.start), int(time_len), int(self.step))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


TemporalCausalRolling = TemporalRegularStride
"""Alias of `TemporalRegularStride` — a forward walk from ``start``."""


@dataclass(eq=False)
class TemporalRandom(TemporalSampler):
    """``n_samples`` uniform-random anchors along the time axis.

    Draws are independent — *with replacement*, so an index can repeat —
    and are yielded in draw order, not sorted, exactly like
    `SpatialRandom`.

    Args:
        n_samples: Number of anchors to draw (``>= 0``).
        seed: Integer seed for reproducible draws — same contract as
            `SpatialRandom.seed`. ``None`` re-seeds each call.
    """

    n_samples: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_samples) != self.n_samples or self.n_samples < 0:
            raise ValueError(
                f"n_samples must be a non-negative integer, got {self.n_samples!r}"
            )

    def anchors(self, time_len: int) -> Iterator[int]:
        if int(time_len) <= 0:
            return
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, int(time_len), size=int(self.n_samples))
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
        every: Thin the valid-origin set in *valid-anchor* space
            (``>= 1``). ``every=2`` keeps every other valid origin.
            Distinct from ``stencil.step``, which is the within-window
            cadence.
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

    def __post_init__(self) -> None:
        if int(self.every) != self.every or self.every < 1:
            raise ValueError(f"every must be an integer >= 1, got {self.every!r}")

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
        coord = np.asarray(coord)
        coord_step(coord)  # 1-D, strictly increasing, evenly spaced
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
            "stencil": axis_envelope(self.stencil),
            **config_from_fields(self, exclude=("stencil",)),
        }


@dataclass(eq=False)
class TemporalExplicit(TemporalSampler):
    """Caller-supplied anchor indices — the universal escape hatch.

    Yields the in-range indices (``0 <= t < time_len``) in the given
    order; out-of-range ones are skipped. Also exported as
    `TemporalEventTriggered` for event-aligned patching (storm tracks,
    plume detections — the natural partner of ``coupling="coupled"`` in
    `SpatioTemporalPatcher`).

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
        return config_from_fields(self)


TemporalEventTriggered = TemporalExplicit
"""Alias of `TemporalExplicit` — event-aligned anchor times."""

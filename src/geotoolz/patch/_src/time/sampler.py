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


class TemporalSampler:
    """Base for temporal anchor placement.

    Subclasses implement `anchors(time_len) -> Iterable[int]`.
    """

    forbid_in_yaml: ClassVar[bool] = False

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
        return {"step": self.step}


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
        return {"step": self.step, "start": self.start}


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
        seed: Optional integer seed for reproducible draws.
    """

    n: int = 1
    seed: int | None = None

    def anchors(self, time_len: int) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, int(time_len), size=int(self.n))
        for t in idx:
            yield int(t)

    def get_config(self) -> dict[str, Any]:
        return {"n": self.n, "seed": self.seed}


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

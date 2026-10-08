"""`temporal.window.Window` — boundary treatment for the time window.

Four windows: `temporal.window.CausalBoxcar` (no taper, hard past cutoff),
`temporal.window.ExponentialDecay` (recency weighting), `temporal.window.TaperedTukey`
(cosine fade-in of the oldest steps), `temporal.window.Periodic` (boxcar weights
tagged with a cycle length for diurnal / annual configs).

Every window gives the newest step (the anchor end of a causal window)
weight 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.temporal.geometry import (
    Geometry as TemporalGeometry,
)


class Window:
    """Base for temporal window functions."""

    forbid_in_yaml: ClassVar[bool] = False

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class CausalBoxcar(Window):
    """Constant 1.0 — no recency weighting, hard past cutoff at the lookback."""

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        return np.ones(int(length), dtype=np.float64)


@dataclass(eq=False)
class ExponentialDecay(Window):
    """Geometric recency weighting — ``w[k] = exp(-k / tau)`` for past-to-present.

    The latest step (largest index) gets weight 1.0; earlier steps decay.

    Args:
        tau: Decay constant in time-axis steps (``> 0``).
    """

    tau: float

    def __post_init__(self) -> None:
        if not self.tau > 0:
            raise ValueError(f"tau must be > 0, got {self.tau!r}")

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.array([], dtype=np.float64)
        # ages: n-1 at the oldest step, 0 at the most recent
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)
        return np.exp(-ages / float(self.tau))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TaperedTukey(Window):
    """One-sided (causal) Tukey taper — the oldest steps fade in.

    The first ``alpha`` fraction of the window, counted from the oldest
    step, rises along a raised cosine; the rest — always including the
    newest step — is 1.0. With ``n`` steps, step ``k`` (oldest ``k = 0``)
    sits at ``x = (k + 1) / n`` and weighs
    ``0.5 * (1 - cos(pi * x / alpha))`` for ``x < alpha``, else 1.0. A
    symmetric taper would zero the newest observation, which is the one a
    causal window exists to keep.

    Args:
        alpha: Taper fraction in ``[0, 1]`` (0 = boxcar, 1 = a raised
            cosine over the whole window).
    """

    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha!r}")

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.array([], dtype=np.float64)
        alpha = float(self.alpha)
        x = np.arange(1, n + 1, dtype=np.float64) / n
        if alpha == 0.0:
            return np.ones(n, dtype=np.float64)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.minimum(x / alpha, 1.0)))
        return np.where(x < alpha, ramp, 1.0)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Periodic(CausalBoxcar):
    """`temporal.window.CausalBoxcar` weights tagged with a cycle length.

    The weights *are* the boxcar's (it inherits them); the ``period`` only
    records the cycle the configuration is built around — pair it with a
    `temporal.geometry.PhaseWindow` of the same period — so YAML configs keep the
    intent.

    Args:
        period: Cycle length in time-axis steps (``>= 1``).
    """

    period: int

    def __post_init__(self) -> None:
        if int(self.period) < 1:
            raise ValueError(f"period must be >= 1, got {self.period!r}")

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)

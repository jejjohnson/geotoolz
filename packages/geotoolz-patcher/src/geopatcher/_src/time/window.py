"""`TemporalWindow` — boundary treatment for the time window.

Four windows: `TemporalCausalBoxcar` (no taper, hard past cutoff),
`TemporalExponentialDecay` (recency weighting), `TemporalTaperedTukey` (spectral
leakage control), `TemporalPeriodic` (cyclic boundary for diurnal / annual).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields
from geopatcher._src.time.geometry import TemporalGeometry


class TemporalWindow:
    """Base for temporal window functions."""

    forbid_in_yaml: ClassVar[bool] = False

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class TemporalCausalBoxcar(TemporalWindow):
    """Constant 1.0 — no recency weighting, hard past cutoff at the lookback."""

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        return np.ones(int(length), dtype=np.float64)


@dataclass(eq=False)
class TemporalExponentialDecay(TemporalWindow):
    """Geometric recency weighting — ``w[k] = exp(-k / tau)`` for past-to-present.

    The latest step (largest index) gets weight 1.0; earlier steps decay.

    Args:
        tau: Decay constant in time-axis steps.
    """

    tau: float

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.array([], dtype=np.float64)
        # ages: n-1 at the oldest step, 0 at the most recent
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)
        return np.exp(-ages / max(self.tau, 1e-12))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalTaperedTukey(TemporalWindow):
    """Tukey-tapered temporal window.

    Args:
        alpha: Taper fraction (0 = Boxcar, 1 = Hann).
    """

    alpha: float = 0.5

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        from scipy.signal.windows import tukey

        n = int(length)
        if n <= 0:
            return np.array([], dtype=np.float64)
        return tukey(n, alpha=self.alpha, sym=False).astype(np.float64)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalPeriodic(TemporalWindow):
    """Boxcar weights with a documented periodic boundary semantic.

    Behaviourally identical to `TemporalCausalBoxcar` at the weight level — the
    "periodic" part is structural (the geometry / aggregation must
    interpret out-of-range steps as wrapping). Carried as a separate
    class so YAML configs preserve the intent.

    Args:
        period: Cycle length in time-axis steps.
    """

    period: int

    def weights(self, geometry: TemporalGeometry, length: int) -> np.ndarray:
        return np.ones(int(length), dtype=np.float64)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)

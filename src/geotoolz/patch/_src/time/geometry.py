"""`TemporalGeometry` — shape of the temporal window around an anchor.

Time is treated as a 1-D axis indexed by integer offsets. The Patcher
asks the geometry "given anchor ``t``, what range of the axis should I
read?" — the answer is a half-open ``(start, stop)`` slice.

Four geometries:

- `TemporalFixedLookback`     — ``(t - length, t]``                  (causal lookback)
- `TemporalLookbackHorizon`   — ``(t - lookback, t + horizon]``      (forecasting)
- `TemporalMultiScale`        — list of nested lookbacks (different scales at once)
- `TemporalPhaseWindow`       — periodic / diurnal slot of width ``phase_width``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


class TemporalGeometry:
    """Base for time-window shapes.

    Subclasses implement ``window(time_len, anchor) -> slice | list[slice]``.
    A list is returned by multi-scale geometries.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def window(self, time_len: int, anchor: int) -> slice | list[slice]:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class TemporalFixedLookback(TemporalGeometry):
    """Causal lookback of fixed ``length`` time steps.

    The returned slice is ``[t - length + 1, t + 1)`` — i.e. ``length``
    steps ending at ``t`` inclusive.
    """

    length: int

    def window(self, time_len: int, anchor: int) -> slice:
        end = int(anchor) + 1
        start = max(0, end - int(self.length))
        return slice(start, end)

    def get_config(self) -> dict[str, Any]:
        return {"length": self.length}


@dataclass(eq=False)
class TemporalLookbackHorizon(TemporalGeometry):
    """Lookback + horizon — the canonical forecasting window shape.

    Returns ``[t - lookback + 1, t + horizon + 1)``. Operators consuming
    this typically split the window into past (``lookback``) and future
    (``horizon``) at index ``lookback``.
    """

    lookback: int
    horizon: int

    def window(self, time_len: int, anchor: int) -> slice:
        start = max(0, int(anchor) - int(self.lookback) + 1)
        end = min(int(time_len), int(anchor) + int(self.horizon) + 1)
        return slice(start, end)

    def get_config(self) -> dict[str, Any]:
        return {"lookback": self.lookback, "horizon": self.horizon}


@dataclass(eq=False)
class TemporalMultiScale(TemporalGeometry):
    """One lookback per scale — for combining hourly + daily + annual context.

    Args:
        scales: List of lookback lengths (in time-axis steps).
    """

    scales: list[int]

    def window(self, time_len: int, anchor: int) -> list[slice]:
        out: list[slice] = []
        for length in self.scales:
            end = int(anchor) + 1
            start = max(0, end - int(length))
            out.append(slice(start, end))
        return out

    def get_config(self) -> dict[str, Any]:
        return {"scales": list(self.scales)}


@dataclass(eq=False)
class TemporalPhaseWindow(TemporalGeometry):
    """Periodic phase slot - all steps within ``phase_width`` of the same phase.

    Args:
        period: Cycle length in time-axis steps (e.g. 24 for hourly diurnal).
        phase_width: Half-width of the slot in steps.
    """

    period: int
    phase_width: int

    def window(self, time_len: int, anchor: int) -> slice:
        # Concrete behaviour for v0.1: return the local ± phase_width
        # window around the anchor. Multi-phase aggregation is a job for
        # `TemporalHierarchicalCombine`.
        start = max(0, int(anchor) - int(self.phase_width))
        end = min(int(time_len), int(anchor) + int(self.phase_width) + 1)
        return slice(start, end)

    def get_config(self) -> dict[str, Any]:
        return {"period": self.period, "phase_width": self.phase_width}

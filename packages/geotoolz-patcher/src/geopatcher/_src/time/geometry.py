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

import numpy as np

from geopatcher._src._serialize import config_from_fields, jsonable_scalar
from geopatcher._src.time.stencils import (
    Stencil,
    build_sampling_slices,
    divide_evenly,
)


class TemporalGeometry:
    """Base for time-window shapes.

    Subclasses implement ``window(time_len, anchor) -> slice | list[slice]``.
    A list is returned by multi-scale geometries.

    Coordinate-aware subclasses (e.g. `TemporalStencilGeometry`) set
    ``needs_coord = True`` and implement
    ``window_coord(coord, anchor_idx) -> slice``. `TemporalPatcher` dispatches
    on the flag and requires a ``coord=`` argument when it is `True`. See
    ADR-004 in ``docs/decisions.md``.
    """

    forbid_in_yaml: ClassVar[bool] = False
    needs_coord: ClassVar[bool] = False

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
        return config_from_fields(self)


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
        return config_from_fields(self)


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
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalStencilGeometry(TemporalGeometry):
    """Coordinate-aware geometry that resolves a `Stencil` against a coord.

    Unlike the integer geometries above, the window is resolved in *coordinate*
    space via `window_coord(coord, anchor_idx)`. `TemporalPatcher` dispatches
    on the `needs_coord` flag and supplies the coord vector through the
    `split(..., coord=)` argument. Calling the integer `window` is a TypeError
    so mis-wiring fails loudly.

    v0.1 supports stride-1 stencils only (the `TemporalWindow.weights` and
    `TemporalAggregation.merge` contracts assume contiguous index ranges).
    Pass ``source_step`` at construction to catch stride > 1 up front; the
    constructor also re-checks at `window_coord` time as a belt-and-braces
    guard for callers that didn't supply it. See ADR-004 in
    ``docs/decisions.md``.

    Args:
        stencil: The `Stencil` (or `TimeStencil`) describing the window shape
            in coordinate units.
        source_step: Optional cadence of the source grid (same units as
            ``stencil.step``). If provided, the constructor raises immediately
            on stride > 1 instead of waiting for `window_coord`.
    """

    stencil: Stencil
    source_step: Any = None
    needs_coord: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.source_step is not None:
            sigma = int(
                divide_evenly(
                    self.stencil.step,
                    self.source_step,
                    label="stencil step / source step",
                ).item()
            )
            if sigma != 1:
                raise ValueError(
                    "v0.1 supports stride-1 stencils only; got "
                    f"stride={sigma}. Use a stencil step equal to the source "
                    "cadence, or wait for v0.2."
                )

    def window_coord(self, coord: np.ndarray, anchor_idx: int) -> slice:
        """Resolve the stencil at the given anchor index → contiguous slice.

        Args:
            coord: 1-D monotonic-ascending coordinate array along the time
                axis (e.g. ``ds["time"].values``).
            anchor_idx: Integer index into ``coord`` marking the origin.

        Returns:
            ``slice(start, stop)`` covering the realised stencil window in
            integer index space.
        """
        origin = coord[int(anchor_idx)]
        (s,) = build_sampling_slices(coord, np.asarray([origin]), self.stencil)
        if s.step is not None and s.step != 1:
            raise ValueError(
                f"v0.1 supports stride-1 stencils only; got stride={s.step}."
            )
        return slice(s.start, s.stop)

    def window(self, time_len: int, anchor: int) -> slice | list[slice]:
        raise TypeError(
            "TemporalStencilGeometry is coordinate-aware; call via "
            "TemporalPatcher.split(..., coord=time_coord) which dispatches to "
            "window_coord. Direct integer window() is not defined."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "stencil": self.stencil.get_config(),
            "source_step": jsonable_scalar(self.source_step),
        }


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
        return config_from_fields(self)

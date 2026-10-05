"""`TemporalGeometry` — shape of the temporal window around an anchor.

Time is treated as a 1-D axis indexed by integer offsets. The Patcher
asks the geometry "given anchor ``t``, what range of the axis should I
read?" — the answer is a half-open ``(start, stop)`` slice, a list of
them (multi-window geometries), or ``None`` when the anchor's window is
dropped by the geometry's ``boundary`` policy.

Four integer geometries:

- `TemporalFixedLookback`     — ``(t - length, t]``                  (causal lookback)
- `TemporalLookbackHorizon`   — ``(t - lookback, t + horizon]``      (forecasting)
- `TemporalMultiScale`        — list of nested lookbacks (different scales at once)
- `TemporalPhaseWindow`       — every ``period``-cycle slot of half-width
  ``phase_width`` centred on the anchor's phase

Each takes ``boundary`` — ``"drop"`` (default), ``"shrink"`` or ``"raise"``
— deciding what happens to a window that overflows the axis. The spatial
geometries' ``"pad"`` / ``"reflect"`` modes have no temporal counterpart.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from geopatcher._src._serialize import axis_envelope, config_from_fields
from geopatcher._src.time.stencils import (
    Stencil,
    _to_timedelta64,
    coord_step,
    delta_config,
    divide_evenly,
    stencil_offsets,
)


TemporalBoundary = Literal["drop", "shrink", "raise"]
"""How an integer temporal geometry treats a window overflowing the axis.

- ``"drop"`` (default): the anchor yields no patch — every emitted window
  is full length, matching the spatial ``boundary="drop"`` default.
- ``"shrink"``: the window is clipped to ``[0, time_len)``, so edge
  windows are shorter (the implicit behaviour before ``boundary`` existed).
- ``"raise"``: a `ValueError` naming the anchor and the window.
"""

_VALID_TEMPORAL_BOUNDARIES = ("drop", "shrink", "raise")


def _check_boundary(boundary: str) -> None:
    if boundary not in _VALID_TEMPORAL_BOUNDARIES:
        raise ValueError(
            f"invalid temporal boundary {boundary!r}; expected one of "
            f"{_VALID_TEMPORAL_BOUNDARIES}"
        )


def _resolve(
    geometry: TemporalGeometry,
    anchor: int,
    start: int,
    stop: int,
    time_len: int,
    boundary: str,
) -> slice | None:
    """Apply ``boundary`` to the half-open window ``[start, stop)``."""
    if start >= 0 and stop <= time_len:
        return slice(start, stop)
    if boundary == "raise":
        raise ValueError(
            f"{type(geometry).__name__} window [{start}, {stop}) for anchor "
            f"{anchor} overflows the time axis [0, {time_len}) under "
            "boundary='raise'; use boundary='drop' or 'shrink'."
        )
    if boundary == "drop":
        return None
    lo, hi = max(0, start), min(time_len, stop)
    return slice(lo, hi) if hi > lo else None


class TemporalGeometry:
    """Base for time-window shapes.

    Subclasses implement
    ``window(time_len, anchor) -> slice | list[slice] | None``. A list is
    returned by multi-window geometries (`TemporalMultiScale`,
    `TemporalPhaseWindow`); ``None`` means the anchor yields no patch
    (its window was dropped under ``boundary="drop"``).

    Coordinate-aware subclasses (e.g. `TemporalStencilGeometry`) set
    ``needs_coord = True`` and implement
    ``window_coord(coord, anchor_idx) -> slice``. `TemporalPatcher` dispatches
    on the flag and requires a ``coord=`` argument when it is `True`. See
    ADR-004 in ``docs/patcher/decisions.md``.
    """

    forbid_in_yaml: ClassVar[bool] = False
    needs_coord: ClassVar[bool] = False

    def window(self, time_len: int, anchor: int) -> slice | list[slice] | None:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


@dataclass(eq=False)
class TemporalFixedLookback(TemporalGeometry):
    """Causal lookback of fixed ``length`` time steps.

    The returned slice is ``[t - length + 1, t + 1)`` — i.e. ``length``
    steps ending at ``t`` inclusive.

    Args:
        length: Number of steps in the lookback (``>= 1``).
        boundary: What to do with an anchor whose lookback starts before
            the axis (``t < length - 1``): ``"drop"`` it (default),
            ``"shrink"`` the window to ``[0, t + 1)``, or ``"raise"``.
    """

    length: int
    boundary: TemporalBoundary = "drop"

    def __post_init__(self) -> None:
        _check_boundary(self.boundary)
        if int(self.length) < 1:
            raise ValueError(f"length must be >= 1; got {self.length}")

    def window(self, time_len: int, anchor: int) -> slice | None:
        end = int(anchor) + 1
        start = end - int(self.length)
        return _resolve(self, int(anchor), start, end, int(time_len), self.boundary)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalLookbackHorizon(TemporalGeometry):
    """Lookback + horizon — the canonical forecasting window shape.

    Returns ``[t - lookback + 1, t + horizon + 1)``. Operators consuming
    this typically split the window into past (``lookback``) and future
    (``horizon``) at index ``lookback``.

    Args:
        lookback: Steps up to and including the anchor (``>= 1``).
        horizon: Steps after the anchor (``>= 0``).
        boundary: What to do with an anchor whose window overflows either
            end of the axis: ``"drop"`` it (default — so the horizon of
            every emitted window is real data), ``"shrink"`` the window to
            the axis, or ``"raise"``.
    """

    lookback: int
    horizon: int
    boundary: TemporalBoundary = "drop"

    def __post_init__(self) -> None:
        _check_boundary(self.boundary)
        if int(self.lookback) < 1:
            raise ValueError(f"lookback must be >= 1; got {self.lookback}")
        if int(self.horizon) < 0:
            raise ValueError(f"horizon must be >= 0; got {self.horizon}")

    def window(self, time_len: int, anchor: int) -> slice | None:
        a = int(anchor)
        start = a - int(self.lookback) + 1
        stop = a + int(self.horizon) + 1
        return _resolve(self, a, start, stop, int(time_len), self.boundary)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class TemporalMultiScale(TemporalGeometry):
    """One lookback per scale — for combining hourly + daily + annual context.

    Emits one window per scale, in ``scales`` order, so the ``k``-th
    patch of an anchor (`TemporalPatch.window_index` ``k``) is scale
    ``scales[k]``.

    Args:
        scales: List of lookback lengths (in time-axis steps, each ``>= 1``).
        boundary: What to do with an anchor whose *longest* lookback starts
            before the axis: ``"drop"`` the whole anchor (default — every
            emitted anchor carries every scale at full length), ``"shrink"``
            each overflowing scale to ``[0, t + 1)``, or ``"raise"``.
    """

    scales: list[int]
    boundary: TemporalBoundary = "drop"

    def __post_init__(self) -> None:
        _check_boundary(self.boundary)
        self.scales = [int(s) for s in self.scales]
        if not self.scales or min(self.scales) < 1:
            raise ValueError(
                f"scales must be a non-empty list of lengths >= 1; got {self.scales}"
            )

    def window(self, time_len: int, anchor: int) -> list[slice] | None:
        end = int(anchor) + 1
        out: list[slice] = []
        for length in self.scales:
            s = _resolve(
                self, int(anchor), end - length, end, int(time_len), self.boundary
            )
            if s is None:
                # "drop": one overflowing scale drops the anchor, so the
                # surviving anchors always carry the full scale set.
                return None
            out.append(s)
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
    ``docs/patcher/decisions.md``.

    The coordinate is validated (1-D, strictly increasing, evenly spaced)
    once per coordinate array, not per anchor: windows are then resolved
    arithmetically from the source step (`stencil_offsets`), so a split
    over a 175 320-step (20-year hourly) axis costs O(N), not O(N²).

    Args:
        stencil: The `Stencil` (or `TimeStencil`) describing the window shape
            in coordinate units.
        source_step: Optional cadence of the source grid (same units as
            ``stencil.step``; a string such as ``"1h"`` is parsed like a
            `TimeStencil` field, as is the ``{"value", "unit"}`` mapping
            `get_config` emits). If provided, the constructor raises
            immediately on stride > 1 or a unit mismatch instead of waiting
            for `window_coord`.
        boundary: What to do with an origin whose stencil overflows the
            coordinate: ``"drop"`` it (default — `TemporalStencilSampler`
            only places origins that fit anyway), ``"shrink"`` the window
            to the axis, or ``"raise"``.
    """

    stencil: Stencil
    source_step: Any = None
    boundary: TemporalBoundary = "drop"
    needs_coord: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _check_boundary(self.boundary)
        # Validated-coord memo: (weakref to the coord array, the stencil's
        # (lo, hi, stride) index offsets on its step).
        self._coord_memo: tuple[Any, tuple[int, int, int]] | None = None
        if isinstance(self.source_step, (str, Mapping)):
            self.source_step = _to_timedelta64(self.source_step)
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
            # Also rejects a source_step that does not divide the bounds.
            stencil_offsets(self.stencil, self.source_step)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_coord_memo"] = None  # weakrefs do not pickle
        return state

    def _offsets_for(self, coord: np.ndarray) -> tuple[int, int, int]:
        """Stencil offsets on ``coord``, validating each coord array only once."""
        memo = self._coord_memo
        if memo is not None and memo[0]() is coord:
            return memo[1]
        offsets = stencil_offsets(self.stencil, coord_step(coord))
        # A non-weak-referenceable coord is simply re-validated every call.
        with contextlib.suppress(TypeError):
            self._coord_memo = (weakref.ref(coord), offsets)
        return offsets

    def window_coord(self, coord: np.ndarray, anchor_idx: int) -> slice | None:
        """Resolve the stencil at the given anchor index → contiguous slice.

        Args:
            coord: 1-D, strictly increasing, evenly spaced coordinate array
                along the time axis (e.g. ``ds["time"].values``).
            anchor_idx: Integer index into ``coord`` marking the origin.

        Returns:
            ``slice(start, stop)`` covering the realised stencil window in
            integer index space, or ``None`` when ``boundary="drop"`` drops
            an overflowing origin.

        Raises:
            ValueError: For an invalid coordinate, a stride > 1, or (under
                ``boundary="raise"``) an overflowing origin.
        """
        coord = np.asarray(coord)
        lo, hi, stride = self._offsets_for(coord)
        if stride != 1:
            raise ValueError(
                f"v0.1 supports stride-1 stencils only; got stride={stride}."
            )
        i = int(anchor_idx)
        return _resolve(self, i, i + lo, i + hi, int(coord.shape[0]), self.boundary)

    def window(self, time_len: int, anchor: int) -> slice | list[slice] | None:
        raise TypeError(
            "TemporalStencilGeometry is coordinate-aware; call via "
            "TemporalPatcher.split(..., coord=time_coord) which dispatches to "
            "window_coord. Direct integer window() is not defined."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "stencil": axis_envelope(self.stencil),
            "source_step": delta_config(self.source_step),
            "boundary": self.boundary,
        }


@dataclass(eq=False)
class TemporalPhaseWindow(TemporalGeometry):
    """Periodic phase slots — every cycle's steps near the anchor's phase.

    For anchor ``t`` with ``phase = t % period`` the geometry returns one
    slot per cycle,
    ``[k * period + phase - phase_width, k * period + phase + phase_width + 1)`` for
    every ``k`` whose slot touches the axis, in time order. E.g. with
    hourly data, ``period=24, phase_width=1`` at a 14:00 anchor yields the
    13:00-15:00 slot of every day — the diurnal-composite shape.

    Args:
        period: Cycle length in time-axis steps (e.g. 24 for hourly diurnal).
        phase_width: Half-width of each slot in steps; slots must not
            overlap, so ``2 * phase_width + 1 <= period``.
        boundary: What to do with a slot that overflows the axis (only the
            first / last cycle can): ``"drop"`` that slot (default — an
            anchor whose every slot is dropped yields no patch),
            ``"shrink"`` it to the axis, or ``"raise"``.
    """

    period: int
    phase_width: int
    boundary: TemporalBoundary = "drop"

    def __post_init__(self) -> None:
        _check_boundary(self.boundary)
        period, w = int(self.period), int(self.phase_width)
        if period < 1:
            raise ValueError(f"period must be >= 1; got {self.period}")
        if w < 0 or 2 * w + 1 > period:
            raise ValueError(
                "phase_width must satisfy 0 <= 2 * phase_width + 1 <= period; "
                f"got phase_width={self.phase_width}, period={self.period}"
            )

    def window(self, time_len: int, anchor: int) -> list[slice] | None:
        period, w, n = int(self.period), int(self.phase_width), int(time_len)
        phase = int(anchor) % period
        out: list[slice] = []
        # Start one cycle early: for a phase within ``w`` of the cycle end
        # the slot centred at ``phase - period`` still reaches into the axis.
        for centre in range(phase - period, n + w, period):
            start, stop = centre - w, centre + w + 1
            if stop <= 0 or start >= n:
                continue
            s = _resolve(self, int(anchor), start, stop, n, self.boundary)
            if s is not None:
                out.append(s)
        return out or None

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright 2026 J. Emmanuel Johnson, distributed under the
# MIT license that covers the rest of geopatcher. Modifications add
# `get_config` for YAML round-trip, an LCD-naming error on uneven division,
# and the `Closed` re-export.
"""Coordinate-aware stencils for sampling 1-D grids in physical units.

Ported from `neuralgcm/terrax`'s `terrax.xreader.stencils`. See ADR-004 in
``docs/decisions.md`` for the design and the v0.1 stride-1 constraint that
sits *outside* this module (it is enforced by `TemporalStencilGeometry`,
not by `build_sampling_slices` itself).

The public surface is four pure functions / dataclasses:

- `Stencil` / `TimeStencil` — the stencil (start, stop, step, closed).
- `divide_evenly` — exact-quotient check; raises with both operands named.
- `build_sampling_slices` — coordinates → `list[slice]`.
- `valid_origin_points` — the trimmed origin set so every window fits.

Reading is not this module's job — these slices feed
`xarray.isel` / `numpy.__getitem__` / `geopatcher.Field.select`.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Literal

import numpy as np

from geopatcher._src._serialize import config_from_fields


Closed = Literal["left", "right", "both", "neither"]

_INCLUDE_START = {"left", "both"}
_INCLUDE_STOP = {"right", "both"}


def divide_evenly(
    x: np.typing.ArrayLike,
    y: np.typing.ArrayLike,
    *,
    label: str = "value",
) -> np.ndarray:
    """Compute ``round(x / y)`` and verify the result is exact.

    For `timedelta64`/`datetime64` operands the equality is exact (NumPy
    promotes both sides to a common resolution before the integer
    comparison). For float operands we allow ``1e-6`` slack.

    Args:
        x: Numerator. May be scalar or array-like.
        y: Denominator. Scalar or broadcastable to ``x``.
        label: Short noun describing what ``y`` represents — used in the
            error message so callers see e.g. ``"step"`` rather than just
            the raw value. Defaults to ``"value"``.

    Returns:
        The integer quotient as ``np.ndarray`` of dtype ``int``.

    Raises:
        ValueError: If ``y`` does not evenly divide ``x``. The message names
            both operands so unit mismatches are obvious from the traceback.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    q = np.around(x / y).astype(int)
    if np.issubdtype(x.dtype, np.timedelta64):
        uneven = q * y != x
    else:
        epsilon = 1e-6
        uneven = abs(q * y - x) > epsilon
    if np.any(uneven):
        raise ValueError(f"{label} {y!r} must evenly divide {x!r}")
    return q


@dataclasses.dataclass(frozen=True)
class Stencil:
    """Sample points relative to an origin, in coordinate units.

    The three numeric fields are deliberately typed `Any` rather than a
    `TypeVar`: the runtime contract is "all three are the same orderable
    type, with `+`/`-`/`*` defined" (Python numerics, NumPy scalars,
    `timedelta64`), but a `TypeVar[T]` confuses static checkers' numeric
    overload resolution. `TimeStencil` is the typed specialisation.

    Args:
        start: Left edge of the window in coordinate units (relative to the
            origin).
        stop: Right edge in coordinate units. Must satisfy ``start <= stop``;
            for the degenerate single-point stencil ``start == stop`` and
            ``step == 0`` with ``closed="both"``.
        step: Spacing between sample points in coordinate units. Must be
            strictly positive when ``start < stop``.
        closed: Which endpoints are included. One of ``"left"``, ``"right"``,
            ``"both"``, ``"neither"``. Defaults to ``"left"``.

    Examples:
        >>> Stencil(start=-2, stop=2, step=0.5, closed='both').points
        array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """

    start: Any
    stop: Any
    step: Any
    closed: Closed = dataclasses.field(default="left", kw_only=True)

    def __post_init__(self) -> None:
        if self.closed not in {"left", "right", "both", "neither"}:
            raise ValueError(f"invalid value for closed: {self.closed!r}")
        if self.start == self.stop:
            if self.step != self.stop - self.start:
                raise ValueError(
                    "For single value stencil ``step`` must equal zero: "
                    f"{self.step=} vs {self.stop - self.start=}"
                )
            if self.closed != "both":
                raise ValueError(
                    'For single value stencil ``closed`` must be "both": '
                    f"{self.closed=}"
                )
        elif self.start > self.stop:
            raise ValueError(
                f"start must not be greater than stop: {self.start} vs {self.stop}"
            )
        else:
            # Non-degenerate window: step must be strictly positive, else
            # `points` produces an inconsistent / reversed grid and slices
            # become silently malformed. Caught here so the error names the
            # field directly.
            zero = self.stop - self.stop
            if self.step <= zero:
                raise ValueError(
                    f"step must be strictly positive when start < stop: {self.step=}"
                )

    @property
    def includes_start(self) -> bool:
        return self.closed in _INCLUDE_START

    @property
    def includes_stop(self) -> bool:
        return self.closed in _INCLUDE_STOP

    @property
    def points(self) -> np.ndarray:
        """Realised sample points after the closedness trim."""
        if self.step:
            num = int(
                divide_evenly(self.stop - self.start, self.step, label="step").item()
            )
        else:
            num = 0
        result = self.start + self.step * np.arange(num + 1)
        if not self.includes_start and self.step:
            result = result[1:]
        if not self.includes_stop and self.step:
            result = result[:-1]
        return result

    def get_config(self) -> dict[str, Any]:
        """YAML-serialisable view of the stencil — geopatcher convention."""
        return config_from_fields(self)


_Td64Unit = Literal["D", "h", "m", "s"]


def _normalize_time_unit(unit: str) -> _Td64Unit:
    if unit in {"D", "day", "days"}:
        return "D"
    if unit in {"h", "hr", "hour", "hours"}:
        return "h"
    if unit in {"m", "min", "minute", "minutes"}:
        return "m"
    if unit in {"s", "sec", "second", "seconds"}:
        return "s"
    raise ValueError(f"unsupported time unit: {unit!r}")


def _to_timedelta64(value: str | np.timedelta64) -> np.timedelta64:
    if isinstance(value, np.timedelta64):
        return value
    match = re.match(r"([+-]?\d+) ?([a-zA-Z]+)", value)
    if not match:
        raise ValueError(f"invalid time delta string: {value}")
    value_int = int(match.group(1))
    unit = _normalize_time_unit(match.group(2))
    return np.timedelta64(value_int, unit)


class TimeStencil(Stencil):
    """`Stencil` specialised to `np.timedelta64`.

    Accepts NumPy timedelta strings (``"-9h"``, ``"3h"``, ``"30min"``,
    ``"2D"``) or pre-built `np.timedelta64` instances.

    Examples:
        >>> TimeStencil(start='-9h', stop='3h', step='1h', closed='both')
        TimeStencil(start='-9 hours', stop='3 hours', step='1 hours', closed='both')
    """

    def __init__(
        self,
        start: str | np.timedelta64,
        stop: str | np.timedelta64,
        step: str | np.timedelta64,
        closed: Closed = "left",
    ) -> None:
        super().__init__(
            _to_timedelta64(start),
            _to_timedelta64(stop),
            _to_timedelta64(step),
            closed=closed,
        )

    def __repr__(self) -> str:
        return (
            f"TimeStencil(start='{self.start}', stop='{self.stop}',"
            f" step='{self.step}', closed='{self.closed}')"
        )


def build_sampling_slices(
    source_points: np.typing.ArrayLike,
    sample_origins: np.typing.ArrayLike,
    stencil: Stencil,
) -> list[slice]:
    """Resolve a stencil at each origin into `slice` objects.

    Args:
        source_points: 1-D, sorted-ascending, evenly-spaced data coordinates.
        sample_origins: 1-D, sorted-ascending origin coordinates. Each must
            be a value present in ``source_points`` (arbitrary gaps OK).
        stencil: The stencil shape to apply at each origin.

    Returns:
        ``list[slice]``, one per origin. Each slice has stride
        ``max(stencil.step / source_step, 1)`` — at most one element per
        source step. Strides > 1 are valid stencil output but rejected by
        the v0.1 `TemporalStencilGeometry` wrapper.

    Raises:
        ValueError: For non-1-D inputs, unsorted points, non-constant source
            step, uneven division of stencil.step by source_step, or any
            origin whose stencil falls outside ``source_points``.
    """
    source_points = np.asarray(source_points)
    sample_origins = np.asarray(sample_origins)

    if source_points.ndim != 1:
        raise ValueError(f"source_points must be 1D, got {source_points.shape=}")

    if sample_origins.ndim != 1:
        raise ValueError(f"sample_origins must be 1D, got {sample_origins.shape=}")

    source_steps = np.diff(source_points)
    if not np.all(source_steps > 0):
        raise ValueError(f"source_points must be sorted: {source_points=}")

    source_step = source_steps[0]
    if np.any(source_steps != source_step):
        raise ValueError(f"source_points must have constant step: {source_points=}")

    if not np.all(np.diff(sample_origins) > 0):
        raise ValueError(f"sample_origins must be sorted: {sample_origins=}")

    start_points = sample_origins + stencil.start
    starts = divide_evenly(
        start_points - source_points[0], source_step, label="source_step"
    )
    if sample_origins[0] + stencil.points[0] < source_points[0]:
        raise ValueError(
            "all points in the stencil centered on the first sample_origin must be "
            "at or after the first source point: "
            f"{sample_origins[0] + stencil.points} vs {source_points[0]}"
        )
    if not stencil.includes_start:
        starts += 1

    stop_points = sample_origins + stencil.stop
    stops = divide_evenly(
        stop_points - source_points[0], source_step, label="source_step"
    )
    if sample_origins[-1] + stencil.points[-1] > source_points[-1]:
        raise ValueError(
            "all points in the stencil centered on the last sample_origin must be"
            " at or before the last source point:"
            f" {sample_origins[-1] + stencil.points} vs {source_points[-1]}"
        )
    if stencil.includes_stop:
        stops += 1

    stride = max(
        divide_evenly(stencil.step, source_step, label="source_step").item(), 1
    )

    return [
        slice(int(start), int(stop), int(stride))
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
    ]


def valid_origin_points(
    source_points: np.typing.ArrayLike, stencil: Stencil
) -> np.ndarray:
    """Source points for which the full stencil fits in-record.

    Trims both ends so every emitted window is full-length — the largest
    anchor set with no truncation.

    Args:
        source_points: 1-D, sorted-ascending, evenly-spaced data coordinates.
        stencil: The stencil shape.

    Returns:
        Subset of ``source_points`` whose stencil fits entirely between
        ``source_points[0]`` and ``source_points[-1]``.
    """
    source_points = np.asarray(source_points)

    min_origin = source_points[0] - stencil.start
    if not stencil.includes_start:
        min_origin -= stencil.step

    max_origin = source_points[-1] - stencil.stop
    if not stencil.includes_stop:
        max_origin += stencil.step

    valid_origins = source_points[
        (source_points >= min_origin) & (source_points <= max_origin)
    ]
    return valid_origins


__all__ = [
    "Closed",
    "Stencil",
    "TimeStencil",
    "build_sampling_slices",
    "divide_evenly",
    "valid_origin_points",
]

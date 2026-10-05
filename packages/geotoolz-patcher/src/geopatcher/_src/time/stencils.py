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
``docs/patcher/decisions.md`` for the design and the v0.1 stride-1 constraint that
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
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from geopatcher._src._serialize import config_from_fields, jsonable_scalar


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

    Operands are numbers or `timedelta64` durations (both of the same
    kind). For `timedelta64` operands the equality is exact (NumPy
    promotes both sides to a common resolution before the integer
    comparison). For float operands we allow ``1e-6`` slack. A
    `datetime64` is a point in time, not a duration — subtract an origin
    first.

    Args:
        x: Numerator. May be scalar or array-like.
        y: Denominator. Scalar or broadcastable to ``x``; must be non-zero.
        label: Short noun describing what ``y`` represents — used in the
            error message so callers see e.g. ``"step"`` rather than just
            the raw value. Defaults to ``"value"``.

    Returns:
        The integer quotient as ``np.ndarray`` of dtype ``int``.

    Raises:
        ValueError: If ``y`` is zero or does not evenly divide ``x``. The
            message names both operands so unit mismatches are obvious from
            the traceback.
        TypeError: If either operand is a `datetime64`, or one is a
            `timedelta64` and the other is not.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.dtype.kind == "M" or y.dtype.kind == "M":
        raise TypeError(
            f"divide_evenly takes durations, not datetime64 points in time: "
            f"{x!r} / {y!r}; subtract an origin first."
        )
    if (x.dtype.kind == "m") != (y.dtype.kind == "m"):
        raise TypeError(
            f"divide_evenly needs both operands to be timedelta64 or both "
            f"numeric; got {x.dtype} / {y.dtype} ({label})"
        )
    if np.any(y == y.dtype.type(0)):
        raise ValueError(f"{label} must be non-zero; got {y!r}")
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
            strictly positive and evenly divide ``stop - start`` when
            ``start < stop``.
        closed: Which endpoints are included. One of ``"left"``, ``"right"``,
            ``"both"``, ``"neither"``. Defaults to ``"left"``.

    Raises:
        ValueError: On an invalid ``closed``, ``start > stop``, a
            non-positive or uneven ``step``, or a stencil whose closedness
            trim leaves no sample point (``Stencil(-1, 0, 1, closed="neither")``).
        TypeError: If the three fields mix `timedelta64` and plain numbers.

    Examples:
        >>> Stencil(start=-2, stop=2, step=0.5, closed='both').points
        array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """

    start: Any
    stop: Any
    step: Any
    closed: Closed = dataclasses.field(default="left", kw_only=True)

    def __post_init__(self) -> None:
        # A `{"value", "unit"}` mapping is the config form of a timedelta64
        # field (see `get_config`); coerce it so `Stencil(**cfg)` rebuilds.
        for name in ("start", "stop", "step"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, _to_timedelta64(value))
        if self.closed not in {"left", "right", "both", "neither"}:
            raise ValueError(f"invalid value for closed: {self.closed!r}")
        kinds = {
            np.asarray(v).dtype.kind == "m" for v in (self.start, self.stop, self.step)
        }
        if len(kinds) > 1:
            raise TypeError(
                "Stencil start / stop / step must all be timedelta64 or all "
                f"numbers; got {self.start!r}, {self.stop!r}, {self.step!r}"
            )
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
            if len(self.points) == 0:
                raise ValueError(
                    f"Stencil({self.start!r}, {self.stop!r}, {self.step!r}, "
                    f"closed={self.closed!r}) has no sample points after the "
                    "closedness trim; widen it or include an endpoint."
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
        """YAML-serialisable view of the stencil — geopatcher convention.

        Numeric fields dump as-is; `np.timedelta64` fields dump as
        ``{"value": int, "unit": str}`` (exact at every resolution), which
        both `Stencil` and `TimeStencil` accept back.
        """
        return {
            "start": delta_config(self.start),
            "stop": delta_config(self.stop),
            "step": delta_config(self.step),
            **config_from_fields(self, exclude=("start", "stop", "step")),
        }


_Td64Unit = Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"]

_UNIT_ALIASES: dict[str, _Td64Unit] = {
    **dict.fromkeys(("Y", "year", "years"), "Y"),
    **dict.fromkeys(("M", "month", "months"), "M"),
    **dict.fromkeys(("W", "week", "weeks"), "W"),
    **dict.fromkeys(("D", "day", "days"), "D"),
    **dict.fromkeys(("h", "hr", "hour", "hours"), "h"),
    **dict.fromkeys(("m", "min", "minute", "minutes"), "m"),
    **dict.fromkeys(("s", "sec", "second", "seconds"), "s"),
    **dict.fromkeys(("ms", "millisecond", "milliseconds"), "ms"),
    **dict.fromkeys(("us", "microsecond", "microseconds"), "us"),
    **dict.fromkeys(("ns", "nanosecond", "nanoseconds"), "ns"),
}


def _normalize_time_unit(unit: str) -> _Td64Unit:
    try:
        return _UNIT_ALIASES[unit]
    except KeyError:
        raise ValueError(f"unsupported time unit: {unit!r}") from None


def delta_config(value: Any) -> Any:
    """Config form of a stencil offset or cadence.

    `np.timedelta64` becomes ``{"value": int, "unit": str}`` — lossless at
    every resolution (``ms`` / ``us`` / ``ns`` / ``W`` / ``M`` / ``Y``
    included); any other value goes through `jsonable_scalar`.
    `_to_timedelta64` inverts the mapping form.

    Args:
        value: A `np.timedelta64`, a Python / NumPy number, or ``None``.

    Returns:
        A JSON-safe value.
    """
    if isinstance(value, np.timedelta64):
        unit, count = np.datetime_data(value.dtype)
        return {"value": int(value.astype(np.int64)) * int(count), "unit": unit}
    return jsonable_scalar(value)


def _to_timedelta64(value: str | Mapping[str, Any] | np.timedelta64) -> np.timedelta64:
    """Coerce a timedelta string / ``{"value", "unit"}`` mapping to `np.timedelta64`.

    Strings are ``"<int><unit>"`` with an optional space between — NumPy's
    own ``str(td)`` form (``"-1000 milliseconds"``, ``"1 weeks"``) included.
    """
    if isinstance(value, np.timedelta64):
        return value
    if isinstance(value, Mapping):
        return np.timedelta64(
            int(value["value"]), _normalize_time_unit(str(value["unit"]))
        )
    match = re.match(r"([+-]?\d+) ?([a-zA-Z]+)", value)
    if not match:
        raise ValueError(f"invalid time delta string: {value}")
    value_int = int(match.group(1))
    unit = _normalize_time_unit(match.group(2))
    return np.timedelta64(value_int, unit)


class TimeStencil(Stencil):
    """`Stencil` specialised to `np.timedelta64`.

    Accepts NumPy timedelta strings (``"-9h"``, ``"3h"``, ``"30min"``,
    ``"2D"``, ``"-1000 milliseconds"``), ``{"value", "unit"}`` mappings (the
    `get_config` form), or pre-built `np.timedelta64` instances.

    Examples:
        >>> TimeStencil(start='-9h', stop='3h', step='1h', closed='both')
        TimeStencil(start='-9 hours', stop='3 hours', step='1 hours', closed='both')
        >>> TimeStencil(**TimeStencil('-9h', '3h', '1h').get_config()).step
        np.timedelta64(1,'h')
    """

    def __init__(
        self,
        start: str | Mapping[str, Any] | np.timedelta64,
        stop: str | Mapping[str, Any] | np.timedelta64,
        step: str | Mapping[str, Any] | np.timedelta64,
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


def coord_step(source_points: np.typing.ArrayLike) -> Any:
    """Validate a 1-D, strictly increasing, evenly spaced coord; return its step.

    The one O(N) pass the stencil path needs per coordinate vector: with
    the step known, every window is resolved arithmetically by
    `stencil_offsets` instead of re-validating the coordinate per origin.

    Raises:
        ValueError: For a non-1-D coordinate, fewer than two points, or
            points that are not strictly increasing or evenly spaced.
    """
    points = np.asarray(source_points)
    if points.ndim != 1:
        raise ValueError(f"coord must be 1-D; got shape {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError(
            f"coord needs at least two points to define a step; got {points.shape[0]}."
        )
    steps = np.diff(points)
    if not np.all(steps > steps.dtype.type(0)):
        raise ValueError(
            "coord must be strictly increasing (sorted ascending, no "
            "duplicates); reverse a descending axis first."
        )
    step = steps[0]
    if np.any(steps != step):
        raise ValueError(
            f"coord must be evenly spaced; first step {step!r}, found "
            f"{np.unique(steps)[:5]!r}."
        )
    return step


def stencil_offsets(stencil: Stencil, source_step: Any) -> tuple[int, int, int]:
    """``(lo, hi, stride)`` index offsets of ``stencil`` on a ``source_step`` grid.

    The window at origin index ``i`` of an evenly spaced coordinate is
    ``slice(i + lo, i + hi, stride)`` — exactly what `build_sampling_slices`
    returns for that origin, computed in O(1).

    Raises:
        ValueError: If ``source_step`` does not evenly divide the stencil's
            bounds or step.
    """
    lo = int(divide_evenly(stencil.start, source_step, label="source_step").item())
    hi = int(divide_evenly(stencil.stop, source_step, label="source_step").item())
    if not stencil.includes_start:
        lo += 1
    if stencil.includes_stop:
        hi += 1
    stride = max(
        int(divide_evenly(stencil.step, source_step, label="source_step").item()), 1
    )
    return lo, hi, stride


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
    "coord_step",
    "divide_evenly",
    "stencil_offsets",
    "valid_origin_points",
]

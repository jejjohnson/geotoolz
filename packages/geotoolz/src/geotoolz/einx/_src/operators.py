"""Tier-B Operators — carrier-aware einx dispatch and pattern presets.

`Einx` is the generic escape hatch: any einx function, any pattern,
with the spatial-survival rule from ``array.py`` deciding whether a
``GeoTensor`` carrier's metadata rides through (`wrap_like`) or the
result drops to a plain ``np.ndarray``. The presets wrap the patterns
we reach for constantly in RS pipelines — channel-order flips, per-band
reductions, and spatial pooling (the one op that *updates* the
geotransform instead of preserving or dropping it).

Design notes:

- Arrays are unwrapped with ``np.asarray`` *before* reaching einx, so
  `GeoTensor.__array_ufunc__` never interacts with einx internals.
- Pattern validation is permissive at construction (einx itself
  validates at dispatch); the op name, bracket balance and the names of
  extra einx keyword arguments are checked eagerly.
- ``einx.vmap`` and other callable-taking forms are not supported by
  `Einx` — configs must stay YAML-serializable.
- Every preset broadcasts over leading axes (``...``), so 2-D ``(H, W)``,
  3-D ``(C, H, W)`` and 4-D ``(T, C, H, W)`` carriers all work.
- Nodata: `SpatialPool` and `PerBandReduce` exclude invalid pixels (see
  `geotoolz._src.valid`) with NaN-aware reductions. `Einx` computes on
  raw values, then writes an explicit output fill into the invalid
  pixels of value-changing, spatially-surviving outputs.
"""

from __future__ import annotations

import numbers
import re
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import einx as _einx
import numpy as np
from pipekit import Operator

from geotoolz._src.config import mapping_from_pairs, mapping_to_pairs
from geotoolz._src.valid import (
    carried_fill,
    carrier_fill_value,
    mask_invalid_to_nan,
    restore_fill,
    valid_pixels,
)
from geotoolz._src.wrap import wrap_like
from geotoolz.einx._src.array import _tokenize, spatial_survives


if TYPE_CHECKING:
    from collections.abc import Callable

    from georeader.geotensor import GeoTensor

#: einx entry points `Einx` refuses because they take callables (not
#: YAML-serializable) or don't return an array.
_UNSUPPORTED_OPS = frozenset({"vmap", "vmap_with_axis", "custom", "trace", "jit"})

#: einx ops that only move values around: their outputs keep the input's
#: values (fill pixels included), so they inherit the carrier's fill.
_VALUE_PRESERVING_OPS = frozenset({"id", "rearrange"})

#: einx keyword arguments `Einx` forwards besides axis sizes.
_EINX_CALL_KWARGS = frozenset({"backend", "keepdims"})

#: einx reduction names accepted by `SpatialPool` / `PerBandReduce`,
#: mapped to the NaN-aware numpy reduction that implements them.
_NAN_REDUCTIONS: dict[str, Callable[..., Any]] = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "prod": np.nanprod,
    "max": np.nanmax,
    "min": np.nanmin,
    "std": np.nanstd,
    "var": np.nanvar,
}

_AXIS_NAME = re.compile(r"[A-Za-z_]\w*")


def _resolve_op(op: str) -> Any:
    """Return the einx callable named ``op``, validating eagerly.

    Args:
        op: Name of an einx entry point (``"id"``, ``"mean"``, ``"sum"``,
            ``"dot"``, ``"rearrange"``, ...).

    Returns:
        The einx function.

    Raises:
        ValueError: Unknown, private, or unsupported op name.
    """
    if op.startswith("_") or op in _UNSUPPORTED_OPS:
        raise ValueError(f"Einx: op {op!r} is not supported")
    fn = getattr(_einx, op, None)
    if not callable(fn):
        raise ValueError(
            f"Einx: {op!r} is not an einx operation "
            "(expected e.g. 'id', 'mean', 'sum', 'max', 'dot', 'add')"
        )
    return fn


def _resolve_reduction(reduce: str, *, name: str) -> Callable[..., Any]:
    """Return a NaN-aware einx reduction for the einx reduction name ``reduce``.

    Args:
        reduce: One of the keys of ``_NAN_REDUCTIONS``.
        name: Operator name, for error messages.

    Returns:
        An einx-notation callable wrapping the ``np.nan*`` reduction.

    Raises:
        ValueError: ``reduce`` is not a supported reduction name.
    """
    fn = _NAN_REDUCTIONS.get(reduce) if isinstance(reduce, str) else None
    if fn is None:
        raise ValueError(
            f"{name}: reduce must be one of {sorted(_NAN_REDUCTIONS)}; got {reduce!r}"
        )
    return _einx.numpy.adapt_numpylike_reduce(fn)


def _pattern_axis_names(pattern: str) -> set[str]:
    """Every axis name mentioned in an einx pattern."""
    return set(_AXIS_NAME.findall(pattern))


def _output_fill(dtype: np.dtype, ref_fill: Any) -> Any:
    """Fill value for a value-changing output of ``dtype``.

    Boolean outputs use ``False``, floating-point outputs ``NaN``.
    Integer outputs keep the input's fill when it is an integer
    representable in ``dtype``; otherwise they declare none.
    """
    if dtype == np.bool_:
        return False
    if np.issubdtype(dtype, np.inexact):
        return np.nan
    if (
        isinstance(ref_fill, numbers.Integral)
        and not isinstance(ref_fill, bool | np.bool_)
        and np.iinfo(dtype).min <= int(ref_fill) <= np.iinfo(dtype).max
    ):
        return ref_fill
    return None


def _pool_factor(factor: Any) -> tuple[int, int]:
    """Validate a `SpatialPool` factor into a ``(row, col)`` pair of ints.

    Raises:
        TypeError: ``factor`` is not an integer or a pair of integers
            (``bool`` is rejected).
        ValueError: Wrong pair length or a factor below 1.
    """

    def is_int(value: Any) -> bool:
        return isinstance(value, numbers.Integral) and not isinstance(
            value, bool | np.bool_
        )

    if is_int(factor):
        pair = (factor, factor)
    elif isinstance(factor, tuple | list):
        if len(factor) != 2:
            raise ValueError(
                "SpatialPool: factor must be one int or a (row, col) pair; "
                f"got {factor!r}"
            )
        pair = tuple(factor)
    else:
        raise TypeError(
            f"SpatialPool: factor must be an int or a (row, col) pair; got {factor!r}"
        )
    if not all(is_int(f) for f in pair):
        raise TypeError(f"SpatialPool: factors must be integers; got {factor!r}")
    fy, fx = int(pair[0]), int(pair[1])
    if fy < 1 or fx < 1:
        raise ValueError(f"SpatialPool: factors must be >= 1; got {factor!r}")
    return fy, fx


def _nan_float(gt: Any, valid: np.ndarray) -> np.ndarray:
    """Float copy of ``gt`` with invalid pixels as NaN (float64 for ints)."""
    dtype = None if np.issubdtype(np.asarray(gt).dtype, np.inexact) else np.float64
    return mask_invalid_to_nan(gt, valid=valid, dtype=dtype)


class Einx(Operator):
    """Apply any einx operation to the carrier, geo-aware.

    The pattern is analyzed once at construction with
    `geotoolz.einx._src.array.spatial_survives`: when the trailing two
    axes of both the output and the carrier's (first) input expression
    are the bare spatial axes (default ``("y", "x")``) and no spatial
    axis is composed anywhere, a ``GeoTensor`` input comes
    back as a ``GeoTensor`` (transform / CRS propagated via
    ``wrap_like``, with fresh ``attrs`` whose per-band keys are dropped
    when the band count changes); otherwise the result is a plain
    ``np.ndarray``. Plain-array input always returns a plain array.

    Nodata: einx runs on the raw values. For a spatially-surviving
    output the pixel grid is unchanged, so:

    - ``"id"`` / ``"rearrange"`` only move values and inherit the
      carrier's ``fill_value_default``;
    - every other op changes the values, so the output gets an explicit
      fill — ``NaN`` for float outputs, ``False`` for boolean outputs,
      the input's fill for integer outputs when representable (else
      none) — written into every pixel that was invalid in the carrier
      (non-finite or equal to its fill in any band; see
      `geotoolz._src.valid`). With ``fill_value_default=0`` on the
      input, 0 is nodata (georeader's convention); pass ``None`` or
      ``NaN`` on the carrier when 0 is real data.

    Non-surviving outputs (e.g. ``"c y x -> c"``) are raw einx results:
    a fill or NaN pixel enters the statistic. Use `PerBandReduce` /
    `SpatialPool` for nodata-aware reductions.

    Multi-input patterns work by passing extra arrays at call time —
    the first argument is the carrier, the rest are unwrapped to plain
    arrays.

    Args:
        op: einx entry-point name (``"id"``, ``"mean"``, ``"sum"``,
            ``"max"``, ``"dot"``, ``"add"``, ...). Callable-taking
            entry points (``"vmap"``, ...) are rejected — configs must
            stay YAML-serializable.
        pattern: The einx pattern string.
        spatial_axes: Names of the (row, column) axes in the pattern.
            Default ``("y", "x")``.
        op_kwargs: Extra keyword arguments forwarded to the einx call:
            axis sizes named in ``pattern`` (``{"py": 2}``) or einx's
            ``backend`` / ``keepdims``. Must be JSON-serializable.
            ``get_config`` nests them under ``"op_kwargs"`` as
            ``[[name, value], ...]`` pairs (``Operator.from_state`` rejects
            dict values); that form is accepted back here.
        **axis_kwargs: Shorthand for ``op_kwargs`` entries
            (``Einx(..., py=2)``); merged into ``op_kwargs`` with the same
            validation, so a misspelled keyword such as ``spatial_axis=``
            raises instead of reaching einx.

    Raises:
        ValueError: Unsupported ``op``, unbalanced pattern, wrong number
            of ``spatial_axes``, or an ``op_kwargs`` name that is neither
            an axis of ``pattern`` nor ``backend`` / ``keepdims``; at call
            time, a pattern that does not match the input's rank / shape.

    Examples:
        >>> import geotoolz as gz
        >>> # Per-pixel mean over bands; spatial survives -> GeoTensor out.
        >>> mean_map = gz.Einx(op="mean", pattern="c y x -> y x")
        >>> # Matched-filter scoring against a signature matrix.
        >>> score = gz.Einx(op="dot", pattern="band y x, sig band -> sig y x")
        >>> scores = score(reflectance_gt, signatures)  # doctest: +SKIP
        >>> # Axis sizes for composed axes.
        >>> pooled = gz.Einx(
        ...     op="mean",
        ...     pattern="c (y py) (x px) -> c y x",
        ...     op_kwargs={"py": 2, "px": 2},
        ... )
    """

    def __init__(
        self,
        *,
        op: str,
        pattern: str,
        spatial_axes: tuple[str, str] | list[str] = ("y", "x"),
        op_kwargs: Mapping[str, Any] | Iterable[Any] | None = None,
        **axis_kwargs: Any,
    ) -> None:
        self.op = op
        self.pattern = pattern
        self.spatial_axes = tuple(spatial_axes)
        if len(self.spatial_axes) != 2:
            raise ValueError(
                f"Einx: spatial_axes must name exactly two axes; got {spatial_axes!r}"
            )
        kwargs = mapping_from_pairs(op_kwargs) or {}
        clash = sorted(set(kwargs) & set(axis_kwargs))
        if clash:
            raise ValueError(f"Einx: {clash} given both in op_kwargs and as keywords")
        kwargs.update(axis_kwargs)
        allowed = _pattern_axis_names(pattern) | _EINX_CALL_KWARGS
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(
                f"Einx: unknown keyword argument(s) {unknown}: op_kwargs must "
                f"name axes of the pattern {pattern!r} or be one of "
                f"{sorted(_EINX_CALL_KWARGS)}"
            )
        self.op_kwargs = kwargs
        self._fn = _resolve_op(op)
        # Also validates bracket balance eagerly (raises ValueError).
        self._survives = spatial_survives(pattern, self.spatial_axes)

    def _apply(
        self, gt: GeoTensor | np.ndarray, *extra: GeoTensor | np.ndarray
    ) -> GeoTensor | np.ndarray:
        arrays = [np.asarray(a) for a in (gt, *extra)]
        try:
            out = np.asarray(self._fn(self.pattern, *arrays, **self.op_kwargs))
        except Exception as exc:
            # einx's rank / solver errors don't say which operator or shape
            # failed; re-raise them as a ValueError that does.
            if not type(exc).__module__.startswith("einx"):
                raise
            shapes = ", ".join(str(a.shape) for a in arrays)
            raise ValueError(
                f"Einx: pattern {self.pattern!r} does not match input shape(s) "
                f"{shapes}: {type(exc).__name__}"
            ) from exc
        if not self._survives:
            return out
        if self.op in _VALUE_PRESERVING_OPS or getattr(gt, "transform", None) is None:
            return wrap_like(gt, out)
        fill = _output_fill(out.dtype, carrier_fill_value(gt))
        valid = self._valid_mask(gt, out)
        if fill is not None:
            out = restore_fill(out, valid, fill)
        return wrap_like(gt, out, fill_value_default=fill)

    def _valid_mask(self, gt: GeoTensor, out: np.ndarray) -> np.ndarray:
        """Carrier validity mask broadcastable against ``out``.

        Per-frame ``(T, H, W)`` when a 4-D carrier's leading (time) axis
        is also the output's leading axis; otherwise a conservative
        ``(H, W)`` mask (a pixel invalid in any frame / band).
        """
        if np.ndim(gt) == 4 and out.ndim >= 3:
            lhs = self.pattern.split("->", 1)[0].split(",", 1)[0]
            in_tokens = _tokenize(lhs, pattern=self.pattern)
            out_tokens = _tokenize(
                self.pattern.rsplit("->", 1)[1], pattern=self.pattern
            )
            same_time = (
                len(in_tokens) == 4
                and len(out_tokens) == out.ndim
                and in_tokens[0] == out_tokens[0]
                and _AXIS_NAME.fullmatch(in_tokens[0]) is not None
            )
            if same_time:
                return valid_pixels(gt, keep_time=True)
        return valid_pixels(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "pattern": self.pattern,
            "spatial_axes": list(self.spatial_axes),
            "op_kwargs": mapping_to_pairs(self.op_kwargs),
        }


class CHWtoHWC(Einx):
    """Reorder a ``(..., C, H, W)`` cube to channels-last ``(..., H, W, C)``.

    Display / ML-interop helper (matplotlib ``imshow``, most vision
    frameworks expect channels-last). Leading axes (e.g. time) are kept.
    The output is always a plain ``np.ndarray``: with channels trailing,
    the spatial axes no longer sit in the trailing-two positions a
    ``GeoTensor`` requires.

    Examples:
        >>> import geotoolz as gz
        >>> rgb = gz.CHWtoHWC()(composite_gt)  # (H, W, C) ndarray  # doctest: +SKIP
    """

    def __init__(self) -> None:
        super().__init__(op="id", pattern="... c y x -> ... y x c")

    def get_config(self) -> dict[str, Any]:
        return {}


class HWCtoCHW(Einx):
    """Reorder a channels-last ``(..., H, W, C)`` array to ``(..., C, H, W)``.

    Inverse of `CHWtoHWC`, for bringing external channels-last arrays
    into geotoolz's channel-first convention. Input is typically a
    plain array (channels-last carriers can't be GeoTensors), and the
    output is a plain array — rewrapping needs a reference carrier's
    metadata, e.g. ``wrap_like(reference_gt, out)`` downstream.

    Examples:
        >>> import geotoolz as gz
        >>> chw = gz.HWCtoCHW()(hwc_array)
    """

    def __init__(self) -> None:
        # `spatial_survives` rejects this pattern (the carrier's input
        # doesn't end in `y x`), so the output is always a plain array.
        super().__init__(op="id", pattern="... y x c -> ... c y x")

    def get_config(self) -> dict[str, Any]:
        return {}


class PerBandReduce(Operator):
    """Reduce each band over its spatial extent to one scalar, ignoring nodata.

    ``(C, H, W)`` in, ``(C,)`` out; ``(T, C, H, W)`` in, ``(T, C)`` out
    (einx pattern ``"... c y x -> ... c"``) — band-wise statistics for QA
    summaries, normalization fitting, or feature vectors. Spatial
    structure is consumed, so the result is always a plain float
    ``np.ndarray`` regardless of carrier.

    NaN-aware, like `geotoolz.normalize.PerBandStats`: invalid pixels —
    non-finite or equal to the carrier's ``fill_value_default`` in any
    band (per frame for 4-D input; see `geotoolz._src.valid`) — are
    excluded from every band's statistic. A band (frame) with no valid
    pixel reduces to ``NaN``. With ``fill_value_default=0`` on the input,
    0 is nodata (georeader's convention); pass ``None`` or ``NaN`` on the
    carrier when 0 is real data.

    Args:
        reduce: einx reduction name: ``"mean"`` (default), ``"sum"``,
            ``"prod"``, ``"max"``, ``"min"``, ``"std"``, ``"var"``.

    Raises:
        ValueError: Unsupported ``reduce``; at call time, an input with
            fewer than three dims (there is no band axis).

    Examples:
        >>> import geotoolz as gz
        >>> band_means = gz.PerBandReduce()(reflectance_gt)  # (C,)  # doctest: +SKIP
    """

    pattern = "... c y x -> ... c"

    def __init__(self, *, reduce: str = "mean") -> None:
        self._fn = _resolve_reduction(reduce, name="PerBandReduce")
        self.reduce = reduce

    def _apply(self, gt: GeoTensor | np.ndarray) -> np.ndarray:
        if np.ndim(gt) < 3:
            raise ValueError(
                "PerBandReduce: expected a (C, H, W) or (T, C, H, W) input; "
                f"got shape {np.shape(gt)}"
            )
        valid = valid_pixels(gt, keep_time=True)
        data = _nan_float(gt, valid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.asarray(self._fn(self.pattern, data))
        # np.nansum / np.nanprod return 0 / 1 for all-NaN slices.
        empty = ~np.asarray(valid.any(axis=(-2, -1)))  # () or (T,)
        if empty.any():
            out = out.copy()
            out[np.broadcast_to(empty[..., None], out.shape)] = np.nan
        return out


class SpatialPool(Operator):
    """Downsample the spatial grid by integer factors, updating the transform.

    Non-overlapping block pooling via einx
    (``"... (y py) (x px) -> ... y x"``), so ``(H, W)``, ``(C, H, W)`` and
    ``(T, C, H, W)`` inputs all work. This is the one einx preset that
    *changes* the geotransform rather than preserving or dropping it: a
    ``GeoTensor`` input returns a ``GeoTensor`` whose pixel size is
    scaled by the pool factors (same origin, coarser grid, fresh
    ``attrs``). Plain-array input returns a plain array.

    Nodata: invalid pixels — non-finite, or equal to the carrier's
    ``fill_value_default`` in any band (per frame for 4-D input; see
    `geotoolz._src.valid`) — are excluded with a NaN-aware reduction, so
    a block pools only its valid pixels. A block with no valid pixel is
    written as the output fill, which the output also declares: ``NaN``
    for float results of an integer input or of an input without a fill,
    otherwise the input's ``fill_value_default``. With
    ``fill_value_default=0`` on the input, 0 is nodata (georeader's
    convention); pass ``None`` or ``NaN`` on the carrier when 0 is real
    data.

    Output dtype: ``"max"`` / ``"min"`` keep an integer input's dtype;
    every other reduction returns floats (``float64`` for integer
    input, the input's float dtype otherwise).

    Args:
        reduce: einx reduction applied per block: ``"mean"`` (default),
            ``"sum"``, ``"prod"``, ``"max"``, ``"min"``, ``"std"``,
            ``"var"``.
        factor: Pool factor — one int for square blocks or a
            ``(row, col)`` pair (numpy integers accepted, ``bool``
            rejected). Spatial dims must divide evenly; otherwise a
            ``ValueError`` names the offending shape (crop or pad first,
            e.g. with ``geom.CropTo`` / ``geom.PadTo``).

    Raises:
        TypeError: ``factor`` is not an int or a pair of ints.
        ValueError: ``factor`` < 1 or not a pair, or unsupported ``reduce``.

    Examples:
        >>> import geotoolz as gz
        >>> coarse = gz.SpatialPool(reduce="mean", factor=4)(scene_gt)  # doctest: +SKIP
    """

    pattern = "... (y py) (x px) -> ... y x"

    def __init__(
        self,
        *,
        reduce: str = "mean",
        factor: int | tuple[int, int] | list[int] = 2,
    ) -> None:
        self.factor = _pool_factor(factor)
        self._fn = _resolve_reduction(reduce, name="SpatialPool")
        self.reduce = reduce

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        if arr.ndim < 2:
            raise ValueError(
                f"SpatialPool: expected an (..., H, W) input; got shape {arr.shape}"
            )
        fy, fx = self.factor
        h, w = arr.shape[-2:]
        if h % fy or w % fx:
            raise ValueError(
                f"SpatialPool: spatial shape ({h}, {w}) is not divisible by "
                f"factor {self.factor}; crop or pad first"
            )
        valid = valid_pixels(gt, keep_time=True)
        data = _nan_float(gt, valid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.asarray(self._fn(self.pattern, data, py=fy, px=fx))
        block_valid = _einx.any(self.pattern, valid, py=fy, px=fx)

        if self.reduce in ("max", "min") and np.issubdtype(arr.dtype, np.integer):
            # Block extrema of integers are integers. Integer pixels are
            # only invalid when they equal the (integer) fill, so
            # all-invalid blocks hold that fill and the cast is exact.
            out_fill = carrier_fill_value(gt)
            out = restore_fill(out, block_valid, out_fill).astype(arr.dtype)
        else:
            # Float result: an integer input's fill would collide with
            # pooled values, so it becomes NaN (see ``carried_fill``).
            out_fill = carried_fill(gt, out.dtype)
            if out_fill is None:
                out_fill = np.nan
            out = restore_fill(out, block_valid, out_fill)

        transform = getattr(gt, "transform", None)
        if transform is None:
            return out
        pooled_transform = transform * type(transform).scale(fx, fy)
        return wrap_like(
            gt, out, transform=pooled_transform, fill_value_default=out_fill
        )

    def get_config(self) -> dict[str, Any]:
        return {"reduce": self.reduce, "factor": list(self.factor)}

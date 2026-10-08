"""Shared notion of an *invalid* (nodata) pixel.

georeader marks nodata through ``GeoTensor.fill_value_default``:
``GeoTensor.validmask()`` is ``values != fill_value_default``. That
comparison never matches a ``NaN`` fill (``NaN != NaN`` is ``True``), so a
NaN-filled raster reports every pixel valid, and it ignores non-finite
values under a numeric fill. The helpers here are the one definition
every operator family uses instead:

* An **element** is invalid when it is non-finite (``NaN`` / ``±inf``)
  or equals the carrier's fill value (``NaN`` fills matched with
  :func:`numpy.isnan`).
* A **pixel** -- one ``(row, col)`` location -- is invalid when *any* of
  its bands is invalid. Readers write nodata into every band of a pixel,
  and a multi-band product (an index, a fitted spectrum, a composite
  score) cannot be trusted when one of its inputs is missing, so the
  conservative "any band" rule is used throughout the package.

Operators exclude invalid pixels from statistics, fits and reductions
(NaN-aware ``np.nan*`` reductions over :func:`mask_invalid_to_nan`
output), then write the output's fill value back into them with
:func:`restore_fill`.

The output's fill value follows its dtype and meaning, never blindly the
input's (see :func:`carried_fill` and :func:`wrap_filled`):

* boolean masks use ``False`` (as georeader's own comparison operators do);
* label / count maps use ``0``;
* float *products* whose values mean something new (indices, scores,
  features, masses, binned statistics) use ``NaN``;
* outputs that carry the input's values (filters, scalings, resampling)
  keep the input's fill when it is representable and unambiguous in the
  output dtype, and switch to ``NaN`` when an integer input is promoted to
  float (an integer fill such as ``0`` would collide with real data).
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from jaxtyping import Bool, Float

from geotoolz._src.wrap import INHERIT, FillValue, wrap_like


__all__ = [
    "carried_fill",
    "carrier_fill_value",
    "invalid_values",
    "is_fill",
    "mask_invalid_to_nan",
    "restore_fill",
    "valid_pixels",
    "wrap_filled",
]


def carrier_fill_value(x: Any, fill_value: FillValue = INHERIT) -> Any:
    """Resolve the fill value that marks nodata in ``x``.

    Args:
        x: A carrier (GeoTensor or plain array).
        fill_value: :data:`~geotoolz._src.wrap.INHERIT` (default) reads
            ``x.fill_value_default`` (``None`` for plain arrays); anything
            else is returned verbatim.

    Returns:
        The fill value, or ``None`` when ``x`` declares none.
    """
    if fill_value is INHERIT:
        return getattr(x, "fill_value_default", None)
    return fill_value


def _is_nan_scalar(value: Any) -> bool:
    return isinstance(value, numbers.Real) and bool(np.isnan(value))


def is_fill(values: np.ndarray, fill_value: Any) -> Bool[np.ndarray, "*dims"]:
    """Elementwise ``values == fill_value``, with a ``NaN`` fill matching ``NaN``.

    Args:
        values: Any array.
        fill_value: The fill value; ``None`` matches nothing.

    Returns:
        A boolean array shaped like ``values``.

    Examples:
        >>> is_fill(np.array([1.0, np.nan, -9999.0]), np.nan).tolist()
        [False, True, False]
        >>> is_fill(np.array([1, -9999]), -9999).tolist()
        [False, True]
    """
    values = np.asarray(values)
    if fill_value is None:
        return np.zeros(values.shape, dtype=bool)
    if _is_nan_scalar(fill_value):
        if np.issubdtype(values.dtype, np.inexact):
            return np.isnan(values)
        return np.zeros(values.shape, dtype=bool)
    return np.asarray(values == fill_value, dtype=bool)


def invalid_values(
    x: Any, *, fill_value: FillValue = INHERIT
) -> Bool[np.ndarray, "*dims"]:
    """Elementwise invalid mask: non-finite, or equal to the fill value.

    Args:
        x: A carrier (GeoTensor or plain array) of any shape.
        fill_value: Fill value to match. :data:`~geotoolz._src.wrap.INHERIT`
            (default) uses ``x.fill_value_default`` (none for plain
            arrays); pass one explicitly to override it.

    Returns:
        A boolean array shaped like ``x``; ``True`` marks invalid elements.
    """
    values = np.asarray(x)
    invalid = is_fill(values, carrier_fill_value(x, fill_value))
    if np.issubdtype(values.dtype, np.inexact):
        invalid |= ~np.isfinite(values)
    return invalid


def valid_pixels(
    x: Any, *, fill_value: FillValue = INHERIT, keep_time: bool = False
) -> Bool[np.ndarray, "*spatial"]:
    """Boolean validity mask over the spatial grid of ``x``.

    A pixel is valid when *every* band at that location is finite and
    differs from the fill value (see the module docstring for why "any
    band invalid" invalidates the pixel). Unlike ``GeoTensor.validmask()``
    this recognises ``NaN`` fills and non-finite values.

    Args:
        x: ``(H, W)``, ``(C, H, W)`` or ``(T, C, H, W)`` carrier.
        fill_value: Fill value override; defaults to
            ``x.fill_value_default`` (none for plain arrays).
        keep_time: For a 4-D input, return a per-frame ``(T, H, W)`` mask
            instead of reducing over time as well. Ignored for 2-D / 3-D.

    Returns:
        ``(H, W)`` boolean mask (``(T, H, W)`` with ``keep_time`` on 4-D
        input); ``True`` marks valid pixels.

    Raises:
        ValueError: If ``x`` has fewer than two dimensions.

    Examples:
        >>> cube = np.array([[[1.0, np.nan]], [[2.0, 3.0]]])  # (2, 1, 2)
        >>> valid_pixels(cube).tolist()
        [[True, False]]
    """
    invalid = invalid_values(x, fill_value=fill_value)
    if invalid.ndim < 2:
        raise ValueError(
            f"valid_pixels needs a carrier with >= 2 dims; got shape {invalid.shape}"
        )
    if invalid.ndim == 4 and keep_time:
        return ~invalid.any(axis=1)
    if invalid.ndim > 2:
        invalid = invalid.any(axis=tuple(range(invalid.ndim - 2)))
    return ~invalid


def _broadcast_valid(valid: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a spatial / per-frame mask against a carrier ``shape``."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim == 3 and len(shape) == 4:
        valid = valid[:, None]
    return np.broadcast_to(valid, shape)


def mask_invalid_to_nan(
    x: Any,
    *,
    fill_value: FillValue = INHERIT,
    valid: np.ndarray | None = None,
    dtype: Any = None,
) -> Float[np.ndarray, "*dims"]:
    """Float copy of ``x`` with every invalid pixel set to ``NaN`` in all bands.

    The result feeds ``np.nan*`` reductions (and NaN-propagating filters)
    so fill pixels never enter a statistic.

    Args:
        x: A carrier (GeoTensor or plain array) with >= 2 dims.
        fill_value: Fill value override (see :func:`valid_pixels`).
        valid: Precomputed validity mask -- ``(H, W)``, per-frame
            ``(T, H, W)`` for 4-D input, or ``x``'s full shape. Computed
            with :func:`valid_pixels` when ``None``.
        dtype: Float dtype of the result. ``None`` keeps a float input's
            dtype and promotes integer input to ``float32`` / ``float64``
            (see :func:`geotoolz._src.dtype.as_float`).

    Returns:
        A new float ndarray shaped like ``x``.
    """
    from geotoolz._src.dtype import as_float

    values = np.asarray(x)
    if valid is None:
        valid = valid_pixels(x, fill_value=fill_value)
    out = (
        as_float(values).copy()
        if dtype is None
        else values.astype(np.result_type(dtype, np.float32), copy=True)
    )
    out[~_broadcast_valid(valid, out.shape)] = np.nan
    return out


def restore_fill(out: Any, valid: np.ndarray, fill_value: Any) -> np.ndarray:
    """Write ``fill_value`` into every invalid pixel of an operator output.

    Args:
        out: The operator's computed output, ``(H, W)``, ``(C, H, W)`` or
            ``(T, C, H, W)``.
        valid: Validity mask from :func:`valid_pixels` -- ``(H, W)``,
            ``(T, H, W)`` for a 4-D ``out``, or ``out``'s full shape.
        fill_value: The *output's* fill value. ``None`` or ``NaN`` writes
            ``NaN`` (floating-point outputs only).

    Returns:
        A copy of ``out`` (as an ndarray) whose invalid pixels hold the
        fill value. When every pixel is valid, ``out`` is returned as-is.

    Raises:
        ValueError: If ``fill_value`` cannot be represented in
            ``out.dtype`` (``None`` / ``NaN`` for an integer or boolean
            output, or an integer fill outside the dtype's range).
    """
    arr = np.asarray(out)
    invalid = ~_broadcast_valid(valid, arr.shape)
    if not invalid.any():
        return arr
    value = np.nan if fill_value is None else fill_value
    if not np.issubdtype(arr.dtype, np.inexact):
        representable = (
            not _is_nan_scalar(value)
            and np.asarray(value).astype(arr.dtype).item() == value
        )
        if not representable:
            raise ValueError(
                f"Cannot write fill value {fill_value!r} into a {arr.dtype} output"
            )
    arr = arr.copy()
    arr[invalid] = value
    return arr


def carried_fill(ref: Any, dtype: Any) -> Any:
    """Fill value for an output that carries ``ref``'s values in ``dtype``.

    For value-preserving operators (filters, scalings, resampling, casts)
    whose output may change dtype. Operators whose output *means*
    something new pass the canonical fill for that meaning instead
    (``False`` / ``0`` / ``NaN``; see the module docstring).

    Args:
        ref: The operator's input carrier.
        dtype: The output dtype.

    Returns:
        ``False`` for a boolean output. For a floating-point output,
        ``NaN`` when ``ref`` is an integer / boolean carrier with a fill
        (the integer fill would collide with promoted real data, e.g.
        ``0``), otherwise ``ref``'s fill (``None`` stays ``None``). For an
        integer output, ``ref``'s fill when it is representable in
        ``dtype``, otherwise ``None``.

    Examples:
        >>> carried_fill(np.zeros(2, dtype=np.uint16), np.float32) is None
        True
        >>> carried_fill(np.zeros(2), bool)
        False
    """
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return False
    fill = carrier_fill_value(ref)
    if fill is None:
        return None
    if np.issubdtype(dtype, np.inexact):
        if _is_nan_scalar(fill) or not np.issubdtype(np.asarray(ref).dtype, np.inexact):
            return np.nan
        return fill
    if _is_nan_scalar(fill) or not isinstance(fill, numbers.Real):
        return None
    try:
        cast = np.asarray(fill).astype(dtype)
    except (OverflowError, ValueError):
        return None
    return fill if cast.item() == fill else None


def wrap_filled(
    ref: Any,
    out: Any,
    *,
    fill_value_default: Any,
    valid: np.ndarray | None = None,
    **wrap_kwargs: Any,
) -> Any:
    """Write an explicit fill into ``ref``'s invalid pixels and rewrap ``out``.

    The one-call form of ``restore_fill`` + ``wrap_like`` that keeps the
    declared ``fill_value_default`` in step with the values written.

    Args:
        ref: The operator's input carrier.
        out: The computed output, on ``ref``'s grid.
        fill_value_default: The output's fill value (``None`` writes ``NaN``
            into a float output, and nothing is declared).
        valid: Validity mask (see :func:`restore_fill`); defaults to
            ``valid_pixels(ref)`` (per frame for 4-D input).
        **wrap_kwargs: Forwarded to :func:`~geotoolz._src.wrap.wrap_like`
            (``attrs``, ``band_names``, ``transform``).

    Returns:
        ``out`` rewrapped like ``ref`` with ``fill_value_default`` declared.
    """
    if valid is None:
        valid = valid_pixels(ref, keep_time=np.ndim(out) == 4)
    filled = restore_fill(out, valid, fill_value_default)
    return wrap_like(ref, filled, fill_value_default=fill_value_default, **wrap_kwargs)

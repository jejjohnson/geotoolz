"""Named reductions over a dim that leave nodata out (GA-B6, #632).

`pipekit_array.reduce` / `argreduce` reduce any named carrier, and know
nothing of nodata: on a `GeoArray` with fill 0, a time mean averages the
zeros in. These versions read the carrier's nodata first, with the
stack's rule (`GeoArray.valid_mask`): a value is invalid when it is
non-finite or equals the fill, a pixel is invalid in every band when any
of its bands is, and on `GeoPoints` a sample that is not ``valid`` is
invalid. Invalid entries become NaN, a named
reduction runs as its NaN-skipping form, and a position with no valid
entry is NaN, the result's fill. They run on NumPy and under ``jax.jit``.

Composites are built from them: a median composite is ``stack`` then
``reduce(stack, "median", "time")``; a max-NDVI composite is
``argreduce(ndvi, "argmax", "time")`` then ``take_along``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pipekit_array


__all__ = ["argreduce", "reduce"]

#: Each named reduction and its NaN-skipping form.
NAN_FORMS: dict[str, str] = {
    "mean": "nanmean",
    "median": "nanmedian",
    "sum": "nansum",
    "prod": "nanprod",
    "max": "nanmax",
    "min": "nanmin",
    "std": "nanstd",
    "var": "nanvar",
}

#: ``argreduce``'s reductions and the value an invalid entry takes so it never wins.
_ARG_FILLS: dict[str, float] = {"argmax": -float("inf"), "argmin": float("inf")}


def reduce(x: Any, fn: str | Callable[..., Any], dim: str | Sequence[str]) -> Any:
    """``pipekit_array.reduce`` over the valid entries of a geoarray carrier.

    Args:
        x: A `GeoArray`, `GeoPoints` or `GeoSwath`: anything with ``data``,
            ``fill_value`` and ``with_data``.
        fn: A named reduction (``"mean"``, ``"median"``, ``"max"``, …, or
            its ``nan*`` form), or a function taking ``axis=``, which gets
            float data with NaN at invalid entries.
        dim: The dim to reduce over, or several.

    Returns:
        The reduced carrier, as `pipekit_array.reduce` returns it (a
        `GeoArray` while ``y`` and ``x`` survive), with float data and a
        NaN fill: NaN where no entry along ``dim`` was valid.

    Raises:
        TypeError: ``x`` is not a geoarray carrier.
        ValueError: ``fn`` names a reduction with no NaN-skipping form.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> from geoops.compositing import reduce
        >>> t = (20.0, 0.0, 0.0, 0.0, -20.0, 0.0)
        >>> stack = GeoArray(
        ...     np.array([[[1.0]], [[0.0]], [[3.0]]]), t, None, fill_value=0.0,
        ...     dims="time y x",
        ... )
        >>> reduce(stack, "mean", "time").data.tolist()  # the 0 is nodata
        [[2.0]]
    """
    masked = _masked(x, "reduce")
    xp = masked.xp
    if callable(fn):
        return pipekit_array.reduce(masked, fn, dim)
    name = NAN_FORMS.get(fn, fn)
    if name not in NAN_FORMS.values():
        raise ValueError(
            f"reduce: {fn!r} has no NaN-skipping form; use one of "
            f"{sorted(NAN_FORMS)}, or pass a function that handles NaN"
        )
    func = getattr(xp, name)

    def skipping(data: Any, axis: Any) -> Any:
        # nodata never counts; a position with no data at all stays NaN
        # (nansum / nanprod would give 0 / 1 there)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = func(data, axis=axis)
        none_valid = xp.all(xp.isnan(data), axis=axis)
        return xp.where(none_valid, xp.asarray(float("nan"), dtype=out.dtype), out)

    return pipekit_array.reduce(masked, skipping, dim)


def argreduce(x: Any, fn: str, dim: str) -> Any:
    """``pipekit_array.argreduce`` that never picks an invalid entry.

    Args:
        x: A `GeoArray`, `GeoPoints` or `GeoSwath`.
        fn: ``"argmax"`` or ``"argmin"``.
        dim: The dim to pick an index along.

    Returns:
        The integer index carrier `pipekit_array.argreduce` returns. Where
        every entry along ``dim`` is invalid the index is 0, so
        ``take_along`` there gathers an invalid entry: the nodata survives.

    Raises:
        TypeError: ``x`` is not a geoarray carrier.
        ValueError: ``fn`` is not ``"argmax"`` or ``"argmin"``.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> from geoops.compositing import argreduce
        >>> t = (20.0, 0.0, 0.0, 0.0, -20.0, 0.0)
        >>> ndvi = GeoArray(
        ...     np.array([[[0.4]], [[9.0]], [[0.7]]]), t, None, fill_value=9.0,
        ...     dims="time y x",
        ... )
        >>> argreduce(ndvi, "argmax", "time").data.tolist()  # 9.0 is the fill
        [[2]]
    """
    if fn not in _ARG_FILLS:
        raise ValueError(f"argreduce: fn is 'argmax' or 'argmin', got {fn!r}")
    masked = _masked(x, "argreduce")
    xp = masked.xp
    loser = _ARG_FILLS[fn]
    func = getattr(xp, fn)

    def skipping(data: Any, axis: Any) -> Any:
        # an invalid entry takes the value that can never win
        return func(xp.where(xp.isnan(data), loser, data), axis=axis)

    return pipekit_array.argreduce(masked, skipping, dim)


def _masked(x: Any, what: str) -> Any:
    """``x`` as float data with NaN at every invalid entry, and a NaN fill."""
    if not all(hasattr(x, a) for a in ("data", "fill_value", "with_data", "xp")):
        raise TypeError(
            f"{what}: needs a geoarray carrier (GeoArray, GeoPoints, GeoSwath), "
            f"got {type(x).__name__}; pipekit_array.{what} reduces any named "
            "carrier without reading nodata"
        )
    xp = x.xp
    data = x.data
    floating = data if data.dtype.kind in "fc" else data.astype(xp.float64)
    # geoops._src.valid holds the same rule on the host (NumPy); written on
    # the carrier's array namespace here, so it also runs under jax.jit
    invalid = ~xp.isfinite(floating)
    fill = x.fill_value
    if fill is not None and fill == fill:  # a NaN fill is already non-finite
        invalid = invalid | (data == fill)
    dims = tuple(getattr(x, "dims", ()))
    if "band" in dims:  # a pixel is nodata in every band when any band is
        axis = dims.index("band")
        invalid = xp.broadcast_to(
            xp.any(invalid, axis=axis, keepdims=True), invalid.shape
        )
    valid = getattr(x, "valid", None)
    if valid is not None:  # GeoPoints: a sample that is not an observation
        invalid = invalid | ~xp.asarray(valid, dtype=bool)
    nan = xp.asarray(float("nan"), dtype=floating.dtype)
    return x.with_data(xp.where(invalid, nan, floating), fill_value=float("nan"))

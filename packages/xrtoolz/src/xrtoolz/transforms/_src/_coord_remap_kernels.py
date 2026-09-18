"""Private numpy kernels for axis remapping.

Implementation detail — no stability guarantees. ``remap_axis``
interpolates values from a source 1D coordinate vector to a target 1D
coordinate vector along a chosen axis, preserving all other dimensions.
Used internally by the Layer 0 xarray wrapper in
:mod:`xrtoolz.transforms._src.coord_remap`.

Backend: numpy. Methods: ``"linear"`` (np.interp on real/imag parts
independently), ``"nearest"``.

:func:`remap_axis_columns` is the per-column generalisation: the source
coordinate has the same shape as ``values`` (terrain-following / hybrid
grids) and an ``extrapolate`` policy decides what happens beyond each
column's own range.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from jaxtyping import Float, Inexact


Extrapolate = Literal["nan", "nearest"]
_EXTRAPOLATE_MODES = ("nan", "nearest")


def _as_floating(
    arr: Inexact[np.ndarray, "*shape"],
) -> Inexact[np.ndarray, "*shape"]:
    """Cast to a floating dtype while preserving complex inputs."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.complexfloating):
        return a if a.dtype == np.complex128 else a.astype(np.complex128)
    if np.issubdtype(a.dtype, np.floating):
        return a
    return a.astype(np.float64)


def remap_axis(
    values: Inexact[np.ndarray, "..."],
    *,
    axis: int = -1,
    source_coords: Float[np.ndarray, "src"],
    target_coords: Float[np.ndarray, "tgt"],
    method: str = "linear",
) -> Inexact[np.ndarray, "..."]:
    """Interpolate values along ``axis`` from ``source_coords`` to ``target_coords``.

    Parameters
    ----------
    values
        Array with one axis whose length matches ``source_coords``.
        Real or complex; complex inputs are interpolated component-wise.
    axis
        Axis to remap along.
    source_coords
        1D monotonic (ascending or descending) coordinate vector.
    target_coords
        1D coordinate vector to interpolate to. Targets outside the
        source range or equal to ``NaN`` produce ``NaN`` in the output.
    method
        ``"linear"`` or ``"nearest"``.

    Returns
    -------
    NDArray
        Same shape as ``values`` except along ``axis``, which is
        replaced by ``len(target_coords)``.
    """
    arr = _as_floating(values)
    src = np.asarray(source_coords, dtype=float)
    tgt = np.asarray(target_coords, dtype=float)

    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("source_coords and target_coords must be 1D")
    if arr.shape[axis] != src.size:
        raise ValueError(
            f"values.shape[axis]={arr.shape[axis]} but len(source_coords)={src.size}"
        )

    # Normalize to ascending source for np.interp / searchsorted.
    diffs = np.diff(src)
    if np.all(diffs > 0):
        ascending = True
    elif np.all(diffs < 0):
        ascending = False
        src = src[::-1]
    else:
        raise ValueError("source_coords must be strictly monotonic")

    moved = np.moveaxis(arr, axis, -1)
    if not ascending:
        moved = moved[..., ::-1]

    flat = moved.reshape(-1, src.size)
    is_complex = np.iscomplexobj(flat)
    out_dtype = flat.dtype if is_complex else float
    out = np.empty((flat.shape[0], tgt.size), dtype=out_dtype)

    # Identify NaN targets up front; numpy.interp/searchsorted have
    # surprising behavior on NaN inputs (np.interp returns the right-hand
    # fill value, argmin/searchsorted treat NaN as larger than any
    # number), so we mask them out and assign NaN explicitly.
    nan_target = np.isnan(tgt)

    if method == "linear":
        if is_complex:
            for i in range(flat.shape[0]):
                real = np.interp(tgt, src, flat[i].real, left=np.nan, right=np.nan)
                imag = np.interp(tgt, src, flat[i].imag, left=np.nan, right=np.nan)
                out[i] = real + 1j * imag
        else:
            for i in range(flat.shape[0]):
                out[i] = np.interp(tgt, src, flat[i], left=np.nan, right=np.nan)
        if nan_target.any():
            out[:, nan_target] = np.nan
    elif method == "nearest":
        # Replace NaN targets with a sentinel so argmin is well-defined;
        # we'll mask the result back to NaN below.
        safe_tgt = np.where(nan_target, src.min(), tgt)
        idx = np.abs(src[None, :] - safe_tgt[:, None]).argmin(axis=1)  # (M,)
        out[:] = flat[:, idx]
        oor = (tgt < src.min()) | (tgt > src.max()) | nan_target
        if oor.any():
            out[:, oor] = np.nan
    else:
        raise ValueError(f"unknown method {method!r}; expected 'linear' or 'nearest'")

    out = out.reshape(*moved.shape[:-1], tgt.size)
    return np.moveaxis(out, -1, axis)


def remap_axis_columns(
    values: Inexact[np.ndarray, "..."],
    *,
    axis: int,
    source_coords: Float[np.ndarray, "..."],
    target_coords: Float[np.ndarray, "tgt"],
    method: str = "linear",
    extrapolate: Extrapolate = "nan",
) -> Inexact[np.ndarray, "..."]:
    """Interpolate every column of ``values`` onto ``target_coords``.

    Unlike :func:`remap_axis`, the source coordinate is an array of the
    same shape as ``values`` — every column (every position along the
    non-``axis`` dims) carries its own set of source levels. Each column
    must be strictly monotone along ``axis`` (ascending or descending;
    the orientation may differ between columns).

    For a column with ascending levels ``z_0 < ... < z_{n-1}`` and a
    target ``zeta`` the linear bracket is ``hi = clip(#{j: z_j <= zeta},
    1, n-1)``, ``lo = hi - 1``, and the value is ``f_lo + (f_hi - f_lo)
    / (z_hi - z_lo) * (zeta - z_lo)`` (np.interp's formula, so a column
    that merely broadcasts a 1-D coordinate reproduces
    :func:`remap_axis` exactly).

    Args:
        values: Array with at least two levels along ``axis``. Real or
            complex; complex inputs are interpolated component-wise.
        axis: Axis to remap along.
        source_coords: Source level values, same shape as ``values``.
        target_coords: 1-D target level values. ``NaN`` targets give
            ``NaN`` output regardless of ``extrapolate``.
        method: ``"linear"`` or ``"nearest"`` (closest level per column).
        extrapolate: ``"nan"`` — targets outside a column's own
            ``[z_0, z_{n-1}]`` range are ``NaN``; ``"nearest"`` — they hold
            that column's end value.

    Returns:
        Same shape as ``values`` except along ``axis``, which is replaced
        by ``len(target_coords)``.

    Raises:
        ValueError: If shapes disagree, ``target_coords`` is not 1-D,
            ``axis`` has fewer than two levels, any column is not strictly
            monotone, or ``method`` / ``extrapolate`` is unknown.
    """
    if method not in ("linear", "nearest"):
        raise ValueError(f"unknown method {method!r}; expected 'linear' or 'nearest'")
    if extrapolate not in _EXTRAPOLATE_MODES:
        raise ValueError(
            f"unknown extrapolate {extrapolate!r}; expected one of "
            f"{_EXTRAPOLATE_MODES!r}"
        )
    arr = _as_floating(values)
    src = np.asarray(source_coords, dtype=float)
    tgt = np.asarray(target_coords, dtype=float)
    if tgt.ndim != 1:
        raise ValueError("target_coords must be 1D")
    if src.shape != arr.shape:
        raise ValueError(
            f"source_coords shape {src.shape} must equal values shape {arr.shape}"
        )
    n_lev = arr.shape[axis]
    if n_lev < 2:
        raise ValueError(
            f"remap_axis_columns needs at least two levels along axis, got {n_lev}"
        )

    moved = np.moveaxis(arr, axis, -1)
    z = np.moveaxis(src, axis, -1).reshape(-1, n_lev)
    f = moved.reshape(-1, n_lev)

    diffs = np.diff(z, axis=1)
    ascending = np.all(diffs > 0, axis=1)
    descending = np.all(diffs < 0, axis=1)
    bad = ~(ascending | descending)
    if bad.any():
        raise ValueError(
            f"source_coords must be strictly monotone along axis in every column; "
            f"{int(bad.sum())} of {bad.size} columns are not"
        )
    # Flip descending columns so every column is ascending.
    flip = descending[:, None]
    z = np.where(flip, z[:, ::-1], z)
    f = np.where(flip, f[:, ::-1], f)

    is_complex = np.iscomplexobj(f)
    out_dtype = f.dtype if is_complex else float
    n_col = f.shape[0]
    out = np.empty((n_col, tgt.size), dtype=out_dtype)
    nan_target = np.isnan(tgt)
    z_first = z[:, 0]
    z_last = z[:, -1]
    f_first = f[:, 0]
    f_last = f[:, -1]

    for k, zk in enumerate(tgt):
        if nan_target[k]:
            out[:, k] = np.nan
            continue
        if method == "linear":
            hi = np.clip((z <= zk).sum(axis=1), 1, n_lev - 1)[:, None]
            lo = hi - 1
            z_lo = np.take_along_axis(z, lo, axis=1)[:, 0]
            z_hi = np.take_along_axis(z, hi, axis=1)[:, 0]
            f_lo = np.take_along_axis(f, lo, axis=1)[:, 0]
            f_hi = np.take_along_axis(f, hi, axis=1)[:, 0]
            slope = (f_hi - f_lo) / (z_hi - z_lo)
            col = slope * (zk - z_lo) + f_lo
            # Exact hits on a level return that level's value (np.interp
            # semantics) and out-of-range targets hold the end values;
            # ``"nan"`` masks the latter below.
            col = np.where(z_lo == zk, f_lo, col)
            col = np.where(z_hi == zk, f_hi, col)
            col = np.where(zk < z_first, f_first, col)
            col = np.where(zk > z_last, f_last, col)
        else:
            idx = np.abs(z - zk).argmin(axis=1)[:, None]
            col = np.take_along_axis(f, idx, axis=1)[:, 0]
        if extrapolate == "nan":
            col = np.where((zk < z_first) | (zk > z_last), np.nan, col)
        out[:, k] = col

    out = out.reshape(*moved.shape[:-1], tgt.size)
    return np.moveaxis(out, -1, axis)


__all__ = ["Extrapolate", "remap_axis", "remap_axis_columns"]

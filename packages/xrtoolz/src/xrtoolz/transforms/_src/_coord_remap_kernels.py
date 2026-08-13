"""Private numpy kernels for axis remapping.

Implementation detail — no stability guarantees. ``remap_axis``
interpolates values from a source 1D coordinate vector to a target 1D
coordinate vector along a chosen axis, preserving all other dimensions.
Used internally by the Layer 0 xarray wrapper in
:mod:`xrtoolz.transforms._src.coord_remap`.

Backend: numpy. Methods: ``"linear"`` (np.interp on real/imag parts
independently), ``"nearest"``.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Inexact


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


__all__ = ["remap_axis"]

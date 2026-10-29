"""Discrete Cosine / Sine transforms via :mod:`scipy.fft`.

Both :func:`dct` and :func:`dst` operate along a single ``DataArray``
dim. The transformed axis keeps its dim name (DCT/DST do not introduce
a frequency coordinate the way FFT does — they're a basis change in
the same length-N space).
"""

from __future__ import annotations

import xarray as xr
from scipy.fft import (
    dct as _scipy_dct,
    dst as _scipy_dst,
    idct as _scipy_idct,
    idst as _scipy_idst,
)


def _output_name(da: xr.DataArray, suffix: str) -> str | None:
    if da.name is None:
        return None
    return f"{da.name}_{suffix}"


def dct(
    da: xr.DataArray,
    dim: str,
    *,
    type: int = 2,
    norm: str | None = "ortho",
) -> xr.DataArray:
    """Discrete Cosine Transform along ``dim``.

    Args:
        da: Input field.
        dim: Dimension to transform.
        type: DCT type ``1, 2, 3, 4`` (default ``2`` — the DCT-II used
            by JPEG and most signal-processing applications).
        norm: ``"ortho"`` (default) makes the transform unitary so that
            ``idct(dct(x)) == x``. Pass ``None`` for the unnormalised
            scipy default.

    Returns:
        DataArray of the same shape, named ``f"{name}_dct"``.
    """
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    axis = da.get_axis_num(dim)
    raw = _scipy_dct(da.values, type=type, axis=axis, norm=norm)
    return xr.DataArray(
        raw,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        attrs=dict(da.attrs),
        name=_output_name(da, "dct"),
    )


def idct(
    da: xr.DataArray,
    dim: str,
    *,
    type: int = 2,
    norm: str | None = "ortho",
) -> xr.DataArray:
    """Inverse DCT along ``dim``. See :func:`dct` for arguments."""
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    axis = da.get_axis_num(dim)
    raw = _scipy_idct(da.values, type=type, axis=axis, norm=norm)
    return xr.DataArray(
        raw,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        attrs=dict(da.attrs),
        name=_output_name(da, "idct"),
    )


def dst(
    da: xr.DataArray,
    dim: str,
    *,
    type: int = 2,
    norm: str | None = "ortho",
) -> xr.DataArray:
    """Discrete Sine Transform along ``dim``. See :func:`dct`."""
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    axis = da.get_axis_num(dim)
    raw = _scipy_dst(da.values, type=type, axis=axis, norm=norm)
    return xr.DataArray(
        raw,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        attrs=dict(da.attrs),
        name=_output_name(da, "dst"),
    )


def idst(
    da: xr.DataArray,
    dim: str,
    *,
    type: int = 2,
    norm: str | None = "ortho",
) -> xr.DataArray:
    """Inverse DST along ``dim``. See :func:`dct`."""
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    axis = da.get_axis_num(dim)
    raw = _scipy_idst(da.values, type=type, axis=axis, norm=norm)
    return xr.DataArray(
        raw,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        attrs=dict(da.attrs),
        name=_output_name(da, "idst"),
    )

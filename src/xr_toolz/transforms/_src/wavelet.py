"""Continuous and discrete wavelet transforms via :mod:`pywt`.

PyWavelets is an optional dependency — install with
``pip install xr_toolz[wavelets]``. The two entry points
(:func:`cwt`, :func:`dwt`) lazy-import :mod:`pywt` and surface a
pointer to the extra in their ``ImportError``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


_WAVELET_INSTALL_HINT = (
    "PyWavelets is required for wavelet transforms in `xr_toolz.transforms` "
    "(for example, `from xr_toolz.transforms import cwt, dwt`). "
    "Install with `pip install xr_toolz[wavelets]` or "
    "`pip install PyWavelets>=1.4`."
)


def _import_pywt():
    """Lazy import :mod:`pywt`, raise a friendly hint if missing."""
    try:
        import pywt  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(_WAVELET_INSTALL_HINT) from exc
    return pywt


def _output_name(da: xr.DataArray, suffix: str) -> str | None:
    if da.name is None:
        return None
    return f"{da.name}_{suffix}"


def cwt(
    da: xr.DataArray,
    dim: str,
    *,
    scales: Sequence[float] | np.ndarray,
    wavelet: str = "morl",
    sampling_period: float = 1.0,
) -> xr.DataArray:
    """Continuous Wavelet Transform along ``dim``.

    Args:
        da: Input field.
        dim: Dimension to transform.
        scales: Wavelet scales (positive floats). Larger scale → lower
            frequency content. See :func:`pywt.scale2frequency`.
        wavelet: Wavelet name (e.g. ``"morl"`` for Morlet,
            ``"mexh"``, ``"cmor"``, ``"gaus1"``).
        sampling_period: Sample spacing along ``dim`` in seconds (or
            equivalent), used to convert scales to frequencies.

    Returns:
        Complex-valued DataArray with dims ``(*outer_dims, scale, dim)``,
        named ``f"{name}_cwt"``. A ``frequency`` 1-D coord is attached
        along the new ``scale`` axis.
    """
    pywt = _import_pywt()
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    scales_arr = np.asarray(scales, dtype=float)
    if scales_arr.ndim != 1 or scales_arr.size == 0:
        raise ValueError("scales must be a non-empty 1-D sequence of positive floats.")
    if (scales_arr <= 0).any():
        raise ValueError("scales must be strictly positive.")

    axis = da.get_axis_num(dim)
    coeffs, freqs = pywt.cwt(
        da.values, scales_arr, wavelet, sampling_period=sampling_period, axis=axis
    )

    new_dims = ("scale", *da.dims)
    coords: dict = {k: da.coords[k] for k in da.coords}
    coords["scale"] = scales_arr
    coords["frequency"] = ("scale", np.asarray(freqs))

    return xr.DataArray(
        coeffs,
        dims=new_dims,
        coords=coords,
        attrs=dict(da.attrs),
        name=_output_name(da, "cwt"),
    )


def dwt(
    da: xr.DataArray,
    dim: str,
    *,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "symmetric",
) -> dict[str, xr.DataArray]:
    """Multi-level discrete wavelet decomposition along ``dim``.

    Returns a dict ``{"approx": ..., "detail_1": ..., ..., "detail_L": ...}``
    where ``L = level`` (or the maximum allowed by the signal length if
    ``level`` is ``None``). Each entry is a DataArray with the
    transformed axis renamed (``dim`` is replaced by ``f"{dim}_dwt_<k>"``)
    since the level-wise outputs have different lengths.

    Args:
        da: Input field.
        dim: Dimension to decompose.
        wavelet: Wavelet name (``"db4"``, ``"sym5"``, ``"haar"``…).
        level: Decomposition depth. ``None`` uses
            :func:`pywt.dwt_max_level`.
        mode: Signal-extension mode at boundaries.

    Returns:
        Dictionary of approximation and detail coefficients as
        DataArrays.
    """
    pywt = _import_pywt()
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")

    axis = da.get_axis_num(dim)
    n = da.sizes[dim]
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    eff_level = max_level if level is None else int(level)
    if eff_level < 1 or eff_level > max_level:
        raise ValueError(
            f"level must be in [1, {max_level}] for n={n} and wavelet={wavelet!r}; "
            f"got {eff_level}."
        )

    coeffs = pywt.wavedec(
        da.values, wavelet=wavelet, level=eff_level, mode=mode, axis=axis
    )
    # ``coeffs[0]`` is the approximation; ``coeffs[1:]`` are details
    # ordered coarsest→finest.
    out: dict[str, xr.DataArray] = {}
    other_dims = tuple(d for d in da.dims if d != dim)
    other_coords = {k: da.coords[k] for k in da.coords if k in other_dims}

    def _wrap(arr: np.ndarray, label: str) -> xr.DataArray:
        new_dim = f"{dim}_dwt_{label}"
        new_dims = tuple(new_dim if d == dim else d for d in da.dims)
        return xr.DataArray(
            arr,
            dims=new_dims,
            coords=other_coords,
            attrs=dict(da.attrs),
            name=_output_name(da, f"dwt_{label}"),
        )

    out["approx"] = _wrap(coeffs[0], "approx")
    for k, det in enumerate(coeffs[1:], start=1):
        out[f"detail_{k}"] = _wrap(det, f"detail_{k}")
    return out

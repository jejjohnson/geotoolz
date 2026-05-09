"""Tier B — value-preserving smoothers on xarray Datasets (D12, F3.3).

Functions take an :class:`xr.Dataset`, a dimension name, and smoother
parameters; they return a Dataset with the same shape and coords, every
numeric data variable smoothed along ``dim``. Variables that don't
carry ``dim`` (or are non-numeric) pass through untouched.

Per D12 the smoothers are deterministic and parameter-free —
``KalmanSmoother`` is out of scope here and lives under future
``assimilate.smooth``.

Tier A array kernels live at :mod:`xr_toolz.interpolate._src.array_smooth`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.interpolate._src import array_smooth as _array


def _apply_along_dim(
    ds: xr.Dataset,
    dim: str,
    fn: Callable[..., Any],
) -> xr.Dataset:
    """Apply a Tier A kernel to every numeric data variable that carries ``dim``."""
    if dim not in ds.dims:
        raise ValueError(f"dim {dim!r} not in Dataset dims {tuple(ds.dims)}")

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if dim not in da.dims or not np.issubdtype(da.dtype, np.number):
            out_vars[str(name)] = da
            continue
        axis = da.get_axis_num(dim)
        smoothed = fn(da.values, axis=axis)
        out_vars[str(name)] = xr.DataArray(
            smoothed,
            dims=da.dims,
            coords=da.coords,
            attrs=dict(da.attrs),
            name=da.name,
        )
    return xr.Dataset(out_vars, coords=ds.coords, attrs=dict(ds.attrs))


def moving_average(
    ds: xr.Dataset,
    *,
    dim: str,
    window: int,
    center: bool = True,
    min_periods: int | None = None,
) -> xr.Dataset:
    """Sliding-window mean along ``dim``.

    See :func:`xr_toolz.interpolate.array.moving_average` for the Tier A
    kernel and parameter semantics.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.moving_average(
            arr,
            axis=axis,
            window=window,
            center=center,
            min_periods=min_periods,
        )

    return _apply_along_dim(ds, dim, _fn)


def gaussian_smooth(
    ds: xr.Dataset,
    *,
    dim: str,
    sigma: float,
    truncate: float = 4.0,
) -> xr.Dataset:
    """Gaussian smoothing along ``dim`` with standard deviation ``sigma``."""

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.gaussian_smooth(arr, axis=axis, sigma=sigma, truncate=truncate)

    return _apply_along_dim(ds, dim, _fn)


def _normalize_dims(dim: str | Sequence[str]) -> tuple[str, ...]:
    dims = (dim,) if isinstance(dim, str) else tuple(dim)
    if not dims:
        raise ValueError("dim must contain at least one dimension")
    if len(set(dims)) != len(dims):
        raise ValueError(f"dim entries must be unique, got {dims}")
    return dims


def _normalize_sigmas(
    sigma: float | Mapping[str, float],
    dims: tuple[str, ...],
) -> tuple[float, ...]:
    if isinstance(sigma, Mapping):
        missing = tuple(d for d in dims if d not in sigma)
        if missing:
            raise ValueError(f"sigma mapping missing dimensions {missing}")
        sigmas = tuple(float(sigma[d]) for d in dims)
    else:
        sigmas = (float(sigma),) * len(dims)
    if any(s <= 0 for s in sigmas):
        raise ValueError(f"all sigma values must be > 0, got {sigmas}")
    return sigmas


def _has_split_core_chunks(da: xr.DataArray, dims: tuple[str, ...]) -> bool:
    if da.chunks is None:
        return False
    chunks_by_dim = dict(zip(da.dims, da.chunks, strict=True))
    return any(dim in chunks_by_dim and len(chunks_by_dim[dim]) > 1 for dim in dims)


def _gaussian_smooth_masked_dataarray(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...],
    sigmas: tuple[float, ...],
    truncate: float,
    mode: str,
    nan_aware: bool,
    min_weight: float,
) -> xr.DataArray:
    if not set(dims) <= set(da.dims) or not np.issubdtype(da.dtype, np.number):
        return da
    if _has_split_core_chunks(da, dims):
        raise ValueError(
            "gaussian_smooth_masked requires each smoothing dimension to be a "
            "single chunk. Rechunk smoothing dims to one chunk, and chunk along "
            "non-smoothing dims."
        )

    def _kernel(arr: np.ndarray) -> np.ndarray:
        # `vectorize=True` ensures apply_ufunc only ever hands us a core-shaped
        # block, so we don't need a manual reshape / stack loop. That also
        # avoids the empty-leading-dim crash that the manual np.stack hit.
        return _array.gaussian_smooth_nd(
            arr,
            sigma=sigmas,
            truncate=truncate,
            mode=mode,
            nan_aware=nan_aware,
            min_weight=min_weight,
        )

    out = xr.apply_ufunc(
        _kernel,
        da,
        input_core_dims=[list(dims)],
        output_core_dims=[list(dims)],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.result_type(da.dtype, np.float64)],
        dask_gufunc_kwargs={"allow_rechunk": False},
        keep_attrs=True,
    )
    return out.transpose(*da.dims)


def gaussian_smooth_masked(
    ds: xr.Dataset | xr.DataArray,
    *,
    dim: str | Sequence[str],
    sigma: float | Mapping[str, float],
    truncate: float = 4.0,
    mode: str = "reflect",
    nan_aware: bool = True,
    min_weight: float = 1e-6,
) -> xr.Dataset | xr.DataArray:
    """NaN-aware N-D Gaussian smoothing along one or more dimensions."""
    if truncate <= 0:
        raise ValueError(f"truncate must be > 0, got {truncate}")
    if min_weight < 0:
        raise ValueError(f"min_weight must be >= 0, got {min_weight}")

    dims = _normalize_dims(dim)
    sigmas = _normalize_sigmas(sigma, dims)

    if isinstance(ds, xr.DataArray):
        missing = tuple(d for d in dims if d not in ds.dims)
        if missing:
            raise ValueError(f"dim entries {missing} not in DataArray dims {ds.dims}")
        return _gaussian_smooth_masked_dataarray(
            ds,
            dims=dims,
            sigmas=sigmas,
            truncate=truncate,
            mode=mode,
            nan_aware=nan_aware,
            min_weight=min_weight,
        )

    missing = tuple(d for d in dims if d not in ds.dims)
    if missing:
        raise ValueError(f"dim entries {missing} not in Dataset dims {tuple(ds.dims)}")

    out_vars = {
        str(name): _gaussian_smooth_masked_dataarray(
            da,
            dims=dims,
            sigmas=sigmas,
            truncate=truncate,
            mode=mode,
            nan_aware=nan_aware,
            min_weight=min_weight,
        )
        for name, da in ds.data_vars.items()
    }
    return xr.Dataset(out_vars, coords=ds.coords, attrs=dict(ds.attrs))


def lowpass_filter(
    ds: xr.Dataset,
    *,
    dim: str,
    cutoff: float | tuple[float, float] | list[float] | np.ndarray,
    order: int = 4,
    btype: str = "low",
) -> xr.Dataset:
    """Zero-phase Butterworth filter along ``dim``.

    ``cutoff`` is the normalized critical frequency (fraction of the
    Nyquist rate). For ``btype`` in ``{"bandpass", "bandstop"}`` pass a
    length-2 ``(low, high)`` sequence. See
    :func:`xr_toolz.interpolate.array.lowpass_filter`.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.lowpass_filter(
            arr, axis=axis, cutoff=cutoff, order=order, btype=btype
        )

    return _apply_along_dim(ds, dim, _fn)


def fir_filter(
    ds: xr.Dataset,
    *,
    dim: str,
    cutoff: float | tuple[float, float] | list[float] | np.ndarray,
    method: str = "lanczos",
    btype: str = "low",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
) -> xr.Dataset:
    """Zero-phase FIR filter along ``dim``.

    See :func:`xr_toolz.interpolate.array.fir_filter` for cutoff, window,
    and tap-count semantics.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _array.fir_filter(
            arr,
            axis=axis,
            cutoff=cutoff,
            method=method,
            btype=btype,
            num_taps=num_taps,
            attenuation_db=attenuation_db,
        )

    return _apply_along_dim(ds, dim, _fn)


__all__ = [
    "fir_filter",
    "gaussian_smooth",
    "gaussian_smooth_masked",
    "lowpass_filter",
    "moving_average",
]

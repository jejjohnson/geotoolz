"""Layer 0 value-preserving smoothers on xarray DataArrays (D12, F3.3).

Per the PR β primitive-flip (``docs/design/xarray-native-primitives.md``),
the Layer-0 smoothers in this module are DataArray-in / DataArray-out:
one variable goes in, one variable comes out. The Dataset loop —
applying the kernel to every numeric data variable that carries
``dim``, while passing other variables through — lives in the Layer-1
``Operator`` wrappers at :mod:`xrtoolz.interpolate.operators`.

Per D12 the smoothers are deterministic and parameter-free —
``KalmanSmoother`` is out of scope here and lives under future
``assimilate.smooth``.

The numpy/scipy compute lives in the private kernel module
:mod:`xrtoolz.interpolate._src._smooth_kernels` (implementation detail,
no stability guarantees).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from xrtoolz.interpolate._src import _smooth_kernels as _kernels


def _apply_kernel_to_dataarray(
    da: xr.DataArray,
    dim: str,
    fn: Callable[..., Any],
) -> xr.DataArray:
    """Apply a private 1-D kernel along ``dim`` of a single DataArray.

    Uses ``apply_ufunc(dask="parallelized")`` so chunked inputs stay lazy.
    ``vectorize=True`` hands the kernel a 1-D core slice with ``dim`` moved to
    the last axis, so it is invoked with ``axis=-1``. The smoothing dim must be
    a single chunk (rechunk it whole and chunk along the other dims instead).
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not in DataArray dims {tuple(da.dims)}")
    if _has_split_core_chunks(da, (dim,)):
        raise ValueError(
            f"smoothing dim {dim!r} must be a single chunk. Rechunk it whole "
            "and chunk along the non-smoothing dims instead."
        )

    def _core(arr: np.ndarray) -> np.ndarray:
        return fn(arr, axis=-1)

    out = xr.apply_ufunc(
        _core,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.result_type(da.dtype, np.float64)],
        dask_gufunc_kwargs={"allow_rechunk": False},
        keep_attrs=True,
    )
    return out.transpose(*da.dims)


def moving_average(
    da: xr.DataArray,
    *,
    dim: str,
    window: int,
    center: bool = True,
    min_periods: int | None = None,
) -> xr.DataArray:
    """Sliding-window mean along ``dim``.

    Computes a centred (``center=True``) or trailing sliding-window mean
    of length ``window`` along ``dim``. ``min_periods`` controls the
    minimum number of valid (non-NaN) samples required inside the window
    for the output to be non-NaN; if unset it defaults to ``window``.
    Inputs with fewer than ``min_periods`` valid samples in a window
    produce NaN at that position.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _kernels.moving_average(
            arr,
            axis=axis,
            window=window,
            center=center,
            min_periods=min_periods,
        )

    return _apply_kernel_to_dataarray(da, dim, _fn)


def gaussian_smooth(
    da: xr.DataArray,
    *,
    dim: str,
    sigma: float,
    truncate: float = 4.0,
) -> xr.DataArray:
    """Gaussian smoothing along ``dim`` with standard deviation ``sigma``."""

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _kernels.gaussian_smooth(arr, axis=axis, sigma=sigma, truncate=truncate)

    return _apply_kernel_to_dataarray(da, dim, _fn)


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
        return _kernels.gaussian_smooth_nd(
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
    da: xr.DataArray,
    *,
    dim: str | Sequence[str],
    sigma: float | Mapping[str, float],
    truncate: float = 4.0,
    mode: str = "reflect",
    nan_aware: bool = True,
    min_weight: float = 1e-6,
) -> xr.DataArray:
    """NaN-aware N-D Gaussian smoothing along one or more dimensions.

    Operates on a single :class:`xr.DataArray`. The Layer-1
    ``GaussianSmoothMasked`` operator handles the Dataset loop.
    """
    if truncate <= 0:
        raise ValueError(f"truncate must be > 0, got {truncate}")
    if min_weight < 0:
        raise ValueError(f"min_weight must be >= 0, got {min_weight}")

    dims = _normalize_dims(dim)
    sigmas = _normalize_sigmas(sigma, dims)

    missing = tuple(d for d in dims if d not in da.dims)
    if missing:
        raise ValueError(f"dim entries {missing} not in DataArray dims {da.dims}")
    return _gaussian_smooth_masked_dataarray(
        da,
        dims=dims,
        sigmas=sigmas,
        truncate=truncate,
        mode=mode,
        nan_aware=nan_aware,
        min_weight=min_weight,
    )


def lowpass_filter(
    da: xr.DataArray,
    *,
    dim: str,
    cutoff: float | tuple[float, float] | list[float] | np.ndarray,
    order: int = 4,
    btype: str = "low",
) -> xr.DataArray:
    """Zero-phase Butterworth filter along ``dim``.

    ``cutoff`` is the normalized critical frequency (fraction of the
    Nyquist rate). For ``btype`` in ``{"bandpass", "bandstop"}`` pass a
    length-2 ``(low, high)`` sequence. The filter is applied with
    ``scipy.signal.sosfiltfilt`` (forward-backward) for zero phase
    distortion; ``order`` is the per-direction order of the SOS sections.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _kernels.lowpass_filter(
            arr, axis=axis, cutoff=cutoff, order=order, btype=btype
        )

    return _apply_kernel_to_dataarray(da, dim, _fn)


def fir_filter(
    da: xr.DataArray,
    *,
    dim: str,
    cutoff: float | tuple[float, float] | list[float] | np.ndarray,
    method: str = "lanczos",
    btype: str = "low",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
) -> xr.DataArray:
    """Zero-phase FIR filter along ``dim``.

    ``cutoff`` is the normalized critical frequency (fraction of the
    Nyquist rate); for ``btype`` in ``{"bandpass", "bandstop"}`` pass a
    length-2 ``(low, high)`` sequence. ``method`` selects the window
    family used to taper the ideal sinc response — ``"lanczos"`` or
    ``"kaiser"``. ``num_taps`` is an odd FIR tap count; if omitted, a
    conservative default is chosen (Lanczos picks from ``cutoff``;
    Kaiser estimates from ``attenuation_db``). ``attenuation_db`` is the
    Kaiser stop-band attenuation target in decibels (default ``60.0``)
    and is ignored when ``method="lanczos"``. The filter is applied with
    ``scipy.signal.filtfilt`` for zero phase.
    """

    def _fn(arr: np.ndarray, *, axis: int) -> np.ndarray:
        return _kernels.fir_filter(
            arr,
            axis=axis,
            cutoff=cutoff,
            method=method,
            btype=btype,
            num_taps=num_taps,
            attenuation_db=attenuation_db,
        )

    return _apply_kernel_to_dataarray(da, dim, _fn)


__all__ = [
    "fir_filter",
    "gaussian_smooth",
    "gaussian_smooth_masked",
    "lowpass_filter",
    "moving_average",
]

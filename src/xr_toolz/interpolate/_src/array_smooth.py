"""Tier A — array kernels for value-preserving smoothers (D11, D12).

Per design decision D11, every arithmetic submodule grows a duck-array
``axis=`` entry point. These kernels operate on raw :class:`numpy.ndarray`
inputs with an explicit ``axis``, returning an array of the same shape
(no reduction).

Backend: numpy + scipy. JAX / CuPy variants are out of scope for the
F3.3 pilot.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


_BUTTER_BTYPES: frozenset[str] = frozenset(
    {"low", "high", "lowpass", "highpass", "bandpass", "bandstop"}
)


def _as_floating(arr: ArrayLike) -> np.ndarray:
    """Cast ``arr`` to a floating dtype while preserving complex inputs."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.complexfloating):
        return a if a.dtype == np.complex128 else a.astype(np.complex128)
    if np.issubdtype(a.dtype, np.floating):
        return a
    return a.astype(np.float64)


def moving_average(
    arr: ArrayLike,
    *,
    axis: int = -1,
    window: int,
    center: bool = True,
    min_periods: int | None = None,
) -> NDArray[np.floating]:
    """Sliding-window mean along ``axis``. NaN-skipping.

    Parameters
    ----------
    arr
        Input array (any shape). Complex inputs are smoothed component-wise.
    axis
        Axis to smooth along.
    window
        Window length (number of samples). Must be a positive integer.
    center
        If True, the window is centered on the output sample; otherwise
        trailing.
    min_periods
        Minimum number of non-NaN samples required inside the window
        for the output to be non-NaN. Defaults to ``window``.

    Returns
    -------
    NDArray
        Smoothed array, same shape as the input.
    """
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
        raise TypeError(f"window must be an integer, got {type(window).__name__}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_periods is not None and min_periods < 0:
        raise ValueError(f"min_periods must be >= 0, got {min_periods}")
    if min_periods is None:
        min_periods = window

    a = _as_floating(arr)
    moved = np.moveaxis(a, axis, -1)
    if center:
        pad_left = (window - 1) // 2
        pad_right = window - 1 - pad_left
    else:
        pad_left = window - 1
        pad_right = 0

    pad_widths = [(0, 0)] * moved.ndim
    pad_widths[-1] = (pad_left, pad_right)
    padded = np.pad(moved, pad_widths, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=-1)

    valid = (~np.isnan(windows)).sum(axis=-1)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(windows, axis=-1)
    out = np.where(valid >= min_periods, means, np.nan)
    return np.moveaxis(out, -1, axis)


def gaussian_smooth(
    arr: ArrayLike,
    *,
    axis: int = -1,
    sigma: float,
    truncate: float = 4.0,
) -> NDArray[np.floating]:
    """Gaussian convolution along ``axis`` with standard deviation ``sigma``.

    Delegates to :func:`scipy.ndimage.gaussian_filter1d`. NaN inputs
    propagate. Complex inputs are filtered component-wise.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if truncate <= 0:
        raise ValueError(f"truncate must be > 0, got {truncate}")
    a = _as_floating(arr)
    return gaussian_filter1d(
        a, sigma=sigma, axis=axis, truncate=truncate, mode="reflect"
    )


def lowpass_filter(
    arr: ArrayLike,
    *,
    axis: int = -1,
    cutoff: float | Sequence[float],
    order: int = 4,
    btype: str = "low",
) -> NDArray[np.floating]:
    """Zero-phase Butterworth filter along ``axis``.

    Parameters
    ----------
    cutoff
        Normalized critical frequency (fraction of the Nyquist rate). For
        ``btype`` in ``{"low", "high"}`` (or aliases), ``cutoff`` is a
        scalar in ``(0, 1)``. For ``btype`` in ``{"bandpass", "bandstop"}``,
        ``cutoff`` is a length-2 sequence ``(low, high)`` with both
        entries in ``(0, 1)``.
    order
        Filter order (positive integer).
    btype
        Filter type — one of ``{"low", "high", "lowpass", "highpass",
        "bandpass", "bandstop"}``, forwarded to :func:`scipy.signal.butter`.

    Notes
    -----
    The filter is applied with :func:`scipy.signal.sosfiltfilt` for zero
    phase and SOS-form numerical stability. Complex inputs are filtered
    component-wise.
    """
    if btype not in _BUTTER_BTYPES:
        raise ValueError(
            f"btype must be one of {sorted(_BUTTER_BTYPES)}, got {btype!r}"
        )
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
        raise TypeError(f"order must be an integer, got {type(order).__name__}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    is_band = btype in {"bandpass", "bandstop"}
    if is_band:
        if np.isscalar(cutoff):
            raise ValueError(
                f"btype={btype!r} requires a length-2 cutoff (low, high); "
                f"got scalar {cutoff}"
            )
        cutoff_arr = np.asarray(cutoff, dtype=float)
        if cutoff_arr.shape != (2,):
            raise ValueError(
                f"btype={btype!r} requires a length-2 cutoff; "
                f"got shape {cutoff_arr.shape}"
            )
        if not np.all((cutoff_arr > 0.0) & (cutoff_arr < 1.0)):
            raise ValueError(
                f"cutoff entries must lie in (0, 1); got {cutoff_arr.tolist()}"
            )
        if cutoff_arr[0] >= cutoff_arr[1]:
            raise ValueError(
                f"cutoff[0] must be < cutoff[1]; got {cutoff_arr.tolist()}"
            )
        sos_cutoff: float | np.ndarray = cutoff_arr
    else:
        if not np.isscalar(cutoff):
            raise ValueError(
                f"btype={btype!r} requires a scalar cutoff; got {cutoff!r}"
            )
        c = float(cutoff)  # type: ignore[arg-type]
        if not (0.0 < c < 1.0):
            raise ValueError(f"cutoff must lie in (0, 1); got {c}")
        sos_cutoff = c

    a = _as_floating(arr)
    sos = butter(order, sos_cutoff, btype=btype, output="sos")
    return sosfiltfilt(sos, a, axis=axis)


__all__ = [
    "gaussian_smooth",
    "lowpass_filter",
    "moving_average",
]

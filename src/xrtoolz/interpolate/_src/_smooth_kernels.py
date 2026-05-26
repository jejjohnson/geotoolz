"""Private numpy/scipy kernels for value-preserving smoothers.

Implementation detail — no stability guarantees. These kernels operate
on raw :class:`numpy.ndarray` inputs with an explicit ``axis``, returning
an array of the same shape (no reduction). Used internally by the
Layer 0 xarray wrappers in :mod:`xrtoolz.interpolate._src.smooth`.

Backend: numpy + scipy. JAX / CuPy variants are out of scope for the
F3.3 pilot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import (
    butter,
    filtfilt,
    kaiser_beta,
    kaiserord,
    sosfiltfilt,
    windows as signal_windows,
)


_BUTTER_BTYPES: frozenset[str] = frozenset(
    {"low", "high", "lowpass", "highpass", "bandpass", "bandstop"}
)
_FIR_BTYPES: frozenset[str] = frozenset(
    {"low", "high", "lowpass", "highpass", "bandpass", "bandstop"}
)
_FIR_METHODS: frozenset[str] = frozenset({"lanczos", "kaiser"})


def _as_floating(arr: ArrayLike) -> np.ndarray:
    """Cast ``arr`` to a floating dtype while preserving complex inputs."""
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.complexfloating):
        return a if a.dtype == np.complex128 else a.astype(np.complex128)
    if np.issubdtype(a.dtype, np.floating):
        return a
    return a.astype(np.float64)


def _validate_num_taps(num_taps: int) -> int:
    if not isinstance(num_taps, (int, np.integer)) or isinstance(num_taps, bool):
        raise TypeError(f"num_taps must be an integer, got {type(num_taps).__name__}")
    if num_taps < 3:
        raise ValueError(f"num_taps must be >= 3, got {num_taps}")
    if num_taps % 2 == 0:
        raise ValueError(f"num_taps must be odd, got {num_taps}")
    return int(num_taps)


def _normalize_fir_cutoff(
    cutoff: float | Sequence[float],
    btype: str,
) -> float | tuple[float, float]:
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
        return (float(cutoff_arr[0]), float(cutoff_arr[1]))

    if not np.isscalar(cutoff):
        raise ValueError(f"btype={btype!r} requires a scalar cutoff; got {cutoff!r}")
    c = float(np.asarray(cutoff, dtype=float).item())
    if not (0.0 < c < 1.0):
        raise ValueError(f"cutoff must lie in (0, 1); got {c}")
    return c


def _default_lanczos_taps(cutoff: float | tuple[float, float]) -> int:
    smallest_cutoff = min(cutoff) if isinstance(cutoff, tuple) else cutoff
    return int(2 * np.ceil(2.0 / smallest_cutoff) + 1)


def _default_kaiser_taps(
    cutoff: float | tuple[float, float],
    attenuation_db: float,
) -> int:
    smallest_cutoff = min(cutoff) if isinstance(cutoff, tuple) else cutoff
    transition_width = max(smallest_cutoff / 2.0, np.finfo(float).eps)
    num_taps, _ = kaiserord(attenuation_db, transition_width)
    if num_taps % 2 == 0:
        num_taps += 1
    return _validate_num_taps(num_taps)


def _lowpass_fir_taps(
    cutoff: float,
    *,
    method: str,
    num_taps: int,
    attenuation_db: float | None,
) -> NDArray[np.floating]:
    m = (num_taps - 1) // 2
    n = np.arange(-m, m + 1, dtype=float)
    taps = cutoff * np.sinc(cutoff * n)
    if method == "lanczos":
        window = np.sinc(n / m)
    else:
        if attenuation_db is None:
            raise ValueError("attenuation_db is required for Kaiser FIR taps")
        beta = kaiser_beta(attenuation_db)
        window = signal_windows.kaiser(num_taps, beta, sym=True)
    taps *= window
    taps /= taps.sum()
    return taps


def _fir_taps(
    *,
    cutoff: float | Sequence[float],
    method: str,
    btype: str,
    num_taps: int | None = None,
    attenuation_db: float | None = None,
) -> NDArray[np.floating]:
    """Design odd-length FIR taps with normalized Nyquist cutoffs."""
    if method not in _FIR_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_FIR_METHODS)}, got {method!r}"
        )
    if btype not in _FIR_BTYPES:
        raise ValueError(f"btype must be one of {sorted(_FIR_BTYPES)}, got {btype!r}")
    btype = {"lowpass": "low", "highpass": "high"}.get(btype, btype)
    normalized_cutoff = _normalize_fir_cutoff(cutoff, btype)

    if method == "kaiser":
        if attenuation_db is None:
            attenuation_db = 60.0
        if attenuation_db <= 0:
            raise ValueError(f"attenuation_db must be > 0, got {attenuation_db}")
    if num_taps is None:
        if method == "lanczos":
            num_taps = _default_lanczos_taps(normalized_cutoff)
        else:
            num_taps = _default_kaiser_taps(normalized_cutoff, attenuation_db)
    else:
        num_taps = _validate_num_taps(num_taps)

    if isinstance(normalized_cutoff, tuple):
        low, high = normalized_cutoff
        taps = _lowpass_fir_taps(
            high, method=method, num_taps=num_taps, attenuation_db=attenuation_db
        ) - _lowpass_fir_taps(
            low, method=method, num_taps=num_taps, attenuation_db=attenuation_db
        )
        if btype == "bandstop":
            taps = -taps
            taps[(num_taps - 1) // 2] += 1.0
        return taps

    taps = _lowpass_fir_taps(
        normalized_cutoff,
        method=method,
        num_taps=num_taps,
        attenuation_db=attenuation_db,
    )
    if btype == "high":
        taps = -taps
        taps[(num_taps - 1) // 2] += 1.0
    return taps


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


def gaussian_smooth_nd(
    arr: ArrayLike,
    *,
    sigma: float | Sequence[float],
    truncate: float = 4.0,
    mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "reflect",
    cval: float = 0.0,
    nan_aware: bool = True,
    min_weight: float = 1e-6,
    mask: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """N-D Gaussian convolution with optional NaN-aware normalization.

    With ``nan_aware=True`` (default), implements normalized convolution:
    NaN or masked pixels are excluded from the weighted sum, and pixels whose
    Gaussian-weighted support falls below ``min_weight`` are returned as NaN.
    Original invalid or masked pixels remain NaN in the output.
    """
    if truncate <= 0:
        raise ValueError(f"truncate must be > 0, got {truncate}")
    if min_weight < 0:
        raise ValueError(f"min_weight must be >= 0, got {min_weight}")

    arr_ndim = np.asarray(arr).ndim
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        if float(sigma_arr) <= 0:
            raise ValueError(f"sigma must be > 0, got {float(sigma_arr)}")
    elif sigma_arr.shape != (arr_ndim,):
        raise ValueError(
            f"sigma must be scalar or length arr.ndim={arr_ndim}; got shape "
            f"{sigma_arr.shape}"
        )
    elif np.any(sigma_arr <= 0):
        raise ValueError(f"all sigma entries must be > 0, got {sigma_arr.tolist()}")

    a = _as_floating(arr)
    if not nan_aware:
        return gaussian_filter(a, sigma=sigma, truncate=truncate, mode=mode, cval=cval)

    if mask is None:
        valid_mask = np.isfinite(a)
    else:
        valid_mask = np.asarray(mask, dtype=bool)
        if valid_mask.shape != a.shape:
            raise ValueError(f"mask shape {valid_mask.shape} != arr shape {a.shape}")
        valid_mask &= np.isfinite(a)

    filled = np.where(valid_mask, a, 0.0)
    numerator = gaussian_filter(
        filled, sigma=sigma, truncate=truncate, mode=mode, cval=cval
    )
    denominator = gaussian_filter(
        valid_mask.astype(float), sigma=sigma, truncate=truncate, mode=mode, cval=cval
    )

    out = np.full_like(numerator, np.nan)
    supported = denominator > min_weight
    np.divide(numerator, denominator, out=out, where=supported)
    out[~valid_mask] = np.nan
    return out


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


def fir_filter(
    arr: ArrayLike,
    *,
    axis: int = -1,
    cutoff: float | Sequence[float],
    method: str = "lanczos",
    btype: str = "low",
    num_taps: int | None = None,
    attenuation_db: float | None = None,
) -> NDArray[np.floating]:
    """Zero-phase FIR filter along ``axis``.

    Args:
        arr: Input array. Complex inputs are filtered component-wise.
        axis: Axis to filter along.
        cutoff: Normalized cutoff frequency (fraction of Nyquist). For
            ``btype`` in ``{"bandpass", "bandstop"}``, pass a length-2
            ``(low, high)`` sequence.
        method: Window family: ``"lanczos"`` or ``"kaiser"``.
        btype: Filter type: ``"low"``, ``"high"``, ``"bandpass"``, or
            ``"bandstop"`` (plus ``"lowpass"``/``"highpass"`` aliases).
        num_taps: Odd FIR tap count. Defaults to a conservative value for
            Lanczos, or is estimated from ``attenuation_db`` for Kaiser.
        attenuation_db: Kaiser stop-band attenuation target in decibels.

    Returns:
        Filtered array with the same shape as ``arr``.
    """
    taps = _fir_taps(
        cutoff=cutoff,
        method=method,
        btype=btype,
        num_taps=num_taps,
        attenuation_db=attenuation_db,
    )
    a = _as_floating(arr)
    return filtfilt(taps, [1.0], a, axis=axis)


__all__ = [
    "fir_filter",
    "gaussian_smooth",
    "gaussian_smooth_nd",
    "lowpass_filter",
    "moving_average",
]

"""Tier A — gap-tolerant segmented 1-D spectral kernels."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy import signal


def _stride(npt: int, overlap: float) -> int:
    if npt <= 0:
        raise ValueError("npt must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")
    return max(1, int(npt * (1.0 - overlap)))


def _segment_bounds(
    n: int,
    *,
    npt: int,
    overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    min_segment_length: int | None = None,
) -> list[tuple[int, int]]:
    stride = _stride(npt, overlap)
    min_len = npt if min_segment_length is None else int(min_segment_length)
    if min_len <= 0:
        raise ValueError("min_segment_length must be positive.")

    if gap_indices is None:
        split_points: list[int] = []
    else:
        gaps = np.asarray(gap_indices, dtype=int).ravel()
        gaps = np.unique(gaps[(gaps >= 0) & (gaps < n - 1)])
        split_points = (gaps + 1).tolist()

    bounds: list[tuple[int, int]] = []
    chunk_start = 0
    for chunk_stop in [*split_points, n]:
        chunk_len = chunk_stop - chunk_start
        if chunk_len >= min_len and chunk_len >= npt:
            for start in range(chunk_start, chunk_stop - npt + 1, stride):
                bounds.append((start, start + npt))
        chunk_start = chunk_stop
    return bounds


def _finite_rows(segments: NDArray[np.floating]) -> NDArray[np.bool_]:
    return np.all(np.isfinite(segments), axis=1)


def _segments_from_bounds(
    values: NDArray[np.floating],
    bounds: list[tuple[int, int]],
) -> NDArray[np.floating]:
    if not bounds:
        return np.empty((0, 0), dtype=float)
    npt = bounds[0][1] - bounds[0][0]
    out = np.empty((len(bounds), npt), dtype=float)
    for i, (start, stop) in enumerate(bounds):
        out[i] = values[start:stop]
    return out


def segment_signal(
    x: ArrayLike,
    *,
    npt: int,
    overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    min_segment_length: int | None = None,
) -> NDArray[np.floating]:
    """Slice a 1-D signal into equal-length, finite, gap-free windows.

    When ``min_segment_length`` is omitted, chunks shorter than ``npt``
    are dropped because they cannot produce a complete equal-length window.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("segment_signal expects a 1-D signal.")

    bounds = _segment_bounds(
        values.size,
        npt=npt,
        overlap=overlap,
        gap_indices=gap_indices,
        min_segment_length=min_segment_length,
    )
    if not bounds:
        return np.empty((0, npt), dtype=float)

    stride = _stride(npt, overlap)
    if gap_indices is None and min_segment_length is None:
        # Fast path: contiguous input needs no explicit bounds indexing.
        segments = sliding_window_view(values, npt)[::stride]
    else:
        segments = _segments_from_bounds(values, bounds)
    return np.asarray(segments[_finite_rows(segments)], dtype=float)


def _paired_segments(
    x: ArrayLike,
    y: ArrayLike,
    *,
    npt: int,
    overlap: float,
    gap_indices: ArrayLike | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("segmented cross-spectral kernels expect 1-D signals.")
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape.")

    bounds = _segment_bounds(
        x_values.size,
        npt=npt,
        overlap=overlap,
        gap_indices=gap_indices,
    )
    x_segments = _segments_from_bounds(x_values, bounds)
    y_segments = _segments_from_bounds(y_values, bounds)
    if x_segments.size == 0:
        return np.empty((0, npt), dtype=float), np.empty((0, npt), dtype=float)

    finite = _finite_rows(x_segments) & _finite_rows(y_segments)
    return x_segments[finite], y_segments[finite]


def segmented_psd(
    x: ArrayLike,
    *,
    fs: float,
    npt: int,
    overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    window: str | tuple[str, float] | ArrayLike = "hann",
    scaling: str = "density",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Mean Welch PSD over finite, gap-free, overlapping 1-D windows."""
    segments = segment_signal(x, npt=npt, overlap=overlap, gap_indices=gap_indices)
    freqs, _ = signal.welch(
        np.zeros(npt),
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling=scaling,
    )
    if segments.size == 0:
        return freqs, np.full(freqs.shape, np.nan, dtype=float)

    freqs, psd = signal.welch(
        segments,
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling=scaling,
        axis=-1,
    )
    return freqs, np.mean(psd, axis=0)


def segmented_csd(
    x: ArrayLike,
    y: ArrayLike,
    *,
    fs: float,
    npt: int,
    overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    window: str | tuple[str, float] | ArrayLike = "hann",
    scaling: str = "density",
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Mean cross spectral density over finite, gap-free windows."""
    x_segments, y_segments = _paired_segments(
        x, y, npt=npt, overlap=overlap, gap_indices=gap_indices
    )
    freqs, _ = signal.csd(
        np.zeros(npt),
        np.zeros(npt),
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling=scaling,
    )
    if x_segments.size == 0:
        return freqs, np.full(freqs.shape, np.nan + 0j, dtype=complex)

    freqs, csd = signal.csd(
        x_segments,
        y_segments,
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling=scaling,
        axis=-1,
    )
    return freqs, np.mean(csd, axis=0)


def segmented_coherence(
    x: ArrayLike,
    y: ArrayLike,
    *,
    fs: float,
    npt: int,
    overlap: float = 0.5,
    gap_indices: ArrayLike | None = None,
    window: str | tuple[str, float] | ArrayLike = "hann",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Magnitude-squared coherence from window-averaged spectra."""
    x_segments, y_segments = _paired_segments(
        x, y, npt=npt, overlap=overlap, gap_indices=gap_indices
    )
    freqs, _ = signal.welch(
        np.zeros(npt),
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling="density",
    )
    if x_segments.size == 0:
        return freqs, np.full(freqs.shape, np.nan, dtype=float)

    freqs, pxx = signal.welch(
        x_segments,
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling="density",
        axis=-1,
    )
    _, pyy = signal.welch(
        y_segments,
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling="density",
        axis=-1,
    )
    _, pxy = signal.csd(
        x_segments,
        y_segments,
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling="density",
        axis=-1,
    )
    return freqs, np.abs(np.mean(pxy, axis=0)) ** 2 / (
        np.mean(pxx, axis=0) * np.mean(pyy, axis=0)
    )


__all__ = [
    "segment_signal",
    "segmented_coherence",
    "segmented_csd",
    "segmented_psd",
]

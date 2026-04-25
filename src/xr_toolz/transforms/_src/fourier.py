"""Fourier-domain transforms — power spectrum, cross-spectrum, STFT,
coherence.

All entry points are ``DataArray``-first: the input is a single
:class:`xr.DataArray` and the output is also a ``DataArray`` with the
spatial / temporal dims replaced by their frequency counterparts (named
``freq_<dim>`` by :mod:`xrft`). Output names follow a single suffix
scheme so callers can identify spectral results downstream:

    f"{name}_psd"      — multi-D power spectrum
    f"{name}_iso_psd"  — radially-averaged (isotropic) power spectrum
    f"{a}_{b}_csd"     — cross-spectrum
    f"{a}_{b}_coh"     — magnitude-squared coherence
    f"{name}_stft"     — short-time Fourier transform
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
import xrft


_DEFAULT_PSD_KWARGS: dict[str, Any] = {
    "scaling": "density",
    "detrend": "linear",
    "window": "tukey",
    "nfactor": 2,
    "window_correction": True,
    "true_amplitude": True,
    "truncate": True,
}


def _output_name(da: xr.DataArray, suffix: str, fallback: str = "field") -> str:
    base = da.name if da.name is not None else fallback
    return f"{base}_{suffix}"


def power_spectrum(
    da: xr.DataArray,
    dim: str | Sequence[str],
    *,
    isotropic: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Power spectral density of ``da`` along ``dim``.

    Args:
        da: Input field.
        dim: Single dim or sequence of dims to transform. Pass a single
            string for 1-D spectra; a tuple/list for multi-D
            (space-time) spectra.
        isotropic: If ``True``, compute the radially-averaged (isotropic)
            power spectrum. Requires exactly two spatial dims.
        **kwargs: Forwarded to :func:`xrft.power_spectrum` /
            :func:`xrft.isotropic_power_spectrum`. Sensible defaults
            (Tukey window, linear detrend, density scaling) fill in
            keys not provided.

    Returns:
        DataArray of the spectrum, named ``f"{da.name}_psd"`` (or
        ``f"{da.name}_iso_psd"`` if ``isotropic=True``).
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    opts = {**_DEFAULT_PSD_KWARGS, **kwargs}
    if isotropic:
        if len(dims) != 2:
            raise ValueError(
                f"isotropic_power_spectrum requires exactly 2 spatial dims; "
                f"got dim={dims}."
            )
        out = xrft.isotropic_power_spectrum(da, dim=dims, **opts)
        out.name = _output_name(da, "iso_psd")
        return out
    out = xrft.power_spectrum(da, dim=dims, **opts)
    out.name = _output_name(da, "psd")
    return out


def isotropic_power_spectrum(
    da: xr.DataArray,
    dim: Sequence[str],
    **kwargs: Any,
) -> xr.DataArray:
    """Radially-averaged 2-D power spectrum. Convenience alias of
    :func:`power_spectrum` with ``isotropic=True``."""
    return power_spectrum(da, dim=dim, isotropic=True, **kwargs)


def cross_spectrum(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    dim: str | Sequence[str],
    **kwargs: Any,
) -> xr.DataArray:
    """Cross-power spectrum of ``da_a`` and ``da_b`` along ``dim``.

    Returns:
        DataArray named ``f"{a.name}_{b.name}_csd"``.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    opts = {**_DEFAULT_PSD_KWARGS, **kwargs}
    out = xrft.cross_spectrum(da_a, da_b, dim=dims, **opts)
    a_name = da_a.name if da_a.name is not None else "a"
    b_name = da_b.name if da_b.name is not None else "b"
    out.name = f"{a_name}_{b_name}_csd"
    return out


def coherence(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    dim: str | Sequence[str],
    **kwargs: Any,
) -> xr.DataArray:
    """Magnitude-squared coherence ``|S_ab|² / (S_aa · S_bb)``.

    Bounded in ``[0, 1]``: 1 means the two signals share identical
    phase-aligned content at that frequency. Computed from the
    :func:`xrft.cross_spectrum` and individual auto-spectra under the
    same averaging window, so the three quantities are commensurable.

    Returns:
        DataArray named ``f"{a.name}_{b.name}_coh"``.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    opts = {**_DEFAULT_PSD_KWARGS, **kwargs}
    s_aa = xrft.power_spectrum(da_a, dim=dims, **opts)
    s_bb = xrft.power_spectrum(da_b, dim=dims, **opts)
    s_ab = xrft.cross_spectrum(da_a, da_b, dim=dims, **opts)
    coh = (np.abs(s_ab) ** 2) / (s_aa * s_bb)
    a_name = da_a.name if da_a.name is not None else "a"
    b_name = da_b.name if da_b.name is not None else "b"
    coh.name = f"{a_name}_{b_name}_coh"
    return coh


def stft(
    da: xr.DataArray,
    dim: str,
    *,
    window_size: int,
    hop: int | None = None,
    window: str = "tukey",
    detrend: str | None = "linear",
) -> xr.DataArray:
    """Short-time Fourier transform along ``dim``.

    Splits the signal into overlapping windows of length ``window_size``
    spaced by ``hop`` samples (default ``window_size // 2``), computes a
    Fourier transform per window, and stacks the result on a new
    ``segment`` axis whose coordinate is the centre time of each
    window.

    Args:
        da: 1-D signal (along ``dim``) plus arbitrary extra dims that
            are broadcast over.
        dim: Time / sample dimension to slide the window along.
        window_size: Length (in samples) of each STFT window.
        hop: Stride between window starts. Defaults to
            ``window_size // 2`` (50 % overlap).
        window: Window name forwarded to :func:`xrft.power_spectrum`'s
            internal window factory (Tukey by default).
        detrend: Per-window detrending. ``None`` to disable.

    Returns:
        DataArray with dims ``(*outer_dims, segment, freq_<dim>)``,
        named ``f"{da.name}_stft"``.
    """
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2; got {window_size}.")
    if hop is None:
        hop = window_size // 2 if window_size > 1 else 1
    if hop < 1:
        raise ValueError(f"hop must be >= 1; got {hop}.")

    n_total = da.sizes[dim]
    if window_size > n_total:
        raise ValueError(f"window_size={window_size} exceeds {dim!r} length {n_total}.")

    starts = np.arange(0, n_total - window_size + 1, hop, dtype=int)
    if starts.size == 0:
        raise ValueError(
            f"No STFT windows fit: window_size={window_size}, hop={hop}, n={n_total}."
        )

    # Use FFT (complex) per segment for true STFT, not power spectrum,
    # so callers can square / phase as they see fit.
    segments = []
    coord = da[dim] if dim in da.coords else None
    centres = []
    for s in starts:
        segment = da.isel({dim: slice(int(s), int(s) + window_size)})
        spec = xrft.fft(
            segment, dim=[dim], window=window, detrend=detrend, true_amplitude=False
        )
        segments.append(spec)
        if coord is not None:
            centres.append(coord.isel({dim: int(s) + window_size // 2}).values.item())
        else:
            centres.append(int(s) + window_size // 2)

    out = xr.concat(segments, dim="segment")
    if centres:
        out = out.assign_coords(segment=("segment", np.asarray(centres)))
    out.name = _output_name(da, "stft")
    return out


def drop_negative_frequencies[T: xr.DataArray | xr.Dataset](
    da: T,
    dims: Sequence[str],
    *,
    drop: bool = True,
) -> T:
    """Mean along ``dims`` after restricting every other freq axis to its
    positive half.

    Used to collapse a multi-D spectrum onto a subset of frequency axes
    without contaminating the average with the zero / negative halves
    of the remaining axes. This is the renamed and clarified form of
    the historical ``conditional_average``.
    """
    dims = list(dims)
    remaining = [d for d in da.dims if d not in dims]
    if not remaining:
        return da.mean(dim=dims)

    cond = da[remaining[0]] > 0.0
    for d in remaining[1:]:
        cond = cond & (da[d] > 0.0)
    return da.mean(dim=dims).where(cond, drop=drop)

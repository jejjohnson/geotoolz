"""Fourier-domain transforms — power spectrum, cross-spectrum, STFT,
coherence, rotary spectra, and spectral flux diagnostics.

All entry points are ``DataArray``-first: the input is a single
:class:`xr.DataArray` and the output is also a ``DataArray`` with the
spatial / temporal dims replaced by their frequency counterparts (named
``freq_<dim>`` by :mod:`xrft`). Output names follow a single suffix
scheme so callers can identify spectral results downstream:

    f"{name}_psd"      — multi-D power spectrum
    f"{name}_iso_psd"  — radially-averaged (isotropic) power spectrum
    f"{a}_{b}_csd"     — cross-spectrum
    f"{a}_{b}_coh"     — magnitude-squared coherence
    "psd_cw" / "psd_ccw" — rotary power-spectrum components
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
# Keep shells with mathematically identical radii together despite roundoff.
_RADIAL_BINNING_PRECISION = 12


def _output_name(da: xr.DataArray, suffix: str, fallback: str = "field") -> str:
    base = da.name if da.name is not None else fallback
    return f"{base}_{suffix}"


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _validate_spatial_dims(dim: Sequence[str]) -> list[str]:
    dims = list(dim)
    if len(dims) != 2:
        raise ValueError(
            f"spectral flux diagnostics require exactly 2 dims; got {dims}."
        )
    return dims


def _fft2(
    da: xr.DataArray,
    dims: Sequence[str],
    *,
    window: str | None,
    detrend: str | None,
) -> xr.DataArray:
    return xrft.fft(
        da,
        dim=list(dims),
        shift=False,
        window=window,
        detrend=detrend,
        true_phase=False,
        true_amplitude=False,
    )


def _ifft2(
    da: xr.DataArray, freq_dims: Sequence[str], template: xr.DataArray | None = None
) -> xr.DataArray:
    out = xrft.ifft(
        da,
        dim=list(freq_dims),
        shift=False,
        true_phase=False,
        true_amplitude=False,
    ).real
    if template is None:
        return out
    coords = {dim: template[dim] for dim in template.dims if dim in out.dims}
    return out.assign_coords(coords)


def _gradient_from_hat(
    field_hat: xr.DataArray,
    freq_dim: str,
    freq_dims: Sequence[str],
    template: xr.DataArray | None = None,
) -> xr.DataArray:
    return _ifft2(2.0j * np.pi * field_hat[freq_dim] * field_hat, freq_dims, template)


def _radial_sum(field: xr.DataArray, freq_dims: Sequence[str]) -> xr.DataArray:
    freq_x, freq_y = freq_dims
    kx, ky = xr.broadcast(field[freq_x], field[freq_y])
    freq_r = np.round(np.hypot(kx.values, ky.values).ravel(), _RADIAL_BINNING_PRECISION)
    stacked = field.stack(_freq_shell=list(freq_dims))
    stacked = stacked.assign_coords(freq_r=("_freq_shell", freq_r))
    out = stacked.groupby("freq_r").sum("_freq_shell").sortby("freq_r")
    return out


def _flux_from_transfer(transfer: xr.DataArray) -> xr.DataArray:
    flux = (
        transfer.isel(freq_r=slice(None, None, -1))
        .cumsum("freq_r")
        .isel(freq_r=slice(None, None, -1))
    )
    return flux.assign_coords(freq_r=transfer["freq_r"])


def _fourier_uv_gradients(
    u: xr.DataArray,
    v: xr.DataArray,
    dims: Sequence[str],
    *,
    window: str | None,
    detrend: str | None,
) -> tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray
]:
    dims = list(dims)
    freq_dims = [f"freq_{dim}" for dim in dims]
    u_hat = _fft2(u, dims, window=window, detrend=detrend)
    v_hat = _fft2(v, dims, window=window, detrend=detrend)
    du_dx = _gradient_from_hat(u_hat, freq_dims[0], freq_dims, u)
    du_dy = _gradient_from_hat(u_hat, freq_dims[1], freq_dims, u)
    dv_dx = _gradient_from_hat(v_hat, freq_dims[0], freq_dims, v)
    dv_dy = _gradient_from_hat(v_hat, freq_dims[1], freq_dims, v)
    return u_hat, v_hat, du_dx, du_dy, dv_dx, dv_dy


def _spectral_flux_dataset(
    transfer_2d: xr.DataArray,
    freq_dims: Sequence[str],
    *,
    avg_dims: Sequence[str] | None,
    return_2d: bool,
) -> xr.Dataset:
    avg = _as_list(avg_dims)
    if avg:
        transfer_2d = transfer_2d.mean(dim=avg)
    transfer = _radial_sum(transfer_2d, freq_dims)
    transfer.name = "transfer"
    flux = _flux_from_transfer(transfer)
    flux.name = "flux"
    data_vars: dict[str, xr.DataArray] = {"transfer": transfer, "flux": flux}
    if return_2d:
        # transfer_2d is already named upstream; just attach it.
        data_vars["transfer_2d"] = transfer_2d
    return xr.Dataset(data_vars)


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


def ke_spectral_flux(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    dim: Sequence[str],
    window: str | None = "tukey",
    detrend: str | None = "linear",
    avg_dims: str | Sequence[str] | None = None,
    return_2d: bool = False,
) -> xr.Dataset:
    """Kinetic-energy spectral flux ``Π(k)``.

    Args:
        u: Zonal velocity component with both spatial dimensions in ``dim``.
        v: Meridional velocity component on the same grid as ``u``.
        dim: Two spatial dimensions to Fourier transform, for example
            ``("x", "y")``.
        window: Optional window forwarded to :func:`xrft.fft`.
        detrend: Optional detrending forwarded to :func:`xrft.fft`.
        avg_dims: Optional non-spectral dimensions to average before radial
            integration, for example ``"time"``.
        return_2d: If ``True``, include the unbinned two-dimensional transfer
            field as ``transfer_2d``.

    Returns:
        Dataset with ``transfer`` as radially summed KE transfer ``T(k)`` and
        ``flux`` as ``Π(k) = Σ_{k' >= k} T(k')``. ``transfer_2d`` is included
        when requested.

    Notes:
        Positive ``flux`` denotes downscale kinetic-energy transfer. The
        nonlinear advection terms are formed in physical space from
        Fourier-space gradients, transformed back to spectral space, radially
        binned by ``freq_r``, and accumulated from high to low wavenumber.
    """
    dims = _validate_spatial_dims(dim)
    freq_dims = [f"freq_{name}" for name in dims]
    u_hat, v_hat, du_dx, du_dy, dv_dx, dv_dy = _fourier_uv_gradients(
        u, v, dims, window=window, detrend=detrend
    )
    # Reconstruct physical-space velocities from the (possibly windowed and
    # detrended) spectra so the advection product mixes fields with matching
    # preprocessing — using raw u/v here would bias the transfer when
    # window/detrend are non-None (which they are by default).
    u_proc = _ifft2(u_hat, freq_dims, u)
    v_proc = _ifft2(v_hat, freq_dims, v)
    phi_u = u_proc * du_dx + v_proc * du_dy
    phi_v = u_proc * dv_dx + v_proc * dv_dy
    phi_u_hat = _fft2(phi_u, dims, window=None, detrend=None)
    phi_v_hat = _fft2(phi_v, dims, window=None, detrend=None)
    norm = float(np.prod([u.sizes[name] for name in dims]) ** 2)
    transfer_2d = (
        -np.real(np.conj(u_hat) * phi_u_hat + np.conj(v_hat) * phi_v_hat) / norm
    )
    transfer_2d.name = "transfer_2d"
    return _spectral_flux_dataset(
        transfer_2d, freq_dims, avg_dims=_as_list(avg_dims), return_2d=return_2d
    )


def enstrophy_spectral_flux(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    dim: Sequence[str],
    window: str | None = "tukey",
    detrend: str | None = "linear",
    avg_dims: str | Sequence[str] | None = None,
    return_2d: bool = False,
) -> xr.Dataset:
    """Enstrophy spectral flux ``Π_Z(k)``.

    Args:
        u: Zonal velocity component with both spatial dimensions in ``dim``.
        v: Meridional velocity component on the same grid as ``u``.
        dim: Two spatial dimensions to Fourier transform.
        window: Optional window forwarded to :func:`xrft.fft`.
        detrend: Optional detrending forwarded to :func:`xrft.fft`.
        avg_dims: Optional non-spectral dimensions to average before radial
            integration.
        return_2d: If ``True``, include the unbinned two-dimensional transfer
            field as ``transfer_2d``.

    Returns:
        Dataset with radially summed enstrophy ``transfer`` and cumulative
        ``flux``. ``transfer_2d`` is included when requested.

    Notes:
        Vorticity is computed spectrally as ``ζ̂ = 2πi(k_x v̂ - k_y û)``.
        Positive ``flux`` denotes downscale enstrophy transfer, accumulated
        from high to low radial wavenumber with the same convention as
        :func:`ke_spectral_flux`.
    """
    dims = _validate_spatial_dims(dim)
    freq_dims = [f"freq_{name}" for name in dims]
    u_hat, v_hat, _du_dx, _du_dy, _dv_dx, _dv_dy = _fourier_uv_gradients(
        u, v, dims, window=window, detrend=detrend
    )
    kx = v_hat[freq_dims[0]]
    ky = u_hat[freq_dims[1]]
    zeta_hat = 2.0j * np.pi * (kx * v_hat - ky * u_hat)
    dzeta_dx = _gradient_from_hat(zeta_hat, freq_dims[0], freq_dims, u)
    dzeta_dy = _gradient_from_hat(zeta_hat, freq_dims[1], freq_dims, u)
    # Match the windowed/detrended preprocessing of zeta_hat in the
    # advection product — see ke_spectral_flux for the same fix.
    u_proc = _ifft2(u_hat, freq_dims, u)
    v_proc = _ifft2(v_hat, freq_dims, v)
    adv_zeta = u_proc * dzeta_dx + v_proc * dzeta_dy
    adv_zeta_hat = _fft2(adv_zeta, dims, window=None, detrend=None)
    norm = float(np.prod([u.sizes[name] for name in dims]) ** 2)
    transfer_2d = -np.real(np.conj(zeta_hat) * adv_zeta_hat) / norm
    transfer_2d.name = "transfer_2d"
    return _spectral_flux_dataset(
        transfer_2d, freq_dims, avg_dims=_as_list(avg_dims), return_2d=return_2d
    )


def integral_scale(
    psd: xr.DataArray,
    *,
    wavenumber_dim: str = "freq_r",
    moment: int = 1,
) -> xr.DataArray:
    """Energy-weighted integral scale or Taylor microscale.

    Args:
        psd: One-dimensional spectrum with a wavenumber coordinate.
        wavenumber_dim: Name of the wavenumber dimension.
        moment: ``1`` for the integral scale, ``2`` for the Taylor microscale.

    Returns:
        DataArray with ``wavenumber_dim`` removed.

    Notes:
        ``moment=1`` returns ``∫ψ dk / ∫kψ dk``. ``moment=2`` returns the
        Taylor microscale ``λ = sqrt(∫ψ dk / ∫k²ψ dk)``. Integrals use
        xarray's coordinate-aware trapezoidal rule, so non-uniform
        wavenumber grids (e.g. radially binned spectra) are handled
        correctly.
    """
    if moment not in (1, 2):
        raise ValueError(
            f"moment must be 1 (integral scale) or 2 (Taylor microscale); got {moment}."
        )
    k = psd[wavenumber_dim]
    numerator = psd.integrate(coord=wavenumber_dim)
    denominator = (psd * k**moment).integrate(coord=wavenumber_dim)
    out = numerator / denominator
    if moment == 2:
        out = out**0.5
    out.name = _output_name(psd, f"moment{moment}_scale", fallback="spectrum")
    return out


def fit_spectral_slope(
    psd: xr.DataArray,
    *,
    wavenumber_dim: str = "freq_r",
    k_min: float,
    k_max: float,
) -> tuple[float, float]:
    """Fit ``log(psd) = slope * log(k) + intercept`` over ``[k_min, k_max]``.

    Args:
        psd: One-dimensional positive spectrum.
        wavenumber_dim: Name of the wavenumber dimension.
        k_min: Lower inclusive wavenumber bound.
        k_max: Upper inclusive wavenumber bound.

    Returns:
        ``(slope, intercept)`` from ``numpy.polyfit`` in log-log space.

    Raises:
        ValueError: If fewer than two positive finite samples remain in the
            requested fit window.

    Notes:
        Use the fitted slope to identify inertial-range power laws such as
        ``-5/3`` inverse-cascade or ``-3`` enstrophy-cascade scaling.
    """
    k = psd[wavenumber_dim]
    subset = psd.where((k >= k_min) & (k <= k_max) & (k > 0) & (psd > 0), drop=True)
    x = np.log(subset[wavenumber_dim].values)
    y = np.log(subset.values)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError(
            f"At least two positive finite spectral samples required in "
            f"[{k_min}, {k_max}]; found {finite.sum()}."
        )
    slope, intercept = np.polyfit(x[finite], y[finite], 1)
    return float(slope), float(intercept)


def compensated_spectrum(
    psd: xr.DataArray,
    *,
    wavenumber_dim: str = "freq_r",
    exponent: float,
) -> xr.DataArray:
    """Return ``psd * k**exponent`` for inertial-range compensation.

    Args:
        psd: Spectrum to compensate.
        wavenumber_dim: Name of the wavenumber dimension.
        exponent: Power-law exponent, for example ``5 / 3`` for Kolmogorov
            scaling.

    Returns:
        DataArray named ``f"{psd.name}_compensated"`` that is flat where
        ``psd ∝ k^{-exponent}``.
    """
    out = psd * psd[wavenumber_dim] ** exponent
    out.name = _output_name(psd, "compensated", fallback="spectrum")
    return out


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


def rotary_spectrum(
    ds: xr.Dataset,
    *,
    u_var: str,
    v_var: str,
    dim: str,
    avg_dims: str | Sequence[str] | None = None,
) -> xr.Dataset:
    """Rotary power spectrum from horizontal velocity components.

    Computes the two-sided FFT of the complex velocity ``u + i v``, splits
    positive and negative wavenumbers into counter-clockwise and clockwise
    components, then folds both onto a shared positive ``wavenumber`` axis.
    The unnormalized FFT is converted to a density with ``dx / n`` scaling so
    integrating ``psd_cw + psd_ccw`` over wavenumber recovers velocity variance.
    Polarization is NaN where total rotary power is negligible.

    Args:
        ds: Dataset containing horizontal velocity components.
        u_var: Zonal/eastward velocity variable.
        v_var: Meridional/northward velocity variable.
        dim: Dimension to Fourier transform.
        avg_dims: Optional dimension or dimensions to average in each output.

    Returns:
        Dataset with ``psd_ccw``, ``psd_cw``, and ``polarization`` where
        ``polarization = (psd_cw - psd_ccw) / (psd_cw + psd_ccw)``.
    """
    if dim not in ds[u_var].dims or dim not in ds[v_var].dims:
        raise ValueError(
            f"dim={dim!r} must be present on both {u_var!r} and {v_var!r}."
        )

    w = ds[u_var] + 1j * ds[v_var]
    # Rectangular window + mean-only detrending keeps the dx / n density scaling
    # Parseval-consistent for variance.
    spec = xrft.fft(w, dim=[dim], window=None, detrend="constant", true_amplitude=False)
    freq_dim = f"freq_{dim}"
    # Datasets can legally have dims without explicit coord variables;
    # fall back to unit spacing in that case rather than KeyError-ing.
    coord = ds[dim] if dim in ds.coords else None
    density_scale = (_coord_spacing(coord) if coord is not None else 1.0) / ds.sizes[
        dim
    ]
    psd = (np.abs(spec) ** 2) * density_scale

    psd_ccw = psd.where(psd[freq_dim] > 0.0, drop=True).rename({freq_dim: "wavenumber"})
    neg = psd.where(psd[freq_dim] < 0.0, drop=True)
    psd_cw = (
        neg.assign_coords({freq_dim: np.abs(neg[freq_dim])})
        .sortby(freq_dim)
        .rename({freq_dim: "wavenumber"})
    )
    # Union the two folds and fill missing bins with 0 so an even-length
    # FFT's lone -Nyquist bin survives, keeping psd_cw + psd_ccw
    # Parseval-consistent with the input variance.
    psd_ccw, psd_cw = xr.align(psd_ccw, psd_cw, join="outer", fill_value=0.0)
    psd_ccw.name = "psd_ccw"
    psd_cw.name = "psd_cw"

    denom = psd_cw + psd_ccw
    polarization = ((psd_cw - psd_ccw) / denom).where(
        denom > _polarization_epsilon(denom)
    )
    polarization.name = "polarization"
    out = xr.Dataset(
        {"psd_ccw": psd_ccw, "psd_cw": psd_cw, "polarization": polarization}
    )
    if avg_dims is not None:
        dims = [avg_dims] if isinstance(avg_dims, str) else list(avg_dims)
        out = out.mean(dim=dims)
    return out


def _coord_spacing(coord: xr.DataArray) -> float:
    if coord.size < 2:
        # A singleton axis has no resolvable sample spacing; use unit spacing so
        # callers still get deterministic density scaling.
        return 1.0
    values = np.asarray(coord.values)
    if np.issubdtype(values.dtype, np.datetime64):
        diffs = np.diff(values).astype("timedelta64[ns]").astype(float) / 1e9
    else:
        diffs = np.diff(values.astype(float))
    return float(np.median(np.abs(diffs)))


def _polarization_epsilon(da: xr.DataArray) -> float:
    dtype = da.dtype
    if np.issubdtype(dtype, np.floating):
        return float(np.finfo(dtype).eps)
    return float(np.finfo(float).eps)


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

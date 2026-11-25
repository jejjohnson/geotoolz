"""Fourier-domain transforms — power spectrum, cross-spectrum, STFT,
coherence, rotary spectra, and spectral flux diagnostics.

The spectra are ``diagnostix.spectra``'s, on the dims of a
:class:`xr.DataArray`: the output has the spatial / temporal dims replaced by
their frequency counterparts, named ``freq_<dim>`` (``freq_r`` for a radial
one), in cycles per unit of the coordinate (cycles per second for dates). The
other dims stay, and stay lazy on dask, where a transformed dim becomes one
chunk. Output names follow a single suffix scheme so callers can identify
spectral results downstream:

    f"{name}_psd"      — power spectrum along a dim, or over two
    f"{name}_iso_psd"  — isotropic power spectrum
    f"{a}_{b}_csd"     — cross-spectrum
    f"{a}_{b}_coh"     — magnitude-squared coherence
    "psd_cw" / "psd_ccw" — rotary power-spectrum components
    f"{name}_stft"     — short-time Fourier transform
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from diagnostix import spectra
from jaxtyping import Float
from scipy import signal


type Window = Literal["hann"] | Float[np.ndarray, "..."] | None
type Detrend = Literal["constant", "linear"] | None


def _output_name(da: xr.DataArray, suffix: str, fallback: str = "field") -> str:
    base = da.name if da.name is not None else fallback
    return f"{base}_{suffix}"


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _positions(da: xr.DataArray, dim: str) -> Float[np.ndarray, " n"]:
    """The coordinate of a dim as floats, dates in seconds.

    Raises:
        ValueError: If there is no coordinate, or fewer than two samples, or
            they are not evenly spaced.
    """
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    if dim not in da.coords:
        raise ValueError(f"dim {dim!r} has no coordinate to measure its spacing.")
    values = np.asarray(da[dim].values)
    if values.size < 2:
        raise ValueError(f"coord {dim!r} must contain at least two samples.")
    if np.issubdtype(values.dtype, np.datetime64):
        # from the first, so the integers of the nanoseconds stay exact
        offsets = (values - values[0]).astype("timedelta64[ns]").astype(np.int64)
        out = offsets.astype(np.float64) / 1e9
    elif values.dtype == object:
        import cftime

        out = cftime.date2num(values, "seconds since 1800-01-01", values[0].calendar)
        out = np.asarray(out, dtype=np.float64)
    else:
        out = values.astype(np.float64)
    steps = np.diff(out)
    if steps[0] == 0.0 or not np.allclose(steps, steps[0], rtol=1e-3, atol=0.0):
        raise ValueError(
            f"coord {dim!r} is not evenly spaced; resample onto a regular grid first."
        )
    return out


def _spacing(da: xr.DataArray, dim: str) -> float:
    positions = _positions(da, dim)
    return float(abs(positions[-1] - positions[0]) / (positions.size - 1))


def _transform(
    kernel: Callable[..., tuple[Any, ...]],
    *fields: xr.DataArray,
    core: Sequence[str],
    out: dict[str, Any],
    dtypes: Sequence[Any],
    replace: dict[str, str] | None = None,
) -> tuple[xr.DataArray, ...]:
    """``kernel`` over the ``core`` dims, each result on the dims of ``out``.

    Args:
        kernel: Takes arrays that end in ``core`` and returns a tuple of arrays
            that end in ``out``.
        *fields: The fields.
        core: The dims of the transform; a field must have them all.
        out: The wavenumbers of each new dim.
        dtypes: The dtype of each result.
        replace: The new dim that takes the place of an old one; the others go
            last.
    """

    def run(*arrays: Any) -> Any:
        made = kernel(*arrays)
        return made if len(made) > 1 else made[0]

    results = xr.apply_ufunc(
        run,
        *fields,
        input_core_dims=[list(core)] * len(fields),
        output_core_dims=[list(out)] * len(dtypes),
        dask="parallelized",
        output_dtypes=list(dtypes),
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {name: np.size(k) for name, k in out.items()},
        },
    )
    results = results if isinstance(results, tuple) else (results,)
    named = replace or {}
    order: dict[Hashable, None] = {}
    for dim in (d for field in fields for d in field.dims):
        if dim in named:
            order[named[str(dim)]] = None
        elif dim not in core:
            order[dim] = None
    order.update(dict.fromkeys(out))
    return tuple(
        r.assign_coords(
            {name: (name, np.asarray(k)) for name, k in out.items()}
        ).transpose(*order)
        for r in results
    )


def _real(*fields: xr.DataArray) -> np.dtype:
    return np.result_type(*(f.dtype for f in fields), np.float32)


def power_spectrum(
    da: xr.DataArray,
    dim: str | Sequence[str],
    *,
    isotropic: bool = False,
    window: Window = "hann",
    detrend: Detrend = "linear",
) -> xr.DataArray:
    """Power spectral density of ``da`` along one dim, or over two.

    The sum of a spectrum times the step of its wavenumbers is the variance of
    the detrended field, on average where there is a taper.

    Args:
        da: Input field.
        dim: One dim, for the one-sided spectrum from wavenumber 0, or two, for
            the two-sided spectrum over both (space-time spectra).
        isotropic: If ``True``, the sum over each ring of wavenumbers of two
            dims (``diagnostix.spectra.isotropic_spectrum``).
        window: ``"hann"``, an array of weights with one per sample of the
            dims, or ``None``.
        detrend: What to take away first: ``"linear"`` (the line or plane),
            ``"constant"`` (the mean) or ``None``.

    Returns:
        DataArray of the spectrum, named ``f"{da.name}_psd"`` (or
        ``f"{da.name}_iso_psd"`` if ``isotropic=True``), on ``freq_<dim>`` in
        place of each dim, or on ``freq_r``. A NaN in a field makes its
        spectrum NaN.

    Raises:
        ValueError: If there are more than two dims, ``isotropic`` has not
            exactly two, or a coordinate is missing or not evenly spaced.
    """
    dims = [dim] if isinstance(dim, str) else list(dim)
    if isotropic and len(dims) != 2:
        raise ValueError(
            f"isotropic_power_spectrum requires exactly 2 spatial dims; got dim={dims}."
        )
    if len(dims) not in (1, 2):
        raise ValueError(f"power_spectrum takes one dim or two; got dim={dims}.")
    steps = [_spacing(da, d) for d in dims]
    shape = tuple(da.sizes[d] for d in dims)
    options: dict[str, Any] = {"window": window, "detrend": detrend}
    blank = np.zeros(shape)
    if isotropic:

        def kernel(x: Any) -> tuple[Any, ...]:
            return (spectra.isotropic_spectrum(x, steps, **options).power,)

        out = {"freq_r": spectra.isotropic_spectrum(blank, steps).wavenumber}
        replace: dict[str, str] = {}
    elif len(dims) == 1:

        def kernel(x: Any) -> tuple[Any, ...]:
            return (spectra.power_spectrum(x, steps[0], **options).power,)

        out = {f"freq_{dims[0]}": spectra.power_spectrum(blank, steps[0]).wavenumber}
        replace = {dims[0]: f"freq_{dims[0]}"}
    else:

        def kernel(x: Any) -> tuple[Any, ...]:
            return (spectra.power_spectrum_2d(x, steps, **options).power,)

        plane = spectra.power_spectrum_2d(blank, steps)
        out = {
            f"freq_{dims[0]}": plane.wavenumber_y,
            f"freq_{dims[1]}": plane.wavenumber_x,
        }
        replace = {d: f"freq_{d}" for d in dims}
    (psd,) = _transform(
        kernel, da, core=dims, out=out, dtypes=[_real(da)], replace=replace
    )
    return psd.rename(_output_name(da, "iso_psd" if isotropic else "psd"))


def isotropic_power_spectrum(
    da: xr.DataArray,
    dim: Sequence[str],
    **kwargs: Any,
) -> xr.DataArray:
    """Isotropic 2-D power spectrum. Convenience alias of
    :func:`power_spectrum` with ``isotropic=True``."""
    return power_spectrum(da, dim=dim, isotropic=True, **kwargs)


def ke_spectral_flux(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    dim: Sequence[str],
    window: Window = None,
    detrend: Detrend = None,
    avg_dims: str | Sequence[str] | None = None,
) -> xr.Dataset:
    """Kinetic-energy spectral flux ``Π(k)``, positive towards small scales.

    The advection is dealiased, so the transfer of a non-divergent velocity in
    a periodic box sums to 0 (``diagnostix.spectra.kinetic_energy_flux``). A
    taper makes no periodic box out of another: a tapered velocity is
    divergent, and its transfer does not sum to 0.

    Args:
        u: Velocity component along the first of ``dim``.
        v: Velocity component along the second, on the same grid.
        dim: The ``(x, y)`` dims to transform.
        window: Taper, as in :func:`power_spectrum`; none by default.
        detrend: As in :func:`power_spectrum`; none by default.
        avg_dims: Optional non-spectral dims to average over, such as
            ``"time"``.

    Returns:
        Dataset with ``transfer``, the gain of each ring of wavenumbers
        ``freq_r``, and ``flux``, ``Π(k) = Σ_{k' >= k} T(k')``.
    """
    return _spectral_flux(
        spectra.kinetic_energy_flux, u, v, dim, window, detrend, avg_dims
    )


def enstrophy_spectral_flux(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    dim: Sequence[str],
    window: Window = None,
    detrend: Detrend = None,
    avg_dims: str | Sequence[str] | None = None,
) -> xr.Dataset:
    """Enstrophy spectral flux ``Π_Z(k)``, positive towards small scales.

    Otherwise as :func:`ke_spectral_flux`, with the vorticity ``∂v/∂x − ∂u/∂y``
    (``diagnostix.spectra.enstrophy_flux``).

    Args:
        u: Velocity component along the first of ``dim``.
        v: Velocity component along the second, on the same grid.
        dim: The ``(x, y)`` dims to transform.
        window: Taper, as in :func:`power_spectrum`; none by default.
        detrend: As in :func:`power_spectrum`; none by default.
        avg_dims: Optional non-spectral dims to average over.

    Returns:
        Dataset with ``transfer`` and ``flux`` on ``freq_r``.
    """
    return _spectral_flux(spectra.enstrophy_flux, u, v, dim, window, detrend, avg_dims)


def _spectral_flux(
    flux: Callable[..., spectra.SpectralFlux],
    u: xr.DataArray,
    v: xr.DataArray,
    dim: Sequence[str],
    window: Window,
    detrend: Detrend,
    avg_dims: str | Sequence[str] | None,
) -> xr.Dataset:
    dims = list(dim)
    if len(dims) != 2:
        raise ValueError(
            f"spectral flux diagnostics require exactly 2 dims; got {dims}."
        )
    # (y, x), each growing: the kernel's derivatives point with the coordinates
    core = dims[::-1]
    for d in core:
        if u[d][0] > u[d][-1]:
            u, v = (da.isel({d: slice(None, None, -1)}) for da in (u, v))
    steps = [_spacing(u, d) for d in core]

    def kernel(u_: Any, v_: Any) -> tuple[Any, ...]:
        out = flux(u_, v_, steps, window=window, detrend=detrend)
        return out.transfer, out.flux

    blank = np.zeros([u.sizes[d] for d in core])
    rings = flux(blank, blank, steps).wavenumber
    transfer, through = _transform(
        kernel, u, v, core=core, out={"freq_r": rings}, dtypes=[_real(u, v)] * 2
    )
    out = xr.Dataset({"transfer": transfer, "flux": through})
    avg = _as_list(avg_dims)
    return out.mean(dim=avg) if avg else out


def _even_spectrum(
    function: Callable[[spectra.Spectrum], Any], psd: xr.DataArray, dim: str
) -> xr.DataArray:
    """A number from each spectrum along ``dim``, of evenly spaced wavenumbers."""
    k = np.asarray(psd[dim].values, dtype=np.float64)
    steps = np.diff(k)
    if not np.allclose(steps, steps[:1], rtol=1e-6, atol=0.0):
        raise ValueError(f"the wavenumbers of {dim!r} must be evenly spaced.")
    k = k.astype(_real(psd))
    return xr.apply_ufunc(
        lambda power: function(spectra.Spectrum(k, power)),
        psd,
        input_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[_real(psd)],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def integral_scale(
    psd: xr.DataArray,
    *,
    wavenumber_dim: str = "freq_r",
    moment: int = 1,
) -> xr.DataArray:
    """Energy-weighted integral scale or Taylor microscale.

    Args:
        psd: Spectrum with a coordinate of evenly spaced wavenumbers.
        wavenumber_dim: Name of the wavenumber dimension.
        moment: ``1`` for the integral scale, ``2`` for the Taylor microscale.

    Returns:
        DataArray with ``wavenumber_dim`` removed.

    Raises:
        ValueError: If ``moment`` is neither, or the wavenumbers are uneven.

    Notes:
        ``(Σψ / Σkᵐψ)^(1/m)`` over the bins of the spectrum
        (``diagnostix.spectra.integral_scale``): a wavelength, with no 2π.
    """
    if moment not in (1, 2):
        raise ValueError(
            f"moment must be 1 (integral scale) or 2 (Taylor microscale); got {moment}."
        )
    out = _even_spectrum(
        lambda spectrum: spectra.integral_scale(spectrum, moment=moment),
        psd,
        wavenumber_dim,
    )
    return out.rename(_output_name(psd, f"moment{moment}_scale", fallback="spectrum"))


def fit_spectral_slope(
    psd: xr.DataArray,
    *,
    wavenumber_dim: str = "freq_r",
    k_min: float,
    k_max: float,
) -> xr.DataArray:
    """The slope ``s`` of ``psd ∝ kˢ`` over ``[k_min, k_max]``.

    The least-squares line of ``log(psd)`` against ``log(k)``
    (``diagnostix.spectra.spectral_slope``). Use it to identify inertial-range
    power laws such as ``-5/3`` inverse-cascade or ``-3`` enstrophy-cascade
    scaling.

    Args:
        psd: Spectrum with a wavenumber coordinate.
        wavenumber_dim: Name of the wavenumber dimension.
        k_min: Lower inclusive wavenumber bound.
        k_max: Upper inclusive wavenumber bound.

    Returns:
        The slope, with ``wavenumber_dim`` removed, NaN where a bin of the band
        has no power.

    Raises:
        ValueError: If the band holds fewer than two wavenumbers.
    """
    k = np.asarray(psd[wavenumber_dim].values, dtype=np.float64)
    count = int(np.count_nonzero((k >= k_min) & (k <= k_max) & (k > 0)))
    if count < 2:
        raise ValueError(
            f"At least two positive wavenumbers required in [{k_min}, {k_max}]; "
            f"found {count}."
        )
    out = xr.apply_ufunc(
        lambda power: spectra.spectral_slope(
            spectra.Spectrum(k.astype(power.dtype), power), k_min, k_max
        ),
        psd.astype(_real(psd)),
        input_core_dims=[[wavenumber_dim]],
        dask="parallelized",
        output_dtypes=[_real(psd)],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return out.rename(_output_name(psd, "slope", fallback="spectrum"))


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
    dim: str,
    *,
    segment: int | None = None,
    overlap: float = 0.5,
    window: Window = "hann",
    detrend: Detrend = "constant",
) -> xr.DataArray:
    """Cross-power spectrum of ``da_a`` and ``da_b`` along ``dim``.

    Args:
        da_a: One series; its phase is positive where it leads.
        da_b: The other series.
        dim: The dim to transform.
        segment: Length in samples of the segments to average over; the whole
            series by default.
        overlap: The share of a segment that the next one repeats.
        window: As in :func:`power_spectrum`, one weight per sample of a
            segment.
        detrend: As in :func:`power_spectrum`, per segment.

    Returns:
        Complex DataArray named ``f"{a.name}_{b.name}_csd"``, one-sided on
        ``freq_<dim>``; the sum of its real part times the step is the
        covariance.
    """
    step = _spacing(da_a, dim)
    options: dict[str, Any] = {
        "segment": segment,
        "overlap": overlap,
        "window": window,
        "detrend": detrend,
    }

    def kernel(a: Any, b: Any) -> tuple[Any, ...]:
        return (spectra.cross_spectrum(a, b, step, **options).cross,)

    blank = np.zeros(da_a.sizes[dim])
    k = spectra.cross_spectrum(blank, blank, step, segment=segment).wavenumber
    (csd,) = _transform(
        kernel,
        da_a,
        da_b,
        core=[dim],
        out={f"freq_{dim}": k},
        dtypes=[np.result_type(_real(da_a, da_b), np.complex64)],
        replace={dim: f"freq_{dim}"},
    )
    return csd.rename(_pair_name(da_a, da_b, "csd"))


def coherence(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    dim: str,
    *,
    average: str | Sequence[str] | None = None,
    segment: int | None = None,
    overlap: float = 0.5,
    window: Window = "hann",
    detrend: Detrend = "constant",
) -> xr.DataArray:
    """Magnitude-squared coherence ``|⟨S_ab⟩|² / (⟨S_aa⟩ ⟨S_bb⟩)``.

    Bounded in ``[0, 1]``: 1 means the two signals share identical
    phase-aligned content at that frequency. Without an average it is 1 for any
    two signals, so one is required; independent noise gives 1 over the number
    of independent transforms.

    Args:
        da_a: One series.
        da_b: The other series.
        dim: The dim to transform.
        average: Dims of realisations to average over.
        segment: Length in samples of the segments to average over.
        overlap: The share of a segment that the next one repeats.
        window: As in :func:`cross_spectrum`.
        detrend: As in :func:`cross_spectrum`.

    Returns:
        DataArray named ``f"{a.name}_{b.name}_coh"``, one-sided on
        ``freq_<dim>``, without the averaged dims.

    Raises:
        ValueError: If ``average`` and ``segment`` give one transform only.
    """
    step = _spacing(da_a, dim)
    over = _as_list(average)
    da_a, da_b = xr.broadcast(da_a, da_b)
    options: dict[str, Any] = {
        "segment": segment,
        "overlap": overlap,
        "window": window,
        "detrend": detrend,
    }

    def kernel(a: Any, b: Any) -> tuple[Any, ...]:
        axes = tuple(range(-1 - len(over), -1)) or None
        out = spectra.coherence(a, b, step, average=axes, **options)
        return (out.coherence,)

    blank = np.zeros(da_a.sizes[dim])
    k = spectra.cross_spectrum(blank, blank, step, segment=segment).wavenumber
    (coh,) = _transform(
        kernel,
        da_a,
        da_b,
        core=[*over, dim],
        out={f"freq_{dim}": k},
        dtypes=[_real(da_a, da_b)],
        replace={dim: f"freq_{dim}"},
    )
    return coh.rename(_pair_name(da_a, da_b, "coh"))


def _pair_name(da_a: xr.DataArray, da_b: xr.DataArray, suffix: str) -> str:
    a = da_a.name if da_a.name is not None else "a"
    b = da_b.name if da_b.name is not None else "b"
    return f"{a}_{b}_{suffix}"


def rotary_spectrum(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    dim: str,
    avg_dims: str | Sequence[str] | None = None,
) -> xr.Dataset:
    """Rotary power spectrum from horizontal velocity components.

    The variance of ``u + i v``, split by the way it turns along ``dim``
    (``diagnostix.spectra.rotary_spectrum``), with the mean removed and no
    taper: the sum of ``psd_cw + psd_ccw`` times the step of the wavenumbers is
    the variance of ``u`` plus that of ``v``. Polarization is NaN where total
    rotary power is negligible.

    Args:
        u: Zonal/eastward velocity DataArray.
        v: Meridional/northward velocity DataArray on the same grid as ``u``.
        dim: Dimension to Fourier transform; without a coordinate, its spacing
            is 1.
        avg_dims: Optional dimension or dimensions to average in each output.

    Returns:
        Dataset with ``psd_ccw``, ``psd_cw``, and ``polarization`` where
        ``polarization = (psd_cw - psd_ccw) / (psd_cw + psd_ccw)``, on
        ``wavenumber`` from 0.
    """
    if dim not in u.dims or dim not in v.dims:
        raise ValueError(
            f"dim={dim!r} must be present on both u (dims={u.dims}) and "
            f"v (dims={v.dims})."
        )
    step = _spacing(u, dim) if dim in u.coords else 1.0

    def kernel(u_: Any, v_: Any) -> tuple[Any, ...]:
        out = spectra.rotary_spectrum(u_, v_, step)
        return out.counter_clockwise, out.clockwise

    psd_ccw, psd_cw = _transform(
        kernel,
        u,
        v,
        core=[dim],
        out={"wavenumber": np.fft.rfftfreq(u.sizes[dim], step)},
        dtypes=[_real(u, v)] * 2,
        replace={dim: "wavenumber"},
    )
    denom = psd_cw + psd_ccw
    polarization = ((psd_cw - psd_ccw) / denom).where(
        denom > float(np.finfo(denom.dtype).eps)
    )
    out = xr.Dataset(
        {"psd_ccw": psd_ccw, "psd_cw": psd_cw, "polarization": polarization}
    )
    avg = _as_list(avg_dims)
    return out.mean(dim=avg) if avg else out


def stft(
    da: xr.DataArray,
    dim: str,
    *,
    window_size: int,
    hop: int | None = None,
    window: str | None = "tukey",
    detrend: Detrend = "linear",
) -> xr.DataArray:
    """Short-time Fourier transform along ``dim``.

    Splits the signal into overlapping windows of length ``window_size``
    spaced by ``hop`` samples (default ``window_size // 2``), computes a
    Fourier transform per window, and stacks the result on a new
    ``segment`` axis whose coordinate is the centre time of each
    window. The phase is that of ``Σ x(t) exp(−2πi f t)``, with ``t`` the
    coordinate of ``dim`` (dates from the first).

    Args:
        da: 1-D signal (along ``dim``) plus arbitrary extra dims that
            are broadcast over.
        dim: Time / sample dimension to slide the window along, increasing.
        window_size: Length (in samples) of each STFT window.
        hop: Stride between window starts. Defaults to
            ``window_size // 2`` (50 % overlap).
        window: Window name or tuple for :func:`scipy.signal.get_window`
            (Tukey by default), periodic. ``None`` for no taper.
        detrend: Per-window detrending. ``None`` to disable.

    Returns:
        Complex DataArray with dims ``(segment, *outer_dims, freq_<dim>)``,
        named ``f"{da.name}_stft"``, with the frequencies ascending.
    """
    if dim not in da.dims:
        raise ValueError(f"dim={dim!r} not present on DataArray with dims={da.dims}.")
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2; got {window_size}.")
    if hop is None:
        hop = window_size // 2
    if hop < 1:
        raise ValueError(f"hop must be >= 1; got {hop}.")

    n_total = da.sizes[dim]
    if window_size > n_total:
        raise ValueError(f"window_size={window_size} exceeds {dim!r} length {n_total}.")

    starts = np.arange(0, n_total - window_size + 1, hop, dtype=int)
    positions = _positions(da, dim)
    step = float((positions[-1] - positions[0]) / (n_total - 1))
    if step < 0.0:
        raise ValueError(f"coord {dim!r} must increase.")
    frequency = np.fft.fftshift(np.fft.fftfreq(window_size, step))
    taper = 1.0 if window is None else signal.get_window(window, window_size)
    # exp(−2πi f t₀) of the first sample of each window
    shift = np.exp(-2j * np.pi * frequency * positions[starts][:, None])

    def kernel(x: Any) -> Any:
        pieces = np.stack([x[..., s : s + window_size] for s in starts], axis=-2)
        if detrend is not None:
            pieces = signal.detrend(pieces, axis=-1, type=detrend)
        z = np.fft.fft(pieces * taper, axis=-1)
        return np.fft.fftshift(z, axes=-1) * shift

    freq_dim = f"freq_{dim}"
    out = xr.apply_ufunc(
        kernel,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["segment", freq_dim]],
        dask="parallelized",
        output_dtypes=[np.complex128],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {"segment": starts.size, freq_dim: window_size},
        },
    )
    centres = [da[dim].values[s + window_size // 2].item() for s in starts]
    out = out.assign_coords({freq_dim: frequency, "segment": np.asarray(centres)})
    return out.transpose("segment", ...).rename(_output_name(da, "stft"))


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

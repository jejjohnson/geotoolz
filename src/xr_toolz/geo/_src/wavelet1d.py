"""Torrence-Compo 1-D continuous wavelet analysis for xarray objects."""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
import xarray as xr
from scipy.fft import fft, fftfreq, ifft
from scipy.special import gammaincinv


Mother = Literal["morlet", "paul", "dog"]
NullModel = Literal["red", "white"]
DominantReduce = Literal["argmax", "median_argmax"]


def cwt1d(
    da: xr.DataArray,
    *,
    dim: str = "time",
    mother: str = "morlet",
    param: float | None = None,
    s0: float | None = None,
    dj: float = 0.25,
    j_max: int | None = None,
    pad: bool = True,
) -> xr.Dataset:
    """Compute a 1-D continuous wavelet transform along one dimension.

    The implementation follows Torrence and Compo (1998): zero-pad to the
    next power of two, multiply the FFT of the signal by the analytic Fourier
    transform of the selected mother wavelet, and strip the padding after the
    inverse FFT. The returned ``power_rect`` variable is Liu et al. (2007)
    rectified power, ``|W|² / scale``.

    Args:
        da: Input time series or array with one transform dimension.
        dim: Dimension to transform.
        mother: Mother wavelet: ``"morlet"``, ``"paul"``, or ``"dog"``.
        param: Mother parameter. Defaults are Morlet ``k0=6``, Paul ``m=4``,
            and DOG ``m=2``.
        s0: Smallest scale. Defaults to ``2 * dt``.
        dj: Scale spacing in octaves. ``0.25`` gives four voices per octave.
        j_max: Largest scale index. Defaults to ``log2(n * dt / s0) / dj``.
        pad: If ``True``, zero-pad to the next power of two before the FFT.

    Returns:
        Dataset with ``wave``, ``power``, ``power_rect``, ``coi``, and
        ``coi_mask`` variables plus ``scale`` and ``period`` coordinates.
    """
    _require_dim(da, dim)
    _raise_if_chunked_along_dim(da, dim, "cwt1d")
    dt = _coord_spacing(da, dim)
    scale = _scale_grid(da.sizes[dim], dt=dt, s0=s0, dj=dj, j_max=j_max)
    mother_name = _normalize_mother(mother)
    param_value = _default_param(mother_name, param)
    fourier_factor, coi_factor, _ = _wavelet_factors(mother_name, param_value)
    period = scale * fourier_factor

    wave = xr.apply_ufunc(
        _cwt1d_numpy,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["scale", dim]],
        kwargs={
            "dt": dt,
            "scale": scale,
            "mother": mother_name,
            "param": param_value,
            "pad": pad,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.complex128],
        dask_gufunc_kwargs={"output_sizes": {"scale": scale.size}},
    )
    outer_dims = tuple(d for d in da.dims if d != dim)
    wave = wave.transpose("scale", dim, *outer_dims)
    coords = {
        "scale": ("scale", scale),
        "period": ("scale", period),
        dim: da[dim],
        "fourier_factor": fourier_factor,
    }
    wave = wave.assign_coords(coords)
    wave.name = "wave"
    wave.attrs.update(
        {
            "mother": mother_name,
            "param": param_value,
            "dt": dt,
            "dj": dj,
            "lag1_autocorrelation": _lag1_autocorrelation(da, dim),
        }
    )
    # Per-series variance preserves outer-dim broadcasting for significance
    # thresholds — a scalar `source_variance` would over/under-flag stacks of
    # series with different amplitudes.
    source_variance = da.var(dim, skipna=True)
    wave = wave.assign_coords(source_variance=source_variance)
    power = (xr.apply_ufunc(np.abs, wave) ** 2).rename("power")
    power.attrs.update(wave.attrs)
    power_rect = (
        power / xr.DataArray(scale, dims=("scale",), coords={"scale": scale})
    ).rename("power_rect")
    power_rect.attrs.update(power.attrs)
    power_rect.attrs["bias_corrected"] = "liu_2007"
    coi = _coi(da[dim], dt=dt, fourier_factor=fourier_factor, coi_factor=coi_factor)
    coi_mask = _coi_mask(scale=scale, period=period, coi=coi, dim=dim)
    return xr.Dataset(
        {
            "wave": wave,
            "power": power,
            "power_rect": power_rect,
            "coi": coi,
            "coi_mask": coi_mask,
        }
    )


def wavelet_significance(
    power: xr.DataArray,
    *,
    dim_time: str = "time",
    dim_scale: str = "scale",
    null: NullModel = "red",
    alpha: float | None = None,
    confidence: float = 0.95,
    mother: str = "morlet",
    param: float | None = None,
) -> xr.DataArray:
    """Return a mask where wavelet power exceeds a red/white-noise threshold.

    Args:
        power: Wavelet power with scale and time dimensions.
        dim_time: Time dimension name.
        dim_scale: Scale dimension name.
        null: Null spectrum, either AR(1) red noise or white noise.
        alpha: Lag-1 autocorrelation. When omitted, the value saved by
            :func:`cwt1d` is used, falling back to white noise.
        confidence: Chi-square confidence level.
        mother: Mother wavelet name, used for the scale-period mapping if no
            ``period`` coordinate is attached.
        param: Optional mother parameter.

    Returns:
        Boolean DataArray with the same dimensions as ``power``.
    """
    _require_dim(power, dim_time)
    _require_dim(power, dim_scale)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    null_name = null.lower()
    if null_name not in {"red", "white"}:
        raise ValueError("null must be either 'red' or 'white'")
    null_model = cast(NullModel, null_name)
    alpha_value = 0.0 if alpha is None else float(alpha)
    if alpha is None:
        alpha_value = float(power.attrs.get("lag1_autocorrelation", 0.0))
    if not -1.0 < alpha_value < 1.0:
        raise ValueError("alpha must be between -1 and 1")

    scale = np.asarray(power[dim_scale].values, dtype=float)
    if "period" in power.coords and power["period"].dims == (dim_scale,):
        period = np.asarray(power["period"].values, dtype=float)
    else:
        mother_name = _normalize_mother(mother)
        param_value = _default_param(mother_name, param)
        fourier_factor, _, _ = _wavelet_factors(mother_name, param_value)
        period = scale * fourier_factor
    # source_variance is per-outer-index when written by cwt1d so the
    # threshold broadcasts correctly across stacks of series. Fall back to
    # the per-series wavelet-power mean if the coord is missing (e.g. when
    # power was constructed by hand).
    if "source_variance" in power.coords:
        variance: xr.DataArray | float = power.coords["source_variance"].astype(float)
    elif "source_variance" in power.attrs:
        variance = float(power.attrs["source_variance"])
    else:
        variance = power.mean(dim_time, skipna=True).mean(dim_scale, skipna=True)
    if null_model == "white":
        background = np.ones_like(period)
    else:
        dt = float(power.attrs.get("dt", 1.0))
        background = (1.0 - alpha_value**2) / (
            1.0 + alpha_value**2 - 2.0 * alpha_value * np.cos(2.0 * np.pi * dt / period)
        )
    background_da = xr.DataArray(
        background,
        dims=(dim_scale,),
        coords={dim_scale: power[dim_scale]},
    )
    threshold = variance * background_da * _chi2_ppf(confidence, df=2.0) / 2.0
    if power.attrs.get("bias_corrected") == "liu_2007":
        threshold = threshold / xr.DataArray(
            scale, dims=(dim_scale,), coords={dim_scale: power[dim_scale]}
        )
    out = (power > threshold).rename("signif_mask")
    out.attrs.update(
        {
            "null": null_model,
            "alpha": alpha_value,
            "confidence": confidence,
        }
    )
    return out


def icwt1d(
    wave: xr.DataArray,
    *,
    band: tuple[float, float] | None = None,
    mother: str = "morlet",
    param: float | None = None,
    dj: float | None = None,
    dt: float | None = None,
) -> xr.DataArray:
    """Reconstruct a signal from 1-D CWT coefficients.

    Args:
        wave: Complex coefficients from :func:`cwt1d`.
        band: Optional ``(min, max)`` band to reconstruct. Interpreted in
            **period** units when ``wave`` carries a ``period`` coordinate
            (the usual case after :func:`cwt1d`); otherwise interpreted in
            **scale** units.
        mother: Mother wavelet name.
        param: Optional mother parameter. Reconstruction constants are
            available for the Torrence-Compo defaults.
        dj: Scale spacing used by :func:`cwt1d`. Defaults to the value
            stored in ``wave.attrs["dj"]`` (falling back to ``0.25`` if
            absent). Reconstruction is only correct when ``dj`` matches
            the value used during decomposition.
        dt: Sample spacing. Defaults to the coefficient ``dt`` attribute.

    Returns:
        Reconstructed DataArray without the scale dimension.
    """
    _require_dim(wave, "scale")
    mother_name = _normalize_mother(mother)
    param_value = _default_param(mother_name, param)
    _, _, cdelta = _wavelet_factors(mother_name, param_value)
    psi0 = _psi0(mother_name, param_value)
    dt_value = float(wave.attrs.get("dt", 1.0) if dt is None else dt)
    dj_value = float(wave.attrs.get("dj", 0.25) if dj is None else dj)
    coeffs = wave
    if band is not None:
        band_coord = coeffs["period"] if "period" in coeffs.coords else coeffs["scale"]
        lo, hi = sorted((float(band[0]), float(band[1])))
        coeffs = coeffs.where((band_coord >= lo) & (band_coord <= hi), other=0.0)
    scale = xr.DataArray(
        np.asarray(coeffs["scale"].values, dtype=float),
        dims=("scale",),
        coords={"scale": coeffs["scale"]},
    )
    reconstructed = (coeffs.real / np.sqrt(scale)).sum("scale") * (
        dj_value * np.sqrt(dt_value) / (cdelta * psi0)
    )
    reconstructed.name = (
        _strip_suffix(str(wave.name) if wave.name is not None else None, "_wave")
        or "icwt"
    )
    return reconstructed


def dominant_period_map(
    power_rect: xr.DataArray,
    *,
    dim_time: str = "time",
    dim_scale: str = "scale",
    coi_mask: xr.DataArray | None = None,
    signif_mask: xr.DataArray | None = None,
    reduce: DominantReduce = "argmax",
) -> xr.DataArray:
    """Return the dominant Fourier period after time-averaging rectified power."""
    _require_dim(power_rect, dim_time)
    _require_dim(power_rect, dim_scale)
    if reduce not in {"argmax", "median_argmax"}:
        raise ValueError("reduce must be 'argmax' or 'median_argmax'")
    masked = power_rect
    if coi_mask is not None:
        masked = masked.where(coi_mask)
    if signif_mask is not None:
        masked = masked.where(signif_mask)
    if reduce == "median_argmax":
        summary = masked.median(dim_time, skipna=True)
    else:
        summary = masked.mean(dim_time, skipna=True)
    peak_scale = summary.idxmax(dim_scale)
    if "period" in power_rect.coords and power_rect["period"].dims == (dim_scale,):
        period_lookup = xr.DataArray(
            np.asarray(power_rect["period"].values, dtype=float),
            dims=(dim_scale,),
            coords={dim_scale: power_rect[dim_scale]},
        )
        out = period_lookup.sel({dim_scale: peak_scale})
    else:
        out = peak_scale
    out = out.rename("dominant_period")
    out.attrs["long_name"] = "dominant Fourier period"
    return out


def _cwt1d_numpy(
    values: np.ndarray,
    *,
    dt: float,
    scale: np.ndarray,
    mother: str,
    param: float,
    pad: bool,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2:
        raise ValueError("CWT requires at least two samples")
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        fill = float(np.nanmean(arr)) if not np.all(nan_mask) else 0.0
        arr = np.where(nan_mask, fill, arr)
    npad = _next_power_of_two(n) if pad else n
    padded = np.zeros(npad, dtype=float)
    padded[:n] = arr
    omega = 2.0 * np.pi * fftfreq(npad, d=dt)
    omega_step = abs(omega[1]) if omega.size > 1 else 1.0
    signal_ft = fft(padded)
    out = np.empty((scale.size, n), dtype=np.complex128)
    for i, s in enumerate(scale):
        daughter = _mother_ft(s * omega, mother=mother, param=param)
        daughter = daughter * np.sqrt(s * omega_step) * np.sqrt(npad)
        out[i] = ifft(signal_ft * daughter)[:n]
    return out


def _mother_ft(eta: np.ndarray, *, mother: str, param: float) -> np.ndarray:
    positive = eta > 0.0
    out = np.zeros_like(eta, dtype=np.complex128)
    if mother == "morlet":
        out[positive] = np.pi ** (-0.25) * np.exp(-0.5 * (eta[positive] - param) ** 2)
    elif mother == "paul":
        m = int(param)
        norm = 2.0**m / np.sqrt(m * np.prod(np.arange(2, 2 * m, dtype=float)))
        out[positive] = norm * eta[positive] ** m * np.exp(-eta[positive])
    elif mother == "dog":
        m = int(param)
        norm = np.sqrt(1.0 / math.gamma(m + 0.5))
        out = ((-1j) ** m) * norm * eta**m * np.exp(-0.5 * eta**2)
    else:
        raise ValueError("mother must be 'morlet', 'paul', or 'dog'")
    return out


def _scale_grid(
    n: int,
    *,
    dt: float,
    s0: float | None,
    dj: float,
    j_max: int | None,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be strictly positive")
    if dj <= 0:
        raise ValueError("dj must be strictly positive")
    s0_value = 2.0 * dt if s0 is None else float(s0)
    if s0_value <= 0:
        raise ValueError("s0 must be strictly positive")
    j_stop = (
        int(np.floor(np.log2(n * dt / s0_value) / dj)) if j_max is None else int(j_max)
    )
    if j_stop < 0:
        raise ValueError("j_max must be non-negative for the chosen s0 and dt")
    return s0_value * 2.0 ** (np.arange(j_stop + 1, dtype=float) * dj)


def _coord_spacing(da: xr.DataArray, dim: str) -> float:
    if dim not in da.coords:
        return 1.0
    values = da[dim].values
    if values.size < 2:
        raise ValueError(f"dim {dim!r} must contain at least two samples")
    if np.issubdtype(values.dtype, np.datetime64):
        numeric = values.astype("datetime64[ns]").astype("int64").astype(float) / 1.0e9
    else:
        numeric = np.asarray(values, dtype=float)
    diffs = np.diff(numeric)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"coord {dim!r} is not uniformly spaced; resample before cwt1d."
        )
    return float(abs(diffs[0]))


def _coi(
    coord: xr.DataArray,
    *,
    dt: float,
    fourier_factor: float,
    coi_factor: float,
) -> xr.DataArray:
    n = coord.size
    distance = np.minimum(
        np.arange(n, dtype=float), np.arange(n - 1, -1, -1, dtype=float)
    )
    coi = np.maximum(distance * dt * fourier_factor / coi_factor, np.finfo(float).tiny)
    return xr.DataArray(coi, dims=coord.dims, coords={coord.dims[0]: coord}, name="coi")


def _coi_mask(
    *,
    scale: np.ndarray,
    period: np.ndarray,
    coi: xr.DataArray,
    dim: str,
) -> xr.DataArray:
    mask = period[:, None] <= np.asarray(coi.values, dtype=float)[None, :]
    return xr.DataArray(
        mask,
        dims=("scale", dim),
        coords={"scale": scale, dim: coi[dim]},
        name="coi_mask",
    )


def _wavelet_factors(mother: str, param: float) -> tuple[float, float, float]:
    if mother == "morlet":
        fourier_factor = 4.0 * np.pi / (param + np.sqrt(2.0 + param**2))
        return float(fourier_factor), np.sqrt(2.0), 0.776
    if mother == "paul":
        m = int(param)
        fourier_factor = 4.0 * np.pi / (2.0 * m + 1.0)
        return float(fourier_factor), np.sqrt(2.0), 1.132
    if mother == "dog":
        m = int(param)
        fourier_factor = 2.0 * np.pi * np.sqrt(2.0 / (2.0 * m + 1.0))
        return float(fourier_factor), np.sqrt(2.0), 3.541
    raise ValueError("mother must be 'morlet', 'paul', or 'dog'")


def _psi0(mother: str, param: float) -> float:
    if mother == "morlet":
        return float(np.pi ** (-0.25))
    if mother == "paul" and int(param) == 4:
        return 1.079
    if mother == "dog" and int(param) == 2:
        return 0.867
    raise ValueError("icwt1d only supports default reconstruction constants")


def _default_param(mother: str, param: float | None) -> float:
    if param is not None:
        return float(param)
    if mother == "morlet":
        return 6.0
    if mother == "paul":
        return 4.0
    if mother == "dog":
        return 2.0
    raise ValueError("mother must be 'morlet', 'paul', or 'dog'")


def _normalize_mother(mother: str) -> Mother:
    name = mother.lower()
    if name not in {"morlet", "paul", "dog"}:
        raise ValueError("mother must be 'morlet', 'paul', or 'dog'")
    return cast(Mother, name)


def _lag1_autocorrelation(da: xr.DataArray, dim: str) -> float:
    current = da.isel({dim: slice(1, None)})
    previous = da.isel({dim: slice(None, -1)})
    current = current.assign_coords({dim: previous[dim]})
    corr = xr.corr(current, previous, dim=dim)
    value = float(corr.mean(skipna=True))
    return value if np.isfinite(value) else 0.0


def _chi2_ppf(confidence: float, *, df: float) -> float:
    return float(2.0 * gammaincinv(df / 2.0, confidence))


def _next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _require_dim(da: xr.DataArray, dim: str) -> None:
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not present on DataArray with dims={da.dims}.")


def _raise_if_chunked_along_dim(da: xr.DataArray, dim: str, func: str) -> None:
    chunksizes = getattr(da, "chunksizes", {})
    chunks = chunksizes.get(dim)
    if chunks is not None and len(chunks) > 1:
        raise ValueError(
            f"{func} requires dimension {dim!r} to be a single chunk; "
            f"got {tuple(chunks)!r}."
        )


def _strip_suffix(name: str | None, suffix: str) -> str | None:
    if name is None:
        return None
    return name.removesuffix(suffix)

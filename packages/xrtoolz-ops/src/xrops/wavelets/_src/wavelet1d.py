"""Torrence-Compo 1-D continuous wavelet analysis for xarray objects."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import xarray as xr
from diagnostix import spectra
from jaxtyping import Float

from xrops.utils._src.spacing import coord_spacing


Mother = Literal["morlet", "paul", "dog"]
NullModel = Literal["red", "white"]
DominantReduce = Literal["argmax", "median_argmax"]

#: The distance of an end that is felt at a scale of 1, as in the records' cone.
COI_REACH = {"morlet": np.sqrt(2.0), "paul": np.sqrt(0.5), "dog": np.sqrt(2.0)}


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
    detrend: Literal["constant", "linear"] | None = None,
) -> xr.Dataset:
    """Compute a 1-D continuous wavelet transform along one dimension.

    Torrence and Compo (1998), from ``diagnostix.spectra.wavelet_transform``.
    The returned ``power_rect`` variable is Liu et al. (2007) rectified power,
    ``|W|² / scale``. A missing sample is filled with the mean of its series.

    Args:
        da: Input time series or array with one transform dimension.
        dim: Dimension to transform.
        mother: Mother wavelet: ``"morlet"``, ``"paul"``, or ``"dog"``.
        param: Mother parameter. Defaults are Morlet ``k0=6``, Paul ``m=4``,
            and DOG ``m=2``.
        s0: Smallest scale. Defaults to ``2 * dt``.
        dj: Scale spacing in octaves. ``0.25`` gives four voices per octave.
        j_max: Largest scale index. Defaults to ``log2(n * dt / s0) / dj``.
        pad: If ``True``, pad with zeros to twice the length, so that nothing
            wraps; if ``False``, the series is periodic and has no cone.
        detrend: What to take away first: the mean (``"constant"``), the line
            (``"linear"``) or nothing. A mean left in is a step at both ends.

    Returns:
        Dataset with ``wave``, ``power``, ``power_rect``, ``coi``, and
        ``coi_mask`` variables plus ``scale`` and ``period`` coordinates.
    """
    _require_dim(da, dim)
    _raise_if_chunked_along_dim(da, dim, "cwt1d")
    dt = coord_spacing(da, dim, require_coord=False)
    scale = _scale_grid(da.sizes[dim], dt=dt, s0=s0, dj=dj, j_max=j_max)
    mother_name = _normalize_mother(mother)
    param_value = _default_param(mother_name, param)
    series = da.fillna(da.mean(dim)) if da.dtype.kind == "f" else da.astype(float)

    def transform(x: Any) -> Any:
        return spectra.wavelet_transform(
            x,
            dt,
            scale,
            mother=mother_name,
            param=param_value,
            detrend=detrend,
            periodic=not pad,
        ).coefficients

    wave = xr.apply_ufunc(
        transform,
        series,
        input_core_dims=[[dim]],
        output_core_dims=[["scale", dim]],
        dask="parallelized",
        output_dtypes=[np.result_type(series.dtype, np.complex64)],
        dask_gufunc_kwargs={"output_sizes": {"scale": scale.size}},
    )
    # the period and the cone do not depend on the coefficients
    record = _record(
        np.broadcast_to(np.complex128(0.0), (scale.size, da.sizes[dim])),
        scale,
        dt=dt,
        mother=mother_name,
        param=param_value,
        periodic=not pad,
    )
    period = record.period
    fourier_factor = float(period[0] / scale[0])
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
            # an integer: NetCDF has no boolean attributes
            "periodic": int(not pad),
            "lag1_autocorrelation": _lag1_autocorrelation(da, dim),
        }
    )
    # Per-series variance preserves outer-dim broadcasting for significance
    # thresholds — a scalar `source_variance` would over/under-flag stacks of
    # series with different amplitudes.
    source_variance = da.var(dim, skipna=True)
    wave = wave.assign_coords(source_variance=source_variance)
    power = (abs(wave) ** 2).rename("power")
    power.attrs.update(wave.attrs)
    power_rect = (
        power / xr.DataArray(scale, dims=("scale",), coords={"scale": scale})
    ).rename("power_rect")
    power_rect.attrs.update(power.attrs)
    power_rect.attrs["bias_corrected"] = "liu_2007"
    coi = _coi(da[dim], record, fourier_factor=fourier_factor)
    coi_mask = xr.DataArray(
        record.inside_cone,
        dims=("scale", dim),
        coords={"scale": scale, dim: da[dim]},
        name="coi_mask",
    )
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
    mother: str | None = None,
    param: float | None = None,
) -> xr.DataArray:
    """Return a mask where wavelet power exceeds a red/white-noise threshold.

    The threshold is the χ² quantile of Torrence and Compo (1998), eqs. 16–18,
    with two degrees of freedom for the complex mothers and one for the real
    ``"dog"`` (``diagnostix.spectra.wavelet_significance``).

    Args:
        power: Wavelet power with scale and time dimensions.
        dim_time: Time dimension name.
        dim_scale: Scale dimension name.
        null: Null spectrum, either AR(1) red noise or white noise.
        alpha: Lag-1 autocorrelation. When omitted, the value saved by
            :func:`cwt1d` is used, falling back to white noise.
        confidence: Chi-square confidence level.
        mother: Mother wavelet name. When omitted, the one saved by
            :func:`cwt1d` is used, falling back to ``"morlet"``.
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
    mother_name, param_value = _mother_of(power, mother, param)
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
    record = _record(
        np.broadcast_to(np.complex128(0.0), (scale.size, 2)),
        scale,
        dt=float(power.attrs.get("dt", 1.0)),
        mother=mother_name,
        param=param_value,
        periodic=False,
    )
    lag1 = 0.0 if null_model == "white" else alpha_value
    threshold = xr.apply_ufunc(
        lambda sigma2: spectra.wavelet_significance(
            record, sigma2, lag1=lag1, confidence=confidence
        ),
        xr.DataArray(variance),
        output_core_dims=[[dim_scale]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {dim_scale: scale.size}},
    ).assign_coords({dim_scale: power[dim_scale]})
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
    mother: str | None = None,
    param: float | None = None,
    dt: float | None = None,
) -> xr.DataArray:
    """Reconstruct a signal from 1-D CWT coefficients.

    The sum over scales of Torrence and Compo (1998), eq. 11, with the constant
    of the mother computed (``diagnostix.spectra.inverse_wavelet_transform``).
    The mean is at no scale, and is not given back.

    Args:
        wave: Complex coefficients from :func:`cwt1d`.
        band: Optional ``(min, max)`` band of periods to reconstruct.
        mother: Mother wavelet name. When omitted, the one saved by
            :func:`cwt1d` is used, falling back to ``"morlet"``.
        param: Optional mother parameter.
        dt: Sample spacing. Defaults to the coefficient ``dt`` attribute.

    Returns:
        Reconstructed DataArray without the scale dimension.
    """
    _require_dim(wave, "scale")
    mother_name, param_value = _mother_of(wave, mother, param)
    dt_value = float(wave.attrs.get("dt", 1.0) if dt is None else dt)
    scale = np.asarray(wave["scale"].values, dtype=float)
    if band is not None:
        band = (min(band), max(band))

    def inverse(coefficients: Any) -> Any:
        record = _record(
            coefficients[..., None],
            scale,
            dt=dt_value,
            mother=mother_name,
            param=param_value,
            periodic=bool(wave.attrs.get("periodic", False)),
        )
        return spectra.inverse_wavelet_transform(record, band=band)[..., 0]

    reconstructed = xr.apply_ufunc(
        inverse,
        wave,
        input_core_dims=[["scale"]],
        dask="parallelized",
        output_dtypes=[wave.real.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
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


def _record(
    coefficients: Any,
    scale: Float[np.ndarray, "scale"],
    *,
    dt: float,
    mother: str,
    param: float,
    periodic: bool,
) -> spectra.WaveletTransform:
    """The coefficients, and what they were made with."""
    return spectra.WaveletTransform(
        coefficients=coefficients,
        scales=tuple(scale.tolist()),
        spacing=dt,
        mother=mother,
        param=param,
        periodic=periodic,
    )


def _mother_of(
    da: xr.DataArray, mother: str | None, param: float | None
) -> tuple[str, float]:
    """The mother and its parameter: those given, or those `cwt1d` saved."""
    if mother is None:
        mother = str(da.attrs.get("mother", "morlet"))
        param = da.attrs.get("param") if param is None else param
    name = _normalize_mother(mother)
    return name, _default_param(name, param)


def _scale_grid(
    n: int,
    *,
    dt: float,
    s0: float | None,
    dj: float,
    j_max: int | None,
) -> Float[np.ndarray, "scale"]:
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


def _coi(
    coord: xr.DataArray, record: spectra.WaveletTransform, *, fourier_factor: float
) -> xr.DataArray:
    """The longest period that the ends do not touch, at each sample."""
    n = coord.size
    distance = record.spacing * np.minimum(np.arange(n), n - 1 - np.arange(n))
    if record.periodic:
        distance = np.full(n, np.inf)
    coi = np.maximum(
        distance * fourier_factor / COI_REACH[record.mother], np.finfo(float).tiny
    )
    return xr.DataArray(coi, dims=coord.dims, coords={coord.dims[0]: coord}, name="coi")


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

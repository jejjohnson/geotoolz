"""Spectral evaluation metrics.

Spectral metrics (``psd_*``) compare the PSD of the prediction against
the PSD of the reference and return a normalized score plus helpers to
locate the resolved-scale crossover.

Frequency-band scores (``evaluate_by_frequency_band`` /
``band_limited_rmse`` and their :class:`Operator` wrappers
:class:`FrequencyBandSkill` / :class:`BandLimitedRMSE`) implement the
band-decomposition slice of validation.md §1. Bands are specified as
``dict[str, tuple[float, float]]`` of ``{name: (low, high)}`` in the
**physical units** of the dim's coordinate (e.g. cycles/day for time,
cycles/km for spatial dims). Lat/lon coordinates whose ``units``
attribute looks like ``degrees_north`` / ``degrees_east`` are converted
to kilometres internally using a meridian-arc length of ``111.0`` km
per degree latitude and an additional ``cos(mean_lat)`` factor for
longitude — adequate for mid-latitude mesoscale demos but the user
should provide ``coord_spacing=`` overrides for high-precision work.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from xr_toolz.core import Operator
from xr_toolz.geo._src.wavelet import wvlt_power_spectrum
from xr_toolz.geo._src.wavelet_utils import scale_to_wavenumber
from xr_toolz.transforms._src.fourier import (
    drop_negative_frequencies,
    power_spectrum,
)


_KM_PER_DEGREE = 111.0
_LAT_UNITS = {
    "degrees_north",
    "degree_north",
    "degrees_n",
    "deg_n",
    "degn",
}
_LON_UNITS = {
    "degrees_east",
    "degree_east",
    "degrees_e",
    "deg_e",
    "dege",
}


# ---------- Layer-0 (xarray) ----------------------------------------------


def psd_error(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    psd_dims: Sequence[str],
    avg_dims: Sequence[str] | None = None,
    isotropic: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """PSD of the prediction error ``pred - ref``.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable to score.
        psd_dims: Dimensions over which to take the PSD.
        avg_dims: Optional dims to conditionally average out after the
            PSD (e.g. average over ``lat`` after computing the lon/time
            spectrum).
        isotropic: If ``True``, use the isotropic power spectrum.
        **kwargs: Forwarded to the underlying PSD function.

    Returns:
        Dataset with a single ``"error"`` variable containing the
        PSD of the error.
    """
    diff = (ds_pred[variable] - ds_ref[variable]).rename("error")
    err = power_spectrum(diff, dim=list(psd_dims), isotropic=isotropic, **kwargs)
    err_ds = err.rename("error").to_dataset()
    if avg_dims is not None:
        err_ds = drop_negative_frequencies(err_ds, dims=avg_dims, drop=True)
    return err_ds


def psd_score(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    psd_dims: Sequence[str],
    avg_dims: Sequence[str] | None = None,
    isotropic: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """Normalized PSD score ``1 - PSD(err) / PSD(ref)``.

    A score of ``1`` means the error has no power at that scale; ``0``
    means the error has as much power as the reference signal.

    Args:
        ds_pred: Prediction dataset.
        ds_ref: Reference dataset.
        variable: Variable to score.
        psd_dims: Dimensions over which to take the PSD.
        avg_dims: Optional conditional-average dims applied after PSD.
        isotropic: If ``True``, use the isotropic power spectrum.
        **kwargs: Forwarded to the underlying PSD function.

    Returns:
        Dataset with a single ``"score"`` variable.
    """
    err = psd_error(
        ds_pred, ds_ref, variable, psd_dims, avg_dims, isotropic=isotropic, **kwargs
    )
    ref_psd = power_spectrum(
        ds_ref[variable], dim=list(psd_dims), isotropic=isotropic, **kwargs
    )
    ref_ds = ref_psd.rename(variable).to_dataset()
    if avg_dims is not None:
        ref_ds = drop_negative_frequencies(ref_ds, dims=avg_dims, drop=True)
    score = 1.0 - err["error"] / ref_ds[variable]
    return score.to_dataset(name="score")


def resolved_scale(
    score: xr.DataArray | xr.Dataset,
    frequency: str,
    level: float = 0.5,
) -> float:
    """Wavelength (``1 / frequency``) at which a PSD score crosses ``level``.

    Args:
        score: DataArray of PSD-score values along ``frequency``, or a
            Dataset containing a ``"score"`` variable.
        frequency: Name of the frequency coordinate.
        level: Score threshold (default 0.5 — the commonly-used
            "resolved scale" threshold).

    Returns:
        Scalar wavelength at which the score first crosses ``level``.
    """
    score_da = score["score"] if isinstance(score, xr.Dataset) else score
    freqs = np.asarray(score_da[frequency].values)
    vals = np.asarray(score_da.values)
    positive = freqs > 0
    freqs = freqs[positive]
    vals = vals[positive]
    wavelengths = 1.0 / freqs
    return find_intercept_1D(x=wavelengths, y=vals, level=level)


def wavelet_psd_score(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    variable: str,
    scales: xr.DataArray,
    *,
    dim: tuple[str, str] = ("y", "x"),
    x0: float = 50e3,
    ntheta: int = 16,
    k0: float = 1.0,
    isotropic: bool = True,
) -> xr.Dataset:
    """Localized wavelet PSD score ``1 - WPSD(err) / WPSD(ref)``.

    Args:
        ds_pred: Prediction dataset containing ``variable``.
        ds_ref: Reference dataset containing ``variable`` on the same grid.
        variable: Data variable to score.
        scales: Positive Morlet scales. The first dimension is used as the
            scale axis.
        dim: Spatial dimensions as ``(y, x)`` on a locally Cartesian grid.
        x0: Reference length scale in the same units as the spatial coords.
        ntheta: Number of evenly spaced Morlet angles.
        k0: Dimensionless central Morlet wavenumber.
        isotropic: If ``True``, average directional power over angle.

    Returns:
        Dataset containing ``score`` with untrusted COI samples masked to NaN.
    """
    if variable not in ds_pred.data_vars:
        raise KeyError(f"prediction missing variable {variable!r}")
    if variable not in ds_ref.data_vars:
        raise KeyError(f"reference missing variable {variable!r}")
    err = (ds_pred[variable] - ds_ref[variable]).rename("error")
    err_psd = wvlt_power_spectrum(
        err,
        scales,
        dim=dim,
        x0=x0,
        ntheta=ntheta,
        k0=k0,
        isotropic=isotropic,
    )
    ref_psd = wvlt_power_spectrum(
        ds_ref[variable],
        scales,
        dim=dim,
        x0=x0,
        ntheta=ntheta,
        k0=k0,
        isotropic=isotropic,
    )
    trust = err_psd["coi_mask"] & ref_psd["coi_mask"]
    score = (1.0 - err_psd / ref_psd).where(trust)
    return score.rename("score").to_dataset()


def wavelet_resolved_scale_map(
    truth: xr.DataArray,
    pred: xr.DataArray,
    scales: xr.DataArray,
    *,
    dim: tuple[str, str] = ("y", "x"),
    x0: float = 50e3,
    ntheta: int = 16,
    k0: float = 1.0,
    threshold: float = 0.5,
    wavelength_scale: float = 1e-3,
    wavelength_units: str = "km",
) -> xr.DataArray:
    """Return the wavelength where local wavelet PSD skill crosses a threshold.

    Args:
        truth: Reference field on a locally Cartesian grid.
        pred: Prediction field on the same grid.
        scales: Positive Morlet scales used to compute the local spectra.
        dim: Spatial dimensions as ``(y, x)``.
        x0: Reference length scale in the same units as the spatial coords.
        ntheta: Number of evenly spaced Morlet angles.
        k0: Dimensionless central Morlet wavenumber.
        threshold: Score level defining the resolved scale.
        wavelength_scale: Multiplier applied to ``1 / wavenumber`` to
            convert into ``wavelength_units``. Defaults to ``1e-3`` for
            metres → km (matches the historical behaviour); pass
            ``1.0`` if your coords are already in km.
        wavelength_units: Unit string written to the output attrs.

    Returns:
        Two-dimensional resolved wavelength in ``wavelength_units``.
        Locations with fewer than two trusted scale samples, or where
        the threshold isn't bracketed by the score column, are NaN.
    """
    ds_pred = pred.rename("field").to_dataset()
    ds_ref = truth.rename("field").to_dataset()
    score = wavelet_psd_score(
        ds_pred,
        ds_ref,
        "field",
        scales,
        dim=dim,
        x0=x0,
        ntheta=ntheta,
        k0=k0,
        isotropic=True,
    )["score"]
    scale_dim = scales.dims[0]
    wavelengths = (
        1.0
        / np.asarray(
            scale_to_wavenumber(scales, x0=x0, k0=k0).values,
            dtype=float,
        )
        * wavelength_scale
    )
    out = xr.apply_ufunc(
        _resolved_scale_column,
        score,
        input_core_dims=[[scale_dim]],
        output_core_dims=[[]],
        kwargs={"wavelengths_km": wavelengths, "threshold": threshold},
        vectorize=True,
        dask="forbidden",
        output_dtypes=[float],
    )
    out.name = "wavelet_resolved_scale"
    out.attrs["units"] = wavelength_units
    out.attrs["threshold"] = threshold
    return out


def find_intercept_1D(
    x: np.ndarray,
    y: np.ndarray,
    level: float = 0.5,
    kind: str = "slinear",
    **kwargs: Any,
) -> float:
    """Invert a 1-D monotone-ish curve at ``y = level`` and return ``x``.

    Uses :class:`scipy.interpolate.interp1d` on ``(y, x)``. Duplicate
    ``y`` values (common for plateaued PSD scores) are collapsed to
    their first occurrence in the sorted order before interpolating,
    because ``interp1d`` requires a strictly monotone x-axis.
    Extrapolates silently when ``level`` falls outside the range of the
    deduplicated ``y``.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    order = np.argsort(y_arr)
    y_sorted = y_arr[order]
    x_sorted = x_arr[order]

    _, first_idx = np.unique(y_sorted, return_index=True)
    first_idx.sort()
    y_unique = y_sorted[first_idx]
    x_unique = x_sorted[first_idx]

    if y_unique.size < 2:
        return float(x_unique.item()) if x_unique.size else float("nan")

    f = interp1d(
        y_unique,
        x_unique,
        fill_value=kwargs.pop("fill_value", "extrapolate"),
        kind=kind,
        **kwargs,
    )
    try:
        return float(np.asarray(f(level)).item())
    except ValueError:
        warnings.warn(
            f"level={level} outside range of y — returning edge value.",
            stacklevel=2,
        )
        y_min, y_max = float(y_unique.min()), float(y_unique.max())
        edge = y_min if level < y_min else y_max
        return float(np.asarray(f(edge)).item())


def _resolved_scale_column(
    score: np.ndarray,
    *,
    wavelengths_km: np.ndarray,
    threshold: float,
) -> float:
    """Find the threshold wavelength for one spatial score column.

    Returns NaN when the column lacks two valid samples *or* when the
    threshold isn't bracketed by the column — without this guard
    ``find_intercept_1D`` extrapolates and reports an arbitrary edge
    wavelength for perfect-skill (all > threshold) or hopeless-skill
    (all < threshold) columns.
    """
    valid = np.isfinite(score)
    if valid.sum() < 2:
        return float("nan")
    valid_scores = score[valid]
    if threshold < valid_scores.min() or threshold > valid_scores.max():
        return float("nan")
    return find_intercept_1D(wavelengths_km[valid], valid_scores, level=threshold)


def find_intercept_2D(
    score: xr.DataArray,
    level: float = 0.5,
    space_dim: str = "freq_lon",
    time_dim: str = "freq_time",
) -> list[xr.DataArray]:
    """Extract the threshold contour of a 2-D ``(space, time)`` score field.

    Counterpart to :func:`find_intercept_1D` for the 2-D space-time
    score: returns the boundary where ``score == level`` as data so it
    can be quoted in a paper or compared across methods quantitatively
    rather than only overlaid on a figure.
    Wraps :func:`skimage.measure.find_contours`. Disconnected boundary
    pieces are returned as separate segments. Coordinate values along
    each segment are interpolated linearly from the input grid.
    Args:
        score: 2-D :class:`xr.DataArray` indexed by ``space_dim`` and
            ``time_dim``.
        level: Score threshold to extract. Default ``0.5``.
        space_dim: Name of the spatial-frequency dim. Default
            ``"freq_lon"``.
        time_dim: Name of the temporal-frequency dim. Default
            ``"freq_time"``.
    Returns:
        List of :class:`xr.DataArray`, one per contour segment, each
        with dims ``("point", "axis")`` where ``axis`` has coordinates
        ``(space_dim, time_dim)`` and contains the segment's coordinate
        polyline.
    """
    from skimage.measure import find_contours

    if score.ndim != 2:
        raise ValueError(
            f"find_intercept_2D expects a 2-D DataArray; got dims {score.dims}."
        )
    if space_dim not in score.dims or time_dim not in score.dims:
        raise ValueError(
            f"score must have dims {space_dim!r} and {time_dim!r}; got {score.dims}."
        )

    # find_contours operates in (row, col) index space; transpose so
    # rows == time_dim (y-axis) and cols == space_dim (x-axis), which
    # matches the conventional space-time score plot orientation.
    s = score.transpose(time_dim, space_dim)
    arr = np.asarray(s.values, dtype=float)
    space_coord = np.asarray(s[space_dim].values, dtype=float)
    time_coord = np.asarray(s[time_dim].values, dtype=float)

    contours = find_contours(arr, level=level)
    segments: list[xr.DataArray] = []
    for c in contours:
        # c[:, 0] is fractional row (time), c[:, 1] is fractional col (space)
        rows = c[:, 0]
        cols = c[:, 1]
        sx = np.interp(cols, np.arange(space_coord.size), space_coord)
        ty = np.interp(rows, np.arange(time_coord.size), time_coord)
        seg = xr.DataArray(
            np.stack([sx, ty], axis=-1),
            dims=("point", "axis"),
            coords={"axis": [space_dim, time_dim]},
        )
        segments.append(seg)
    return segments


# ---------- Layer-1 (Operator wrappers) -----------------------------------


class PSDScore(Operator):
    """Two-input PSD score operator."""

    def __init__(
        self,
        variable: str,
        psd_dims: Sequence[str],
        avg_dims: Sequence[str] | None = None,
        isotropic: bool = False,
        **kwargs: Any,
    ):
        self.variable = variable
        self.psd_dims = list(psd_dims)
        self.avg_dims = None if avg_dims is None else list(avg_dims)
        self.isotropic = isotropic
        self.kwargs = dict(kwargs)

    def _apply(self, ds_pred, ds_ref):
        return psd_score(
            ds_pred,
            ds_ref,
            self.variable,
            self.psd_dims,
            avg_dims=self.avg_dims,
            isotropic=self.isotropic,
            **self.kwargs,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "psd_dims": list(self.psd_dims),
            "avg_dims": None if self.avg_dims is None else list(self.avg_dims),
            "isotropic": self.isotropic,
            **self.kwargs,
        }


class WaveletPSDScore(Operator):
    """Two-input localized wavelet PSD score operator."""

    def __init__(
        self,
        variable: str,
        scales: Sequence[float] | xr.DataArray,
        *,
        dim: tuple[str, str] = ("y", "x"),
        x0: float = 50e3,
        ntheta: int = 16,
        k0: float = 1.0,
        isotropic: bool = True,
    ) -> None:
        self.variable = variable
        self.scales = scales
        self.dim = tuple(dim)
        self.x0 = float(x0)
        self.ntheta = int(ntheta)
        self.k0 = float(k0)
        self.isotropic = bool(isotropic)

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.Dataset:
        return wavelet_psd_score(
            ds_pred,
            ds_ref,
            self.variable,
            self.scales,
            dim=self.dim,
            x0=self.x0,
            ntheta=self.ntheta,
            k0=self.k0,
            isotropic=self.isotropic,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "scales": "<xr object>",
            "dim": list(self.dim),
            "x0": self.x0,
            "ntheta": self.ntheta,
            "k0": self.k0,
            "isotropic": self.isotropic,
        }


# ---------- Frequency-band scores -----------------------------------------


def _physical_spacing(
    ds: xr.Dataset,
    dims: Sequence[str],
    *,
    coord_spacing: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Spacing per dim in physical units expected by ``bands``.

    Reads the dim coordinate's ``units`` attribute and converts
    ``degrees_north`` / ``degrees_east`` into kilometres so that bands
    can be expressed in cycles/km. Other units pass through (the user
    is responsible for ensuring band frequencies match coord units).

    A per-dim override may be supplied via ``coord_spacing``; when
    given, the override wins and unit detection is skipped for that
    dim.
    """
    out: dict[str, float] = {}
    for d in dims:
        if coord_spacing is not None and d in coord_spacing:
            out[d] = float(coord_spacing[d])
            continue
        if d not in ds.coords:
            raise ValueError(
                f"dim {d!r} has no coordinate; FrequencyBandSkill needs the "
                "coord values to compute spacing — attach one or pass "
                "coord_spacing= explicitly."
            )
        coord = ds[d]
        units = str(coord.attrs.get("units", "")).strip().lower()
        if not units and (coord_spacing is None or d not in coord_spacing):
            raise ValueError(
                f"coord {d!r} is missing a `units` attribute; bands must be "
                "in physical coord units, so attach one (e.g. with "
                "`xr_toolz.geo.ValidateCoords()`) or pass `coord_spacing=` "
                "per dim."
            )
        vals = np.asarray(coord.values, dtype=float)
        if vals.size < 2:
            raise ValueError(f"dim {d!r} has < 2 coord samples; cannot fft")
        diffs = np.diff(vals)
        # FFT assumes uniform sampling; raise instead of silently using the
        # mean spacing on a stretched grid.
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"coord {d!r} is not uniformly spaced (FFT-based banding "
                "requires regular sampling). Either resample onto a regular "
                "grid first, or pass `coord_spacing=` to declare the "
                "intended spacing."
            )
        raw = float(np.abs(np.mean(diffs)))
        if units in _LAT_UNITS:
            out[d] = raw * _KM_PER_DEGREE
        elif units in _LON_UNITS:
            cos_factor = 1.0
            for lat_name in ("lat", "latitude"):
                if lat_name in ds.coords:
                    lat_vals = np.asarray(ds[lat_name].values, dtype=float)
                    cos_factor = float(np.cos(np.deg2rad(lat_vals.mean())))
                    break
            out[d] = raw * _KM_PER_DEGREE * max(cos_factor, 1e-6)
        else:
            out[d] = raw
    return out


def _bandpass(
    da: xr.DataArray,
    dims: Sequence[str],
    spacing: Mapping[str, float],
    lo: float,
    hi: float,
) -> xr.DataArray:
    """FFT ``da`` along ``dims``, zero coefficients outside ``[lo, hi]``,
    inverse-FFT and return a real-valued DataArray with the same coords
    and dims as ``da``.

    NaN values (typical on masked oceanic grids) are filled with the
    field's nan-mean before the FFT so they do not propagate through
    the inverse transform; the original NaN mask is reapplied to the
    output so downstream metrics still skip masked cells.

    .. note::
       This implementation eagerly materialises ``da.values`` and uses
       ``numpy.fft``; it is **not lazy** under dask. For very large
       gridded inputs, slice down to a manageable region before scoring
       (a follow-up will route through ``xrft.fft`` for chunked arrays).
    """
    arr = np.asarray(da.values, dtype=float)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        fill = float(np.nanmean(arr)) if not np.all(nan_mask) else 0.0
        arr = np.where(nan_mask, fill, arr)
    # Build the radial-frequency mask in *axis-position* order (not the
    # caller-supplied `dims` order) so the broadcast against `arr` is
    # always correct, even when `dims=("lat", "lon")` but the data is
    # stored as `(time, lon, lat)`.
    sorted_dims = sorted(dims, key=da.get_axis_num)
    axes = tuple(da.get_axis_num(d) for d in sorted_dims)
    freqs = [
        np.fft.fftfreq(arr.shape[a], d=spacing[d])
        for d, a in zip(sorted_dims, axes, strict=True)
    ]
    grid = np.meshgrid(*freqs, indexing="ij")
    kmag = np.sqrt(sum(g**2 for g in grid))
    mask = (kmag >= lo) & (kmag < hi)
    shape = [1] * arr.ndim
    for a in axes:
        shape[a] = arr.shape[a]
    mask_b = mask.reshape(shape)
    fft_arr = np.fft.fftn(arr, axes=axes)
    out = np.fft.ifftn(fft_arr * mask_b, axes=axes).real
    if nan_mask.any():
        out = np.where(nan_mask, np.nan, out)
    return xr.DataArray(
        out, dims=da.dims, coords=da.coords, name=da.name, attrs=dict(da.attrs)
    )


def _validate_bands(bands: Mapping[str, tuple[float, float]]) -> None:
    if not bands:
        raise ValueError("bands must be a non-empty mapping {name: (low, high)}")
    for name, edge in bands.items():
        if not (isinstance(edge, tuple | list) and len(edge) == 2):
            raise ValueError(f"band {name!r}: expected (low, high) tuple, got {edge!r}")
        lo, hi = float(edge[0]), float(edge[1])
        if lo < 0 or hi <= lo:
            raise ValueError(
                f"band {name!r}: require 0 <= low < high; got ({lo}, {hi})"
            )


def evaluate_by_frequency_band(
    prediction: xr.Dataset,
    reference: xr.Dataset,
    *,
    variable: str,
    bands: Mapping[str, tuple[float, float]],
    dims: Sequence[str],
    metric: Operator | None = None,
    coord_spacing: Mapping[str, float] | None = None,
) -> xr.Dataset:
    """Filter ``prediction`` and ``reference`` to each band and apply ``metric``.

    Args:
        prediction: Prediction dataset.
        reference: Reference dataset.
        variable: Variable to score; must exist in both inputs.
        bands: ``{name: (low, high)}`` in physical coord units of
            ``dims``. ``low`` is included, ``high`` is excluded; both
            must be ``>= 0``.
        dims: Dimensions to FFT along. Coords must carry ``units``
            metadata (or per-dim ``coord_spacing`` overrides must be
            supplied).
        metric: Inner :class:`Operator` evaluated on the band-passed
            ``(pred, ref)`` pair. Defaults to ``RMSE(variable, dims=dims)``.
        coord_spacing: Optional per-dim spacing overrides (in the same
            physical units as ``bands``); bypasses unit auto-detection.

    Returns:
        Dataset stacked along a ``"band"`` dim with auxiliary
        ``"band_low"`` / ``"band_high"`` coords. Bands whose ``low``
        exceeds the Nyquist magnitude produce NaN entries plus a
        :class:`UserWarning` (no exception).
    """
    if variable not in prediction.data_vars:
        raise KeyError(f"prediction missing variable {variable!r}")
    if variable not in reference.data_vars:
        raise KeyError(f"reference missing variable {variable!r}")
    dim_list = list(dims)
    for d in dim_list:
        if d not in prediction.dims:
            raise ValueError(f"prediction is missing dim {d!r}")
        if d not in reference.dims:
            raise ValueError(f"reference is missing dim {d!r}")
    # Bands are physical-frequency, so prediction and reference must
    # share the dim coordinates — otherwise a single ``spacing`` derived
    # from one of them would band-pass the other with the wrong cutoff.
    for d in dim_list:
        if d in prediction.coords and d in reference.coords:
            p = np.asarray(prediction[d].values)
            r = np.asarray(reference[d].values)
            if p.shape != r.shape or not np.allclose(p, r):
                raise ValueError(
                    f"prediction and reference disagree on coord {d!r}; "
                    "regrid one onto the other (e.g. with "
                    "`xr_toolz.interpolate.RegridLike`) before scoring."
                )
    _validate_bands(bands)

    if metric is None:
        from xr_toolz.metrics._src.pixel import RMSE

        metric = RMSE(variable, dims=tuple(dim_list))
    elif not isinstance(metric, Operator):
        raise TypeError(
            "metric must be an Operator instance (e.g. RMSE(...)) so its "
            "configuration is introspectable."
        )

    spacing = _physical_spacing(prediction, dim_list, coord_spacing=coord_spacing)
    nyquist_mag = float(np.sqrt(sum((0.5 / spacing[d]) ** 2 for d in dim_list)))

    pred_da = prediction[variable]
    ref_da = reference[variable]

    pieces: list[xr.DataArray | xr.Dataset] = []
    band_names = list(bands.keys())
    for name in band_names:
        lo, hi = float(bands[name][0]), float(bands[name][1])
        if lo > nyquist_mag:
            warnings.warn(
                f"band {name!r} low={lo} exceeds Nyquist magnitude "
                f"{nyquist_mag:.4g} — emitting NaN.",
                UserWarning,
                stacklevel=2,
            )
            template = metric(prediction, reference)
            pieces.append(xr.full_like(template, np.nan, dtype=float))
            continue
        pred_band = _bandpass(pred_da, dim_list, spacing, lo, hi)
        ref_band = _bandpass(ref_da, dim_list, spacing, lo, hi)
        pred_ds = prediction.assign({variable: pred_band})
        ref_ds = reference.assign({variable: ref_band})
        pieces.append(metric(pred_ds, ref_ds))

    band_index = xr.DataArray(band_names, dims=("band",), name="band")
    out = xr.concat(pieces, dim=band_index)
    band_low = xr.DataArray(
        [float(bands[k][0]) for k in band_names], dims=("band",), name="band_low"
    )
    band_high = xr.DataArray(
        [float(bands[k][1]) for k in band_names], dims=("band",), name="band_high"
    )
    out = out.assign_coords(band_low=band_low, band_high=band_high)
    if isinstance(out, xr.DataArray):
        return out.to_dataset(name=out.name or "score")
    return out


def band_limited_rmse(
    prediction: xr.Dataset,
    reference: xr.Dataset,
    *,
    variable: str,
    bands: Mapping[str, tuple[float, float]],
    dims: Sequence[str],
    coord_spacing: Mapping[str, float] | None = None,
) -> xr.Dataset:
    """RMSE of ``prediction - reference`` filtered to each band.

    Convenience wrapper around :func:`evaluate_by_frequency_band` with
    ``metric=RMSE(variable, dims=dims)``.
    """
    from xr_toolz.metrics._src.pixel import RMSE

    return evaluate_by_frequency_band(
        prediction,
        reference,
        variable=variable,
        bands=bands,
        dims=dims,
        metric=RMSE(variable, dims=tuple(dims)),
        coord_spacing=coord_spacing,
    )


class FrequencyBandSkill(Operator):
    """Apply an inner metric to band-passed (pred, ref) pairs.

    Args:
        variable: Variable to score.
        dims: Dimensions to FFT along (passed through to inner metric
            as well when ``metric`` is omitted).
        bands: ``{name: (low, high)}`` in physical coord units.
        metric: Inner :class:`Operator` (default ``RMSE(variable, dims=dims)``).
        coord_spacing: Optional per-dim spacing override (in the same
            physical units as ``bands``).
    """

    def __init__(
        self,
        variable: str,
        dims: Sequence[str],
        bands: Mapping[str, tuple[float, float]],
        *,
        metric: Operator | None = None,
        coord_spacing: Mapping[str, float] | None = None,
    ) -> None:
        if metric is None:
            from xr_toolz.metrics._src.pixel import RMSE

            metric = RMSE(variable, dims=tuple(dims))
        elif not isinstance(metric, Operator):
            raise TypeError(
                "metric must be an Operator instance (e.g. RMSE(...)) so its "
                "configuration is introspectable."
            )
        _validate_bands(bands)
        self.variable = variable
        self.dims = tuple(dims)
        self.bands = {k: (float(v[0]), float(v[1])) for k, v in bands.items()}
        self.metric = metric
        self.coord_spacing = (
            None
            if coord_spacing is None
            else {k: float(v) for k, v in coord_spacing.items()}
        )

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.Dataset:
        return evaluate_by_frequency_band(
            ds_pred,
            ds_ref,
            variable=self.variable,
            bands=self.bands,
            dims=self.dims,
            metric=self.metric,
            coord_spacing=self.coord_spacing,
        )

    def get_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "variable": self.variable,
            "dims": list(self.dims),
            "bands": {k: list(v) for k, v in self.bands.items()},
            "metric": {
                "class": type(self.metric).__name__,
                "config": self.metric.get_config(),
            },
        }
        if self.coord_spacing is not None:
            cfg["coord_spacing"] = dict(self.coord_spacing)
        return cfg


class BandLimitedRMSE(FrequencyBandSkill):
    """Fixed-metric convenience subclass: ``FrequencyBandSkill`` with RMSE."""

    def __init__(
        self,
        variable: str,
        dims: Sequence[str],
        bands: Mapping[str, tuple[float, float]],
        *,
        coord_spacing: Mapping[str, float] | None = None,
    ) -> None:
        from xr_toolz.metrics._src.pixel import RMSE

        super().__init__(
            variable,
            dims,
            bands,
            metric=RMSE(variable, dims=tuple(dims)),
            coord_spacing=coord_spacing,
        )


__all__ = [
    "BandLimitedRMSE",
    "FrequencyBandSkill",
    "PSDScore",
    "band_limited_rmse",
    "evaluate_by_frequency_band",
    "find_intercept_1D",
    "psd_error",
    "psd_score",
    "resolved_scale",
]

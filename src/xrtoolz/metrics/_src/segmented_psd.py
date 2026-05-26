"""Gap-tolerant along-track segmented PSD score drivers."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.stats import circmean

from xrtoolz._operator import Operator
from xrtoolz.metrics._src._segmented_psd_kernels import _segment_bounds


_EARTH_RADIUS_KM = 6371.0088
_DEFAULT_MAX_GAP = np.timedelta64(2, "s")
_DEFAULT_LAT_CENTERS = np.arange(-80, 91, 1)
_DEFAULT_LON_CENTERS = np.arange(0, 360, 1)


def _coord_values_from_da(
    da: xr.DataArray, *, name: str, dim: str
) -> NDArray[np.floating]:
    values = np.asarray(da.transpose(dim).values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name!r} must be 1-D along {dim!r}.")
    return values


def _gap_indices_from_da(
    time_da: xr.DataArray, *, dim: str, max_gap: Any
) -> NDArray[np.integer]:
    if max_gap is None:
        return np.empty(0, dtype=int)
    values = np.asarray(time_da.transpose(dim).values)
    return np.flatnonzero(np.diff(values) > max_gap)


def _median_dx_km(lon: NDArray[np.floating], lat: NDArray[np.floating]) -> float:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    # Wrap to (-π, π] so dateline crossings (e.g. 179° → -179°) report
    # the shortest arc rather than the raw ~358° jump. Haversine's
    # sin²(Δλ/2) happens to be 2π-periodic so this is defensive, but
    # being explicit avoids future formula tweaks silently breaking it.
    dlon = (np.diff(lon_rad) + np.pi) % (2.0 * np.pi) - np.pi
    dlat = np.diff(lat_rad)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * (
        np.sin(dlon / 2.0) ** 2
    )
    dx = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dx = dx[np.isfinite(dx) & (dx > 0.0)]
    if dx.size == 0:
        raise ValueError("could not infer positive along-track spacing.")
    return float(np.median(dx))


def _segment_stack(
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


def _segment_lons(
    lon: NDArray[np.floating],
    bounds: list[tuple[int, int]],
) -> NDArray[np.floating]:
    out = np.empty(len(bounds), dtype=float)
    for i, (start, stop) in enumerate(bounds):
        out[i] = circmean(lon[start:stop], high=360.0, low=0.0)
    return out


def _empty_spectra(
    *,
    npt: int,
    fs: float,
    window: str | tuple[str, float] | ArrayLike,
    scaling: str,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    freqs, _ = signal.welch(
        np.zeros(npt),
        fs=fs,
        window=window,
        nperseg=npt,
        noverlap=0,
        scaling=scaling,
    )
    shape = (0, freqs.size)
    return (
        freqs,
        np.empty(shape, dtype=float),
        np.empty(shape, dtype=float),
        np.empty(shape, dtype=float),
        np.empty(shape, dtype=float),
    )


def along_track_psd_score(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    lon: xr.DataArray,
    lat: xr.DataArray,
    time: xr.DataArray | None = None,
    dim: str = "num_lines",
    npt: int,
    overlap: float = 0.5,
    max_gap: np.timedelta64 | None = _DEFAULT_MAX_GAP,
    spacing_km: float | None = None,
    window: str | tuple[str, float] | ArrayLike = "hann",
    scaling: str = "density",
) -> xr.Dataset:
    """Compute per-window along-track PSD score from prediction/reference tracks.

    Args:
        pred: Prediction DataArray (1-D along ``dim``).
        ref: Reference DataArray (1-D along ``dim``).
        lon: Longitude DataArray aligned with ``pred``/``ref`` on ``dim``.
        lat: Latitude DataArray aligned with ``pred``/``ref`` on ``dim``.
        time: Time DataArray; required when ``max_gap`` is not ``None``.
        dim: Track dimension name (default ``"num_lines"``).
        npt: Window length in samples.
        overlap: Welch overlap fraction.
        max_gap: Maximum allowed time gap before splitting a segment; pass
            ``None`` to disable gap detection.
        spacing_km: Override along-track spacing; otherwise inferred from
            ``lon``/``lat`` via haversine.
        window: Welch window spec.
        scaling: Welch ``scaling`` (``"density"`` or ``"spectrum"``).

    Returns:
        Per-segment Dataset with ``psd_ref``, ``psd_pred``, ``psd_err``,
        ``psd_score`` and ``coherence`` variables.
    """
    if dim not in pred.dims or dim not in ref.dims:
        raise ValueError(f"{dim!r} is not a dimension of pred/ref.")

    ref_values = np.asarray(ref.transpose(dim).values, dtype=float)
    pred_values = np.asarray(pred.transpose(dim).values, dtype=float)
    if ref_values.ndim != 1 or pred_values.ndim != 1:
        raise ValueError("pred and ref must be 1-D along the track dimension.")
    if ref_values.shape != pred_values.shape:
        raise ValueError("pred and ref must have the same shape.")

    lon_values = _coord_values_from_da(lon, name="lon", dim=dim)
    lat_values = _coord_values_from_da(lat, name="lat", dim=dim)
    spacing = (
        _median_dx_km(lon_values, lat_values)
        if spacing_km is None
        else float(spacing_km)
    )
    fs = 1.0 / spacing
    if max_gap is not None:
        if time is None:
            raise ValueError("time DataArray is required when max_gap is not None.")
        gaps = _gap_indices_from_da(time, dim=dim, max_gap=max_gap)
    else:
        gaps = np.empty(0, dtype=int)
    bounds = _segment_bounds(
        ref_values.size, npt=npt, overlap=overlap, gap_indices=gaps
    )

    ref_segments = _segment_stack(ref_values, bounds)
    pred_segments = _segment_stack(pred_values, bounds)
    lon_segments = _segment_stack(lon_values, bounds)
    lat_segments = _segment_stack(lat_values, bounds)
    if ref_segments.size == 0:
        freqs, psd_ref, psd_pred, psd_err, coherence = _empty_spectra(
            npt=npt, fs=fs, window=window, scaling=scaling
        )
        segment_lons = np.empty(0, dtype=float)
        segment_lats = np.empty(0, dtype=float)
    else:
        finite = np.all(
            np.isfinite(ref_segments)
            & np.isfinite(pred_segments)
            & np.isfinite(lon_segments)
            & np.isfinite(lat_segments),
            axis=1,
        )
        bounds = [bounds[i] for i in np.flatnonzero(finite)]
        ref_segments = ref_segments[finite]
        pred_segments = pred_segments[finite]
        if ref_segments.shape[0] == 0:
            freqs, psd_ref, psd_pred, psd_err, coherence = _empty_spectra(
                npt=npt, fs=fs, window=window, scaling=scaling
            )
            segment_lons = np.empty(0, dtype=float)
            segment_lats = np.empty(0, dtype=float)
        else:
            err_segments = pred_segments - ref_segments
            freqs, psd_ref = signal.welch(
                ref_segments,
                fs=fs,
                window=window,
                nperseg=npt,
                noverlap=0,
                scaling=scaling,
                axis=-1,
            )
            _, psd_pred = signal.welch(
                pred_segments,
                fs=fs,
                window=window,
                nperseg=npt,
                noverlap=0,
                scaling=scaling,
                axis=-1,
            )
            _, psd_err = signal.welch(
                err_segments,
                fs=fs,
                window=window,
                nperseg=npt,
                noverlap=0,
                scaling=scaling,
                axis=-1,
            )
            _, csd = signal.csd(
                ref_segments,
                pred_segments,
                fs=fs,
                window=window,
                nperseg=npt,
                noverlap=0,
                scaling=scaling,
                axis=-1,
            )
            coherence = np.abs(csd) ** 2 / (psd_ref * psd_pred)
            segment_lons = _segment_lons(lon_values, bounds)
            segment_lats = np.asarray(
                [np.nanmedian(lat_values[start:stop]) for start, stop in bounds],
                dtype=float,
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(
            np.isfinite(psd_ref) & (psd_ref > 0.0),
            1.0 - psd_err / psd_ref,
            np.nan,
        )
        wavelength = 1.0 / freqs

    coords = {
        "segment": np.arange(segment_lons.size),
        "wavenumber": ("wavenumber", freqs, {"units": "cycles/km"}),
        "wavelength": ("wavenumber", wavelength, {"units": "km"}),
        "segment_lon": ("segment", segment_lons, {"units": "degrees_east"}),
        "segment_lat": ("segment", segment_lats, {"units": "degrees_north"}),
    }
    return xr.Dataset(
        data_vars={
            "psd_ref": (("segment", "wavenumber"), psd_ref),
            "psd_pred": (("segment", "wavenumber"), psd_pred),
            "psd_err": (("segment", "wavenumber"), psd_err),
            "psd_score": (("segment", "wavenumber"), score),
            "coherence": (("segment", "wavenumber"), coherence),
        },
        coords=coords,
        attrs={"spacing_km": spacing},
    )


def psd_score_by_region(
    ds_segments: xr.Dataset,
    *,
    lat_centers: ArrayLike = _DEFAULT_LAT_CENTERS,
    lon_centers: ArrayLike = _DEFAULT_LON_CENTERS,
    delta_lat: float = 10.0,
    delta_lon: float = 10.0,
    min_segments: int = 2,
) -> xr.Dataset:
    """Average per-segment PSDs into overlapping circular-longitude boxes."""
    lat_values = np.asarray(ds_segments["segment_lat"].values, dtype=float)
    lon_values = np.mod(
        np.asarray(ds_segments["segment_lon"].values, dtype=float), 360.0
    )
    lat_out = np.asarray(lat_centers, dtype=float)
    # Normalize to [0, 360) so callers can pass either [-180, 180] or
    # [0, 360] convention; the circular-distance computation downstream
    # assumes a common modulus.
    lon_out = np.mod(np.asarray(lon_centers, dtype=float), 360.0)
    wavenumber = np.asarray(ds_segments["wavenumber"].values, dtype=float)

    data_names = [
        name
        for name in ("psd_ref", "psd_pred", "psd_err", "coherence")
        if name in ds_segments
    ]
    out = {
        name: np.full(
            (lat_out.size, lon_out.size, wavenumber.size), np.nan, dtype=float
        )
        for name in data_names
    }
    counts = np.zeros((lat_out.size, lon_out.size), dtype=int)

    for i, lat_center in enumerate(lat_out):
        lat_mask = np.abs(lat_values - lat_center) <= delta_lat / 2.0
        for j, lon_center in enumerate(lon_out):
            # Wrap to [-180, 180] to get shortest circular longitude distance.
            lon_delta = np.abs((lon_values - lon_center + 180.0) % 360.0 - 180.0)
            mask = lat_mask & (lon_delta <= delta_lon / 2.0)
            counts[i, j] = int(np.count_nonzero(mask))
            if counts[i, j] < min_segments:
                continue
            for name in data_names:
                out[name][i, j] = np.nanmean(ds_segments[name].values[mask], axis=0)

    coords = {
        "lat": ("lat", lat_out, {"units": "degrees_north"}),
        "lon": ("lon", lon_out, {"units": "degrees_east"}),
        "wavenumber": ("wavenumber", wavenumber, ds_segments["wavenumber"].attrs),
    }
    if "wavelength" in ds_segments:
        coords["wavelength"] = (
            "wavenumber",
            ds_segments["wavelength"].values,
            ds_segments["wavelength"].attrs,
        )

    data_vars: dict[str, Any] = {
        name: (("lat", "lon", "wavenumber"), values) for name, values in out.items()
    }
    # psd_score is only well-defined when both reference and error PSDs
    # are present; otherwise emit NaN rather than raising on a missing key.
    if "psd_ref" in out and "psd_err" in out:
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(
                np.isfinite(out["psd_ref"]) & (out["psd_ref"] > 0.0),
                1.0 - out["psd_err"] / out["psd_ref"],
                np.nan,
            )
        data_vars["psd_score"] = (("lat", "lon", "wavenumber"), score)
    data_vars["n_segments"] = (("lat", "lon"), counts)
    return xr.Dataset(data_vars=data_vars, coords=coords)


class SegmentedPSDScore(Operator):
    """Single-input operator for along-track segmented PSD score."""

    def __init__(
        self,
        *,
        var_ref: str,
        var_pred: str,
        npt: int,
        dim: str = "num_lines",
        overlap: float = 0.5,
        max_gap: np.timedelta64 | None = _DEFAULT_MAX_GAP,
        spacing_km: float | None = None,
        lon: str = "lon",
        lat: str = "lat",
        time: str = "time",
        window: str | tuple[str, float] | ArrayLike = "hann",
        scaling: str = "density",
    ):
        self.var_ref = var_ref
        self.var_pred = var_pred
        self.npt = npt
        self.dim = dim
        self.overlap = overlap
        self.max_gap = max_gap
        self.spacing_km = spacing_km
        self.lon = lon
        self.lat = lat
        self.time = time
        self.window = window
        self.scaling = scaling

    def _apply(self, ds_track: xr.Dataset) -> xr.Dataset:
        if self.dim not in ds_track.dims:
            raise ValueError(f"{self.dim!r} is not a dimension in the track Dataset.")
        for name in (self.var_ref, self.var_pred, self.lon, self.lat):
            if name not in ds_track:
                raise ValueError(f"{name!r} is not present in the track Dataset.")
        time_da = ds_track.get(self.time, None)
        if self.max_gap is not None and time_da is None:
            raise ValueError(f"{self.time!r} is required when max_gap is not None.")
        return along_track_psd_score(
            ds_track[self.var_pred],
            ds_track[self.var_ref],
            lon=ds_track[self.lon],
            lat=ds_track[self.lat],
            time=time_da,
            dim=self.dim,
            npt=self.npt,
            overlap=self.overlap,
            max_gap=self.max_gap,
            spacing_km=self.spacing_km,
            window=self.window,
            scaling=self.scaling,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "var_ref": self.var_ref,
            "var_pred": self.var_pred,
            "npt": self.npt,
            "dim": self.dim,
            "overlap": self.overlap,
            "max_gap": self.max_gap,
            "spacing_km": self.spacing_km,
            "lon": self.lon,
            "lat": self.lat,
            "time": self.time,
            "window": self.window,
            "scaling": self.scaling,
        }


__all__ = [
    "SegmentedPSDScore",
    "along_track_psd_score",
    "psd_score_by_region",
]

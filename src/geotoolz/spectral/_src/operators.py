"""Tier-B Operators for band-space spectral operations.

Band names are resolved from an explicit ``band_names=`` constructor argument
when present; otherwise operators look for ``gt.attrs["band_names"]``. Wavelength
dependent operators follow the same convention with explicit
``source_wavelengths=`` / ``wavelengths=`` arguments first, then
``gt.attrs["wavelengths"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from georeader.reflectance import srf, transform_to_srf

from geotoolz.core import Operator
from geotoolz.spectral._src.array import (
    band_ratio,
    continuum_removal,
    evaluate_band_math,
    normalized_difference,
    reorder_bands,
    select_bands,
    spectral_binning,
    spectral_smoothing,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


BandKey = int | str


def _attrs(gt: GeoTensor) -> dict[str, Any]:
    return dict(getattr(gt, "attrs", None) or {})


def _band_names(gt: GeoTensor, band_names: list[str] | None) -> list[str] | None:
    if band_names is not None:
        return list(band_names)
    names = _attrs(gt).get("band_names")
    if names is None:
        return None
    return [str(name) for name in names]


def _default_band_names(n_bands: int) -> list[str]:
    return [f"B{idx}" for idx in range(n_bands)]


def _resolve_band(key: BandKey, names: list[str] | None) -> int:
    if isinstance(key, int):
        return key
    if names is None:
        raise ValueError(
            f"Cannot resolve band name {key!r}: band_names must be provided "
            "or present in gt.attrs['band_names']"
        )
    try:
        return names.index(key)
    except ValueError as exc:
        raise ValueError(f"Band name {key!r} is not present in band_names") from exc


def _resolve_bands(keys: list[BandKey], names: list[str] | None) -> list[int]:
    return [_resolve_band(key, names) for key in keys]


def _jsonable_array(values: np.ndarray | list[float]) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _wrap_like(
    gt: GeoTensor,
    values: np.ndarray,
    *,
    band_names: list[str] | None = None,
    wavelengths: np.ndarray | None = None,
) -> GeoTensor:
    out = gt.array_as_geotensor(values)
    new_attrs = _attrs(gt)
    if band_names is not None:
        new_attrs["band_names"] = band_names
    if wavelengths is not None:
        new_attrs["wavelengths"] = _jsonable_array(wavelengths)
    out.attrs = new_attrs
    return out


def _wavelengths(
    gt: GeoTensor, wavelengths: np.ndarray | list[float] | None
) -> np.ndarray:
    if wavelengths is None:
        wavelengths = _attrs(gt).get("wavelengths")
    if wavelengths is None:
        raise ValueError(
            "wavelengths must be provided or available as gt.attrs['wavelengths']"
        )
    return np.asarray(wavelengths, dtype=float)


class SelectBands(Operator):
    """Select bands by integer index or band name."""

    def __init__(self, *, indexes: list[BandKey], axis: int = 0) -> None:
        self.indexes = indexes
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        names = _band_names(gt, None)
        indexes = _resolve_bands(self.indexes, names)
        selected_names = [names[idx] for idx in indexes] if names is not None else None
        wavelengths = _attrs(gt).get("wavelengths")
        selected_wavelengths = (
            np.asarray(wavelengths, dtype=float)[indexes]
            if wavelengths is not None
            else None
        )
        out = select_bands(np.asarray(gt), indexes, axis=self.axis)
        return _wrap_like(
            gt, out, band_names=selected_names, wavelengths=selected_wavelengths
        )

    def get_config(self) -> dict[str, Any]:
        return {"indexes": self.indexes, "axis": self.axis}


class ReorderBands(SelectBands):
    """Reorder bands by integer index or band name."""

    def __init__(self, *, order: list[BandKey], axis: int = 0) -> None:
        super().__init__(indexes=order, axis=axis)
        self.order = order

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        names = _band_names(gt, None)
        order = _resolve_bands(self.order, names)
        ordered_names = [names[idx] for idx in order] if names is not None else None
        wavelengths = _attrs(gt).get("wavelengths")
        ordered_wavelengths = (
            np.asarray(wavelengths, dtype=float)[order]
            if wavelengths is not None
            else None
        )
        out = reorder_bands(np.asarray(gt), order, axis=self.axis)
        return _wrap_like(
            gt, out, band_names=ordered_names, wavelengths=ordered_wavelengths
        )

    def get_config(self) -> dict[str, Any]:
        return {"order": self.order, "axis": self.axis}


class StackBands(Operator):
    """Concatenate GeoTensors along the band axis."""

    def __init__(self, *, axis: int = 0, along: str = "band") -> None:
        self.axis = axis
        self.along = along

    def _apply(self, tensors: list[GeoTensor]) -> GeoTensor:
        if not tensors:
            raise ValueError("StackBands requires at least one GeoTensor")
        first = tensors[0]
        arrays = []
        names: list[str] = []
        wavelengths: list[float] = []
        have_names = True
        have_wavelengths = True
        for gt in tensors:
            if gt.shape[-2:] != first.shape[-2:]:
                raise ValueError(
                    "All GeoTensors must share spatial shape; "
                    f"got {first.shape[-2:]} and {gt.shape[-2:]}"
                )
            if gt.transform != first.transform or gt.crs != first.crs:
                raise ValueError("All GeoTensors must share transform and CRS")
            arr = np.asarray(gt)
            expanded = np.expand_dims(arr, axis=self.axis) if arr.ndim == 2 else arr
            arrays.append(expanded)
            gt_names = _attrs(gt).get("band_names")
            gt_wavelengths = _attrs(gt).get("wavelengths")
            have_names = have_names and gt_names is not None
            have_wavelengths = have_wavelengths and gt_wavelengths is not None
            if gt_names is not None:
                names.extend(str(name) for name in gt_names)
            if gt_wavelengths is not None:
                wavelengths.extend(float(wavelength) for wavelength in gt_wavelengths)
        out = np.concatenate(arrays, axis=self.axis)
        return _wrap_like(
            first,
            out,
            band_names=names if have_names else None,
            wavelengths=np.asarray(wavelengths) if have_wavelengths else None,
        )

    def get_config(self) -> dict[str, Any]:
        return {"axis": self.axis, "along": self.along}


class SplitBands(Operator):
    """Return a list of single-band GeoTensors."""

    def __init__(self, *, names: list[str] | None = None, axis: int = 0) -> None:
        self.names = names
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> list[GeoTensor]:
        arr = np.asarray(gt)
        axis = self.axis + arr.ndim if self.axis < 0 else self.axis
        n_bands = arr.shape[axis]
        source_names = _band_names(gt, self.names)
        if source_names is not None and len(source_names) != n_bands:
            raise ValueError("names length must match the number of bands")
        wavelengths = _attrs(gt).get("wavelengths")
        outputs = []
        for idx in range(n_bands):
            out = np.take(arr, [idx], axis=axis)
            band_name = [source_names[idx]] if source_names is not None else None
            band_wavelength = (
                np.asarray([np.asarray(wavelengths, dtype=float)[idx]])
                if wavelengths is not None
                else None
            )
            outputs.append(
                _wrap_like(gt, out, band_names=band_name, wavelengths=band_wavelength)
            )
        return outputs

    def get_config(self) -> dict[str, Any]:
        return {"names": self.names, "axis": self.axis}


class BandMath(Operator):
    """Evaluate a restricted arithmetic expression over named bands."""

    def __init__(
        self,
        *,
        expression: str,
        band_names: list[str] | None = None,
        axis: int = 0,
    ) -> None:
        self.expression = expression
        self.band_names = band_names
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt)
        names = _band_names(gt, self.band_names) or _default_band_names(
            arr.shape[self.axis]
        )
        variables = {
            name: np.take(arr, idx, axis=self.axis) for idx, name in enumerate(names)
        }
        out = evaluate_band_math(self.expression, variables)
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "band_names": self.band_names,
            "axis": self.axis,
        }


class NormalizedDifference(Operator):
    """Compute ``(a - b) / (a + b + eps)`` by band index or name."""

    def __init__(
        self, *, a: BandKey, b: BandKey, eps: float = 1e-6, axis: int = 0
    ) -> None:
        self.a = a
        self.b = b
        self.eps = eps
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        names = _band_names(gt, None)
        out = normalized_difference(
            np.asarray(gt),
            _resolve_band(self.a, names),
            _resolve_band(self.b, names),
            axis=self.axis,
            eps=self.eps,
        )
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "eps": self.eps, "axis": self.axis}


class BandRatio(Operator):
    """Compute ``numerator / (denominator + eps)`` by band index or name."""

    def __init__(
        self,
        *,
        numerator: BandKey,
        denominator: BandKey,
        eps: float = 1e-6,
        axis: int = 0,
    ) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.eps = eps
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        names = _band_names(gt, None)
        out = band_ratio(
            np.asarray(gt),
            _resolve_band(self.numerator, names),
            _resolve_band(self.denominator, names),
            axis=self.axis,
            eps=self.eps,
        )
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "eps": self.eps,
            "axis": self.axis,
        }


class ApplySRF(Operator):
    """Convolve band-first hyperspectral data to target Gaussian SRFs."""

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list[float],
        target_fwhm: np.ndarray | list[float],
        source_wavelengths: np.ndarray | list[float],
        band_names: list[str] | None = None,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.band_names = band_names

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        source_wavelengths = np.asarray(self.source_wavelengths, dtype=float)
        target_center_wavelengths = np.asarray(
            self.target_center_wavelengths, dtype=float
        )
        responses = srf(target_center_wavelengths, self.target_fwhm, source_wavelengths)
        names = self.band_names or [
            f"B{idx}" for idx in range(target_center_wavelengths.size)
        ]
        srf_df = pd.DataFrame(responses, index=source_wavelengths, columns=names)
        out = transform_to_srf(
            gt,
            srf_df,
            source_wavelengths.tolist(),
            fill_value_default=gt.fill_value_default,
        )
        out.attrs = _attrs(gt)
        out.attrs["band_names"] = names
        out.attrs["wavelengths"] = _jsonable_array(target_center_wavelengths)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": _jsonable_array(
                self.target_center_wavelengths
            ),
            "target_fwhm": _jsonable_array(self.target_fwhm),
            "source_wavelengths": _jsonable_array(self.source_wavelengths),
            "band_names": self.band_names,
        }


class GaussianSRF(Operator):
    """Convolve to synthetic Gaussian SRFs using source wavelengths from attrs."""

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list[float],
        target_fwhm: np.ndarray | list[float],
        source_wavelengths: np.ndarray | list[float] | None = None,
        band_names: list[str] | None = None,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.band_names = band_names

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return ApplySRF(
            target_center_wavelengths=self.target_center_wavelengths,
            target_fwhm=self.target_fwhm,
            source_wavelengths=_wavelengths(gt, self.source_wavelengths),
            band_names=self.band_names,
        )(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": _jsonable_array(
                self.target_center_wavelengths
            ),
            "target_fwhm": _jsonable_array(self.target_fwhm),
            "source_wavelengths": (
                None
                if self.source_wavelengths is None
                else _jsonable_array(self.source_wavelengths)
            ),
            "band_names": self.band_names,
        }


class ContinuumRemoval(Operator):
    """Hull-quotient continuum removal along the band axis."""

    def __init__(
        self,
        *,
        method: str = "convex_hull",
        wavelengths: np.ndarray | list[float] | None = None,
        axis: int = 0,
    ) -> None:
        self.method = method
        self.wavelengths = wavelengths
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = continuum_removal(
            np.asarray(gt),
            _wavelengths(gt, self.wavelengths),
            axis=self.axis,
            method=self.method,
        )
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "wavelengths": None
            if self.wavelengths is None
            else _jsonable_array(self.wavelengths),
            "axis": self.axis,
        }


class SpectralBinning(Operator):
    """Aggregate source bands into wavelength-centered bins."""

    def __init__(
        self,
        *,
        target_wavelengths: np.ndarray | list[float],
        width: float | np.ndarray | list[float],
        method: str = "mean",
        source_wavelengths: np.ndarray | list[float] | None = None,
        axis: int = 0,
    ) -> None:
        self.target_wavelengths = target_wavelengths
        self.width = width
        self.method = method
        self.source_wavelengths = source_wavelengths
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        target_wavelengths = np.asarray(self.target_wavelengths, dtype=float)
        out = spectral_binning(
            np.asarray(gt),
            _wavelengths(gt, self.source_wavelengths),
            target_wavelengths,
            self.width,
            axis=self.axis,
            method=self.method,
        )
        return _wrap_like(gt, out, wavelengths=target_wavelengths)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_wavelengths": _jsonable_array(self.target_wavelengths),
            "width": _jsonable_array(np.asarray(self.width, dtype=float)),
            "method": self.method,
            "source_wavelengths": (
                None
                if self.source_wavelengths is None
                else _jsonable_array(self.source_wavelengths)
            ),
            "axis": self.axis,
        }


class SpectralSmoothing(Operator):
    """Smooth spectra along the band axis."""

    def __init__(
        self,
        *,
        method: str = "savgol",
        window: int = 7,
        polyorder: int = 2,
        axis: int = 0,
    ) -> None:
        self.method = method
        self.window = window
        self.polyorder = polyorder
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = spectral_smoothing(
            np.asarray(gt),
            axis=self.axis,
            method=self.method,
            window=self.window,
            polyorder=self.polyorder,
        )
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "window": self.window,
            "polyorder": self.polyorder,
            "axis": self.axis,
        }

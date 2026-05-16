"""Carrier-aware plume operators for retrieval post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from rasterio import features
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

from geotoolz.core import Operator
from geotoolz.plume._src.array import (
    ColumnUnit,
    Connectivity,
    ThresholdMode,
    convert_column_units,
    label_components,
    pixel_area,
    pixel_centers,
    plume_length,
    plume_mask,
    squeeze_single_band,
    wind_advection_cone,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


S2_BAND_TO_INDEX = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B9": 9,
    "B11": 10,
    "B12": 11,
}


def _band_index(band: int | str) -> int:
    if isinstance(band, str):
        key = band.upper()
        if key not in S2_BAND_TO_INDEX:
            raise ValueError(f"unknown Sentinel-2 band name {band!r}")
        return S2_BAND_TO_INDEX[key]
    return int(band)


def _extract_and_clip_band(arr: np.ndarray, band: int | str, axis: int) -> np.ndarray:
    return np.maximum(np.take(arr, _band_index(band), axis=axis), 0.0)


class SBMP(Operator):
    """Single-band multi-pass Sentinel-2 SWIR ratio retrieval.

    The operator emits a single-band enhancement score. With a
    ``reference_scene`` it returns the change in log(SWIR1 / SWIR2)
    between the input and reference scenes. Without a reference scene it
    falls back to a same-scene normalized SWIR contrast.

    The reference-scene formulation is
    ``log((SWIR1 + eps) / (SWIR2 + eps)) -
    log((ref_SWIR1 + eps) / (ref_SWIR2 + eps))``. Inputs are expected to
    be non-negative radiance or reflectance values; negative values are
    clipped to zero, then ``eps`` is added to prevent division by zero in
    the log-ratio.
    """

    def __init__(
        self,
        *,
        swir1: int | str = "B11",
        swir2: int | str = "B12",
        reference_scene: GeoTensor | None = None,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.swir1 = swir1
        self.swir2 = swir2
        self.reference_scene = reference_scene
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = np.asarray(gt, dtype=float)
        swir1 = _extract_and_clip_band(arr, self.swir1, self.axis)
        swir2 = _extract_and_clip_band(arr, self.swir2, self.axis)
        ratio = np.log((swir1 + self.eps) / (swir2 + self.eps))
        if self.reference_scene is not None:
            ref = np.asarray(self.reference_scene, dtype=float)
            ref_swir1 = _extract_and_clip_band(ref, self.swir1, self.axis)
            ref_swir2 = _extract_and_clip_band(ref, self.swir2, self.axis)
            ref_ratio = np.log((ref_swir1 + self.eps) / (ref_swir2 + self.eps))
            out = ratio - ref_ratio
        else:
            out = (swir1 - swir2) / (swir1 + swir2 + self.eps)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "swir1": self.swir1,
            "swir2": self.swir2,
            "reference_scene": None,
            "axis": self.axis,
            "eps": self.eps,
        }
        if self.reference_scene is not None:
            config["reference_scene"] = {
                "shape": list(np.asarray(self.reference_scene).shape),
                "dtype": str(np.asarray(self.reference_scene).dtype),
            }
        return config


class PlumeMask(Operator):
    """Binary mask from a single-band score or enhancement map."""

    def __init__(
        self,
        *,
        threshold: ThresholdMode = "otsu",
        min_area: int = 50,
        connectivity: Connectivity = 8,
    ) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.connectivity = connectivity

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        mask = plume_mask(
            np.asarray(gt),
            threshold=self.threshold,
            min_area=self.min_area,
            connectivity=self.connectivity,
        )
        return gt.array_as_geotensor(mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "min_area": self.min_area,
            "connectivity": self.connectivity,
        }


class PlumeContours(Operator):
    """Connected-component labelling of a plume mask."""

    def __init__(
        self,
        *,
        min_area: int = 50,
        return_labels: bool = True,
        connectivity: Connectivity = 8,
    ) -> None:
        self.min_area = min_area
        self.return_labels = return_labels
        self.connectivity = connectivity

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        labels = label_components(
            squeeze_single_band(np.asarray(gt)).astype(bool),
            min_area=self.min_area,
            connectivity=self.connectivity,
        )
        out = labels if self.return_labels else labels > 0
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "return_labels": self.return_labels,
            "connectivity": self.connectivity,
        }


class PlumeFootprint(Operator):
    """Vectorize a plume mask into polygons with per-plume metadata."""

    def __init__(
        self,
        *,
        min_area_m2: float = 500.0,
        simplify_tolerance: float | None = 15.0,
        enhancement: GeoTensor | None = None,
    ) -> None:
        self.min_area_m2 = min_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.enhancement = enhancement

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        mask_arr = squeeze_single_band(np.asarray(gt))
        if mask_arr.dtype == bool:
            labels = label_components(mask_arr, min_area=1, connectivity=8)
        else:
            labels = mask_arr.astype(np.int32, copy=False)
        enh = (
            None
            if self.enhancement is None
            else squeeze_single_band(np.asarray(self.enhancement))
        )

        rows: list[dict[str, Any]] = []
        for label_id in sorted(int(v) for v in np.unique(labels) if v != 0):
            component = labels == label_id
            n_pixels = int(component.sum())
            geometries = [
                shape(geom)
                for geom, value in features.shapes(
                    labels.astype(np.int32), mask=component, transform=gt.transform
                )
                if int(value) == label_id
            ]
            if not geometries:
                continue
            geometry = unary_union(geometries)
            if self.simplify_tolerance is not None:
                geometry = geometry.simplify(
                    self.simplify_tolerance, preserve_topology=True
                )
            area_m2 = float(geometry.area)
            if area_m2 < self.min_area_m2:
                continue
            component_values = None if enh is None else enh[component]
            rows.append(
                {
                    "geometry": geometry,
                    "area_m2": area_m2,
                    "centroid": geometry.centroid,
                    "mean_enhancement": None
                    if component_values is None
                    else float(np.nanmean(component_values)),
                    "max_enhancement": None
                    if component_values is None
                    else float(np.nanmax(component_values)),
                    "n_pixels": n_pixels,
                    "label_id": label_id,
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gt.crs)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area_m2": self.min_area_m2,
            "simplify_tolerance": self.simplify_tolerance,
            "enhancement": None
            if self.enhancement is None
            else {
                "shape": list(np.asarray(self.enhancement).shape),
                "dtype": str(np.asarray(self.enhancement).dtype),
            },
        }


class WindAdvectionCone(Operator):
    """Mask likely downwind plume extent from a source point and wind vector."""

    def __init__(
        self,
        *,
        source: tuple[float, float],
        wind_u: float,
        wind_v: float,
        half_angle_deg: float = 30.0,
        max_distance: float = 5000.0,
        crs: str | None = None,
    ) -> None:
        self.source = source
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.half_angle_deg = half_angle_deg
        self.max_distance = max_distance
        self.crs = crs

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        arr = squeeze_single_band(np.asarray(gt))
        source = self.source
        if self.crs is not None and gt.crs is not None:
            src_crs = CRS.from_user_input(self.crs)
            dst_crs = CRS.from_user_input(gt.crs)
            if src_crs != dst_crs:
                transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                source = transformer.transform(*source)
        mask = wind_advection_cone(
            arr.shape,
            gt.transform,
            source=source,
            wind_u=self.wind_u,
            wind_v=self.wind_v,
            half_angle_deg=self.half_angle_deg,
            max_distance=self.max_distance,
        )
        return gt.array_as_geotensor(mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "wind_u": self.wind_u,
            "wind_v": self.wind_v,
            "half_angle_deg": self.half_angle_deg,
            "max_distance": self.max_distance,
            "crs": self.crs,
        }


class IMEEstimate(Operator):
    """Estimate emission rate with Q = IME * U_eff / L.

    When uncertainty output is requested, ``uncertainty_fraction`` is
    applied as a simple fractional uncertainty on the emission rate.
    """

    def __init__(
        self,
        *,
        plume_mask: GeoTensor,
        wind_speed: float,
        length_method: Literal["max_axis", "convex_hull", "skeleton"] = "max_axis",
        pixel_area_m2: float | None = None,
        return_uncertainty: bool = True,
        uncertainty_fraction: float = 0.5,
    ) -> None:
        self.plume_mask = plume_mask
        self.wind_speed = wind_speed
        self.length_method = length_method
        self.pixel_area_m2 = pixel_area_m2
        self.return_uncertainty = return_uncertainty
        if uncertainty_fraction < 0.0:
            raise ValueError("uncertainty_fraction must be non-negative")
        self.uncertainty_fraction = uncertainty_fraction

    def _apply(self, gt: GeoTensor) -> dict[str, float]:
        mask = squeeze_single_band(np.asarray(self.plume_mask)).astype(bool)
        enhancement = squeeze_single_band(np.asarray(gt, dtype=float))
        area = (
            self.pixel_area_m2
            if self.pixel_area_m2 is not None
            else pixel_area(gt.transform)
        )
        ime_kg = float(np.nansum(enhancement[mask] * area))
        length_m = plume_length(mask, gt.transform, method=self.length_method)
        rate = 0.0 if length_m == 0.0 else ime_kg * self.wind_speed / length_m
        out = {
            "ime_kg": ime_kg,
            "length_m": float(length_m),
            "wind_speed_m_s": float(self.wind_speed),
            "emission_rate_kg_s": float(rate),
        }
        if self.return_uncertainty:
            out["emission_rate_uncertainty_kg_s"] = float(
                abs(rate) * self.uncertainty_fraction
            )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "plume_mask": {
                "shape": list(np.asarray(self.plume_mask).shape),
                "dtype": str(np.asarray(self.plume_mask).dtype),
            },
            "wind_speed": self.wind_speed,
            "length_method": self.length_method,
            "pixel_area_m2": self.pixel_area_m2,
            "return_uncertainty": self.return_uncertainty,
            "uncertainty_fraction": self.uncertainty_fraction,
        }


class CrossSectionalFlux(Operator):
    """Estimate cross-sectional flux at downwind transects."""

    def __init__(
        self,
        *,
        plume_mask: GeoTensor,
        source: tuple[float, float],
        wind_u: float,
        wind_v: float,
        n_transects: int = 5,
        transect_spacing_m: float = 100.0,
    ) -> None:
        self.plume_mask = plume_mask
        self.source = source
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.n_transects = n_transects
        self.transect_spacing_m = transect_spacing_m

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        wind_norm = float(np.hypot(self.wind_u, self.wind_v))
        if wind_norm == 0.0:
            raise ValueError("wind vector must be non-zero")
        mask = squeeze_single_band(np.asarray(self.plume_mask)).astype(bool)
        enhancement = squeeze_single_band(np.asarray(gt, dtype=float))
        area = pixel_area(gt.transform)
        width = float(np.sqrt(area))
        xs, ys = pixel_centers(mask.shape, gt.transform)
        dx = xs - self.source[0]
        dy = ys - self.source[1]
        along = (dx * self.wind_u + dy * self.wind_v) / wind_norm
        across = (-dx * self.wind_v + dy * self.wind_u) / wind_norm
        rows: list[dict[str, Any]] = []
        for idx in range(self.n_transects):
            distance = (idx + 1) * self.transect_spacing_m
            on_transect = mask & (np.abs(along - distance) <= width / 2.0)
            flux = float(np.nansum(enhancement[on_transect] * wind_norm * width))
            half_len = max(
                float(np.max(np.abs(across[mask]))) if mask.any() else width,
                width,
            )
            cx = self.source[0] + (self.wind_u / wind_norm) * distance
            cy = self.source[1] + (self.wind_v / wind_norm) * distance
            px = -self.wind_v / wind_norm
            py = self.wind_u / wind_norm
            line = LineString(
                [
                    (cx - px * half_len, cy - py * half_len),
                    (cx + px * half_len, cy + py * half_len),
                ]
            )
            rows.append(
                {
                    "geometry": line,
                    "transect_id": idx + 1,
                    "distance_m": float(distance),
                    "flux_kg_s": flux,
                    "n_pixels": int(on_transect.sum()),
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gt.crs)

    def get_config(self) -> dict[str, Any]:
        return {
            "plume_mask": {
                "shape": list(np.asarray(self.plume_mask).shape),
                "dtype": str(np.asarray(self.plume_mask).dtype),
            },
            "source": self.source,
            "wind_u": self.wind_u,
            "wind_v": self.wind_v,
            "n_transects": self.n_transects,
            "transect_spacing_m": self.transect_spacing_m,
        }


class ColumnToMass(Operator):
    """Convert column enhancement units to mass density or related units."""

    def __init__(
        self,
        *,
        gas: str = "CH4",
        units_in: ColumnUnit = "ppm_m",
        units_out: ColumnUnit = "kg_m2",
    ) -> None:
        self.gas = gas
        self.units_in = units_in
        self.units_out = units_out

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = convert_column_units(
            np.asarray(gt),
            gas=self.gas,
            units_in=self.units_in,
            units_out=self.units_out,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gas": self.gas,
            "units_in": self.units_in,
            "units_out": self.units_out,
        }

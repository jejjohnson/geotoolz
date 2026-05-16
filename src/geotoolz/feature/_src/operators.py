"""Carrier-aware wrappers around :mod:`skimage.feature` primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from skimage.feature import (
    blob_dog,
    blob_doh,
    blob_log,
    canny,
    corner_harris,
    corner_peaks,
    hog,
    multiscale_basic_features,
    peak_local_max,
    structure_tensor,
    structure_tensor_eigenvalues,
)
from skimage.transform import (
    hough_circle,
    hough_circle_peaks,
    hough_line,
    hough_line_peaks,
)

from geotoolz.core import Operator


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _single_band(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError("feature operator expects shape (H, W) or (1, H, W)")


def _xy(transform: Any, row: float, col: float) -> tuple[float, float]:
    x = transform.c + transform.a * (col + 0.5) + transform.b * (row + 0.5)
    y = transform.f + transform.d * (col + 0.5) + transform.e * (row + 0.5)
    return float(x), float(y)


def _points(
    gt: GeoTensor,
    rows: np.ndarray,
    cols: np.ndarray,
    data: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    frame = pd.DataFrame({"row": rows, "col": cols, **(data or {})})
    geometry = [
        Point(_xy(gt.transform, row, col)) for row, col in zip(rows, cols, strict=True)
    ]
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=gt.crs)


class PeakLocalMax(Operator):
    """Find local maxima and return point features with pixel scores."""

    def __init__(
        self,
        *,
        min_distance: int = 1,
        threshold_abs: float | None = None,
        threshold_rel: float | None = None,
        exclude_border: bool | int = True,
    ) -> None:
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.exclude_border = exclude_border

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        image = _single_band(np.asarray(gt, dtype=float))
        coords = peak_local_max(
            image,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            threshold_rel=self.threshold_rel,
            exclude_border=self.exclude_border,
        )
        if coords.size == 0:
            return gpd.GeoDataFrame(
                {"row": [], "col": [], "score": []}, geometry=[], crs=gt.crs
            )
        rows, cols = coords[:, 0], coords[:, 1]
        return _points(gt, rows, cols, {"score": image[rows, cols]})

    def get_config(self) -> dict[str, Any]:
        return {
            "min_distance": self.min_distance,
            "threshold_abs": self.threshold_abs,
            "threshold_rel": self.threshold_rel,
            "exclude_border": self.exclude_border,
        }


class _BlobBase(Operator):
    _func: Any

    def __init__(
        self,
        *,
        min_sigma: float = 1.0,
        max_sigma: float = 50.0,
        num_sigma: int = 10,
        threshold: float = 0.2,
    ) -> None:
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        blobs = self._func(
            _single_band(np.asarray(gt, dtype=float)),
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
        )
        if blobs.size == 0:
            return gpd.GeoDataFrame(
                {"row": [], "col": [], "sigma": []}, geometry=[], crs=gt.crs
            )
        return _points(
            gt,
            blobs[:, 0],
            blobs[:, 1],
            {"sigma": blobs[:, 2], "radius": blobs[:, 2] * np.sqrt(2.0)},
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "num_sigma": self.num_sigma,
            "threshold": self.threshold,
        }


class BlobLoG(_BlobBase):
    """Blob detection via Laplacian of Gaussian."""

    _func = staticmethod(blob_log)


class BlobDOG(_BlobBase):
    """Blob detection via Difference of Gaussian."""

    _func = staticmethod(blob_dog)


class BlobDoH(_BlobBase):
    """Blob detection via Determinant of Hessian."""

    _func = staticmethod(blob_doh)


class Canny(Operator):
    """Canny edge detection returning a boolean GeoTensor."""

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> None:
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        edges = canny(
            _single_band(np.asarray(gt, dtype=float)),
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
        return gt.array_as_geotensor(edges)

    def get_config(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
        }


class CornerHarris(Operator):
    """Harris corner response plus peak selection as point features."""

    def __init__(self, *, min_distance: int = 1, threshold_rel: float = 0.1) -> None:
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        response = corner_harris(_single_band(np.asarray(gt, dtype=float)))
        coords = corner_peaks(
            response,
            min_distance=self.min_distance,
            threshold_rel=self.threshold_rel,
        )
        if coords.size == 0:
            return gpd.GeoDataFrame(
                {"row": [], "col": [], "response": []}, geometry=[], crs=gt.crs
            )
        rows, cols = coords[:, 0], coords[:, 1]
        return _points(gt, rows, cols, {"response": response[rows, cols]})

    def get_config(self) -> dict[str, Any]:
        return {"min_distance": self.min_distance, "threshold_rel": self.threshold_rel}


class HOG(Operator):
    """Histogram of Oriented Gradients descriptor."""

    def __init__(
        self,
        *,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (3, 3),
    ) -> None:
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def _apply(self, gt: GeoTensor) -> np.ndarray:
        return hog(
            _single_band(np.asarray(gt, dtype=float)),
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "orientations": self.orientations,
            "pixels_per_cell": self.pixels_per_cell,
            "cells_per_block": self.cells_per_block,
        }


class StructureTensor(Operator):
    """Local structure-tensor eigenvalue stack."""

    def __init__(self, *, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        tensor = structure_tensor(
            _single_band(np.asarray(gt, dtype=float)),
            sigma=self.sigma,
        )
        eigvals = np.asarray(structure_tensor_eigenvalues(tensor))
        return gt.array_as_geotensor(eigvals)

    def get_config(self) -> dict[str, Any]:
        return {"sigma": self.sigma}


class MultiscaleBasicFeatures(Operator):
    """General-purpose multiscale feature stack."""

    def __init__(
        self,
        *,
        intensity: bool = True,
        edges: bool = True,
        texture: bool = True,
        sigma_min: float = 0.5,
        sigma_max: float = 16.0,
    ) -> None:
        self.intensity = intensity
        self.edges = edges
        self.texture = texture
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        features = multiscale_basic_features(
            _single_band(np.asarray(gt, dtype=float)),
            intensity=self.intensity,
            edges=self.edges,
            texture=self.texture,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            channel_axis=None,
        )
        return gt.array_as_geotensor(np.moveaxis(features, -1, 0))

    def get_config(self) -> dict[str, Any]:
        return {
            "intensity": self.intensity,
            "edges": self.edges,
            "texture": self.texture,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
        }


class HoughLines(Operator):
    """Detect prominent straight lines with the Hough transform."""

    def __init__(self, *, num_peaks: int = 10) -> None:
        self.num_peaks = num_peaks

    def _apply(self, gt: GeoTensor) -> pd.DataFrame:
        hspace, angles, distances = hough_line(_single_band(np.asarray(gt)))
        accum, angle_peaks, dist_peaks = hough_line_peaks(
            hspace, angles, distances, num_peaks=self.num_peaks
        )
        return pd.DataFrame(
            {"accumulator": accum, "angle": angle_peaks, "distance": dist_peaks}
        )

    def get_config(self) -> dict[str, Any]:
        return {"num_peaks": self.num_peaks}


class HoughCircles(Operator):
    """Detect circles with the circular Hough transform."""

    def __init__(self, *, radii: list[int], total_num_peaks: int = 10) -> None:
        self.radii = radii
        self.total_num_peaks = total_num_peaks

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        hspaces = hough_circle(_single_band(np.asarray(gt)), self.radii)
        accum, cx, cy, radii = hough_circle_peaks(
            hspaces,
            self.radii,
            total_num_peaks=self.total_num_peaks,
        )
        return _points(gt, cy, cx, {"radius": radii, "accumulator": accum})

    def get_config(self) -> dict[str, Any]:
        return {"radii": self.radii, "total_num_peaks": self.total_num_peaks}

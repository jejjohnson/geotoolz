"""Carrier-aware wrappers around :mod:`skimage.feature` primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pipekit import Operator
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

from geotoolz._src.shape import single_band
from geotoolz._src.wrap import wrap_like


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _require_geotensor(gt: Any, name: str) -> None:
    """Raise when a geo-dependent operator receives a plain array."""
    if not hasattr(gt, "transform"):
        raise TypeError(
            f"{name} requires a georeferenced GeoTensor input; got a plain array"
        )


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
    """Find local maxima and return point features with pixel scores.

    Wraps :func:`skimage.feature.peak_local_max` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image and returns a ``GeoDataFrame``
    with ``row``/``col`` pixel indices, the peak ``score``, and a
    ``Point`` geometry in world coordinates. Geo-dependent: requires a
    georeferenced ``GeoTensor`` input (plain arrays raise ``TypeError``).

    Args:
        min_distance: Minimum pixel distance between reported peaks.
        threshold_abs: Minimum absolute intensity of a peak.
        threshold_rel: Minimum intensity relative to the image maximum.
        exclude_border: Exclude peaks within this distance of the border
            (``True`` uses ``min_distance``).

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

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
        _require_geotensor(gt, "PeakLocalMax")
        image = single_band(np.asarray(gt, dtype=float), name="PeakLocalMax")
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
    """Common base for ``skimage.feature.blob_*`` wrappers.

    Subclasses set ``_func`` to the underlying skimage callable and
    override :meth:`_extra_kwargs` / :meth:`_radius_from_sigma` to track
    each detector's actual signature and radius convention.

    Detections are returned as a ``GeoDataFrame`` with ``row``/``col``
    pixel indices, ``sigma``/``radius`` columns, and a ``Point`` geometry
    in world coordinates. Geo-dependent: requires a georeferenced
    ``GeoTensor`` input (plain arrays raise ``TypeError``).

    Args:
        min_sigma: Smallest blob scale (Gaussian sigma) considered.
        max_sigma: Largest blob scale considered.
        threshold: Detector response threshold; lower values detect
            fainter blobs.

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

    _func: Any

    def __init__(
        self,
        *,
        min_sigma: float = 1.0,
        max_sigma: float = 50.0,
        threshold: float = 0.2,
    ) -> None:
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold

    def _extra_kwargs(self) -> dict[str, Any]:
        """Detector-specific keyword arguments for the underlying call."""
        return {}

    def _radius_from_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """Convert returned sigma values to approximate blob radii."""
        return sigma * np.sqrt(2.0)

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        name = type(self).__name__
        _require_geotensor(gt, name)
        blobs = self._func(
            single_band(np.asarray(gt, dtype=float), name=name),
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            threshold=self.threshold,
            **self._extra_kwargs(),
        )
        if blobs.size == 0:
            return gpd.GeoDataFrame(
                {"row": [], "col": [], "sigma": []}, geometry=[], crs=gt.crs
            )
        return _points(
            gt,
            blobs[:, 0],
            blobs[:, 1],
            {"sigma": blobs[:, 2], "radius": self._radius_from_sigma(blobs[:, 2])},
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "threshold": self.threshold,
        }


class BlobLoG(_BlobBase):
    """Blob detection via Laplacian of Gaussian.

    See :class:`_BlobBase` for the shared parameters and carrier
    behavior (GeoTensor-only).

    Args:
        num_sigma: Number of sigma steps between ``min_sigma`` and
            ``max_sigma``.
    """

    _func = staticmethod(blob_log)

    def __init__(
        self,
        *,
        min_sigma: float = 1.0,
        max_sigma: float = 50.0,
        num_sigma: int = 10,
        threshold: float = 0.2,
    ) -> None:
        super().__init__(min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        self.num_sigma = num_sigma

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"num_sigma": self.num_sigma}

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "num_sigma": self.num_sigma}


class BlobDOG(_BlobBase):
    """Blob detection via Difference of Gaussian.

    See :class:`_BlobBase` for the shared parameters and carrier
    behavior (GeoTensor-only).

    Args:
        sigma_ratio: Ratio between the sigmas of successive Gaussians.
    """

    _func = staticmethod(blob_dog)

    def __init__(
        self,
        *,
        min_sigma: float = 1.0,
        max_sigma: float = 50.0,
        sigma_ratio: float = 1.6,
        threshold: float = 0.2,
    ) -> None:
        super().__init__(min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        self.sigma_ratio = sigma_ratio

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"sigma_ratio": self.sigma_ratio}

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "sigma_ratio": self.sigma_ratio}


class BlobDoH(_BlobBase):
    """Blob detection via Determinant of Hessian.

    Unlike LoG/DoG, ``blob_doh`` already returns ``sigma`` values that
    approximate blob radii directly, so no ``sqrt(2)`` scaling is
    applied when populating the ``radius`` column.

    See :class:`_BlobBase` for the shared parameters and carrier
    behavior (GeoTensor-only).

    Args:
        num_sigma: Number of sigma steps between ``min_sigma`` and
            ``max_sigma``.
    """

    _func = staticmethod(blob_doh)

    def __init__(
        self,
        *,
        min_sigma: float = 1.0,
        max_sigma: float = 30.0,
        num_sigma: int = 10,
        threshold: float = 0.01,
    ) -> None:
        super().__init__(min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        self.num_sigma = num_sigma

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"num_sigma": self.num_sigma}

    def _radius_from_sigma(self, sigma: np.ndarray) -> np.ndarray:
        return np.asarray(sigma, dtype=float)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "num_sigma": self.num_sigma}


class Canny(Operator):
    """Canny edge detection returning a boolean edge map.

    Wraps :func:`skimage.feature.canny` over a single-band ``(H, W)`` or
    ``(1, H, W)`` image. Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns a boolean edge map in the same carrier
    kind.

    Args:
        sigma: Width of the Gaussian smoothing kernel, in pixels.
        low_threshold: Lower hysteresis threshold; ``None`` uses the
            skimage default.
        high_threshold: Upper hysteresis threshold; ``None`` uses the
            skimage default.
    """

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        edges = canny(
            single_band(np.asarray(gt, dtype=float), name="Canny"),
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
        return wrap_like(gt, edges)

    def get_config(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
        }


class CornerHarris(Operator):
    """Harris corner response plus peak selection as point features.

    Runs :func:`skimage.feature.corner_harris` then
    :func:`skimage.feature.corner_peaks` over a single-band ``(H, W)``
    or ``(1, H, W)`` image and returns a ``GeoDataFrame`` with
    ``row``/``col`` pixel indices, the Harris ``response``, and a
    ``Point`` geometry in world coordinates. Geo-dependent: requires a
    georeferenced ``GeoTensor`` input (plain arrays raise ``TypeError``).

    Args:
        min_distance: Minimum pixel distance between reported corners.
        threshold_rel: Minimum response relative to the strongest corner.

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

    def __init__(self, *, min_distance: int = 1, threshold_rel: float = 0.1) -> None:
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        _require_geotensor(gt, "CornerHarris")
        response = corner_harris(
            single_band(np.asarray(gt, dtype=float), name="CornerHarris")
        )
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
    """Histogram of Oriented Gradients descriptor.

    Wraps :func:`skimage.feature.hog` over a single-band ``(H, W)`` or
    ``(1, H, W)`` image. The output is a flat 1-D feature vector (a
    plain ``np.ndarray``), so both ``GeoTensor`` and plain-array
    carriers are accepted.

    Args:
        orientations: Number of orientation histogram bins.
        pixels_per_cell: Cell size in pixels, ``(rows, cols)``.
        cells_per_block: Block size in cells, ``(rows, cols)``.
    """

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> np.ndarray:
        return hog(
            single_band(np.asarray(gt, dtype=float), name="HOG"),
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
    """Local structure-tensor eigenvalue stack.

    Runs :func:`skimage.feature.structure_tensor` then
    :func:`skimage.feature.structure_tensor_eigenvalues` over a
    single-band ``(H, W)`` or ``(1, H, W)`` image. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns the ``(2, H, W)``
    eigenvalue stack in the same carrier kind.

    Args:
        sigma: Width of the Gaussian window used to average the
            gradient products, in pixels.
    """

    def __init__(self, *, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        tensor = structure_tensor(
            single_band(np.asarray(gt, dtype=float), name="StructureTensor"),
            sigma=self.sigma,
        )
        eigvals = np.asarray(structure_tensor_eigenvalues(tensor))
        return wrap_like(gt, eigvals)

    def get_config(self) -> dict[str, Any]:
        return {"sigma": self.sigma}


class MultiscaleBasicFeatures(Operator):
    """General-purpose multiscale feature stack.

    Wraps :func:`skimage.feature.multiscale_basic_features` over a
    single-band ``(H, W)`` or ``(1, H, W)`` image. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns the ``(F, H, W)``
    channel-first feature stack in the same carrier kind.

    Args:
        intensity: Include Gaussian-smoothed intensity features.
        edges: Include gradient-magnitude (edge) features.
        texture: Include Hessian-eigenvalue (texture) features.
        sigma_min: Smallest smoothing scale, in pixels.
        sigma_max: Largest smoothing scale, in pixels.
    """

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        features = multiscale_basic_features(
            single_band(np.asarray(gt, dtype=float), name="MultiscaleBasicFeatures"),
            intensity=self.intensity,
            edges=self.edges,
            texture=self.texture,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            channel_axis=None,
        )
        return wrap_like(gt, np.moveaxis(features, -1, 0))

    def get_config(self) -> dict[str, Any]:
        return {
            "intensity": self.intensity,
            "edges": self.edges,
            "texture": self.texture,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
        }


class HoughLines(Operator):
    """Detect prominent straight lines with the Hough transform.

    Runs :func:`skimage.transform.hough_line` /
    :func:`~skimage.transform.hough_line_peaks` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image. The output is a plain
    ``pd.DataFrame`` of pixel-space line parameters (``accumulator``,
    ``angle``, ``distance``), so both ``GeoTensor`` and plain-array
    carriers are accepted.

    Args:
        num_peaks: Maximum number of line peaks to return.
    """

    def __init__(self, *, num_peaks: int = 10) -> None:
        self.num_peaks = num_peaks

    def _apply(self, gt: GeoTensor | np.ndarray) -> pd.DataFrame:
        hspace, angles, distances = hough_line(
            single_band(np.asarray(gt), name="HoughLines")
        )
        accum, angle_peaks, dist_peaks = hough_line_peaks(
            hspace, angles, distances, num_peaks=self.num_peaks
        )
        return pd.DataFrame(
            {"accumulator": accum, "angle": angle_peaks, "distance": dist_peaks}
        )

    def get_config(self) -> dict[str, Any]:
        return {"num_peaks": self.num_peaks}


class HoughCircles(Operator):
    """Detect circles with the circular Hough transform.

    Runs :func:`skimage.transform.hough_circle` /
    :func:`~skimage.transform.hough_circle_peaks` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image and returns a ``GeoDataFrame``
    with ``row``/``col`` centre pixels, ``radius`` and ``accumulator``
    columns, and a ``Point`` geometry in world coordinates.
    Geo-dependent: requires a georeferenced ``GeoTensor`` input (plain
    arrays raise ``TypeError``).

    Args:
        radii: Candidate circle radii, in pixels.
        total_num_peaks: Maximum number of circles to return.

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

    def __init__(self, *, radii: list[int], total_num_peaks: int = 10) -> None:
        self.radii = radii
        self.total_num_peaks = total_num_peaks

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        _require_geotensor(gt, "HoughCircles")
        hspaces = hough_circle(
            single_band(np.asarray(gt), name="HoughCircles"), self.radii
        )
        accum, cx, cy, radii = hough_circle_peaks(
            hspaces,
            self.radii,
            total_num_peaks=self.total_num_peaks,
        )
        return _points(gt, cy, cx, {"radius": radii, "accumulator": accum})

    def get_config(self) -> dict[str, Any]:
        return {"radii": self.radii, "total_num_peaks": self.total_num_peaks}

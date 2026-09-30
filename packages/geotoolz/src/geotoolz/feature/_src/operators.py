"""Carrier-aware wrappers around :mod:`skimage.feature` primitives.

Raster outputs mark nodata input pixels (non-finite or equal to the
input's ``fill_value_default``; see :mod:`geotoolz._src.valid`) with a
fill that matches the output's dtype: ``False`` for the boolean
:class:`Canny` edge map, ``NaN`` for the float feature stacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import einx
import geopandas as gpd
import numpy as np
import pandas as pd
from pipekit import Operator
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

from geotoolz._src.config import as_tuple
from geotoolz._src.geo import pixel_xy, require_geotensor
from geotoolz._src.shape import single_band
from geotoolz._src.valid import wrap_filled


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _points(
    gt: GeoTensor,
    rows: np.ndarray,
    cols: np.ndarray,
    data: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    frame = pd.DataFrame({"row": rows, "col": cols, **(data or {})})
    xs, ys = pixel_xy(gt.transform, rows, cols)
    geometry = gpd.points_from_xy(xs, ys)
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

    _terminal: ClassVar[bool] = True

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
        require_geotensor(gt, "PeakLocalMax")
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

    _terminal: ClassVar[bool] = True

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
        require_geotensor(gt, name)
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
    kind; nodata input pixels are ``False`` and a GeoTensor output
    declares ``fill_value_default=False``.

    The input is handed to skimage in its own dtype, so the result is
    identical to ``skimage.feature.canny`` on the same array. skimage
    rescales the image with ``img_as_float`` internally, which makes the
    threshold units dtype-dependent:

    * integer input (e.g. ``uint16`` DN): explicit thresholds are in
      input units (DN), and the ``None`` defaults are 10 % / 20 % of the
      dtype maximum (``6553.5`` / ``13107`` for ``uint16``);
    * float input: thresholds are absolute gradient magnitudes and the
      ``None`` defaults are ``0.1`` / ``0.2`` — sensible for ``[0, 1]``
      reflectance, too low for unscaled float DN.

    ``int64`` / ``uint64`` input, which skimage rejects, is cast to
    ``float64`` first and so follows the float convention.

    Args:
        sigma: Width of the Gaussian smoothing kernel, in pixels.
        low_threshold: Lower hysteresis threshold, in the units above;
            ``None`` uses the skimage default.
        high_threshold: Upper hysteresis threshold, in the units above;
            ``None`` uses the skimage default.
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
        arr = np.asarray(gt)
        # skimage.feature.canny rejects 64-bit integers; everything else
        # goes through untouched so its dtype-relative defaults apply.
        if arr.dtype in (np.int64, np.uint64):
            arr = arr.astype(np.float64)
        edges = canny(
            single_band(arr, name="Canny"),
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
        return wrap_filled(gt, edges, fill_value_default=False)


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

    _terminal: ClassVar[bool] = True

    def __init__(self, *, min_distance: int = 1, threshold_rel: float = 0.1) -> None:
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        require_geotensor(gt, "CornerHarris")
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

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (3, 3),
    ) -> None:
        self.orientations = orientations
        self.pixels_per_cell = as_tuple(pixels_per_cell)
        self.cells_per_block = as_tuple(cells_per_block)

    def _apply(self, gt: GeoTensor | np.ndarray) -> np.ndarray:
        return hog(
            single_band(np.asarray(gt, dtype=float), name="HOG"),
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True,
        )


class StructureTensor(Operator):
    """Local structure-tensor eigenvalue stack.

    Runs :func:`skimage.feature.structure_tensor` then
    :func:`skimage.feature.structure_tensor_eigenvalues` over a
    single-band ``(H, W)`` or ``(1, H, W)`` image. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns the ``(2, H, W)``
    eigenvalue stack in the same carrier kind; nodata input pixels are
    ``NaN`` (``fill_value_default=NaN``).

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
        return wrap_filled(gt, eigvals, fill_value_default=np.nan)


class MultiscaleBasicFeatures(Operator):
    """General-purpose multiscale feature stack.

    Wraps :func:`skimage.feature.multiscale_basic_features` over a
    single-band ``(H, W)`` or ``(1, H, W)`` image. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns the ``(F, H, W)``
    channel-first feature stack in the same carrier kind; nodata input
    pixels are ``NaN`` (``fill_value_default=NaN``).

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
        stack = einx.id("h w f -> f h w", features)
        return wrap_filled(gt, stack, fill_value_default=np.nan)


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

    _terminal: ClassVar[bool] = True

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

    _terminal: ClassVar[bool] = True

    def __init__(self, *, radii: list[int], total_num_peaks: int = 10) -> None:
        self.radii = radii
        self.total_num_peaks = total_num_peaks

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        require_geotensor(gt, "HoughCircles")
        hspaces = hough_circle(
            single_band(np.asarray(gt), name="HoughCircles"), self.radii
        )
        accum, cx, cy, radii = hough_circle_peaks(
            hspaces,
            self.radii,
            total_num_peaks=self.total_num_peaks,
        )
        return _points(gt, cy, cx, {"radius": radii, "accumulator": accum})

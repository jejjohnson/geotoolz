"""Carrier-aware wrappers around :mod:`skimage.measure` primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import geopandas as gpd
import numpy as np
from pipekit import Operator
from shapely.geometry import LineString
from skimage.measure import find_contours, profile_line, ransac, shannon_entropy

from geotoolz._src.config import as_tuple
from geotoolz._src.geo import pixel_xy, require_geotensor
from geotoolz._src.shape import single_band
from geotoolz._src.valid import valid_pixels
from geotoolz._src.wrap import wrap_like
from geotoolz.measure._src.array import (
    DEFAULT_REGIONPROPS,
    Connectivity,
    label_components,
    regionprops_frame,
    regionprops_to_crs_units,
    skeleton_length,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class LabelConnectedComponents(Operator):
    """Convert a binary mask into an int32 connected-component label map.

    Delegates to :func:`geotoolz.measure.label_components`. Expects a
    single-band ``(H, W)`` or ``(1, H, W)`` mask; pixels different from
    ``background`` are foreground. Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns an ``int32`` label map (contiguous labels
    ``1..K`` in raster order) in the same carrier kind, with
    ``fill_value_default=0``. Nodata input pixels (non-finite or equal to
    the input's fill) are background ``0`` and never join components.

    Args:
        connectivity: ``4`` (edge neighbours, default) or ``8`` (edge +
            diagonal neighbours).
        background: Pixel value treated as background and labelled ``0``.
        min_area_px: Minimum component size in pixels; smaller components
            become background. ``0`` keeps every component.
    """

    def __init__(
        self,
        *,
        connectivity: Connectivity = 4,
        background: int = 0,
        min_area_px: int = 0,
    ) -> None:
        self.connectivity = connectivity
        self.background = background
        self.min_area_px = min_area_px

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        values = single_band(np.asarray(gt), name="LabelConnectedComponents")
        labels = label_components(
            (values != self.background) & single_band(valid_pixels(gt)),
            connectivity=self.connectivity,
            min_area_px=self.min_area_px,
        )
        return wrap_like(gt, labels, fill_value_default=0)


class RegionProps(Operator):
    """Extract a GeoDataFrame of per-region statistics from a label map.

    Wraps :func:`skimage.measure.regionprops_table` over a single-band
    ``(H, W)`` or ``(1, H, W)`` integer label map. When the ``centroid``
    property is requested (the default), each row carries a ``Point``
    geometry at the region centroid in world coordinates. Geo-dependent:
    the output geometry needs the carrier's transform/CRS, so the input
    must be a georeferenced ``GeoTensor`` (plain arrays raise
    ``TypeError``).

    Units: the ``geometry`` column is in CRS units. The property columns
    are in **pixel units** by default -- areas in pixels, lengths
    (``perimeter``, ``major_axis_length``, ...) in pixel widths, second
    moments (``inertia_tensor*``) in squared pixel widths, positions
    (``centroid-*``, ``bbox-*``) as ``(row, col)`` indices. With
    ``scale_to_crs=True`` areas, lengths and second moments are converted
    to CRS units (m^2 / m / m^2 for a projected metric CRS); positions,
    angles and dimensionless ratios are unchanged.

    Args:
        intensity_image: Optional single-band intensity image aligned
            with the label map, enabling intensity-based properties.
        properties: Property names passed to ``regionprops_table``;
            defaults to :data:`DEFAULT_REGIONPROPS`.
        extra_properties: Optional callables computing custom per-region
            properties (see skimage docs). Not YAML-serialisable, so
            instances are forbidden in YAML. Their outputs are never
            rescaled.
        scale_to_crs: Convert area / length / second-moment columns from
            pixel to CRS units using the carrier's transform.

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
        ValueError: If ``scale_to_crs=True`` and a length or moment
            column is requested on non-square (anisotropic or sheared)
            pixels, where no single pixel length exists.
    """

    _terminal: ClassVar[bool] = True

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        intensity_image: GeoTensor | None = None,
        properties: Sequence[str] | None = None,
        extra_properties: Sequence[Callable[..., Any]] | None = None,
        scale_to_crs: bool = False,
    ) -> None:
        self.intensity_image = intensity_image
        self.properties = tuple(
            DEFAULT_REGIONPROPS if properties is None else properties
        )
        self.extra_properties = (
            None if extra_properties is None else tuple(extra_properties)
        )
        self.scale_to_crs = scale_to_crs

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        require_geotensor(gt, "RegionProps")
        labels = single_band(np.asarray(gt), name="RegionProps").astype(
            np.int32, copy=False
        )
        intensity = (
            None
            if self.intensity_image is None
            else single_band(self.intensity_image, name="RegionProps intensity_image")
        )
        frame = regionprops_frame(
            labels,
            intensity_image=intensity,
            properties=self.properties,
            extra_properties=self.extra_properties,
        )
        if self.scale_to_crs:
            frame = regionprops_to_crs_units(frame, gt.transform)
        if frame.empty:
            return gpd.GeoDataFrame(frame, geometry=[], crs=gt.crs)
        if {"centroid-0", "centroid-1"}.issubset(frame.columns):
            xs, ys = pixel_xy(gt.transform, frame["centroid-0"], frame["centroid-1"])
            geometry = gpd.points_from_xy(xs, ys)
            return gpd.GeoDataFrame(frame, geometry=geometry, crs=gt.crs)
        # Caller omitted centroid: still return a valid GeoDataFrame with an
        # empty (None-valued) geometry column rather than raising in the
        # ``GeoDataFrame`` constructor for missing geometry.
        return gpd.GeoDataFrame(frame, geometry=[None] * len(frame), crs=gt.crs)

    def get_config(self) -> dict[str, Any]:
        return {
            "intensity_image": None
            if self.intensity_image is None
            else {
                "shape": list(np.asarray(self.intensity_image).shape),
                "dtype": str(np.asarray(self.intensity_image).dtype),
            },
            "properties": list(self.properties),
            "extra_properties": None
            if self.extra_properties is None
            else [
                getattr(func, "__name__", repr(func)) for func in self.extra_properties
            ],
            "scale_to_crs": self.scale_to_crs,
        }


class FindContours(Operator):
    """Extract iso-value contours as LineString geometries.

    Wraps :func:`skimage.measure.find_contours` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image and maps each contour's pixel
    coordinates into world coordinates. Geo-dependent: the output
    geometries need the carrier's transform/CRS, so the input must be a
    georeferenced ``GeoTensor`` (plain arrays raise ``TypeError``).

    Units: the ``geometry`` column is in CRS units (contour vertices are
    sub-pixel positions mapped through the transform, pixel centres at
    integer ``(row, col)``); ``contour_id`` is a 1-based index. There are
    no pixel-unit columns, so there is no ``scale_to_crs`` switch.

    Args:
        level: Iso-value along which to find contours; ``None`` uses the
            midpoint of the image's value range.
        fully_connected: Whether high- or low-valued pixels are
            considered fully connected (``"low"`` or ``"high"``).

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        level: float | None = None,
        fully_connected: str = "low",
    ) -> None:
        self.level = level
        self.fully_connected = fully_connected

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        require_geotensor(gt, "FindContours")
        contours = find_contours(
            single_band(np.asarray(gt, dtype=float), name="FindContours"),
            level=self.level,
            fully_connected=self.fully_connected,
        )
        rows = []
        for contour_id, coords in enumerate(contours, start=1):
            if len(coords) >= 2:
                xs, ys = pixel_xy(gt.transform, coords[:, 0], coords[:, 1])
                geometry = LineString(np.column_stack([xs, ys]))
                rows.append({"contour_id": contour_id, "geometry": geometry})
        if not rows:
            # No contours (constant raster, out-of-range level, or all
            # segments degenerate to <2 points). Return an empty
            # GeoDataFrame with the expected schema instead of letting the
            # constructor raise on a missing ``geometry`` column.
            return gpd.GeoDataFrame(
                {"contour_id": [], "geometry": []},
                geometry="geometry",
                crs=gt.crs,
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gt.crs)


class ProfileLine(Operator):
    """Sample values along a line between two pixel coordinates.

    Wraps :func:`skimage.measure.profile_line` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image. Endpoints are pixel ``(row, col)``
    coordinates and the returned profile is a plain 1-D ``np.ndarray``,
    so both ``GeoTensor`` and plain-array carriers are accepted.

    Args:
        src: ``(row, col)`` start pixel of the profile.
        dst: ``(row, col)`` end pixel of the profile.
        linewidth: Width of the sampling band, in pixels.
        order: Spline interpolation order (0 = nearest neighbour).
        mode: Boundary handling mode for samples outside the image.
    """

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        src: tuple[int, int],
        dst: tuple[int, int],
        linewidth: int = 1,
        order: int = 1,
        mode: str = "reflect",
    ) -> None:
        self.src = as_tuple(src)
        self.dst = as_tuple(dst)
        self.linewidth = linewidth
        self.order = order
        self.mode = mode

    def _apply(self, gt: GeoTensor | np.ndarray) -> np.ndarray:
        return profile_line(
            single_band(np.asarray(gt), name="ProfileLine"),
            self.src,
            self.dst,
            linewidth=self.linewidth,
            order=self.order,
            mode=self.mode,
        )


class RANSAC(Operator):
    """Robust model fitting via :func:`skimage.measure.ransac`.

    Carrier-agnostic: the input is whatever data layout the chosen model
    class consumes (e.g. an ``(N, D)`` point array), and the output is
    the ``(model, inlier_mask)`` pair returned by ``ransac``.

    Args:
        model_class: skimage model class to fit (e.g.
            ``skimage.measure.LineModelND``). Not YAML-serialisable, so
            instances are forbidden in YAML.
        min_samples: Minimum number of samples per model estimate.
        residual_threshold: Maximum residual for a sample to count as an
            inlier.
        **kwargs: Extra keyword arguments forwarded to
            :func:`skimage.measure.ransac` (e.g. ``max_trials``, ``rng``).
    """

    _terminal: ClassVar[bool] = True

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model_class: type[Any],
        min_samples: int,
        residual_threshold: float,
        **kwargs: Any,
    ) -> None:
        self.model_class = model_class
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.kwargs = kwargs

    def _apply(self, data: Any) -> Any:
        return ransac(
            data,
            self.model_class,
            min_samples=self.min_samples,
            residual_threshold=self.residual_threshold,
            **self.kwargs,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "model_class": getattr(
                self.model_class,
                "__name__",
                repr(self.model_class),
            ),
            "min_samples": self.min_samples,
            "residual_threshold": self.residual_threshold,
            **self.kwargs,
        }


class SkeletonLength(Operator):
    """Longest path through the skeleton of a binary mask.

    Delegates to :func:`geotoolz.measure.skeleton_length`: the mask is
    skeletonised with :func:`skimage.morphology.skeletonize` and the
    longest geodesic path through the 8-connected skeleton is measured
    between pixel centres, each step weighted by its Euclidean length
    (a diagonal step counts ``sqrt(2)``). This is the geodesic "fiber
    length" of Pérez Carrasco et al. (2026) with Euclidean rather than
    unit step weights.

    Units: pixel widths by default, so both ``GeoTensor`` and plain-array
    carriers are accepted. With ``scale_to_crs=True`` steps are measured
    through the carrier's transform (anisotropic and rotated grids
    included) and the result is in CRS units (m for a projected metric
    CRS); the input must then be a georeferenced ``GeoTensor``.

    Returns ``0.0`` for empty masks or skeletons that collapse to a single
    pixel.

    Args:
        scale_to_crs: Measure in CRS units via the carrier's transform
            instead of pixel widths.

    Raises:
        TypeError: If ``scale_to_crs=True`` and the input is not a
            georeferenced GeoTensor.
    """

    _terminal: ClassVar[bool] = True

    def __init__(self, *, scale_to_crs: bool = False) -> None:
        self.scale_to_crs = scale_to_crs

    def _apply(self, gt: GeoTensor | np.ndarray) -> float:
        step: Any = (1.0, 1.0)
        if self.scale_to_crs:
            step = require_geotensor(gt, "SkeletonLength(scale_to_crs=True)").transform
        return skeleton_length(
            single_band(np.asarray(gt), name="SkeletonLength"), step=step
        )


class ShannonEntropy(Operator):
    """Compute Shannon entropy of the input image.

    Wraps :func:`skimage.measure.shannon_entropy` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image. The result is a plain float, so
    both ``GeoTensor`` and plain-array carriers are accepted.

    Args:
        base: Logarithm base for the entropy (``2.0`` gives bits).
    """

    _terminal: ClassVar[bool] = True

    def __init__(self, *, base: float = 2.0) -> None:
        self.base = base

    def _apply(self, gt: GeoTensor | np.ndarray) -> float:
        return float(
            shannon_entropy(
                single_band(np.asarray(gt), name="ShannonEntropy"), base=self.base
            )
        )

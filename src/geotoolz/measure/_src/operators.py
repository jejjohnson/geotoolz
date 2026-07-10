"""Carrier-aware wrappers around :mod:`skimage.measure` primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
from jaxtyping import Shaped
from pipekit import Operator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import LineString, Point
from skimage.measure import (
    find_contours,
    label,
    profile_line,
    ransac,
    regionprops_table,
    shannon_entropy,
)
from skimage.morphology import skeletonize

from geotoolz._src.shape import single_band
from geotoolz._src.wrap import wrap_like


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


DEFAULT_REGIONPROPS: tuple[str, ...] = (
    "label",
    "area",
    "area_convex",
    "area_filled",
    "centroid",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "eccentricity",
    "solidity",
    "perimeter",
    "bbox",
    "inertia_tensor_eigvals",
)


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


class LabelConnectedComponents(Operator):
    """Convert a binary mask into an int32 connected-component label map.

    Wraps :func:`skimage.measure.label`. Expects a single-band ``(H, W)``
    or ``(1, H, W)`` mask (values are cast to bool). Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns an ``int32``
    label map in the same carrier kind.

    Args:
        connectivity: Maximum orthogonal hops for two pixels to count as
            neighbours (``1`` = 4-connectivity, ``2`` = 8-connectivity;
            ``None`` = full connectivity).
        background: Pixel value treated as background and labelled ``0``.
    """

    def __init__(self, *, connectivity: int | None = 1, background: int = 0) -> None:
        self.connectivity = connectivity
        self.background = background

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        labels = label(
            single_band(np.asarray(gt), name="LabelConnectedComponents").astype(bool),
            connectivity=self.connectivity,
            background=self.background,
        )
        return wrap_like(gt, labels.astype(np.int32, copy=False))

    def get_config(self) -> dict[str, Any]:
        return {"connectivity": self.connectivity, "background": self.background}


class RegionProps(Operator):
    """Extract a GeoDataFrame of per-region statistics from a label map.

    Wraps :func:`skimage.measure.regionprops_table` over a single-band
    ``(H, W)`` or ``(1, H, W)`` integer label map. When the ``centroid``
    property is requested (the default), each row carries a ``Point``
    geometry at the region centroid in world coordinates. Geo-dependent:
    the output geometry needs the carrier's transform/CRS, so the input
    must be a georeferenced ``GeoTensor`` (plain arrays raise
    ``TypeError``).

    Args:
        intensity_image: Optional single-band intensity image aligned
            with the label map, enabling intensity-based properties.
        properties: Property names passed to ``regionprops_table``;
            defaults to :data:`DEFAULT_REGIONPROPS`.
        extra_properties: Optional callables computing custom per-region
            properties (see skimage docs). Not YAML-serialisable, so
            instances are forbidden in YAML.

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        intensity_image: GeoTensor | None = None,
        properties: Sequence[str] | None = None,
        extra_properties: Sequence[Callable[..., Any]] | None = None,
    ) -> None:
        self.intensity_image = intensity_image
        self.properties = tuple(
            DEFAULT_REGIONPROPS if properties is None else properties
        )
        self.extra_properties = (
            None if extra_properties is None else tuple(extra_properties)
        )

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        _require_geotensor(gt, "RegionProps")
        labels = single_band(np.asarray(gt), name="RegionProps").astype(
            np.int32, copy=False
        )
        intensity = (
            None
            if self.intensity_image is None
            else single_band(self.intensity_image, name="RegionProps intensity_image")
        )
        props = regionprops_table(
            labels,
            intensity_image=intensity,
            properties=self.properties,
            extra_properties=self.extra_properties,
        )
        frame = pd.DataFrame(props)
        if frame.empty:
            return gpd.GeoDataFrame(frame, geometry=[], crs=gt.crs)
        if {"centroid-0", "centroid-1"}.issubset(frame.columns):
            geometry = [
                Point(_xy(gt.transform, row, col))
                for row, col in zip(
                    frame["centroid-0"],
                    frame["centroid-1"],
                    strict=True,
                )
            ]
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
        }


class FindContours(Operator):
    """Extract iso-value contours as LineString geometries.

    Wraps :func:`skimage.measure.find_contours` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image and maps each contour's pixel
    coordinates into world coordinates. Geo-dependent: the output
    geometries need the carrier's transform/CRS, so the input must be a
    georeferenced ``GeoTensor`` (plain arrays raise ``TypeError``).

    Args:
        level: Iso-value along which to find contours; ``None`` uses the
            midpoint of the image's value range.
        fully_connected: Whether high- or low-valued pixels are
            considered fully connected (``"low"`` or ``"high"``).

    Raises:
        TypeError: If the input is not a georeferenced GeoTensor.
    """

    def __init__(
        self,
        *,
        level: float | None = None,
        fully_connected: str = "low",
    ) -> None:
        self.level = level
        self.fully_connected = fully_connected

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        _require_geotensor(gt, "FindContours")
        contours = find_contours(
            single_band(np.asarray(gt, dtype=float), name="FindContours"),
            level=self.level,
            fully_connected=self.fully_connected,
        )
        rows = []
        for contour_id, coords in enumerate(contours, start=1):
            xy = [_xy(gt.transform, float(row), float(col)) for row, col in coords]
            if len(xy) >= 2:
                rows.append({"contour_id": contour_id, "geometry": LineString(xy)})
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

    def get_config(self) -> dict[str, Any]:
        return {"level": self.level, "fully_connected": self.fully_connected}


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

    def __init__(
        self,
        *,
        src: tuple[int, int],
        dst: tuple[int, int],
        linewidth: int = 1,
        order: int = 1,
        mode: str = "reflect",
    ) -> None:
        self.src = src
        self.dst = dst
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

    def get_config(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "linewidth": self.linewidth,
            "order": self.order,
            "mode": self.mode,
        }


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


def _skeleton_diameter_pixels(mask: Shaped[np.ndarray, "h w"]) -> float:
    """Longest shortest-path (graph diameter) of the skeleton of a binary
    mask, with 8-connected adjacency and unit edge weight.

    Returns ``0.0`` when the mask is empty or the skeleton collapses to a
    single pixel. NaNs in the input are treated as background.
    """
    binary = np.asarray(mask)
    bin_clean = binary if binary.dtype == bool else np.nan_to_num(binary, nan=0.0) > 0
    if not bin_clean.any():
        return 0.0
    skel = skeletonize(bin_clean)
    ys, xs = np.nonzero(skel)
    n = ys.size
    if n < 2:
        return 0.0

    coord_to_idx: dict[tuple[int, int], int] = {
        (int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs, strict=True))
    }
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        y, x = int(ys[i]), int(xs[i])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = coord_to_idx.get((y + dy, x + dx))
                # Add each edge once (j > i) — the graph is built as
                # symmetric below by setting ``directed=False``.
                if j is not None and j > i:
                    rows.append(i)
                    cols.append(j)
    if not rows:
        return 0.0
    data = np.ones(len(rows), dtype=np.int64)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    distances = shortest_path(graph, directed=False, unweighted=True)
    finite = distances[np.isfinite(distances)]
    return float(finite.max()) if finite.size else 0.0


class SkeletonLength(Operator):
    """Longest path through the skeleton of a binary mask, in pixels.

    The mask is skeletonized via :func:`skimage.morphology.skeletonize` and
    the result is treated as an 8-connected unit-weight graph; the operator
    returns the graph diameter (longest shortest-path between any pair of
    skeleton pixels). For tree-like skeletons this equals the geodesic
    "fiber length" — closely matching ``calculate_fiber_length_from_mask``
    from Pérez Carrasco et al. (2026), which uses
    :func:`networkx.all_pairs_shortest_path_length` for the same purpose.

    Returns ``0.0`` for empty masks or skeletons that collapse to a single
    pixel. The result is a plain float in pixel units, so both
    ``GeoTensor`` and plain-array carriers are accepted.
    """

    def _apply(self, gt: GeoTensor | np.ndarray) -> float:
        return _skeleton_diameter_pixels(
            single_band(np.asarray(gt), name="SkeletonLength")
        )

    def get_config(self) -> dict[str, Any]:
        return {}


class ShannonEntropy(Operator):
    """Compute Shannon entropy of the input image.

    Wraps :func:`skimage.measure.shannon_entropy` over a single-band
    ``(H, W)`` or ``(1, H, W)`` image. The result is a plain float, so
    both ``GeoTensor`` and plain-array carriers are accepted.

    Args:
        base: Logarithm base for the entropy (``2.0`` gives bits).
    """

    def __init__(self, *, base: float = 2.0) -> None:
        self.base = base

    def _apply(self, gt: GeoTensor | np.ndarray) -> float:
        return float(
            shannon_entropy(
                single_band(np.asarray(gt), name="ShannonEntropy"), base=self.base
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {"base": self.base}

"""Carrier-aware wrappers around :mod:`skimage.measure` primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
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


def _single_band(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError("measure operators expect shape (H, W) or (1, H, W)")


def _xy(transform: Any, row: float, col: float) -> tuple[float, float]:
    x = transform.c + transform.a * (col + 0.5) + transform.b * (row + 0.5)
    y = transform.f + transform.d * (col + 0.5) + transform.e * (row + 0.5)
    return float(x), float(y)


class LabelConnectedComponents(Operator):
    """Convert a binary mask into an int32 connected-component label map."""

    def __init__(self, *, connectivity: int | None = 1, background: int = 0) -> None:
        self.connectivity = connectivity
        self.background = background

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        labels = label(
            _single_band(np.asarray(gt)).astype(bool),
            connectivity=self.connectivity,
            background=self.background,
        )
        return gt.array_as_geotensor(labels.astype(np.int32, copy=False))

    def get_config(self) -> dict[str, Any]:
        return {"connectivity": self.connectivity, "background": self.background}


class RegionProps(Operator):
    """Extract a GeoDataFrame of per-region statistics from a label map."""

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
        labels = _single_band(np.asarray(gt)).astype(np.int32, copy=False)
        intensity = (
            None
            if self.intensity_image is None
            else _single_band(np.asarray(self.intensity_image))
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
    """Extract iso-value contours as LineString geometries."""

    def __init__(
        self,
        *,
        level: float | None = None,
        fully_connected: str = "low",
    ) -> None:
        self.level = level
        self.fully_connected = fully_connected

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        contours = find_contours(
            _single_band(np.asarray(gt, dtype=float)),
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
    """Sample values along a line between two pixel coordinates."""

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

    def _apply(self, gt: GeoTensor) -> np.ndarray:
        return profile_line(
            _single_band(np.asarray(gt)),
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
    """Robust model fitting via :func:`skimage.measure.ransac`."""

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


def _skeleton_diameter_pixels(mask: np.ndarray) -> float:
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
    pixel.
    """

    def _apply(self, gt: GeoTensor) -> float:
        return _skeleton_diameter_pixels(_single_band(np.asarray(gt)))

    def get_config(self) -> dict[str, Any]:
        return {}


class ShannonEntropy(Operator):
    """Compute Shannon entropy of the input image."""

    def __init__(self, *, base: float = 2.0) -> None:
        self.base = base

    def _apply(self, gt: GeoTensor) -> float:
        return float(shannon_entropy(_single_band(np.asarray(gt)), base=self.base))

    def get_config(self) -> dict[str, Any]:
        return {"base": self.base}

"""Carrier-aware wrappers around :mod:`skimage.measure` primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from skimage.measure import (
    find_contours,
    label,
    profile_line,
    ransac,
    regionprops_table,
    shannon_entropy,
)

from pipekit import Operator


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


class ShannonEntropy(Operator):
    """Compute Shannon entropy of the input image."""

    def __init__(self, *, base: float = 2.0) -> None:
        self.base = base

    def _apply(self, gt: GeoTensor) -> float:
        return float(shannon_entropy(_single_band(np.asarray(gt)), base=self.base))

    def get_config(self) -> dict[str, Any]:
        return {"base": self.base}

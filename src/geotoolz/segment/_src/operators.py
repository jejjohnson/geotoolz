"""Carrier-aware wrappers around :mod:`skimage.segmentation` primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from georeader.geotensor import GeoTensor
from skimage.segmentation import (
    chan_vese,
    expand_labels,
    felzenszwalb,
    mark_boundaries,
    quickshift,
    random_walker,
    slic,
    watershed,
)

from pipekit import Operator


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor as GeoTensorType


def _as_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != shape:
        raise ValueError(f"mask shape {arr.shape} does not match image shape {shape}")
    return arr.astype(bool)


def _finite_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.isfinite(image)
    return np.all(np.isfinite(image), axis=0)


def _fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    # Compute fill statistic from finite values only so +/-inf cannot leak
    # into the result via ``np.nanmedian`` (which ignores NaN but not inf).
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def _single_band(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError("operator expects shape (H, W) or (1, H, W)")


def _labels(
    gt: GeoTensorType,
    labels: np.ndarray,
    mask: np.ndarray | None = None,
) -> GeoTensorType:
    out = np.asarray(labels, dtype=np.int32)
    if mask is not None:
        out = np.where(mask, out, 0).astype(np.int32, copy=False)
    return gt.array_as_geotensor(out, fill_value_default=0)


class SLIC(Operator):
    """SLIC superpixels via :func:`skimage.segmentation.slic`."""

    def __init__(
        self,
        *,
        n_segments: int = 100,
        compactness: float = 10.0,
        sigma: float = 0.0,
        channel_axis: int | None = 0,
        start_label: int = 1,
        mask: Any = None,
    ) -> None:
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.channel_axis = channel_axis
        self.start_label = start_label
        self.mask = mask
        # SLIC's `mask` is a non-JSON-safe carrier; forbid yaml round-trip
        # for this instance whenever the user provided one.
        if mask is not None:
            self.forbid_in_yaml = True

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        image = _fill_nan(np.asarray(gt))
        valid = _finite_mask(np.asarray(gt))
        mask = _as_mask(self.mask, gt.shape[-2:])
        if mask is not None:
            valid &= mask
        labels = slic(
            image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            channel_axis=self.channel_axis,
            start_label=self.start_label,
            mask=valid,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "compactness": self.compactness,
            "sigma": self.sigma,
            "channel_axis": self.channel_axis,
            "start_label": self.start_label,
            "mask": None if self.mask is None else "provided",
        }


class Felzenszwalb(Operator):
    """Graph-based segmentation via :func:`skimage.segmentation.felzenszwalb`."""

    def __init__(
        self,
        *,
        scale: float = 1.0,
        sigma: float = 0.8,
        min_size: int = 20,
        channel_axis: int | None = 0,
        mask: Any = None,
    ) -> None:
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.channel_axis = channel_axis
        self.mask = mask
        if mask is not None:
            self.forbid_in_yaml = True

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        valid = _finite_mask(np.asarray(gt))
        mask = _as_mask(self.mask, gt.shape[-2:])
        if mask is not None:
            valid &= mask
        labels = felzenszwalb(
            _fill_nan(np.asarray(gt)),
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_size,
            channel_axis=self.channel_axis,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "sigma": self.sigma,
            "min_size": self.min_size,
            "channel_axis": self.channel_axis,
            "mask": None if self.mask is None else "provided",
        }


class Quickshift(Operator):
    """Mode-seeking superpixels via :func:`skimage.segmentation.quickshift`."""

    def __init__(
        self,
        *,
        kernel_size: float = 5.0,
        max_dist: float = 10.0,
        ratio: float = 1.0,
        sigma: float = 0.0,
        channel_axis: int | None = 0,
        convert2lab: bool = False,
        mask: Any = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        self.sigma = sigma
        self.channel_axis = channel_axis
        # Default to False so non-RGB / multispectral / single-band inputs
        # work out of the box. skimage's quickshift defaults convert2lab=True,
        # which raises on any input that is not exactly 3-channel RGB.
        self.convert2lab = convert2lab
        self.mask = mask
        if mask is not None:
            self.forbid_in_yaml = True

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        valid = _finite_mask(np.asarray(gt))
        mask = _as_mask(self.mask, gt.shape[-2:])
        if mask is not None:
            valid &= mask
        labels = quickshift(
            _fill_nan(np.asarray(gt)),
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            ratio=self.ratio,
            sigma=self.sigma,
            channel_axis=self.channel_axis,
            convert2lab=self.convert2lab,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "kernel_size": self.kernel_size,
            "max_dist": self.max_dist,
            "ratio": self.ratio,
            "sigma": self.sigma,
            "channel_axis": self.channel_axis,
            "convert2lab": self.convert2lab,
            "mask": None if self.mask is None else "provided",
        }


class Watershed(Operator):
    """Watershed segmentation via :func:`skimage.segmentation.watershed`."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        markers: Any = None,
        connectivity: int = 1,
        compactness: float = 0.0,
        watershed_line: bool = False,
        mask: Any = None,
    ) -> None:
        self.markers = markers
        self.connectivity = connectivity
        self.compactness = compactness
        self.watershed_line = watershed_line
        self.mask = mask

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        image = _single_band(_fill_nan(np.asarray(gt)))
        valid = np.isfinite(_single_band(np.asarray(gt)))
        mask = _as_mask(self.mask, gt.shape[-2:])
        if mask is not None:
            valid &= mask
        markers = (
            None if self.markers is None else _single_band(np.asarray(self.markers))
        )
        labels = watershed(
            image,
            markers=markers,
            connectivity=self.connectivity,
            mask=valid,
            compactness=self.compactness,
            watershed_line=self.watershed_line,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "markers": None if self.markers is None else "provided",
            "connectivity": self.connectivity,
            "compactness": self.compactness,
            "watershed_line": self.watershed_line,
            "mask": None if self.mask is None else "provided",
        }


class ChanVese(Operator):
    """Active-contour segmentation via :func:`skimage.segmentation.chan_vese`."""

    def __init__(
        self,
        *,
        mu: float = 0.25,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        tol: float = 1e-3,
        max_num_iter: int = 500,
    ) -> None:
        self.mu = mu
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tol = tol
        self.max_num_iter = max_num_iter

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        band = _single_band(np.asarray(gt))
        valid = np.isfinite(band)
        labels = chan_vese(
            _single_band(_fill_nan(np.asarray(gt))),
            mu=self.mu,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            tol=self.tol,
            max_num_iter=self.max_num_iter,
        )
        return _labels(gt, labels.astype(np.int32), mask=valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "tol": self.tol,
            "max_num_iter": self.max_num_iter,
        }


class RandomWalker(Operator):
    """Seeded segmentation via :func:`skimage.segmentation.random_walker`."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        markers: Any,
        beta: float = 130.0,
        mode: str = "cg_j",
        tol: float = 1e-3,
    ) -> None:
        self.markers = markers
        self.beta = beta
        self.mode = mode
        self.tol = tol

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        labels = random_walker(
            _single_band(_fill_nan(np.asarray(gt))),
            _single_band(np.asarray(self.markers)).astype(np.int32, copy=False),
            beta=self.beta,
            mode=self.mode,
            tol=self.tol,
        )
        return _labels(gt, labels)

    def get_config(self) -> dict[str, Any]:
        return {
            "markers": "provided",
            "beta": self.beta,
            "mode": self.mode,
            "tol": self.tol,
        }


class ExpandLabels(Operator):
    """Grow label regions by a fixed pixel distance without overlap."""

    def __init__(self, *, distance: float = 1.0) -> None:
        self.distance = distance

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        labels = expand_labels(
            _single_band(np.asarray(gt)).astype(np.int32, copy=False),
            distance=self.distance,
        )
        return _labels(gt, labels)

    def get_config(self) -> dict[str, Any]:
        return {"distance": self.distance}


class MarkBoundaries(Operator):
    """Overlay segmentation boundaries on an image for visual inspection."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        label_img: Any,
        color: tuple[float, float, float] = (1.0, 1.0, 0.0),
        mode: str = "thick",
    ) -> None:
        self.label_img = label_img
        self.color = color
        self.mode = mode

    def _apply(self, gt: GeoTensorType) -> GeoTensorType:
        image = np.asarray(gt)
        if image.ndim == 3:
            image = np.moveaxis(image, 0, -1)
        marked = mark_boundaries(
            image,
            _single_band(np.asarray(self.label_img)).astype(np.int32, copy=False),
            color=self.color,
            mode=self.mode,
        )
        if marked.ndim == 3:
            marked = np.moveaxis(marked, -1, 0)
        return gt.array_as_geotensor(marked)

    def get_config(self) -> dict[str, Any]:
        return {
            "label_img": "provided",
            "color": self.color,
            "mode": self.mode,
        }

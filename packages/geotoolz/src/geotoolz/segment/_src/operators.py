"""Carrier-aware wrappers around :mod:`skimage.segmentation` primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import einx
import numpy as np
from jaxtyping import Bool, Float, Int, Num
from pipekit import Operator
from scipy import ndimage
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

from geotoolz._src.shape import single_band
from geotoolz._src.wrap import wrap_like


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor as GeoTensorType


def _as_mask(
    mask: Any,
    shape: tuple[int, int],
    *,
    name: str = "mask",
) -> Bool[np.ndarray, "h w"] | None:
    if mask is None:
        return None
    arr = single_band(mask, name=name)
    if arr.shape != shape:
        raise ValueError(f"mask shape {arr.shape} does not match image shape {shape}")
    return arr.astype(bool)


def _finite_mask(
    image: Num[np.ndarray, "h w"] | Num[np.ndarray, "c h w"],
) -> Bool[np.ndarray, "h w"]:
    if image.ndim == 2:
        return np.isfinite(image)
    return np.all(np.isfinite(image), axis=0)


def _fill_nan(image: Num[np.ndarray, "*dims"]) -> Float[np.ndarray, "*dims"]:
    arr = np.asarray(image, dtype=float)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    # Compute fill statistic from finite values only so +/-inf cannot leak
    # into the result via ``np.nanmedian`` (which ignores NaN but not inf).
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def _labels(
    gt: GeoTensorType | np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray | None = None,
) -> GeoTensorType | np.ndarray:
    out = np.asarray(labels, dtype=np.int32)
    if mask is not None:
        out = np.where(mask, out, 0).astype(np.int32, copy=False)
    return wrap_like(gt, out, fill_value_default=0)


class SLIC(Operator):
    """SLIC superpixels via :func:`skimage.segmentation.slic`.

    Non-finite pixels are median-filled before clustering and forced to
    label ``0`` in the output. Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` (channel-first) and returns an ``int32`` label map in
    the same carrier kind.

    Args:
        n_segments: Approximate number of superpixels to produce.
        compactness: Balance between color and spatial proximity; higher
            values yield more compact (squarer) segments.
        sigma: Width of the Gaussian pre-smoothing kernel, in pixels.
        channel_axis: Axis holding channels (``0`` for channel-first
            cubes, ``None`` for 2-D single-band input).
        start_label: First label assigned to a superpixel.
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``. Providing a mask forbids YAML
            round-trip for the instance.
    """

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
            # The parent's `forbid_in_yaml` is a ClassVar; the per-instance
            # override here is intentional. ty rejects it, hence the ignore.
            self.forbid_in_yaml = True  # ty: ignore[invalid-attribute-access]

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        image = _fill_nan(np.asarray(gt))
        valid = _finite_mask(np.asarray(gt))
        mask = _as_mask(self.mask, gt.shape[-2:], name="SLIC mask")
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
    """Graph-based segmentation via :func:`skimage.segmentation.felzenszwalb`.

    Non-finite pixels are median-filled before segmentation and forced
    to label ``0`` in the output. Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` (channel-first) and returns an ``int32`` label map in
    the same carrier kind.

    Args:
        scale: Free parameter controlling segment size; larger values
            produce larger segments.
        sigma: Width of the Gaussian pre-smoothing kernel, in pixels.
        min_size: Minimum segment size, enforced by postprocessing.
        channel_axis: Axis holding channels (``0`` for channel-first
            cubes, ``None`` for 2-D single-band input).
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``. Providing a mask forbids YAML
            round-trip for the instance.
    """

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
            # The parent's `forbid_in_yaml` is a ClassVar; the per-instance
            # override here is intentional. ty rejects it, hence the ignore.
            self.forbid_in_yaml = True  # ty: ignore[invalid-attribute-access]

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = _finite_mask(np.asarray(gt))
        mask = _as_mask(self.mask, gt.shape[-2:], name="Felzenszwalb mask")
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
    """Mode-seeking superpixels via :func:`skimage.segmentation.quickshift`.

    Non-finite pixels are median-filled before segmentation and forced
    to label ``0`` in the output. Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` (channel-first) and returns an ``int32`` label map in
    the same carrier kind.

    Args:
        kernel_size: Width of the Gaussian kernel used to estimate the
            local density.
        max_dist: Cut-off point for data distances; higher values mean
            fewer clusters.
        ratio: Balance (0-1) between color-space and image-space
            proximity.
        sigma: Width of the Gaussian pre-smoothing kernel, in pixels.
        channel_axis: Axis holding channels (``0`` for channel-first
            cubes, ``None`` for 2-D single-band input).
        convert2lab: Convert the image to LAB space first. Defaults to
            ``False`` (unlike skimage) so non-RGB / multispectral /
            single-band inputs work out of the box.
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``. Providing a mask forbids YAML
            round-trip for the instance.
    """

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
            # The parent's `forbid_in_yaml` is a ClassVar; the per-instance
            # override here is intentional. ty rejects it, hence the ignore.
            self.forbid_in_yaml = True  # ty: ignore[invalid-attribute-access]

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = _finite_mask(np.asarray(gt))
        mask = _as_mask(self.mask, gt.shape[-2:], name="Quickshift mask")
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
    """Watershed segmentation via :func:`skimage.segmentation.watershed`.

    Expects a single-band ``(H, W)`` or ``(1, H, W)`` image. Non-finite
    pixels are median-filled before flooding and forced to label ``0``
    in the output. Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and
    returns an ``int32`` label map in the same carrier kind.

    Args:
        markers: Optional single-band integer marker array seeding the
            basins; when None, local minima of the image are used.
        connectivity: Neighbourhood connectivity used for flooding.
        compactness: Compactness parameter; higher values produce more
            regularly-shaped basins.
        watershed_line: Separate basins with a zero-labelled line.
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``.
    """

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

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        image = single_band(_fill_nan(np.asarray(gt)), name="Watershed")
        valid = np.isfinite(single_band(np.asarray(gt), name="Watershed"))
        mask = _as_mask(self.mask, gt.shape[-2:], name="Watershed mask")
        if mask is not None:
            valid &= mask
        markers = (
            None
            if self.markers is None
            else single_band(self.markers, name="Watershed markers")
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
    """Active-contour segmentation via :func:`skimage.segmentation.chan_vese`.

    Expects a single-band ``(H, W)`` or ``(1, H, W)`` image. Non-finite
    pixels are median-filled before evolution and forced to label ``0``
    in the output. Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and
    returns an ``int32`` label map (0 = outside, 1 = inside) in the same
    carrier kind.

    Args:
        mu: Edge-length penalty weight; higher values give smoother
            boundaries.
        lambda1: Weight of the interior fitting term.
        lambda2: Weight of the exterior fitting term.
        tol: Convergence tolerance on the level set.
        max_num_iter: Maximum number of iterations.
    """

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

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        band = single_band(np.asarray(gt), name="ChanVese")
        valid = np.isfinite(band)
        labels = chan_vese(
            single_band(_fill_nan(np.asarray(gt)), name="ChanVese"),
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
    """Seeded segmentation via :func:`skimage.segmentation.random_walker`.

    Expects a single-band ``(H, W)`` or ``(1, H, W)`` image; non-finite
    pixels are median-filled before diffusion. Accepts a ``GeoTensor``
    or a plain ``np.ndarray`` and returns an ``int32`` label map in the
    same carrier kind.

    Args:
        markers: Single-band integer array of seed labels (0 = unseeded);
            required. Not YAML-serialisable, so instances are forbidden
            in YAML.
        beta: Penalisation coefficient for the random-walk motion;
            higher values make diffusion harder across intensity edges.
        mode: Linear-system solver mode (see skimage docs).
        tol: Solver convergence tolerance.
    """

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

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        labels = random_walker(
            single_band(_fill_nan(np.asarray(gt)), name="RandomWalker"),
            single_band(self.markers, name="RandomWalker markers").astype(
                np.int32, copy=False
            ),
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
    """Grow label regions by a fixed pixel distance without overlap.

    Wraps :func:`skimage.segmentation.expand_labels`. Expects a
    single-band ``(H, W)`` or ``(1, H, W)`` integer label map. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns an ``int32``
    label map in the same carrier kind.

    Args:
        distance: Euclidean distance (in pixels) by which each labelled
            region is grown into the background.
    """

    def __init__(self, *, distance: float = 1.0) -> None:
        self.distance = distance

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        labels = expand_labels(
            single_band(np.asarray(gt), name="ExpandLabels").astype(
                np.int32, copy=False
            ),
            distance=self.distance,
        )
        return _labels(gt, labels)

    def get_config(self) -> dict[str, Any]:
        return {"distance": self.distance}


def _bbox_edge_distance(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> float:
    """Minimum edge-to-edge Euclidean distance between two ``(y1, x1, y2, x2)``
    boxes. Returns 0 when the boxes overlap or touch."""
    y1a, x1a, y2a, x2a = box1
    y1b, x1b, y2b, x2b = box2
    dx = max(0, x1a - x2b, x1b - x2a)
    dy = max(0, y1a - y2b, y1b - y2a)
    return float(np.hypot(dx, dy))


def _bbox_iou(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> float:
    """Box IoU using exclusive ``(y2, x2)`` convention."""
    y1a, x1a, y2a, x2a = box1
    y1b, x1b, y2b, x2b = box2
    yi1, xi1 = max(y1a, y1b), max(x1a, x1b)
    yi2, xi2 = min(y2a, y2b), min(x2a, x2b)
    if yi2 <= yi1 or xi2 <= xi1:
        return 0.0
    inter = (yi2 - yi1) * (xi2 - xi1)
    area_a = (y2a - y1a) * (x2a - x1a)
    area_b = (y2b - y1b) * (x2b - x1b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _merge_nearby_instances(
    labels: Num[np.ndarray, "h w"],
    *,
    distance_threshold: float,
    iou_threshold_min: float,
    iou_threshold_max: float,
    classes: Mapping[int, int] | None,
    start_label: int,
) -> Int[np.ndarray, "h w"]:
    """Merge instance labels whose bboxes are close and partially overlap.

    Connectivity rule (from Pérez Carrasco et al. 2026):
    two instances are connected iff their bbox edge-to-edge distance is
    strictly below ``distance_threshold`` *and* their box IoU lies in the
    open interval ``(iou_threshold_min, iou_threshold_max)``.
    Connected components are then merged via pixel-wise union.
    """
    work = np.where(labels > 0, labels, 0).astype(np.int64, copy=False)
    ids = np.unique(work)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros_like(work, dtype=np.int64)

    # ``find_objects`` returns a list indexed by ``label - 1`` whose entries are
    # ``(slice_y, slice_x)`` with exclusive ``stop`` — matching the paper's
    # ``[y1, x1, y2, x2]`` convention used by ``_bbox_edge_distance``/``_bbox_iou``.
    slices = ndimage.find_objects(work)
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for lbl_value in ids:
        lbl = int(lbl_value)
        sl = slices[lbl - 1]
        if sl is None:
            continue
        y_sl, x_sl = sl
        boxes[lbl] = (
            int(y_sl.start),
            int(x_sl.start),
            int(y_sl.stop),
            int(x_sl.stop),
        )

    instance_ids = list(boxes.keys())
    n = len(instance_ids)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        bi = boxes[instance_ids[i]]
        ci = None if classes is None else classes.get(instance_ids[i])
        for j in range(i + 1, n):
            if classes is not None:
                cj = classes.get(instance_ids[j])
                if ci != cj:
                    continue
            bj = boxes[instance_ids[j]]
            if _bbox_edge_distance(bi, bj) >= distance_threshold:
                continue
            iou = _bbox_iou(bi, bj)
            if not (iou_threshold_min < iou < iou_threshold_max):
                continue
            union(i, j)

    root_to_new: dict[int, int] = {}
    remap: dict[int, int] = {}
    next_label = start_label
    for i, lbl in enumerate(instance_ids):
        root = find(i)
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        remap[lbl] = root_to_new[root]

    max_label = int(work.max())
    lookup = np.zeros(max_label + 1, dtype=np.int64)
    for old_lbl, new_lbl in remap.items():
        lookup[old_lbl] = new_lbl
    return lookup[work]


class MergeNearbyInstances(Operator):
    """Merge instance labels whose bboxes are close and partially overlap.

    Adapted from Pérez Carrasco et al. (2026), *Plume Segmentation from
    MethaneSAT with Cross-Sensor Transfer Learning and Physics-Informed
    Postprocessing* (``merging.merge_spatial_fragments_v2``). Used there as
    Mask R-CNN postprocessing to stitch instance fragments split across
    sliding-window patches.

    Two instances are connected when their bounding boxes are within
    ``distance_threshold`` pixels edge-to-edge **and** their box IoU is in
    the open interval ``(iou_threshold_min, iou_threshold_max)``. Connected
    components are then merged into single instances via pixel-wise union.
    Note that the IoU lower bound is strict, so two close-but-disjoint
    boxes (IoU = 0) never merge — match the paper's behavior; loosen
    ``iou_threshold_min`` to ``0.0`` to recover proximity-only merging.

    Args:
        distance_threshold: Maximum edge-to-edge bbox distance (pixels).
        iou_threshold_min: Lower (exclusive) bound on box IoU.
        iou_threshold_max: Upper (exclusive) bound on box IoU.
        classes: Optional ``{instance_label: class_id}`` mapping; when
            provided, only same-class pairs are eligible to merge.
        start_label: Starting integer label for the relabeled output.

    Accepts a single-band ``GeoTensor`` or a plain ``np.ndarray`` label
    map and returns an ``int32`` label map in the same carrier kind.
    """

    def __init__(
        self,
        *,
        distance_threshold: float = 40.0,
        iou_threshold_min: float = 0.01,
        iou_threshold_max: float = 0.65,
        classes: Mapping[int, int] | None = None,
        start_label: int = 1,
    ) -> None:
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative")
        if not 0.0 <= iou_threshold_min < iou_threshold_max <= 1.0:
            raise ValueError("require 0 <= iou_threshold_min < iou_threshold_max <= 1")
        if start_label < 1:
            raise ValueError("start_label must be >= 1")
        self.distance_threshold = float(distance_threshold)
        self.iou_threshold_min = float(iou_threshold_min)
        self.iou_threshold_max = float(iou_threshold_max)
        self.classes = None if classes is None else dict(classes)
        self.start_label = int(start_label)

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        labels_in = single_band(np.asarray(gt), name="MergeNearbyInstances")
        merged = _merge_nearby_instances(
            labels_in,
            distance_threshold=self.distance_threshold,
            iou_threshold_min=self.iou_threshold_min,
            iou_threshold_max=self.iou_threshold_max,
            classes=self.classes,
            start_label=self.start_label,
        )
        return wrap_like(gt, merged.astype(np.int32, copy=False))

    def get_config(self) -> dict[str, Any]:
        return {
            "distance_threshold": self.distance_threshold,
            "iou_threshold_min": self.iou_threshold_min,
            "iou_threshold_max": self.iou_threshold_max,
            "classes": None if self.classes is None else dict(self.classes),
            "start_label": self.start_label,
        }


def _mask_nms(
    masks: Num[np.ndarray, "n h w"] | Bool[np.ndarray, "n h w"],
    scores: Float[np.ndarray, " n"] | None,
    iou_threshold: float,
    start_label: int,
) -> Int[np.ndarray, "h w"]:
    """Suppress overlapping mask predictions via mask-IoU and stitch
    survivors into a 2-D label map (suppressed planes -> background).

    Mirrors Pérez Carrasco et al. (2026), ``non_max_suppression_masks``.
    When ``scores`` is None instances are ranked by area, largest first.
    """
    if masks.ndim != 3:
        raise ValueError(f"expected a (N, H, W) mask stack, got shape {masks.shape}")
    n, h, w = masks.shape
    if n == 0:
        return np.zeros((h, w), dtype=np.int64)
    bool_masks = masks.astype(bool, copy=False)
    areas = bool_masks.reshape(n, -1).sum(axis=1).astype(np.int64)
    if scores is None:
        rank = np.argsort(-areas, kind="stable")
    else:
        if scores.shape != (n,):
            raise ValueError(
                f"scores length {scores.shape} does not match mask count ({n})"
            )
        rank = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    suppressed = np.zeros(n, dtype=bool)
    for idx in rank:
        i = int(idx)
        if suppressed[i] or areas[i] == 0:
            continue
        keep.append(i)
        mi = bool_masks[i]
        ai = int(areas[i])
        for jdx in rank:
            j = int(jdx)
            if j == i or suppressed[j] or j in keep:
                continue
            inter = int(np.logical_and(mi, bool_masks[j]).sum())
            if inter == 0:
                continue
            union = ai + int(areas[j]) - inter
            if union <= 0:
                continue
            if (inter / union) > iou_threshold:
                suppressed[j] = True

    # Higher-ranked masks claim pixels first; later survivors below the
    # suppression threshold but with some overlap must not overwrite that
    # claim, or downstream area/statistics for the highest-confidence
    # detection would silently shrink. Paste in keep order, into pixels
    # still unassigned.
    out = np.zeros((h, w), dtype=np.int64)
    for new_offset, src in enumerate(keep):
        target = bool_masks[src] & (out == 0)
        out[target] = start_label + new_offset
    return out


class MaskNMS(Operator):
    """Mask-IoU non-maximum suppression over a stack of instance masks.

    Adapted from Pérez Carrasco et al. (2026), ``metrics_instance_segmentation.
    non_max_suppression_masks``. The operator ranks instance masks by
    their score (or by pixel area when no scores are supplied), then
    suppresses any later mask whose mask-IoU against a kept mask exceeds
    ``iou_threshold``. Surviving masks are pasted into a 2-D label map
    numbered from ``start_label`` in keep order.

    Input is a ``(N, H, W)`` mask stack carrier — one plane per candidate
    instance. This is the natural shape for raw detector output before
    fragments have been merged into a single non-overlapping label map;
    use this *before* :class:`MergeNearbyInstances` when multiple
    detections may cover the same object.

    Args:
        iou_threshold: Pairs with mask-IoU above this are suppressed.
        scores: Optional length-``N`` array of per-instance scores. When
            None, instances are ranked by pixel area (largest first).
        start_label: Starting integer label for the renumbered output.

    Accepts a ``GeoTensor`` or a plain ``np.ndarray`` mask stack and
    returns an ``int32`` label map in the same carrier kind.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.1,
        scores: np.ndarray | None = None,
        start_label: int = 1,
    ) -> None:
        if not 0.0 < iou_threshold < 1.0:
            raise ValueError("iou_threshold must be in (0, 1)")
        if start_label < 1:
            raise ValueError("start_label must be >= 1")
        self.iou_threshold = float(iou_threshold)
        self.scores = None if scores is None else np.asarray(scores, dtype=float)
        self.start_label = int(start_label)

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        masks = np.asarray(gt)
        out = _mask_nms(
            masks,
            self.scores,
            iou_threshold=self.iou_threshold,
            start_label=self.start_label,
        )
        return _labels(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "scores": None if self.scores is None else self.scores.tolist(),
            "start_label": self.start_label,
        }


class MarkBoundaries(Operator):
    """Overlay segmentation boundaries on an image for visual inspection.

    Wraps :func:`skimage.segmentation.mark_boundaries`. Channel-first
    inputs are moved to channel-last for skimage and back afterwards.
    Accepts a ``GeoTensor`` or a plain ``np.ndarray`` and returns the
    RGB overlay in the same carrier kind.

    Args:
        label_img: Single-band integer label map whose region boundaries
            are drawn; required. Not YAML-serialisable, so instances are
            forbidden in YAML.
        color: RGB color (floats in 0-1) of the boundary lines.
        mode: Boundary style — ``"thick"``, ``"inner"``, ``"outer"`` or
            ``"subpixel"``.
    """

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

    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        image = np.asarray(gt)
        if image.ndim == 3:
            image = einx.id("c h w -> h w c", image)
        marked = mark_boundaries(
            image,
            single_band(self.label_img, name="MarkBoundaries label_img").astype(
                np.int32, copy=False
            ),
            color=self.color,
            mode=self.mode,
        )
        if marked.ndim == 3:
            marked = einx.id("h w c -> c h w", marked)
        return wrap_like(gt, marked)

    def get_config(self) -> dict[str, Any]:
        return {
            "label_img": "provided",
            "color": self.color,
            "mode": self.mode,
        }

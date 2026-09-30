"""Carrier-aware wrappers around :mod:`skimage.segmentation` primitives.

The image segmenters (SLIC, Felzenszwalb, Quickshift, Watershed, ChanVese,
RandomWalker) judge nodata with :func:`geotoolz._src.valid.valid_pixels`
-- a pixel is invalid when any band is non-finite or equals the carrier's
``fill_value_default``. Invalid pixels are median-filled (from valid pixels
only) before the skimage call, kept out of it where the algorithm takes a
mask, and always carry the "no segment" label ``0`` in the output (whose
``fill_value_default`` is ``0``).

A ``(T, C, H, W)`` time stack is segmented frame by frame: the output is
the per-frame label maps (or boundary images) stacked as
``(T, 1, H, W)`` (``(T, 3, H, W)`` for :class:`MarkBoundaries`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import einx
import numpy as np
from jaxtyping import Bool
from pipekit import Operator
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

from geotoolz._src.config import (
    mapping_from_pairs,
    mapping_to_pairs,
    reject_config_summary,
)
from geotoolz._src.labels import Connectivity, connectivity_structure
from geotoolz._src.shape import over_frames, single_band
from geotoolz._src.valid import (
    carried_fill,
    mask_invalid_to_nan,
    valid_pixels,
    wrap_filled,
)
from geotoolz._src.wrap import wrap_like
from geotoolz.segment._src.array import (
    ThresholdMode,
    fill_invalid,
    mask_nms,
    merge_nearby_instances,
    threshold_mask,
)


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


def _array_summary(value: Any) -> dict[str, Any] | None:
    """Debug-config summary of a runtime array argument (``None`` passes)."""
    if value is None:
        return None
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


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

    Nodata pixels (non-finite or ``fill_value_default`` in any band) are
    median-filled, excluded from clustering and forced to label ``0`` in
    the output. Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` (channel-first) and returns an ``int32`` label map in
    the same carrier kind.

    Args:
        n_segments: Approximate number of superpixels to produce.
        compactness: Balance between color and spatial proximity; higher
            values yield more compact (squarer) segments.
        sigma: Width of the Gaussian pre-smoothing kernel, in pixels.
        axis: Position of the band (channel) axis (``0`` for channel-first
            cubes, ``None`` for 2-D single-band input). Ignored for 2-D
            input, which is always treated as single-band.
        start_label: First label assigned to a superpixel.
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``. ``get_config`` summarises a mask
            as ``{"shape", "dtype"}``, which ``Operator.from_state``
            refuses to rebuild.
    """

    def __init__(
        self,
        *,
        n_segments: int = 100,
        compactness: float = 10.0,
        sigma: float = 0.0,
        axis: int | None = 0,
        start_label: int = 1,
        mask: Any = None,
    ) -> None:
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.axis = axis
        self.start_label = start_label
        self.mask = reject_config_summary(mask, "mask")

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = valid_pixels(gt)
        image = fill_invalid(np.asarray(gt), valid)
        mask = _as_mask(self.mask, gt.shape[-2:], name="SLIC mask")
        if mask is not None:
            valid &= mask
        # A 2-D image has no channel axis; skimage raises if one is given.
        channel_axis = None if image.ndim == 2 else self.axis
        labels = slic(
            image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            channel_axis=channel_axis,
            start_label=self.start_label,
            mask=valid,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "compactness": self.compactness,
            "sigma": self.sigma,
            "axis": self.axis,
            "start_label": self.start_label,
            "mask": _array_summary(self.mask),
        }


class Felzenszwalb(Operator):
    """Graph-based segmentation via :func:`skimage.segmentation.felzenszwalb`.

    Nodata pixels (non-finite or ``fill_value_default`` in any band) are
    median-filled before segmentation and forced to label ``0`` in the
    output; valid segments are labelled from ``1``.
    Accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` (channel-first) and returns an ``int32`` label map in
    the same carrier kind.

    Args:
        scale: Free parameter controlling segment size; larger values
            produce larger segments.
        sigma: Width of the Gaussian pre-smoothing kernel, in pixels.
        min_area_px: Minimum segment area in pixels, enforced by
            postprocessing (skimage's ``min_size``).
        axis: Position of the band (channel) axis (``0`` for channel-first
            cubes, ``None`` for 2-D single-band input).
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``. ``get_config`` summarises a mask
            as ``{"shape", "dtype"}``, which ``Operator.from_state``
            refuses to rebuild.
    """

    def __init__(
        self,
        *,
        scale: float = 1.0,
        sigma: float = 0.8,
        min_area_px: int = 20,
        axis: int | None = 0,
        mask: Any = None,
    ) -> None:
        self.scale = scale
        self.sigma = sigma
        self.min_area_px = min_area_px
        self.axis = axis
        self.mask = reject_config_summary(mask, "mask")

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = valid_pixels(gt)
        image = fill_invalid(np.asarray(gt), valid)
        mask = _as_mask(self.mask, gt.shape[-2:], name="Felzenszwalb mask")
        if mask is not None:
            valid &= mask
        labels = felzenszwalb(
            image,
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_area_px,
            channel_axis=self.axis,
        )
        # skimage numbers segments from 0, which collides with the
        # invalid-pixel label; shift so valid segments start at 1.
        return _labels(gt, labels + 1, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "sigma": self.sigma,
            "min_area_px": self.min_area_px,
            "axis": self.axis,
            "mask": _array_summary(self.mask),
        }


class Quickshift(Operator):
    """Mode-seeking superpixels via :func:`skimage.segmentation.quickshift`.

    Nodata pixels (non-finite or ``fill_value_default`` in any band) are
    median-filled before segmentation and forced to label ``0`` in the
    output; valid segments are labelled from ``1``.
    Accepts a ``GeoTensor`` or a plain
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
        axis: Position of the band (channel) axis (``0`` for channel-first
            cubes, ``None`` for 2-D single-band input).
        convert2lab: Convert the image to LAB space first. Defaults to
            ``False`` (unlike skimage) so non-RGB / multispectral /
            single-band inputs work out of the box.
        mask: Optional ``(H, W)`` (or ``(1, H, W)``) boolean mask; pixels
            outside it get label ``0``. ``get_config`` summarises a mask
            as ``{"shape", "dtype"}``, which ``Operator.from_state``
            refuses to rebuild.
    """

    def __init__(
        self,
        *,
        kernel_size: float = 5.0,
        max_dist: float = 10.0,
        ratio: float = 1.0,
        sigma: float = 0.0,
        axis: int | None = 0,
        convert2lab: bool = False,
        mask: Any = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        self.sigma = sigma
        self.axis = axis
        # Default to False so non-RGB / multispectral / single-band inputs
        # work out of the box. skimage's quickshift defaults convert2lab=True,
        # which raises on any input that is not exactly 3-channel RGB.
        self.convert2lab = convert2lab
        self.mask = reject_config_summary(mask, "mask")

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = valid_pixels(gt)
        image = fill_invalid(np.asarray(gt), valid)
        mask = _as_mask(self.mask, gt.shape[-2:], name="Quickshift mask")
        if mask is not None:
            valid &= mask
        labels = quickshift(
            image,
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            ratio=self.ratio,
            sigma=self.sigma,
            channel_axis=self.axis,
            convert2lab=self.convert2lab,
        )
        # skimage numbers segments from 0, which collides with the
        # invalid-pixel label; shift so valid segments start at 1.
        return _labels(gt, labels + 1, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "kernel_size": self.kernel_size,
            "max_dist": self.max_dist,
            "ratio": self.ratio,
            "sigma": self.sigma,
            "axis": self.axis,
            "convert2lab": self.convert2lab,
            "mask": _array_summary(self.mask),
        }


class Watershed(Operator):
    """Watershed segmentation via :func:`skimage.segmentation.watershed`.

    Expects a single-band ``(H, W)`` or ``(1, H, W)`` image. Nodata pixels
    (non-finite or ``fill_value_default``) are median-filled, masked out of
    the flooding and forced to label ``0`` in the output. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns an ``int32`` label
    map in the same carrier kind.

    Args:
        markers: Optional single-band integer marker array seeding the
            basins; when None, local minima of the image are used.
        connectivity: ``4`` (edge neighbours, default) or ``8`` (edge +
            diagonal neighbours) used for flooding -- the package-wide
            spelling, converted to a footprint for skimage (whose ``1`` /
            ``2`` spelling is not accepted).
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
        connectivity: Connectivity = 4,
        compactness: float = 0.0,
        watershed_line: bool = False,
        mask: Any = None,
    ) -> None:
        connectivity_structure(connectivity)  # validate 4 | 8 up front
        self.markers = markers
        self.connectivity = connectivity
        self.compactness = compactness
        self.watershed_line = watershed_line
        self.mask = reject_config_summary(mask, "mask")

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = valid_pixels(gt)
        image = single_band(fill_invalid(np.asarray(gt), valid), name="Watershed")
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
            connectivity=connectivity_structure(self.connectivity),
            mask=valid,
            compactness=self.compactness,
            watershed_line=self.watershed_line,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "markers": _array_summary(self.markers),
            "connectivity": self.connectivity,
            "compactness": self.compactness,
            "watershed_line": self.watershed_line,
            "mask": _array_summary(self.mask),
        }


class ChanVese(Operator):
    """Active-contour segmentation via :func:`skimage.segmentation.chan_vese`.

    Expects a single-band ``(H, W)`` or ``(1, H, W)`` image. Nodata pixels
    (non-finite or ``fill_value_default``) are median-filled before
    evolution and forced to label ``0`` in the output. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns an ``int32`` label
    map (0 = outside, 1 = inside) in the same carrier kind.

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

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = valid_pixels(gt)
        labels = chan_vese(
            single_band(fill_invalid(np.asarray(gt), valid), name="ChanVese"),
            mu=self.mu,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            tol=self.tol,
            max_num_iter=self.max_num_iter,
        )
        return _labels(gt, labels.astype(np.int32), mask=valid)


class RandomWalker(Operator):
    """Seeded segmentation via :func:`skimage.segmentation.random_walker`.

    Expects a single-band ``(H, W)`` or ``(1, H, W)`` image. Nodata pixels
    (non-finite or ``fill_value_default``) are median-filled, marked
    inactive (removed from the diffusion graph) and labelled ``0`` in the
    output. Accepts a ``GeoTensor``
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

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        valid = valid_pixels(gt)
        markers = single_band(self.markers, name="RandomWalker markers").astype(
            np.int32, copy=True
        )
        # Negative markers are inactive in skimage: nodata pixels are removed
        # from the diffusion graph so no label can spread through them.
        markers[~valid] = -1
        labels = random_walker(
            single_band(fill_invalid(np.asarray(gt), valid), name="RandomWalker"),
            markers,
            beta=self.beta,
            mode=self.mode,
            tol=self.tol,
        )
        return _labels(gt, labels, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "markers": _array_summary(self.markers),
            "beta": self.beta,
            "mode": self.mode,
            "tol": self.tol,
        }


class Threshold(Operator):
    """Binary mask of the pixels strictly above a global threshold.

    Delegates to :func:`geotoolz.segment.threshold_mask`: the threshold is
    an absolute number, Otsu's (1979) between-class-variance optimum, or
    a percentile of the input, and the mask is ``values > t``. The
    data-driven modes use one threshold over every band and pixel of a
    frame; a ``(T, C, H, W)`` time stack is thresholded frame by frame.

    Nodata pixels (non-finite or equal to the input's fill value, in any
    band) are excluded from the Otsu / percentile statistic and are always
    ``False``; the output declares ``fill_value_default=False``. Accepts a
    ``GeoTensor`` or a plain ``np.ndarray`` and returns a boolean mask of
    the input's shape in the same carrier kind.

    :class:`geotoolz.plume.PlumeMask` is this thresholding followed by a
    minimum-component-size filter.

    Args:
        threshold: ``float`` (absolute), ``"otsu"``, or
            ``"percentile:<p>"`` with ``p`` in ``[0, 100]``.
        nbins: Histogram bins for the ``"otsu"`` threshold; ignored by
            the other modes. Default ``256``.

    Raises:
        ValueError: If ``threshold`` is a string other than ``"otsu"`` or
            ``"percentile:<p>"``.

    Examples:
        >>> mask = gz.segment.Threshold(threshold="otsu")(score_map)
        >>> hot = gz.segment.Threshold(threshold="percentile:99")(score_map)
        >>> above = gz.segment.Threshold(threshold=0.5)(probability)
    """

    def __init__(self, *, threshold: ThresholdMode = "otsu", nbins: int = 256) -> None:
        if isinstance(threshold, str) and not (
            threshold == "otsu" or threshold.startswith("percentile:")
        ):
            raise ValueError(
                "threshold must be a number, 'otsu', or 'percentile:<p>'; "
                f"got {threshold!r}"
            )
        self.threshold = threshold
        self.nbins = nbins

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        mask = threshold_mask(mask_invalid_to_nan(gt), self.threshold, nbins=self.nbins)
        return wrap_like(gt, mask, fill_value_default=False)


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

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        labels = expand_labels(
            single_band(np.asarray(gt), name="ExpandLabels").astype(
                np.int32, copy=False
            ),
            distance=self.distance,
        )
        return _labels(gt, labels)


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
        classes: Optional ``{instance_label: class_id}`` mapping (or the
            ``[[instance_label, class_id], ...]`` pairs ``get_config``
            emits); when provided, only same-class pairs are eligible to
            merge.
        start_label: Starting integer label for the relabeled output.

    Accepts a single-band ``GeoTensor`` or a plain ``np.ndarray`` label
    map and returns an ``int32`` label map in the same carrier kind, with
    ``fill_value_default=0``; nodata input pixels are ``0``.
    """

    def __init__(
        self,
        *,
        distance_threshold: float = 40.0,
        iou_threshold_min: float = 0.01,
        iou_threshold_max: float = 0.65,
        classes: Mapping[int, int] | list[list[int]] | None = None,
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
        self.classes = mapping_from_pairs(classes)
        self.start_label = int(start_label)

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        labels_in = single_band(np.asarray(gt), name="MergeNearbyInstances")
        merged = merge_nearby_instances(
            labels_in,
            distance_threshold=self.distance_threshold,
            iou_threshold_min=self.iou_threshold_min,
            iou_threshold_max=self.iou_threshold_max,
            classes=self.classes,
            start_label=self.start_label,
        )
        # Label map: 0 is "no instance", including at nodata input pixels.
        return _labels(gt, merged, valid_pixels(gt))

    def get_config(self) -> dict[str, Any]:
        return {
            "distance_threshold": self.distance_threshold,
            "iou_threshold_min": self.iou_threshold_min,
            "iou_threshold_max": self.iou_threshold_max,
            "classes": mapping_to_pairs(self.classes),
            "start_label": self.start_label,
        }


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
    returns an ``int32`` label map in the same carrier kind, with
    ``fill_value_default=0``; pixels that are nodata in any plane are ``0``.
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

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        masks = np.asarray(gt)
        out = mask_nms(
            masks,
            self.scores,
            iou_threshold=self.iou_threshold,
            start_label=self.start_label,
        )
        return _labels(gt, out, valid_pixels(gt))

    def get_config(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "scores": None if self.scores is None else self.scores.tolist(),
            "start_label": self.start_label,
        }


class MarkBoundaries(Operator):
    """Overlay segmentation boundaries on an image for visual inspection.

    Wraps :func:`skimage.segmentation.mark_boundaries`. Accepts a
    single-band ``(H, W)`` / ``(1, H, W)`` image or a 3-band ``(3, H, W)``
    RGB image; channel-first inputs are moved to channel-last for skimage
    and back afterwards. The output is always a ``(3, H, W)`` RGB overlay.
    Any other band count raises ``ValueError``. Accepts a ``GeoTensor``
    or a plain ``np.ndarray`` and returns the overlay in the same carrier
    kind.

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

    @over_frames
    def _apply(self, gt: GeoTensorType | np.ndarray) -> GeoTensorType | np.ndarray:
        image = np.asarray(gt)
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = single_band(image, name="MarkBoundaries")
            elif image.shape[0] == 3:
                image = einx.id("c h w -> h w c", image)
            else:
                raise ValueError(
                    "MarkBoundaries expects a single-band (H, W) / (1, H, W) "
                    f"image or a 3-band (3, H, W) RGB image; got shape {image.shape}"
                )
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
        return wrap_filled(
            gt, marked, fill_value_default=carried_fill(gt, np.asarray(marked).dtype)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "label_img": _array_summary(self.label_img),
            "color": self.color,
            "mode": self.mode,
        }

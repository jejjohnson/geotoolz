"""Tier-A primitives for :mod:`geotoolz.segment` -- pure numpy, no ``GeoTensor``.

* :func:`fill_invalid` -- median-fill nodata before a skimage segmenter.
* :func:`otsu_threshold` / :func:`resolve_threshold` /
  :func:`threshold_mask` -- global thresholding (absolute, Otsu (1979) or
  percentile), the simplest binary segmentation of a score map; also
  used by :class:`geotoolz.plume.PlumeMask`.
* :func:`merge_nearby_instances` / :func:`mask_nms` -- instance
  post-processing after Pérez Carrasco et al. (2026).

The carrier-aware wrappers live in
:mod:`geotoolz.segment._src.operators`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from jaxtyping import Bool, Float, Int, Num
from scipy import ndimage


ThresholdMode = float | int | str


def fill_invalid(
    image: Num[np.ndarray, "*dims"], valid: Bool[np.ndarray, "h w"]
) -> Float[np.ndarray, "*dims"]:
    """Float copy of ``image`` with every invalid pixel set to the valid median.

    The fill statistic is the median over valid pixels only, so neither a
    ``fill_value_default`` sentinel nor ``+/-inf`` can leak into it. The
    segmenters run on the filled image, because skimage has no notion of
    nodata.

    Args:
        image: ``(H, W)`` or ``(C, H, W)`` image.
        valid: ``(H, W)`` validity; ``False`` pixels are filled in every
            band.

    Returns:
        A float copy (``image`` itself, as float, when all pixels are
        valid). An image with no valid pixel is filled with ``0``.

    Examples:
        >>> image = np.array([[1.0, 2.0], [3.0, -9999.0]])
        >>> fill_invalid(image, image != -9999.0)
        array([[1., 2.],
               [3., 2.]])
    """
    arr = np.asarray(image, dtype=float)
    if valid.all():
        return arr
    invalid = np.broadcast_to(~valid, arr.shape)
    fill = float(np.median(arr[~invalid])) if valid.any() else 0.0
    arr = arr.copy()
    arr[invalid] = fill
    return arr


def otsu_threshold(values: Num[np.ndarray, "*dims"], *, nbins: int = 256) -> float:
    """Compute Otsu's between-class-variance threshold, ignoring NaNs.

    Builds an ``nbins``-bin histogram of the finite values and returns
    the bin center that maximises the between-class variance
    ``w_bg * w_fg * (mu_bg - mu_fg)^2`` (Otsu, 1979). A constant input
    returns that constant.

    Args:
        values: Array of any shape; non-finite entries are ignored.
        nbins: Number of histogram bins. Default ``256``.

    Returns:
        The threshold; pixels strictly above it are foreground.

    Raises:
        ValueError: If ``values`` contains no finite entries.

    Examples:
        >>> values = np.r_[np.zeros(50), np.ones(50)]
        >>> 0.0 < otsu_threshold(values) < 1.0
        True
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot compute an Otsu threshold on all-NaN data")
    if np.all(finite == finite[0]):
        return float(finite[0])

    hist, bin_edges = np.histogram(finite, bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = finite.size - weight_bg

    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.divide(
        np.cumsum(hist * centers),
        weight_bg,
        out=np.zeros_like(centers, dtype=float),
        where=weight_bg > 0,
    )
    mean_fg = np.divide(
        np.cumsum((hist * centers)[::-1])[::-1] - hist * centers,
        weight_fg,
        out=np.zeros_like(centers, dtype=float),
        where=weight_fg > 0,
    )
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance[~valid] = -np.inf
    return float(centers[int(np.argmax(variance))])


def resolve_threshold(
    values: Num[np.ndarray, "*dims"], threshold: ThresholdMode, *, nbins: int = 256
) -> float:
    """Resolve an absolute, Otsu, or percentile threshold to a float.

    Args:
        values: Data the data-driven modes are evaluated on.
        threshold: A number (returned as-is), ``"otsu"`` (see
            :func:`otsu_threshold`), or ``"percentile:<p>"`` with ``p``
            in ``[0, 100]`` (NaN-aware percentile of ``values``).
        nbins: Histogram bins for the ``"otsu"`` mode, passed to
            :func:`otsu_threshold`. Default ``256``.

    Returns:
        The resolved threshold value.

    Raises:
        ValueError: If a string threshold is neither ``"otsu"`` nor a
            valid ``"percentile:<p>"`` spec.

    Examples:
        >>> resolve_threshold(np.arange(101.0), "percentile:90")
        90.0
        >>> resolve_threshold(np.arange(101.0), 3)
        3.0
    """
    if isinstance(threshold, str):
        if threshold == "otsu":
            return otsu_threshold(values, nbins=nbins)
        prefix = "percentile:"
        if threshold.startswith(prefix):
            percentile = float(threshold.removeprefix(prefix))
            if not 0.0 <= percentile <= 100.0:
                raise ValueError("percentile threshold must be in [0, 100]")
            return float(np.nanpercentile(values, percentile))
        raise ValueError("threshold must be a number, 'otsu', or 'percentile:<p>'")
    return float(threshold)


def threshold_mask(
    values: Num[np.ndarray, "*dims"],
    threshold: ThresholdMode = "otsu",
    *,
    nbins: int = 256,
) -> Bool[np.ndarray, "*dims"]:
    """Binary mask of the samples strictly above a resolved threshold.

        mask = values > t,   t = resolve_threshold(values, threshold)

    Non-finite samples are ignored by the data-driven modes and are never
    foreground (``NaN > t`` is ``False``).

    Args:
        values: Array of any shape; mark nodata as ``NaN``.
        threshold: A number, ``"otsu"`` or ``"percentile:<p>"``; see
            :func:`resolve_threshold`.
        nbins: Histogram bins for the ``"otsu"`` mode. Default ``256``.

    Returns:
        Boolean array shaped like ``values``.

    Examples:
        >>> threshold_mask(np.array([0.0, np.nan, 5.0]), 1.0)
        array([False, False,  True])
    """
    arr = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        return arr > resolve_threshold(arr, threshold, nbins=nbins)


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


def merge_nearby_instances(
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

    Args:
        labels: ``(H, W)`` instance label map; ``<= 0`` is background.
        distance_threshold: Maximum edge-to-edge bbox distance (pixels),
            exclusive.
        iou_threshold_min: Lower (exclusive) bound on box IoU.
        iou_threshold_max: Upper (exclusive) bound on box IoU.
        classes: Optional ``{instance_label: class_id}``; only same-class
            pairs may merge.
        start_label: First label of the relabelled output.

    Returns:
        ``int64`` label map, merged groups numbered from ``start_label``
        in order of their first member label.

    Examples:
        >>> labels = np.full((4, 4), 2)  # an L-shaped 1 wrapping a block of 2
        >>> labels[0, :], labels[:, 0] = 1, 1
        >>> np.unique(merge_nearby_instances(
        ...     labels, distance_threshold=5, iou_threshold_min=0.0,
        ...     iou_threshold_max=1.0, classes=None, start_label=1))
        array([1])
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


def mask_nms(
    masks: Num[np.ndarray, "n h w"] | Bool[np.ndarray, "n h w"],
    scores: Float[np.ndarray, " n"] | None,
    iou_threshold: float,
    start_label: int,
) -> Int[np.ndarray, "h w"]:
    """Suppress overlapping mask predictions via mask-IoU and stitch
    survivors into a 2-D label map (suppressed planes -> background).

    Mirrors Pérez Carrasco et al. (2026), ``non_max_suppression_masks``.
    When ``scores`` is None instances are ranked by area, largest first.

    Args:
        masks: ``(N, H, W)`` instance mask stack.
        scores: ``(N,)`` confidence per mask, or ``None`` to rank by area.
        iou_threshold: Masks whose IoU with a kept mask exceeds this are
            suppressed.
        start_label: Label of the first kept mask.

    Returns:
        ``(H, W)`` ``int64`` label map; earlier-ranked survivors keep
        overlapping pixels.

    Raises:
        ValueError: If ``masks`` is not 3-D or ``scores`` has the wrong
            length.

    Examples:
        >>> masks = np.zeros((2, 3, 3), dtype=bool)
        >>> masks[0, :2], masks[1, :3] = True, True
        >>> mask_nms(masks, None, 0.5, 1)
        array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]])
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


__all__ = [
    "ThresholdMode",
    "fill_invalid",
    "mask_nms",
    "merge_nearby_instances",
    "otsu_threshold",
    "resolve_threshold",
    "threshold_mask",
]

"""Tests for segmentation operators."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:4326",
        fill_value_default=0,
    )


def test_slic_returns_int_labels_and_masks_nan_pixels() -> None:
    values = np.zeros((1, 8, 8), dtype=float)
    values[:, :, 4:] = 1.0
    values[:, 0, 0] = np.nan
    gt = _gt(values)

    labels = gz.segment.SLIC(n_segments=4, compactness=1.0)(gt)

    assert labels.shape == gt.shape[-2:]
    assert labels.transform == gt.transform
    assert np.asarray(labels).dtype == np.int32
    assert np.asarray(labels)[0, 0] == 0
    assert np.asarray(labels)[0, 1] > 0
    assert np.asarray(labels).max() > 0


def test_watershed_separates_marker_basins() -> None:
    image = _gt(np.array([[3.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0]]))
    markers = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int32)

    labels = gz.segment.Watershed(markers=markers)(image)

    assert set(np.unique(np.asarray(labels))) >= {1, 2}


def _checker_gt(shape: tuple[int, int] = (8, 8)) -> GeoTensor:
    h, w = shape
    values = np.zeros((1, h, w), dtype=float)
    half = w // 2
    values[:, :, half:] = 1.0
    return _gt(values)


def test_felzenszwalb_metadata_and_nan_mask() -> None:
    gt = _checker_gt()
    arr = np.asarray(gt).copy()
    arr[:, 0, 0] = np.nan
    gt_nan = _gt(arr)

    out = gz.segment.Felzenszwalb(scale=1.0, min_size=1)(gt_nan)

    assert out.transform == gt_nan.transform
    assert out.crs == gt_nan.crs
    assert np.asarray(out).dtype == np.int32
    # Non-finite pixel must round-trip to label 0.
    assert np.asarray(out)[0, 0] == 0


def test_quickshift_runs_on_single_band_geotensor() -> None:
    # Codex flag: skimage quickshift defaults convert2lab=True which only
    # accepts 3-channel RGB. Our wrapper defaults convert2lab=False so this
    # single-band call must succeed instead of raising.
    gt = _checker_gt()

    out = gz.segment.Quickshift(kernel_size=2.0, max_dist=4.0, ratio=0.5)(gt)

    assert out.shape == gt.shape[-2:]
    assert out.transform == gt.transform
    assert np.asarray(out).dtype == np.int32


def test_quickshift_convert2lab_true_still_works_for_rgb() -> None:
    # Three-channel input remains valid when callers opt back in to LAB.
    rgb = np.random.default_rng(0).random((3, 8, 8))
    gt = _gt(rgb)

    out = gz.segment.Quickshift(
        kernel_size=2.0, max_dist=4.0, ratio=0.5, convert2lab=True
    )(gt)

    assert out.shape == gt.shape[-2:]


def test_chanvese_forces_non_finite_pixels_to_zero_label() -> None:
    gt = _checker_gt()
    arr = np.asarray(gt).copy()
    arr[:, 0, 0] = np.nan
    arr[:, 0, 1] = np.inf
    gt_nan = _gt(arr)

    out = gz.segment.ChanVese(max_num_iter=10)(gt_nan)

    out_arr = np.asarray(out)
    assert out_arr.dtype == np.int32
    assert out_arr[0, 0] == 0
    assert out_arr[0, 1] == 0


def test_random_walker_round_trips_labels() -> None:
    gt = _checker_gt((6, 6))
    markers = np.zeros((6, 6), dtype=np.int32)
    markers[0, 0] = 1
    markers[0, 5] = 2

    op = gz.segment.RandomWalker(markers=markers, beta=10.0, mode="cg_j")
    out = op(gt)

    assert out.transform == gt.transform
    assert np.asarray(out).dtype == np.int32
    assert set(np.unique(np.asarray(out))) >= {1, 2}
    # Carrier with non-serializable markers must be forbidden in YAML.
    assert gz.segment.RandomWalker.forbid_in_yaml is True


def test_expand_labels_grows_regions() -> None:
    labels_in = np.zeros((1, 5, 5), dtype=np.int32)
    labels_in[0, 2, 2] = 7
    gt = _gt(labels_in)

    out = gz.segment.ExpandLabels(distance=1.0)(gt)

    out_arr = np.asarray(out)
    assert out_arr.dtype == np.int32
    # Original pixel keeps its label, and at least one neighbor gets it too.
    assert out_arr[2, 2] == 7
    assert (out_arr == 7).sum() > 1


def test_mark_boundaries_uses_array_as_geotensor() -> None:
    rgb = np.tile(np.linspace(0, 1, 6, dtype=float), (3, 6, 1))
    gt = _gt(rgb)
    label_img = np.zeros((6, 6), dtype=np.int32)
    label_img[:, :3] = 1
    label_img[:, 3:] = 2

    op = gz.segment.MarkBoundaries(label_img=label_img)
    out = op(gt)

    # Metadata propagation comes from array_as_geotensor.
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out.fill_value_default == gt.fill_value_default
    assert np.asarray(out).shape[-2:] == gt.shape[-2:]
    # Non-serializable label_img -> forbid_in_yaml at the class level.
    assert gz.segment.MarkBoundaries.forbid_in_yaml is True


def _labels_gt(values: np.ndarray) -> GeoTensor:
    return _gt(values.astype(np.int32, copy=False))


def _stamp(arr: np.ndarray, lbl: int, y0: int, y1: int, x0: int, x1: int) -> None:
    arr[y0:y1, x0:x1] = lbl


def test_merge_nearby_instances_merges_close_overlapping_boxes() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    # Two boxes whose bounding rectangles overlap moderately: IoU sits inside
    # the (0.01, 0.65) gate and edges are within 40 px.
    _stamp(labels, 1, 2, 12, 2, 12)
    _stamp(labels, 2, 6, 16, 6, 16)
    gt = _labels_gt(labels)

    out = gz.segment.MergeNearbyInstances()(gt)
    out_arr = np.asarray(out)

    non_bg = out_arr[out_arr > 0]
    assert non_bg.size > 0
    assert np.unique(non_bg).size == 1
    # Pixel-wise union preserved coverage of both inputs.
    assert (out_arr > 0).sum() == (labels > 0).sum()


def test_merge_nearby_instances_leaves_distant_boxes_alone() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    _stamp(labels, 1, 0, 5, 0, 5)
    _stamp(labels, 2, 55, 60, 55, 60)
    gt = _labels_gt(labels)

    out = gz.segment.MergeNearbyInstances(distance_threshold=10.0)(gt)
    out_arr = np.asarray(out)

    assert set(np.unique(out_arr)) == {0, 1, 2}


def test_merge_nearby_instances_skips_disjoint_boxes_by_default() -> None:
    # The paper's gate is ``iou_min < iou < iou_max`` (exclusive on the low
    # side), so two close-but-non-overlapping boxes have IoU == 0 and do NOT
    # merge under the default settings — a faithfulness check.
    labels = np.zeros((10, 30), dtype=np.int32)
    _stamp(labels, 1, 2, 8, 2, 8)
    _stamp(labels, 2, 2, 8, 10, 16)
    gt = _labels_gt(labels)

    out = gz.segment.MergeNearbyInstances()(gt)
    assert set(np.unique(np.asarray(out))) == {0, 1, 2}


def test_merge_nearby_instances_class_restriction() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    _stamp(labels, 1, 2, 12, 2, 12)
    _stamp(labels, 2, 6, 16, 6, 16)
    gt = _labels_gt(labels)

    # Identical geometry as the merge test but the instances live in
    # different classes, so the operator must keep them separate.
    out = gz.segment.MergeNearbyInstances(classes={1: 0, 2: 1})(gt)
    out_arr = np.asarray(out)
    assert set(np.unique(out_arr[out_arr > 0])) == {1, 2}


def test_merge_nearby_instances_transitive_chain() -> None:
    # A overlaps B, B overlaps C, but A and C do not overlap directly. The
    # connected-component merge must still collapse all three.
    labels = np.zeros((20, 40), dtype=np.int32)
    _stamp(labels, 1, 2, 12, 2, 12)
    _stamp(labels, 2, 4, 14, 8, 18)
    _stamp(labels, 3, 6, 16, 14, 24)
    gt = _labels_gt(labels)

    out = gz.segment.MergeNearbyInstances()(gt)
    out_arr = np.asarray(out)
    assert np.unique(out_arr[out_arr > 0]).size == 1


def test_merge_nearby_instances_empty_input() -> None:
    gt = _labels_gt(np.zeros((5, 5), dtype=np.int32))
    out = gz.segment.MergeNearbyInstances()(gt)
    out_arr = np.asarray(out)
    assert out_arr.dtype == np.int32
    assert out_arr.shape == (5, 5)
    assert (out_arr == 0).all()


def test_merge_nearby_instances_preserves_carrier_metadata() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    _stamp(labels, 1, 1, 4, 1, 4)
    gt = _labels_gt(labels)

    out = gz.segment.MergeNearbyInstances(start_label=7)(gt)
    out_arr = np.asarray(out)

    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out_arr.dtype == np.int32
    assert set(np.unique(out_arr)) == {0, 7}


def test_merge_nearby_instances_rejects_invalid_iou_window() -> None:
    import pytest

    with pytest.raises(ValueError):
        gz.segment.MergeNearbyInstances(iou_threshold_min=0.5, iou_threshold_max=0.4)


def test_merge_nearby_instances_get_config_round_trip() -> None:
    op = gz.segment.MergeNearbyInstances(
        distance_threshold=12.5,
        iou_threshold_min=0.05,
        iou_threshold_max=0.5,
        classes={1: 0, 2: 1},
        start_label=3,
    )
    cfg = op.get_config()
    assert cfg == {
        "distance_threshold": 12.5,
        "iou_threshold_min": 0.05,
        "iou_threshold_max": 0.5,
        "classes": {1: 0, 2: 1},
        "start_label": 3,
    }


def _mask_stack_gt(masks: np.ndarray) -> GeoTensor:
    return _gt(masks.astype(bool, copy=False))


def test_mask_nms_suppresses_lower_scored_overlap() -> None:
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:10, 0:6] = True  # 60-pixel mask
    masks[1, 0:10, 4:10] = True  # 60-pixel mask, overlaps in cols 4-5

    # Scores favor mask 0 — mask 1 should be suppressed because the IoU
    # (20 / (60 + 60 - 20) = 0.2) clears the default 0.1 gate.
    out = gz.segment.MaskNMS(scores=np.array([0.9, 0.5]))(_mask_stack_gt(masks))
    out_arr = np.asarray(out)
    assert set(np.unique(out_arr[out_arr > 0])) == {1}
    assert (out_arr > 0).sum() == 60


def test_mask_nms_no_overlap_keeps_everything() -> None:
    masks = np.zeros((2, 6, 12), dtype=bool)
    masks[0, 0:6, 0:4] = True
    masks[1, 0:6, 8:12] = True

    out = gz.segment.MaskNMS(scores=np.array([0.9, 0.5]))(_mask_stack_gt(masks))
    assert set(np.unique(np.asarray(out))) == {0, 1, 2}


def test_mask_nms_falls_back_to_area_ranking() -> None:
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:10, 0:3] = True  # 30 pixels
    masks[1, 0:10, 1:10] = True  # 90 pixels, overlap with mask 0 is 20 px
    # Without scores the larger mask (index 1) should win since IoU = 20 /
    # (30 + 90 - 20) = 0.2 > default 0.1 gate. Output renumbers from 1.
    out = gz.segment.MaskNMS(iou_threshold=0.1)(_mask_stack_gt(masks))
    out_arr = np.asarray(out)
    assert set(np.unique(out_arr[out_arr > 0])) == {1}
    assert (out_arr > 0).sum() == 90  # the larger mask survived


def test_mask_nms_rejects_invalid_iou_threshold() -> None:
    import pytest

    with pytest.raises(ValueError):
        gz.segment.MaskNMS(iou_threshold=0.0)
    with pytest.raises(ValueError):
        gz.segment.MaskNMS(iou_threshold=1.0)


def test_mask_nms_higher_ranked_mask_keeps_overlapping_pixels() -> None:
    """Regression (PR #87 review): in keep-order paste, the
    higher-ranked surviving mask must retain pixels it shares with a
    lower-ranked survivor below the IoU gate — otherwise its area in
    the stitched label map silently shrinks."""
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:10, 0:7] = True  # 70 px, higher score
    masks[1, 0:10, 6:10] = True  # 40 px, overlaps with mask 0 in col 6
    # IoU = 10 / (70 + 40 - 10) = 0.1 — exactly at default gate, so
    # *neither* is suppressed (gate is strict >). Both survive, but the
    # higher-ranked mask must keep all of its 70 pixels.
    out = gz.segment.MaskNMS(iou_threshold=0.2, scores=np.array([0.9, 0.5]))(
        _mask_stack_gt(masks)
    )
    out_arr = np.asarray(out)
    assert (out_arr == 1).sum() == 70
    assert (out_arr == 2).sum() == 30  # mask 1 loses col 6 to mask 0


def test_mask_nms_rejects_non_3d_input() -> None:
    import pytest

    flat = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="N, H, W"):
        gz.segment.MaskNMS()(_gt(flat))


def test_slic_forbid_in_yaml_when_mask_provided() -> None:
    mask = np.ones((8, 8), dtype=bool)
    op_with_mask = gz.segment.SLIC(n_segments=4, mask=mask)
    op_without_mask = gz.segment.SLIC(n_segments=4)

    assert op_with_mask.forbid_in_yaml is True
    # Class-level default should remain falsy when no mask is provided.
    assert getattr(op_without_mask, "forbid_in_yaml", False) is False

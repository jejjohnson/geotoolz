"""Tests for segmentation operators."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor
from pipekit import Operator

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    # NaN fill: the toy images hold real zeros, which a ``0`` fill would
    # mark as nodata (the segmenters honour ``fill_value_default``).
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:4326",
        fill_value_default=np.nan,
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

    labels = gz.segment.Watershed()(image, markers)

    assert set(np.unique(np.asarray(labels))) >= {1, 2}


@pytest.mark.parametrize(("connectivity", "skimage_connectivity"), [(4, 1), (8, 2)])
def test_watershed_connectivity_is_4_or_8(
    connectivity: int, skimage_connectivity: int
) -> None:
    """``connectivity`` uses the package-wide 4 | 8 spelling, not skimage's 1 | 2."""
    from skimage.segmentation import watershed

    rng = np.random.default_rng(0)
    image = rng.uniform(size=(9, 9))
    markers = np.zeros((9, 9), dtype=np.int32)
    markers[0, 0], markers[-1, -1], markers[0, -1] = 1, 2, 3

    labels = gz.segment.Watershed(connectivity=connectivity)(image, markers)

    expected = watershed(image, markers=markers, connectivity=skimage_connectivity)
    np.testing.assert_array_equal(np.asarray(labels), expected)
    with pytest.raises(ValueError, match="4 or 8"):
        gz.segment.Watershed(connectivity=1)  # ty: ignore[invalid-argument-type]


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

    out = gz.segment.Felzenszwalb(scale=1.0, min_area_px=1)(gt_nan)

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


@pytest.mark.parametrize(
    "op",
    [
        gz.segment.Felzenszwalb(scale=1.0, min_area_px=1),
        gz.segment.Quickshift(kernel_size=2.0, max_dist=4.0),
    ],
    ids=["felzenszwalb", "quickshift"],
)
def test_felzenszwalb_quickshift_valid_labels_start_at_one(op: Operator) -> None:
    # skimage numbers these segments from 0; label 0 is reserved for
    # invalid pixels, so every finite pixel must carry a label >= 1.
    # NaN in the last pixel so skimage's segment 0 (which starts at the
    # top-left) is a real, finite segment.
    arr = np.asarray(_checker_gt()).copy()
    arr[:, -1, -1] = np.nan
    gt_nan = _gt(arr)

    labels = np.asarray(op(gt_nan))

    valid = np.isfinite(arr[0])
    assert labels[~valid].tolist() == [0]
    assert labels[valid].min() >= 1


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

    op = gz.segment.RandomWalker(beta=10.0, mode="cg_j")
    out = op(gt, markers)

    assert out.transform == gt.transform
    assert np.asarray(out).dtype == np.int32
    assert set(np.unique(np.asarray(out))) >= {1, 2}
    # The markers are a call-time carrier, so the operator round-trips (#141).
    assert gz.segment.RandomWalker.forbid_in_yaml is False
    assert Operator.from_state(op.state).get_config() == op.get_config()


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


def test_mark_boundaries_preserves_grid_and_fill() -> None:
    rgb = np.tile(np.linspace(0, 1, 6, dtype=float), (3, 6, 1))
    gt = _gt(rgb)
    label_img = np.zeros((6, 6), dtype=np.int32)
    label_img[:, :3] = 1
    label_img[:, 3:] = 2

    op = gz.segment.MarkBoundaries()
    out = op(gt, label_img)

    # The overlay keeps the input grid and its (NaN) fill via wrap_like.
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    np.testing.assert_equal(out.fill_value_default, gt.fill_value_default)
    assert np.asarray(out).shape[-2:] == gt.shape[-2:]
    # The label image is a call-time carrier, so the operator round-trips.
    assert gz.segment.MarkBoundaries.forbid_in_yaml is False


def test_mark_boundaries_single_band_input() -> None:
    values = np.tile(np.linspace(0, 1, 6, dtype=float), (1, 6, 1))
    gt = _gt(values)
    label_img = np.zeros((6, 6), dtype=np.int32)
    label_img[:, 3:] = 1

    out = gz.segment.MarkBoundaries(color=(1.0, 0.0, 0.0))(gt, label_img)

    out_arr = np.asarray(out)
    assert out_arr.shape == (3, 6, 6)
    assert out.transform == gt.transform
    # Boundary pixels carry the requested colour; interior pixels keep
    # the grey value replicated across the three channels.
    np.testing.assert_allclose(out_arr[:, 0, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out_arr[:, 0, 0], [0.0, 0.0, 0.0])
    # Plain (H, W) ndarray follows the same single-band path.
    out_2d = gz.segment.MarkBoundaries()(values[0], label_img)
    assert np.asarray(out_2d).shape == (3, 6, 6)


def test_mark_boundaries_rejects_non_rgb_multiband() -> None:
    label_img = np.zeros((6, 6), dtype=np.int32)
    with pytest.raises(ValueError, match=r"\(3, H, W\) RGB"):
        gz.segment.MarkBoundaries()(np.zeros((4, 6, 6)), label_img)


def test_slic_accepts_2d() -> None:
    values = np.zeros((8, 8), dtype=float)
    values[:, 4:] = 1.0

    labels = np.asarray(gz.segment.SLIC(n_segments=4, compactness=1.0)(values))

    assert labels.shape == (8, 8)
    assert labels.dtype == np.int32
    assert labels.min() >= 1
    assert labels[0, 0] != labels[0, -1]


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
        "classes": [[1, 0], [2, 1]],
        "start_label": 3,
    }
    assert gz.segment.MergeNearbyInstances(**cfg).classes == {1: 0, 2: 1}


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


def _step_cube() -> np.ndarray:
    values = np.zeros((1, 8, 8), dtype=float)
    values[:, :, 4:] = 1.0
    return values


def _merge_labels() -> np.ndarray:
    labels = np.zeros((8, 8), dtype=np.int32)
    _stamp(labels, 1, 1, 4, 1, 4)
    return labels


def _nms_stack() -> np.ndarray:
    masks = np.zeros((2, 6, 8), dtype=bool)
    masks[0, :, 0:3] = True
    masks[1, :, 5:8] = True
    return masks


@pytest.mark.parametrize(
    ("op", "values", "extra"),
    [
        pytest.param(
            gz.segment.SLIC(n_segments=4, compactness=1.0),
            _step_cube(),
            (),
            id="slic",
        ),
        pytest.param(
            gz.segment.Watershed(),
            np.array([[3.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0]]),
            (np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int32),),
            id="watershed",
        ),
        pytest.param(
            gz.segment.ExpandLabels(distance=1.0),
            np.eye(5, dtype=np.int32)[None],
            (),
            id="expand-labels",
        ),
        pytest.param(
            gz.segment.MergeNearbyInstances(start_label=3),
            _merge_labels(),
            (),
            id="merge-nearby-instances",
        ),
        pytest.param(
            gz.segment.MaskNMS(scores=np.array([0.9, 0.5])),
            _nms_stack(),
            (),
            id="mask-nms",
        ),
        pytest.param(
            gz.segment.MarkBoundaries(),
            np.tile(np.linspace(0.0, 1.0, 6), (3, 6, 1)),
            (np.eye(6, dtype=np.int32),),
            id="mark-boundaries",
        ),
    ],
)
def test_plain_ndarray_in_plain_ndarray_out(
    op: gz.Operator, values: np.ndarray, extra: tuple[np.ndarray, ...]
) -> None:
    """Segment ops are metadata-independent: ndarray in -> ndarray out,
    with values identical to the GeoTensor path."""
    out_plain = op(values, *extra)
    out_geo = op(_gt(values), *extra)

    assert type(out_plain) is np.ndarray
    np.testing.assert_array_equal(out_plain, np.asarray(out_geo))


@pytest.mark.parametrize(
    "cls", [gz.segment.SLIC, gz.segment.Felzenszwalb, gz.segment.Quickshift]
)
def test_segment_mask_is_a_call_argument(cls: type) -> None:
    """The optional mask is a positional carrier, not configuration (#141).

    The operator round-trips through its state, the mask zeroes the labels
    outside it, and a mask on another grid is rejected by name.
    """
    with pytest.raises(TypeError, match="mask"):
        cls(mask=np.ones((8, 8), dtype=bool))
    op = cls()
    assert "mask" not in op.get_config()
    assert Operator.from_state(json.loads(json.dumps(op.state))).get_config() == (
        op.get_config()
    )
    image = toy_geotensor(_step_cube())
    mask = np.ones((8, 8), dtype=bool)
    mask[:, :2] = False
    labels = np.asarray(op(image, mask))
    assert (labels[:, :2] == 0).all()
    assert (labels[:, 2:] >= 1).all()
    shifted = toy_geotensor(
        mask,
        transform=image.transform * rasterio.Affine.translation(1, 0),
        fill_value_default=None,
    )
    with pytest.raises(ValueError, match=cls.__name__):
        op(image, shifted)


def _fill_step_image() -> np.ndarray:
    values = np.full((1, 8, 8), 1.0)
    values[:, :, 4:] = 2.0
    return values


def _row_markers() -> np.ndarray:
    markers = np.zeros((8, 8), dtype=np.int32)
    markers[3, 1] = 1
    markers[3, 6] = 2
    return markers


@pytest.mark.parametrize(
    ("op", "extra"),
    [
        (gz.segment.SLIC(n_segments=4, compactness=1.0), ()),
        (gz.segment.Felzenszwalb(scale=1.0, min_area_px=1), ()),
        (gz.segment.Quickshift(kernel_size=2.0, max_dist=4.0), ()),
        (gz.segment.Watershed(), (_row_markers(),)),
        (gz.segment.ChanVese(max_num_iter=10), ()),
        (gz.segment.RandomWalker(beta=10.0), (_row_markers(),)),
    ],
    ids=["slic", "felzenszwalb", "quickshift", "watershed", "chanvese", "rw"],
)
def test_fill_pixels_are_excluded(op: Operator, extra: tuple[np.ndarray, ...]) -> None:
    """#145: a ``fill_value_default`` pixel is nodata, not a region.

    It gets the "no segment" label 0, and the labels match those obtained
    when the same pixels are NaN (the nodata path that already worked).
    """
    fill = fill_pixel_mask((8, 8))
    gt = toy_geotensor(_fill_step_image(), with_fill_pixels=True)
    as_nan = _fill_step_image()
    as_nan[:, fill] = np.nan

    labels = op(gt, *extra)

    out = np.asarray(labels)
    assert out.dtype == np.int32
    assert labels.fill_value_default == 0
    assert (out[fill] == 0).all()
    np.testing.assert_array_equal(out, np.asarray(op(as_nan, *extra)))
    if not isinstance(op, gz.segment.ChanVese):  # ChanVese: 0 = "outside"
        assert (out[~fill] >= 1).all()


def test_4d_time_stack() -> None:
    """Multi-band segmenters label each frame of a stack (#147)."""
    from _helpers import frames, time_stack

    stack = time_stack((2, 3, 16, 16))
    op = gz.segment.SLIC(n_segments=4, compactness=10.0)
    out = op(stack)
    assert out.shape == (2, 1, 16, 16)
    for t, frame in enumerate(frames(stack)):
        np.testing.assert_array_equal(np.asarray(out)[t, 0], np.asarray(op(frame)))


# ---------------------------------------------------------------------------
# Threshold (moved from plume in #162)
# ---------------------------------------------------------------------------


def _bimodal() -> np.ndarray:
    values = np.zeros((1, 8, 8), dtype=float)
    values[:, :, 4:] = 10.0
    return values


def test_threshold_otsu_splits_bimodal_map_and_keeps_carrier() -> None:
    gt = _gt(_bimodal())
    mask = gz.segment.Threshold()(gt)
    assert mask.shape == gt.shape
    assert mask.transform == gt.transform
    assert np.asarray(mask).dtype == bool
    np.testing.assert_array_equal(np.asarray(mask), _bimodal() > 5.0)
    assert mask.fill_value_default is False


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(4.0, 32), ("percentile:0", 32), ("percentile:100", 0)],
)
def test_threshold_absolute_and_percentile_modes(
    threshold: float | str, expected: int
) -> None:
    mask = gz.segment.Threshold(threshold=threshold)(_bimodal())
    assert int(np.asarray(mask).sum()) == expected


def test_threshold_nodata_is_false_and_excluded_from_statistic() -> None:
    values = _bimodal()
    gt = toy_geotensor(values, with_fill_pixels=True)
    invalid = fill_pixel_mask(gt.shape)
    mask = np.asarray(gz.segment.Threshold()(gt))
    assert not mask[..., invalid].any()
    np.testing.assert_array_equal(mask[..., ~invalid], (values > 5.0)[..., ~invalid])


def test_threshold_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="otsu"):
        gz.segment.Threshold(threshold="triangle")


def test_threshold_round_trips_and_is_top_level() -> None:
    op = gz.Threshold(threshold="percentile:95", nbins=64)
    assert gz.Threshold is gz.segment.Threshold
    rebuilt = Operator.from_state(json.loads(json.dumps(op.state)))
    assert rebuilt.get_config() == op.get_config()


def test_plume_mask_is_threshold_plus_area_filter() -> None:
    """``PlumeMask`` shares the segment thresholding primitives."""
    # One home per public name: plume uses, but does not re-export, them.
    assert not hasattr(gz.plume, "otsu_threshold")
    assert not hasattr(gz.plume, "resolve_threshold")
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1, 16, 16))
    for threshold in ("otsu", "percentile:90", 0.5):
        plume = gz.plume.PlumeMask(threshold=threshold, min_area_px=0)(values)
        seg = gz.segment.Threshold(threshold=threshold)(values)
        np.testing.assert_array_equal(np.asarray(plume)[None], np.asarray(seg))

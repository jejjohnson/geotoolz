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


def test_slic_forbid_in_yaml_when_mask_provided() -> None:
    mask = np.ones((8, 8), dtype=bool)
    op_with_mask = gz.segment.SLIC(n_segments=4, mask=mask)
    op_without_mask = gz.segment.SLIC(n_segments=4)

    assert op_with_mask.forbid_in_yaml is True
    # Class-level default should remain falsy when no mask is provided.
    assert getattr(op_without_mask, "forbid_in_yaml", False) is False

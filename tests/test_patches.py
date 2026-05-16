"""Tests for `geotoolz.patches`."""

from __future__ import annotations

import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray | None = None) -> GeoTensor:
    if values is None:
        values = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    return GeoTensor(
        values,
        transform=Affine(1, 0, 10, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )


def test_patches_module_is_public() -> None:
    assert gz.patches.ExtractPatches is gz.patches.ExtractPatches


def test_extract_then_stitch_identity_without_overlap() -> None:
    gt = _gt()
    patches = gz.patches.ExtractPatches(
        size=(2, 2),
        stride=(2, 2),
        drop_incomplete=True,
    )(gt)

    stitched = gz.patches.StitchPatches(
        target_shape=gt.shape[-2:],
        target_transform=gt.transform,
        target_crs=str(gt.crs),
    )(patches)

    assert stitched.shape == gt.shape
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))
    assert stitched.transform == gt.transform
    assert str(stitched.crs) == str(gt.crs)


def test_extract_nan_cutoff_keeps_expected_patches() -> None:
    values = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    values[..., :2, :2] = np.nan
    gt = _gt(values)

    patches = gz.patches.ExtractPatches(
        size=(2, 2),
        stride=(2, 2),
        nan_cutoff=0.5,
        drop_incomplete=True,
    )(gt)

    assert len(patches) == 3
    assert all(np.isnan(np.asarray(patch)).mean() <= 0.5 for patch in patches)


def test_sliding_window_is_lazy_iterator() -> None:
    gt = _gt()

    patches = gz.patches.SlidingWindow(size=(2, 2), stride=(1, 1))(gt)

    assert iter(patches) is patches
    assert next(patches).shape == (1, 2, 2)


def test_random_crop_seed_is_reproducible() -> None:
    gt = _gt()
    op = gz.patches.RandomCrop(size=(2, 2), n_samples=3, seed=0)

    first = op(gt)
    second = op(gt)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        assert left.transform == right.transform


def test_balanced_sampler_returns_n_per_class() -> None:
    gt = _gt()
    labels = GeoTensor(
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.uint8,
        ),
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=255,
    )

    patches = gz.patches.BalancedSampler(
        labels=labels, n_per_class=1, size=(2, 2), seed=0
    )(gt)

    assert len(patches) == 4
    assert sorted(patch.attrs["class_label"] for patch in patches) == [0, 1, 2, 3]
    label_arr = np.asarray(labels)
    for patch in patches:
        row = round((patch.transform.f - gt.transform.f) / gt.transform.e)
        col = round((patch.transform.c - gt.transform.c) / gt.transform.a)
        center = label_arr[row + patch.height // 2, col + patch.width // 2]
        assert int(center) == patch.attrs["class_label"]


def test_sample_points_nearest_matches_hand_indexed_reference() -> None:
    gt = _gt()
    points = np.array([[10.5, 19.5], [12.5, 17.5]])

    samples = gz.patches.SamplePoints(points=points, crs="EPSG:4326")(gt)

    expected = np.array([[0.0], [10.0]], dtype=np.float32)
    np.testing.assert_array_equal(samples, expected)


def test_sample_along_track_spacing_has_monotonic_distance() -> None:
    gt = _gt()
    track = np.array([[10.5, 19.5], [13.5, 19.5], [13.5, 16.5]])

    result = gz.patches.SampleAlongTrack(
        track=track,
        crs="EPSG:4326",
        spacing=1.0,
    )(gt)

    assert np.all(np.diff(result["distance"]) > 0)
    assert result["samples"].shape[0] == result["distance"].shape[0]


def test_stitch_feather_preserves_gradient_shape() -> None:
    gt = _gt(np.arange(1 * 6 * 6, dtype=np.float32).reshape(1, 6, 6))
    patches = gz.patches.ExtractPatches(
        size=(3, 3), stride=(2, 2), drop_incomplete=False
    )(gt)

    stitched = gz.patches.StitchPatches(
        target_shape=gt.shape[-2:],
        target_transform=gt.transform,
        target_crs=str(gt.crs),
        blend="feather",
        feather_width=1,
    )(patches)

    assert stitched.shape[-2:] == gt.shape[-2:]
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))

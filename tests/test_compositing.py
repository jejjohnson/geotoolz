"""Tests for explicit compositing operators."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz.compositing import (
    BAPComposite,
    CloudFreeComposite,
    MaxNDVIComposite,
    MedianComposite,
    MinCloudComposite,
)


def _gt(values: np.ndarray, *, transform: rasterio.Affine | None = None) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=transform
        or rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=np.nan,
    )


def test_compositing_namespace_is_available() -> None:
    assert gz.compositing.BAPComposite is BAPComposite
    assert gz.compositing.CloudFreeComposite is CloudFreeComposite
    assert gz.compositing.MaxNDVIComposite is MaxNDVIComposite
    assert gz.compositing.MedianComposite is MedianComposite
    assert gz.compositing.MinCloudComposite is MinCloudComposite
    assert gz.BAPComposite is BAPComposite
    assert gz.CloudFreeComposite is CloudFreeComposite
    assert gz.MaxNDVIComposite is MaxNDVIComposite
    assert gz.MedianComposite is MedianComposite
    assert gz.MinCloudComposite is MinCloudComposite


def test_median_composite_ignores_nan_and_returns_count() -> None:
    scenes = [
        _gt(np.array([[[1.0, np.nan], [3.0, 4.0]]], dtype=np.float32)),
        _gt(np.array([[[3.0, 2.0], [np.nan, 8.0]]], dtype=np.float32)),
        _gt(np.array([[[5.0, 4.0], [7.0, np.nan]]], dtype=np.float32)),
    ]

    median, count = MedianComposite(return_count=True)(scenes)

    np.testing.assert_allclose(np.asarray(median), [[[3.0, 3.0], [5.0, 6.0]]])
    np.testing.assert_array_equal(np.asarray(count), [[[3, 2], [2, 2]]])
    assert np.asarray(count).dtype == np.int64
    assert median.transform == scenes[0].transform
    assert str(median.crs) == "EPSG:32629"


def test_max_ndvi_composite_returns_frame_with_peak_ndvi() -> None:
    low = _gt(
        np.array(
            [
                [[0.2, 0.2], [0.2, 0.2]],
                [[0.4, 0.4], [0.4, 0.4]],
            ],
            dtype=np.float32,
        )
    )
    high = _gt(
        np.array(
            [
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.9, 0.9], [0.9, 0.9]],
            ],
            dtype=np.float32,
        )
    )
    low.attrs["descriptions"] = ("red", "nir")
    high.attrs["descriptions"] = ("red", "nir")

    out, index = MaxNDVIComposite(red="red", nir="nir", return_index=True)([low, high])

    np.testing.assert_allclose(np.asarray(out), np.asarray(high))
    np.testing.assert_array_equal(np.asarray(index), np.ones((2, 2), dtype=np.int64))


def test_max_ndvi_composite_outputs_nan_when_all_ndvi_is_invalid() -> None:
    scene1 = _gt(np.full((2, 2, 2), np.nan, dtype=np.float32))
    scene2 = _gt(np.full((2, 2, 2), np.nan, dtype=np.float32))

    out = MaxNDVIComposite(red=0, nir=1)([scene1, scene2])

    assert np.isnan(np.asarray(out)).all()


def test_max_ndvi_composite_rejects_two_dimensional_geotensors() -> None:
    flat = _gt(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="requires multi-band"):
        MaxNDVIComposite(red=0, nir=1)([flat, flat])


def test_cloud_free_composite_respects_masks_and_min_valid() -> None:
    scene1 = _gt(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    scene2 = _gt(np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32))
    mask1 = np.array([[False, True], [False, False]])
    mask2 = np.array([[False, False], [True, True]])

    out, count = CloudFreeComposite(min_valid=2, return_count=True)(
        [(scene1, mask1), (scene2, mask2)]
    )

    expected = np.array([[[5.5, np.nan], [np.nan, np.nan]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(out), expected, equal_nan=True)
    np.testing.assert_array_equal(np.asarray(count), [[[2, 1], [1, 1]]])


def test_bap_composite_uses_weighted_scores() -> None:
    scene1 = _gt(np.full((1, 2, 2), 1.0, dtype=np.float32))
    scene2 = _gt(np.full((1, 2, 2), 2.0, dtype=np.float32))
    scene3 = _gt(np.full((1, 2, 2), 3.0, dtype=np.float32))
    metadata = [
        {
            "view_angle_score": 0.2,
            "recency_score": 0.1,
            "cloud_distance_score": 0.1,
            "opacity_score": 0.1,
        },
        {
            "view_angle_score": 0.1,
            "recency_score": 0.9,
            "cloud_distance_score": 0.1,
            "opacity_score": 0.1,
        },
        {
            "view_angle_score": 0.1,
            "recency_score": 0.1,
            "cloud_distance_score": 0.1,
            "opacity_score": 0.1,
        },
    ]

    out, score = BAPComposite(target_doy=196, return_score=True)(
        list(zip([scene1, scene2, scene3], metadata, strict=True))
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(scene2))
    np.testing.assert_allclose(
        np.asarray(score), np.full((2, 2), 0.42, dtype=np.float32)
    )


def test_min_cloud_composite_prefers_clear_pixels_from_least_cloudy_frame() -> None:
    scene1 = _gt(np.full((1, 2, 2), 1.0, dtype=np.float32))
    scene2 = _gt(np.full((1, 2, 2), 2.0, dtype=np.float32))
    mask1 = np.array([[False, True], [False, True]])
    mask2 = np.array([[False, False], [True, False]])

    out, count = MinCloudComposite(return_count=True)(
        [(scene1, mask1), (scene2, mask2)]
    )

    np.testing.assert_allclose(np.asarray(out), [[[2.0, 2.0], [1.0, 2.0]]])
    np.testing.assert_array_equal(np.asarray(count), [[[2, 1], [1, 1]]])


@pytest.mark.parametrize(
    ("operator", "payload"),
    [
        (MedianComposite(), "frames"),
        (MaxNDVIComposite(red=0, nir=1), "frames"),
        (CloudFreeComposite(), "masks"),
        (BAPComposite(target_doy=196), "metadata"),
        (MinCloudComposite(), "masks"),
    ],
)
def test_composites_raise_on_mismatched_grid(
    operator: MedianComposite
    | MaxNDVIComposite
    | CloudFreeComposite
    | BAPComposite
    | MinCloudComposite,
    payload: str,
) -> None:
    base = _gt(np.ones((1, 2, 2), dtype=np.float32))
    shifted = _gt(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
    )
    inputs = {
        "frames": [base, shifted],
        "masks": [
            (base, np.zeros((2, 2), dtype=bool)),
            (shifted, np.zeros((2, 2), dtype=bool)),
        ],
        "metadata": [(base, {}), (shifted, {})],
    }[payload]

    with pytest.raises(ValueError, match="shape, transform, and CRS"):
        operator(inputs)  # type: ignore[arg-type]


def test_max_ndvi_composite_rejects_identical_red_and_nir_bands() -> None:
    scene = _gt(np.ones((2, 2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="distinct red and NIR bands"):
        MaxNDVIComposite(red=0, nir=0)([scene, scene])


def test_cloud_free_composite_accepts_one_channel_mask_for_two_d_input() -> None:
    scene1 = _gt(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    scene2 = _gt(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
    mask1 = np.zeros((1, 2, 2), dtype=bool)
    mask2 = np.array([[[False, False], [True, True]]])

    out = CloudFreeComposite()([(scene1, mask1), (scene2, mask2)])

    np.testing.assert_allclose(
        np.asarray(out), [[5.5, 11.0], [3.0, 4.0]]
    )


def test_bap_composite_rejects_mixed_cloud_distance_inputs() -> None:
    scene1 = _gt(np.full((1, 2, 2), 1.0, dtype=np.float32))
    scene2 = _gt(np.full((1, 2, 2), 2.0, dtype=np.float32))
    pairs = [
        (scene1, {"cloud_distance": 50.0}),
        (scene2, {"cloud_distance_score": 0.5}),
    ]

    with pytest.raises(ValueError, match="mix of raw 'cloud_distance'"):
        BAPComposite(target_doy=196)(pairs)


def test_compositing_get_config_is_json_safe() -> None:
    ops = [
        MedianComposite(return_count=True),
        MaxNDVIComposite(red="red", nir="nir", return_index=True),
        CloudFreeComposite(min_valid=2, return_count=True),
        BAPComposite(target_doy=196, return_score=True),
        MinCloudComposite(return_count=True),
    ]
    for op in ops:
        json.dumps(op.get_config())

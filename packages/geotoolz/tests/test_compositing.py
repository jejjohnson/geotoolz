"""Tests for explicit compositing operators."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import rasterio
from _helpers import toy_geotensor
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
    return toy_geotensor(values, transform=transform, fill_value_default=np.nan)


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

    np.testing.assert_allclose(np.asarray(out), [[5.5, 11.0], [3.0, 4.0]])


def test_bap_composite_rejects_mixed_cloud_distance_inputs() -> None:
    scene1 = _gt(np.full((1, 2, 2), 1.0, dtype=np.float32))
    scene2 = _gt(np.full((1, 2, 2), 2.0, dtype=np.float32))
    pairs = [
        (scene1, {"cloud_distance": 50.0}),
        (scene2, {"cloud_distance_score": 0.5}),
    ]

    with pytest.raises(ValueError, match="mix of raw 'cloud_distance'"):
        BAPComposite(target_doy=196)(pairs)


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
def test_composites_accept_plain_ndarray_frames(
    operator: MedianComposite
    | MaxNDVIComposite
    | CloudFreeComposite
    | BAPComposite
    | MinCloudComposite,
    payload: str,
) -> None:
    frame1 = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [4.0, 3.0]]], dtype=np.float32
    )
    frame2 = np.array(
        [[[5.0, 6.0], [7.0, 8.0]], [[6.0, 5.0], [8.0, 7.0]]], dtype=np.float32
    )
    mask1 = np.array([[False, True], [False, False]])
    mask2 = np.array([[False, False], [True, False]])
    meta1 = {"view_angle": 10.0, "doy": 100}
    meta2 = {"view_angle": 2.0, "doy": 190}

    def inputs(a: object, b: object) -> list:
        return {
            "frames": [a, b],
            "masks": [(a, mask1), (b, mask2)],
            "metadata": [(a, meta1), (b, meta2)],
        }[payload]

    out_plain = operator(inputs(frame1, frame2))
    out_geo = operator(inputs(_gt(frame1), _gt(frame2)))

    assert type(out_plain) is np.ndarray
    np.testing.assert_allclose(out_plain, np.asarray(out_geo), equal_nan=True)


def test_max_ndvi_composite_named_bands_require_geotensor_attrs() -> None:
    frame = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(TypeError, match="named band references"):
        MaxNDVIComposite(red="red", nir="nir")([frame, frame])


def test_composites_raise_on_mismatched_plain_array_shapes() -> None:
    with pytest.raises(ValueError, match="shape, transform, and CRS"):
        MedianComposite()(
            [
                np.ones((1, 2, 2), dtype=np.float32),
                np.ones((1, 4, 4), dtype=np.float32),
            ]
        )


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


# ---------------------------------------------------------------------------
# #145: nodata (fill_value_default) frame-pixels are excluded
# ---------------------------------------------------------------------------

_FILL = -9999.0


def _fill_frames() -> list[GeoTensor]:
    """Three ``(2, 1, 3)`` frames with a ``-9999`` fill.

    Pixel 0 is valid everywhere, pixel 1 is nodata in frame 0 only, and
    pixel 2 is nodata in every frame. Band 0 reproduces the #145 median
    example ``[1, -9999], [3, 5], [5, 7] -> [3, 6]``; read as (red, NIR)
    the valid frames at pixel 1 have *negative* NDVI, which the fill
    frame's NDVI of 0 used to beat.
    """
    band0 = [[1.0, _FILL, _FILL], [3.0, 5.0, _FILL], [5.0, 7.0, _FILL]]
    band1 = [[2.0, _FILL, _FILL], [1.0, 1.0, _FILL], [2.0, 2.0, _FILL]]
    return [
        toy_geotensor(
            np.array([[b0], [b1]], dtype=np.float32), fill_value_default=_FILL
        )
        for b0, b1 in zip(band0, band1, strict=True)
    ]


def _assert_composite(out: Any, pixel0: list[float], pixel1: list[float]) -> None:
    """Valid pixels match the fill-free reduction; all-nodata pixel is fill."""
    arr = np.asarray(out)
    np.testing.assert_allclose(arr[:, 0, :2], np.array([pixel0, pixel1]).T)
    assert (arr[:, 0, 2] == _FILL).all()
    assert out.fill_value_default == _FILL


def _case_median() -> None:
    out, count = MedianComposite(return_count=True)(_fill_frames())
    _assert_composite(out, [3.0, 2.0], [6.0, 1.5])
    np.testing.assert_array_equal(np.asarray(count)[:, 0], [[3, 2, 0], [3, 2, 0]])
    # A contributor count declares 0 (no frame), not the frames' fill (#146).
    assert count.fill_value_default == 0


def _case_max_ndvi() -> None:
    out, index = MaxNDVIComposite(red=0, nir=1, return_index=True)(_fill_frames())
    # Pixel 1: frame 2 (NDVI -0.56) beats frame 1 (-0.67); the fill
    # frame's NDVI of 0 must not win.
    assert np.asarray(index)[0, 1] == 2
    # No frame is valid at pixel 2; frame 0 is a real index, so -1 marks it.
    assert np.asarray(index)[0, 2] == -1
    assert index.fill_value_default == -1
    _assert_composite(out, [1.0, 2.0], [7.0, 2.0])


def _case_cloud_free() -> None:
    clear = np.zeros((1, 3), dtype=bool)
    out, count = CloudFreeComposite(return_count=True)(
        [(frame, clear) for frame in _fill_frames()]
    )
    _assert_composite(out, [3.0, 5.0 / 3.0], [6.0, 1.5])
    np.testing.assert_array_equal(np.asarray(count)[:, 0], [[3, 2, 0], [3, 2, 0]])


def _case_bap() -> None:
    frames = _fill_frames()
    # A NaN band also makes a frame-pixel nodata: frame 0 has the best
    # score, but its pixel 0 is unusable, so frame 1 must win there.
    values0 = np.asarray(frames[0]).copy()
    values0[1, 0, 0] = np.nan
    frames[0] = toy_geotensor(values0, fill_value_default=_FILL)
    metadata = [{"view_angle": 0.0}, {"view_angle": 1.0}, {"view_angle": 2.0}]
    out, score = BAPComposite(target_doy=196, return_score=True)(
        list(zip(frames, metadata, strict=True))
    )
    _assert_composite(out, [3.0, 1.0], [5.0, 1.0])
    # A score is a new quantity: NaN nodata, never the frames' fill (#146).
    assert np.isnan(score.fill_value_default)
    assert np.isnan(np.asarray(score)[0, 2])


def _case_min_cloud() -> None:
    clear = np.zeros((1, 3), dtype=bool)
    cloudy = np.array([[True, False, False]])
    out = MinCloudComposite()(
        list(zip(_fill_frames(), [clear, cloudy, clear], strict=True))
    )
    # Pixel 1: frame 0 is nodata -> least-cloudy valid clear frame (2).
    _assert_composite(out, [1.0, 2.0], [7.0, 2.0])


def _case_blend_matched() -> None:
    out = gz.compositing.BlendMatched()(_fill_frames())
    _assert_composite(out, [3.0, 5.0 / 3.0], [6.0, 1.5])


def _case_stack_matched() -> None:
    other = toy_geotensor(
        np.array([[[4.0, 0.0, 6.0]]], dtype=np.float32), fill_value_default=0.0
    )
    out = gz.compositing.StackMatched()([_fill_frames()[0], other])
    # The second input's nodata (its own fill, 0) is rewritten to the
    # output's fill so it stays marked as nodata.
    np.testing.assert_array_equal(np.asarray(out)[2, 0], [4.0, _FILL, 6.0])
    assert out.fill_value_default == _FILL


@pytest.mark.parametrize(
    "case",
    [
        _case_median,
        _case_max_ndvi,
        _case_cloud_free,
        _case_bap,
        _case_min_cloud,
        _case_blend_matched,
        _case_stack_matched,
    ],
    ids=lambda case: case.__name__.removeprefix("_case_"),
)
def test_fill_pixels_are_excluded(case: Callable[[], None]) -> None:
    """#145: a frame's nodata never enters a composite; all-nodata -> fill."""
    case()

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

    with pytest.raises(ValueError, match=r"Composite: the frame 1 pixel grid"):
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


def _single_score(**kwargs: Any) -> Callable[[dict[str, Any]], float]:
    """Winning score of a one-frame BAP with only the given weight non-zero."""
    weights = {
        "w_view_angle": 0.0,
        "w_recency": 0.0,
        "w_cloud_distance": 0.0,
        "w_opacity": 0.0,
    }
    weights.update(kwargs.pop("weights"))
    op = BAPComposite(return_score=True, **weights, **kwargs)
    scene = _gt(np.ones((1, 1, 1), dtype=np.float32))

    def score(metadata: dict[str, Any]) -> float:
        _, best = op([(scene, metadata)])
        return float(np.asarray(best)[0, 0])

    return score


def test_bap_recency_wraps_year() -> None:
    # DOY 360 vs target 5 is 10 days apart (not 355): exp(-½ (10/30)²).
    score = _single_score(target_doy=5, weights={"w_recency": 1.0})
    np.testing.assert_allclose(
        score({"doy": 360}), np.exp(-0.5 * (10 / 30) ** 2), rtol=1e-6
    )
    np.testing.assert_allclose(score({"doy": 360}), score({"doy": 15}), rtol=1e-6)
    # The farthest point on the circle is half a year away.
    np.testing.assert_allclose(
        score({"doy": 187.5}), np.exp(-0.5 * (182.5 / 30) ** 2), rtol=1e-5
    )


@pytest.mark.parametrize("doy", [1, 30, 100, 196, 250, 300, 365])
@pytest.mark.parametrize("sigma", [10.0, 45.0])
def test_bap_doy_score_matches_gaussian(doy: int, sigma: float) -> None:
    target = 20
    delta = abs(doy - target) % 365
    d = min(delta, 365 - delta)
    score = _single_score(
        target_doy=target, doy_sigma=sigma, weights={"w_recency": 1.0}
    )
    np.testing.assert_allclose(
        score({"day_of_year": doy}),
        np.exp(-0.5 * (d / sigma) ** 2),
        rtol=1e-6,
        atol=1e-12,  # far tails underflow float32
    )


@pytest.mark.parametrize("theta", [-12.0, 0.0, 7.5, 30.0])
def test_bap_view_angle_score_is_gaussian_in_degrees(theta: float) -> None:
    score = _single_score(target_doy=1, weights={"w_view_angle": 1.0})
    np.testing.assert_allclose(
        score({"view_angle": theta}), np.exp(-0.5 * (theta / 15.0) ** 2), rtol=1e-6
    )


@pytest.mark.parametrize("distance", [0.0, 10.0, 28.0, 40.0, 60.0, 500.0])
def test_bap_cloud_distance_score_is_griffiths_sigmoid(distance: float) -> None:
    d_req, d_min, k = 60.0, 4.0, 0.15
    score = _single_score(
        target_doy=1,
        cloud_distance_req=d_req,
        cloud_distance_min=d_min,
        cloud_distance_slope=k,
        weights={"w_cloud_distance": 1.0},
    )
    expected = 1.0 / (1.0 + np.exp(-k * (min(distance, d_req) - (d_req - d_min) / 2)))
    np.testing.assert_allclose(score({"cloud_distance": distance}), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("opacity", "expected"),
    [(0.0, 1.0), (0.2, 1.0), (0.225, 0.75), (0.25, 0.5), (0.3, 0.0), (0.9, 0.0)],
)
def test_bap_opacity_score_is_linear_ramp(opacity: float, expected: float) -> None:
    score = _single_score(target_doy=1, weights={"w_opacity": 1.0})
    np.testing.assert_allclose(score({"opacity": opacity}), expected, atol=1e-6)


def test_bap_late_december_beats_spring_for_january_target() -> None:
    december = _gt(np.full((1, 2, 2), 1.0, dtype=np.float32))
    spring = _gt(np.full((1, 2, 2), 2.0, dtype=np.float32))
    pairs = [(spring, {"doy": 60}), (december, {"doy": 358})]

    out, score = BAPComposite(target_doy=10, return_score=True)(pairs)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(december))
    # Default weights; view/opacity at their best (1), cloud_distance 0.
    s_cloud = 1.0 / (1.0 + np.exp(0.2 * 25.0))
    expected = 0.3 + 0.4 * np.exp(-0.5 * (17 / 30) ** 2) + 0.2 * s_cloud + 0.1
    np.testing.assert_allclose(np.asarray(score), expected, rtol=1e-6)


def test_bap_composite_accepts_mixed_cloud_distance_inputs() -> None:
    # Raw distances map through the sigmoid onto [0, 1], so they share the
    # precomputed scores' scale and may be mixed across frames.
    scene1 = _gt(np.full((1, 2, 2), 1.0, dtype=np.float32))
    scene2 = _gt(np.full((1, 2, 2), 2.0, dtype=np.float32))
    pairs = [
        (scene1, {"cloud_distance": 50.0}),
        (scene2, {"cloud_distance_score": 0.5}),
    ]
    op = BAPComposite(
        target_doy=196,
        w_view_angle=0.0,
        w_recency=0.0,
        w_opacity=0.0,
        w_cloud_distance=1.0,
        return_score=True,
    )

    out, score = op(pairs)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(scene1))
    np.testing.assert_allclose(np.asarray(score), 1.0 / (1.0 + np.exp(-5.0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"doy_sigma": 0.0},
        {"view_angle_sigma": -1.0},
        {"cloud_distance_slope": 0.0},
        {"cloud_distance_req": 5.0, "cloud_distance_min": 5.0},
        {"opacity_low": 0.3, "opacity_high": 0.3},
    ],
)
def test_bap_composite_rejects_degenerate_score_parameters(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="must"):
        BAPComposite(target_doy=1, **kwargs)


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
    with pytest.raises(ValueError, match=r"Composite: the frame 1 pixel grid"):
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


def test_4d_time_stack() -> None:
    """Composites reduce a (T, C, H, W) stack over time (#147)."""
    from _helpers import frames, time_stack

    stack = time_stack(with_fill_pixels=True)
    for op in (MedianComposite(), MaxNDVIComposite(red=0, nir=1)):
        out = op(stack)
        expected = op(frames(stack))
        assert isinstance(out, GeoTensor)
        assert out.shape == (3, 4, 4)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
        assert out.transform == stack.transform
        assert out.fill_value_default == stack.fill_value_default
        assert out.attrs == stack.attrs and out.attrs is not stack.attrs
    with pytest.raises(ValueError, match="MedianComposite takes co-registered frames"):
        MedianComposite()(stack.isel({"time": 0}))


# ---------------------------------------------------------------------------
# N-ary reducers (#141)
# ---------------------------------------------------------------------------


def test_composites_take_frames_positionally() -> None:
    """``op(f1, f2)`` equals ``op([f1, f2])`` for every composite (#141)."""
    from geotoolz.compositing import BlendMatched, StackMatched

    rng = np.random.default_rng(0)
    frames = [
        _gt(rng.uniform(0.1, 1.0, (2, 3, 3)).astype(np.float32)) for _ in range(3)
    ]
    masks = [np.zeros((3, 3), dtype=bool) for _ in frames]
    masks[0][0, 0] = True
    cases: list[tuple[Any, list[Any]]] = [
        (MedianComposite(), frames),
        (MaxNDVIComposite(red=0, nir=1), frames),
        (CloudFreeComposite(), list(zip(frames, masks, strict=True))),
        (MinCloudComposite(), list(zip(frames, masks, strict=True))),
        (
            BAPComposite(target_doy=180),
            [(frame, {"doy": 170 + t}) for t, frame in enumerate(frames)],
        ),
        (StackMatched(), frames),
        (BlendMatched(), frames),
    ]
    for op, inputs in cases:
        np.testing.assert_array_equal(
            np.asarray(op(*inputs)), np.asarray(op(inputs)), err_msg=repr(op)
        )


def test_pair_composites_read_a_lone_pair_as_one_frame() -> None:
    """A single ``(frame, mask)`` argument is one pair, not two inputs."""
    frame = _gt(np.ones((1, 2, 2), dtype=np.float32))
    mask = np.zeros((2, 2), dtype=bool)
    np.testing.assert_array_equal(
        np.asarray(CloudFreeComposite()((frame, mask))), np.asarray(frame)
    )


def test_blend_matched_ivw_takes_tensor_variance_pairs() -> None:
    """``BlendMatched(method="ivw")`` pairs each tensor with its variance (#141)."""
    from geotoolz.compositing import BlendMatched

    a = _gt(np.full((2, 2), 10.0, dtype=np.float32))
    b = _gt(np.full((2, 2), 20.0, dtype=np.float32))
    var_a = np.full((2, 2), 1.0)
    var_b = np.full((2, 2), 3.0)
    spread = BlendMatched(method="ivw")((a, var_a), (b, var_b))
    mapping = BlendMatched(method="ivw")({"a": (a, var_a), "b": (b, var_b)})
    # Inverse-variance weights 1 and 1/3: (10 + 20 / 3) / (1 + 1 / 3) = 12.5.
    np.testing.assert_allclose(np.asarray(spread), 12.5)
    np.testing.assert_allclose(np.asarray(mapping), 12.5)

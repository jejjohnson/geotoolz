"""Tests for the shared pixel-flattening helpers (``geotoolz._src.samples``).

``cube_to_samples`` / ``samples_to_cube`` are the one implementation of
the ``(c, h, w) <-> (n, c)`` layout change used by ``learn``,
``matched_filter`` and ``restore``.
"""

from __future__ import annotations

import einx
import numpy as np
import pytest
from _helpers import toy_geotensor
from sklearn.cluster import KMeans

import geotoolz as gz
from geotoolz._src.samples import (
    SampleLayout,
    cube_to_samples,
    sample_layout,
    samples_to_cube,
)
from geotoolz.matched_filter._src.array import _cube_samples
from geotoolz.restore._src.array import fit_pca, inverse_pca


def _cube(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def _legacy(cube: np.ndarray, axis: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """The pre-consolidation ``moveaxis`` + ``reshape`` flattening."""
    moved = np.moveaxis(cube, axis, -1)
    return moved.reshape(-1, moved.shape[-1]), moved.shape[:-1]


def test_canonical_cube_is_pixel_major_c_h_w_to_hw_c() -> None:
    cube = _cube(4, 5, 6)
    samples, layout = cube_to_samples(cube)

    assert samples.shape == (30, 4)
    np.testing.assert_array_equal(samples, einx.id("c h w -> (h w) c", cube))
    np.testing.assert_array_equal(samples[7], cube[:, 1, 1])
    assert layout == SampleLayout(shape=(4, 5, 6), sample_axes=(1, 2), band_axes=(0,))
    assert layout.sample_shape == (5, 6)
    assert (layout.n_samples, layout.n_bands) == (30, 4)


@pytest.mark.parametrize(
    ("shape", "band_axis"),
    [((4, 5, 6), -3), ((4, 5, 6), 0), ((5, 6, 4), -1), ((3, 4, 5, 6), -3)],
)
def test_matches_legacy_flattening_and_round_trips(
    shape: tuple[int, ...], band_axis: int
) -> None:
    cube = _cube(*shape)
    samples, layout = cube_to_samples(cube, band_axis=band_axis)
    expected, spatial = _legacy(cube, band_axis)

    np.testing.assert_array_equal(samples, expected)
    assert layout.sample_shape == spatial
    np.testing.assert_array_equal(samples_to_cube(samples, layout), cube)


def test_4d_time_stack_flattens_time_into_rows() -> None:
    stack = _cube(3, 4, 5, 6)  # (t, c, h, w)
    samples, layout = cube_to_samples(stack)

    assert samples.shape == (3 * 5 * 6, 4)
    assert layout.sample_axes == (0, 2, 3)
    np.testing.assert_array_equal(samples[5 * 6 + 1], stack[1, :, 0, 1])
    np.testing.assert_array_equal(samples_to_cube(samples, layout), stack)
    # Per-row scalars map back onto the (t, h, w) sample grid.
    assert samples_to_cube(samples[:, 0], layout).shape == (3, 5, 6)


def test_nodata_rows_stay_in_place_and_round_trip() -> None:
    cube = _cube(3, 4, 5)
    cube[:, 1, 2] = np.nan  # a nodata pixel
    cube[0, 3, 3] = np.nan  # one bad band
    samples, layout = cube_to_samples(cube)

    finite = np.isfinite(samples).all(axis=1)
    assert (~finite).sum() == 2
    np.testing.assert_array_equal(
        samples_to_cube(finite, layout), np.isfinite(cube).all(axis=0)
    )
    # A mask flattened with the same layout lines up row for row.
    invalid, _ = cube_to_samples(~np.isfinite(cube))
    np.testing.assert_array_equal(invalid.any(axis=1), ~finite)
    np.testing.assert_array_equal(samples_to_cube(samples, layout), cube)


def test_nodata_4d_round_trip() -> None:
    stack = _cube(2, 3, 4, 5)
    stack[1, :, 0, 0] = np.nan
    samples, layout = cube_to_samples(stack, band_axis=1)
    assert np.isnan(samples[4 * 5]).all()
    np.testing.assert_array_equal(samples_to_cube(samples, layout), stack)


def test_k_output_columns_replace_the_band_axis() -> None:
    cube = _cube(5, 4, 3)
    samples, layout = cube_to_samples(cube, band_axis=-1)
    scores = samples[:, :2]
    out = samples_to_cube(scores, layout)
    assert out.shape == (5, 4, 2)
    np.testing.assert_array_equal(out, cube[..., :2])


def test_multiple_band_axes_and_explicit_sample_order() -> None:
    arr = _cube(2, 3, 4, 5)  # (t, c, h, w); features (t, c), samples (w, h)
    samples, layout = cube_to_samples(arr, band_axis=(0, 1), sample_axes=(3, 2))

    assert samples.shape == (20, 6)
    np.testing.assert_array_equal(samples[1], arr[:, :, 1, 0].reshape(-1))
    # Results come back in the input's (h, w) order, not the row order.
    assert samples_to_cube(samples[:, 0], layout).shape == (4, 5)
    np.testing.assert_array_equal(samples_to_cube(samples, layout)[:, 1, 0], samples[1])


def test_no_band_axis_gives_one_column_and_channel_first_output() -> None:
    image = _cube(4, 5)
    samples, layout = cube_to_samples(image, band_axis=())
    assert samples.shape == (20, 1)
    np.testing.assert_array_equal(samples[:, 0], image.reshape(-1))
    assert samples_to_cube(np.hstack([samples, samples]), layout).shape == (2, 4, 5)


def test_dtype_coercion_and_layout_rebuild() -> None:
    cube = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    samples, layout = cube_to_samples(cube, dtype=float)
    assert samples.dtype == np.float64
    assert sample_layout(cube.shape) == layout


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"band_axis": 3}, "out of range"),
        ({"band_axis": 0, "sample_axes": (0, 1, 2)}, "exactly once"),
        ({"band_axis": 0, "sample_axes": (1,)}, "exactly once"),
    ],
)
def test_invalid_axes_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cube_to_samples(_cube(2, 3, 4), **kwargs)


def test_samples_to_cube_rejects_wrong_row_count() -> None:
    _, layout = cube_to_samples(_cube(2, 3, 4))
    with pytest.raises(ValueError, match="n=12"):
        samples_to_cube(np.zeros((11, 2)), layout)
    with pytest.raises(ValueError, match="n=12"):
        samples_to_cube(np.zeros((12, 2, 1)), layout)


# --- callers ----------------------------------------------------------------


@pytest.mark.parametrize("axis", [-3, 0, -1])
def test_matched_filter_flattening_matches_legacy(axis: int) -> None:
    cube = _cube(4, 5, 6)
    samples, spatial = _cube_samples(cube, axis)
    expected, expected_spatial = _legacy(cube.astype(float), axis)
    np.testing.assert_array_equal(samples, expected)
    assert spatial == expected_spatial
    with pytest.raises(ValueError, match="at least two dimensions"):
        _cube_samples(np.ones(3), -1)


@pytest.mark.parametrize(("shape", "axis"), [((4, 5, 6), -3), ((2, 4, 5, 6), 1)])
def test_restore_pca_round_trips_with_nodata(shape: tuple[int, ...], axis: int) -> None:
    cube = _cube(*shape)
    nan_at = (0,) * len(shape)
    cube[nan_at] = np.nan
    state = fit_pca(cube, axis=axis)

    assert state["scores"].shape[0] == shape[axis]
    restored = inverse_pca(state["scores"], state)
    assert restored.shape == shape
    assert np.isnan(restored[nan_at])
    finite = np.isfinite(cube)
    np.testing.assert_allclose(restored[finite], cube[finite], atol=1e-10)


def test_learn_custom_sample_order_returns_input_grid_order() -> None:
    """1-D per-pixel outputs come back on the input (H, W) grid even when
    ``sample_axes`` lists the spatial axes out of order (they used to come
    back as ``(W, H)``, unlike 2-D outputs)."""
    scene = toy_geotensor(_cube(3, 4, 5), fill_value_default=None)
    est = gz.learn.GeoTensorEstimator(
        KMeans(n_clusters=3, n_init=2, random_state=0),
        mode="custom",
        sample_axes=("W", "H"),
        feature_axes=("C",),
    )
    est.fit(scene)
    rows_hw = np.moveaxis(np.asarray(scene), 0, -1).reshape(-1, 3)
    expected = est.estimator.predict(rows_hw).reshape(4, 5)

    out = np.asarray(est.predict(scene))

    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out, expected)

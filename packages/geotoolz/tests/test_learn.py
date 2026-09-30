"""Tests for `geotoolz.learn` scikit-learn adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor
from sklearn.cluster import KMeans as SKKMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import IsolationForest as SKIsolationForest
from sklearn.impute import KNNImputer as SKKNNImputer
from sklearn.preprocessing import StandardScaler

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    """NaN-filled toy carrier (learn ops treat NaN as missing data)."""
    return toy_geotensor(values, fill_value_default=np.nan)


def test_pixel_pca_fits_clean_pixels_and_restores_nan_sample() -> None:
    arr = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    arr[:, 0, 0] = np.nan
    scene = _gt(arr)

    out = gz.learn.PixelwisePCA(
        estimator=PCA(n_components=2),
        nan_fit="drop",
        nan_transform="propagate",
    )(scene)

    assert out.shape == (2, 4, 5)
    assert out.transform == scene.transform
    assert str(out.crs) == str(scene.crs)
    assert np.isnan(np.asarray(out)[:, 0, 0]).all()
    assert np.isfinite(np.asarray(out)[:, 1:, 1:]).all()


def test_pixel_time_mode_restores_output_feature_axis_position() -> None:
    arr = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    scene = _gt(arr)

    out = gz.learn.SklearnOp(
        estimator=StandardScaler(),
        mode="pixel_time",
        task="transform",
        nan_fit="error",
        nan_transform="error",
    )(scene)

    assert out.shape == scene.shape
    np.testing.assert_allclose(
        np.nanmean(np.asarray(out), axis=(0, 2, 3)),
        np.zeros(3),
        atol=1e-12,
    )


def test_impute_simple_strategy_fills_before_estimator() -> None:
    arr = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr[0, 0, 0] = np.nan
    scene = _gt(arr)

    out = gz.learn.SklearnOp(
        estimator=StandardScaler(),
        nan_fit="impute_simple",
        nan_transform="impute_simple",
        task="transform",
    )(scene)

    assert out.shape == scene.shape
    assert np.isfinite(np.asarray(out)).all()


def test_custom_axes_fit_predict_returns_sample_shape() -> None:
    arr = np.stack(
        [
            np.zeros((3, 4), dtype=float),
            np.ones((3, 4), dtype=float),
            np.full((3, 4), 10.0, dtype=float),
        ]
    )
    scene = _gt(arr)

    out = gz.learn.SklearnOp(
        estimator=SKKMeans(
            n_clusters=2,
            n_init=1,
            random_state=0,
        ),
        mode="custom",
        sample_axes=("C",),
        feature_axes=("H", "W"),
        task="fit_predict",
    )(scene)

    assert out.shape == (3,)
    assert set(np.asarray(out).tolist()) == {0, 1}


def test_state_roundtrip_saves_joblib_and_metadata(tmp_path: Path) -> None:
    scene = _gt(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    op = gz.learn.PixelwisePCA(estimator=PCA(n_components=2))
    expected = op(scene)
    state_path = tmp_path / "pca.joblib"

    op.save_state(state_path)
    loaded = gz.learn.PixelwisePCA(
        estimator=PCA(n_components=2),
        fit_mode="pre_fit",
        state_path=state_path,
    )
    actual = loaded(scene)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert state_path.exists()
    assert state_path.with_suffix(".joblib.meta.json").exists()
    json.dumps(loaded.get_config())


def test_fit_streaming_requires_partial_fit_estimator() -> None:
    with pytest.raises(TypeError, match="partial_fit"):
        gz.learn.SklearnOp(estimator=PCA(n_components=2), fit_mode="fit_streaming")


def test_top_level_exports_learn_symbols() -> None:
    assert gz.SklearnOp is gz.learn.SklearnOp
    assert gz.GeoTensorEstimator is gz.learn.GeoTensorEstimator


def test_drop_vs_propagate_produce_different_shapes() -> None:
    arr = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr[:, 0, 0] = np.nan
    scene = _gt(arr)

    propagated = gz.learn.SklearnOp(
        estimator=StandardScaler(),
        nan_fit="drop",
        nan_transform="propagate",
        task="transform",
    )(scene)
    dropped = gz.learn.SklearnOp(
        estimator=StandardScaler(),
        nan_fit="drop",
        nan_transform="drop",
        task="transform",
    )(scene)

    # ``propagate`` preserves the spatial layout (and refills NaN positions),
    # so the GeoTensor contract holds and the original shape is kept.
    assert propagated.shape == scene.shape
    assert np.isnan(np.asarray(propagated)[:, 0, 0]).all()
    # ``drop`` collapses the sample axis to the valid rows only, which
    # cannot be reshaped back into (H, W); a bare ndarray is returned.
    assert isinstance(dropped, np.ndarray)
    assert dropped.shape[0] == int((~np.isnan(arr).any(axis=0)).sum())


def test_propagate_raw_passes_nan_rows_to_estimator() -> None:
    arr = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr[:, 0, 0] = np.nan
    scene = _gt(arr)

    # KNNImputer is NaN-tolerant; ``propagate_raw`` should hand it the NaN
    # rows directly instead of stripping them.
    out = gz.learn.SklearnOp(
        estimator=SKKNNImputer(n_neighbors=2),
        nan_fit="propagate_raw",
        nan_transform="propagate_raw",
        task="transform",
    )(scene)

    assert out.shape == scene.shape
    assert np.isfinite(np.asarray(out)).all()


def test_impute_transform_refuses_to_refit_on_inference_batch() -> None:
    """Reject ``nan_transform=impute_*`` when no imputer has been fitted."""
    arr = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr[0, 0, 0] = np.nan
    scene = _gt(arr)

    est = gz.GeoTensorEstimator(
        StandardScaler(),
        nan_fit="drop",  # no imputer fitted
        nan_transform="impute_simple",
    )
    est.fit(scene)
    with pytest.raises(RuntimeError, match="fitted imputer"):
        est.transform(scene)


def test_pre_fit_without_state_path_raises() -> None:
    with pytest.raises(ValueError, match="state_path"):
        gz.learn.SklearnOp(estimator=PCA(n_components=2), fit_mode="pre_fit")


def test_get_config_records_resolved_task() -> None:
    op = gz.learn.SklearnOp(estimator=PCA(n_components=2), mode="pixel")  # task=None
    cfg = op.get_config()
    assert cfg["task"] is None
    assert cfg["estimator"]["resolved_task"] == "transform"
    assert cfg["estimator"]["params"]["n_components"] == 2


def test_ipca_streaming_fits_via_partial_fit() -> None:
    scene = _gt(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    op = gz.learn.PixelwiseIPCA(estimator=IncrementalPCA(n_components=2))
    out = op(scene)
    assert out.shape == (2, 4, 5)
    assert op._geo_estimator.is_fitted


def test_kmeans_predict_labels_pixels() -> None:
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 4, 5))
    arr[:, :, 2:] += 10  # two clusters in feature space
    scene = _gt(arr)
    op = gz.learn.PixelwiseKMeans(
        estimator=SKKMeans(n_clusters=2, n_init=1, random_state=0)
    )
    out = op(scene)
    assert out.shape == (4, 5)
    assert set(np.unique(np.asarray(out)).tolist()) == {0, 1}


def test_isolation_forest_decision_function_returns_scores() -> None:
    rng = np.random.default_rng(0)
    scene = _gt(rng.normal(size=(2, 4, 5)))
    op = gz.learn.PixelwiseIsolationForest(
        estimator=SKIsolationForest(n_estimators=10, random_state=0),
    )
    out = op(scene)
    assert out.shape == (4, 5)
    assert np.asarray(out).dtype.kind == "f"


@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(
            lambda: gz.learn.PixelwisePCA(estimator=PCA(n_components=2)),
            id="pca-transform",
        ),
        pytest.param(
            lambda: gz.learn.PixelwiseKMeans(
                estimator=SKKMeans(n_clusters=2, n_init=1, random_state=0)
            ),
            id="kmeans-predict",
        ),
        pytest.param(
            lambda: gz.learn.SklearnOp(estimator=StandardScaler(), task="transform"),
            id="scaler-transform",
        ),
    ],
)
def test_plain_ndarray_in_plain_ndarray_out(
    make_op: Callable[[], gz.learn.SklearnOp],
) -> None:
    """ndarray in -> ndarray out, values equal to the GeoTensor path."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(3, 4, 5))
    arr[:, :, 3:] += 5.0  # two clusters so KMeans labels are stable

    op = make_op()
    from_gt = op(_gt(arr.copy()))
    from_arr = op(arr.copy())  # same fitted state -> identical values

    assert type(from_arr) is np.ndarray
    np.testing.assert_allclose(np.asarray(from_arr), np.asarray(from_gt))

    fitted_on_array = make_op()(arr.copy())  # fitting from an ndarray works too
    assert type(fitted_on_array) is np.ndarray
    assert fitted_on_array.shape == from_arr.shape


# --- #112 metadata / nodata hygiene (#145-#148) -------------------------------


def test_2d_input_returns_channel_first_geotensor() -> None:
    """A single-band ``(H, W)`` raster maps to ``(k, H, W)``, not ``(H, W, k)``."""
    rng = np.random.default_rng(0)
    scene = _gt(rng.normal(size=(4, 5)))

    out = gz.learn.SklearnOp(estimator=PCA(n_components=1), task="transform")(scene)

    assert isinstance(out, GeoTensor)
    assert out.shape == (1, 4, 5)
    assert out.transform == scene.transform

    from_arr = gz.learn.SklearnOp(estimator=StandardScaler(), task="transform")(
        np.asarray(scene)
    )
    assert type(from_arr) is np.ndarray
    assert from_arr.shape == (1, 4, 5)


def test_out_band_names_applied() -> None:
    """``out_band_names`` names the output bands on fresh attrs (#148)."""
    rng = np.random.default_rng(0)
    scene = toy_geotensor(
        rng.normal(size=(3, 4, 5)),
        fill_value_default=np.nan,
        attrs={"band_names": ["b1", "b2", "b3"], "wavelengths": [1.0, 2.0, 3.0]},
    )

    out = gz.learn.PixelwisePCA(
        estimator=PCA(n_components=2), out_band_names=["pc1", "pc2"]
    )(scene)

    assert out.shape == (2, 4, 5)
    assert out.attrs["band_names"] == ["pc1", "pc2"]
    assert "wavelengths" not in out.attrs
    assert out.attrs is not scene.attrs
    assert scene.attrs["band_names"] == ["b1", "b2", "b3"]

    labels = gz.learn.PixelwiseKMeans(
        estimator=SKKMeans(n_clusters=2, n_init=1, random_state=0),
        out_band_names=["cluster"],
    )(scene)
    assert labels.attrs["band_names"] == ["cluster"]

    with pytest.raises(ValueError, match="band_names"):
        gz.learn.PixelwisePCA(
            estimator=PCA(n_components=2), out_band_names=["only_one"]
        )(scene)


def test_partial_fit_imputer_created_once() -> None:
    """Streaming ``partial_fit`` fits the imputer on the first batch only (#148)."""
    rng = np.random.default_rng(0)
    first = rng.normal(size=(2, 4, 5))
    first[0, 0, 0] = np.nan
    second = rng.normal(loc=100.0, size=(2, 4, 5))
    second[0, 1, 1] = np.nan

    est = gz.GeoTensorEstimator(
        IncrementalPCA(n_components=1),
        nan_fit="impute_simple",
        nan_transform="impute_simple",
    )
    est.partial_fit(_gt(first))
    imputer = est.imputer
    assert imputer is not None
    stats = imputer.statistics_.copy()
    est.partial_fit(_gt(second))

    assert est.imputer is imputer
    np.testing.assert_array_equal(imputer.statistics_, stats)

    # ``fit`` starts over and refits the imputer on its own input.
    est.fit(_gt(second))
    assert est.imputer is not imputer


def test_fill_pixels_are_excluded() -> None:
    """Numeric fill pixels never enter a fit and come back as nodata (#145, #146)."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 4, 5))
    scene = toy_geotensor(arr, fill_value_default=-9999, with_fill_pixels=True)
    fill = fill_pixel_mask(scene.shape)

    op = gz.learn.SklearnOp(
        estimator=StandardScaler(), task="transform", nan_fit="drop"
    )
    out = op(scene)

    scaler = op._geo_estimator.estimator
    np.testing.assert_allclose(scaler.mean_, arr[:, ~fill].mean(axis=1))
    assert op._geo_estimator.fit_n_samples == int((~fill).sum())
    assert np.isnan(out.fill_value_default)
    assert np.isnan(np.asarray(out)[:, fill]).all()
    assert np.isfinite(np.asarray(out)[:, ~fill]).all()


def test_kmeans_labels_are_integer_with_minus_one_fill() -> None:
    """Label maps stay integer; nodata pixels hold the ``-1`` label fill (#146)."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 4, 5))
    arr[:, :, 2:] += 10
    scene = toy_geotensor(arr, fill_value_default=-9999, with_fill_pixels=True)
    fill = fill_pixel_mask(scene.shape)

    out = gz.learn.PixelwiseKMeans(
        estimator=SKKMeans(n_clusters=2, n_init=1, random_state=0)
    )(scene)

    assert out.shape == (4, 5)
    assert np.asarray(out).dtype.kind == "i"
    assert out.fill_value_default == -1
    assert (np.asarray(out)[fill] == -1).all()
    assert set(np.asarray(out)[~fill].tolist()) == {0, 1}
    np.testing.assert_array_equal(np.asarray(out.validmask()), ~fill)

    clean = gz.learn.PixelwiseKMeans(
        estimator=SKKMeans(n_clusters=2, n_init=1, random_state=0), label_fill_value=-99
    )(toy_geotensor(arr, fill_value_default=-9999))
    assert np.asarray(clean).dtype.kind == "i"
    assert clean.fill_value_default == -99


def test_gmm_probabilities_use_nan_fill() -> None:
    """Probability maps mark nodata with NaN, never an ambiguous ``0`` (#146)."""
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 4, 5))
    arr[:, :, 2:] += 10
    scene = toy_geotensor(arr, fill_value_default=0, with_fill_pixels=True)
    fill = fill_pixel_mask(scene.shape)

    out = gz.learn.PixelwiseGMM(
        estimator=GaussianMixture(n_components=2, random_state=0)
    )(scene)

    assert out.shape == (2, 4, 5)
    assert np.isnan(out.fill_value_default)
    assert np.isnan(np.asarray(out)[:, fill]).all()
    np.testing.assert_allclose(np.asarray(out)[:, ~fill].sum(axis=0), 1.0)


def test_4d_time_stack() -> None:
    """``(T, C, H, W)`` stacks: per-frame fills, band axis at ``-3`` (#147)."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 3, 4, 5))
    arr[1, :, 2, 3] = -9999  # nodata in one frame only
    scene = toy_geotensor(arr, fill_value_default=-9999)

    per_frame = gz.learn.SklearnOp(
        estimator=PCA(n_components=2),
        mode="pixel_time",
        task="transform",
        out_band_names=["pc1", "pc2"],
    )(scene)
    assert isinstance(per_frame, GeoTensor)
    assert per_frame.shape == (2, 2, 4, 5)
    assert per_frame.attrs["band_names"] == ["pc1", "pc2"]
    assert np.isnan(np.asarray(per_frame)[1, :, 2, 3]).all()
    assert np.isfinite(np.asarray(per_frame)[0]).all()

    stacked = gz.learn.PixelwisePCA(estimator=PCA(n_components=2))(scene)
    assert stacked.shape == (2, 4, 5)
    assert np.isnan(np.asarray(stacked)[:, 2, 3]).all()


@pytest.mark.parametrize(
    ("mode", "shape", "match"),
    [
        ("pixel_time", (3, 4, 5), r"pixel_time.*4-D"),
        ("temporal", (4, 5), r"temporal.*T"),
        ("pixel", (5,), r"at least 2.*got a 1-D"),
    ],
)
def test_rank_errors_name_mode_and_shape(
    mode: str, shape: tuple[int, ...], match: str
) -> None:
    """Modes that need a time axis reject lower-rank input up front (#147)."""
    est = gz.GeoTensorEstimator(StandardScaler(), mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        est.fit(np.ones(shape))

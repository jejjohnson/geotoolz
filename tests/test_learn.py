"""Tests for `geotoolz.learn` scikit-learn adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from sklearn.cluster import KMeans as SKKMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import IsolationForest as SKIsolationForest
from sklearn.impute import KNNImputer as SKKNNImputer
from sklearn.preprocessing import StandardScaler

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=Affine(1, 0, 10, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=np.nan,
    )


def test_pixel_pca_fits_clean_pixels_and_restores_nan_sample() -> None:
    arr = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    arr[:, 0, 0] = np.nan
    scene = _gt(arr)

    out = gz.learn.PCA(
        PCA(n_components=2),
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
        StandardScaler(),
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
        StandardScaler(),
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
    op = gz.learn.PCA(PCA(n_components=2))
    expected = op(scene)
    state_path = tmp_path / "pca.joblib"

    op.save_state(state_path)
    loaded = gz.learn.PCA(
        PCA(n_components=2),
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
        gz.learn.SklearnOp(PCA(n_components=2), fit_mode="fit_streaming")


def test_top_level_exports_learn_symbols() -> None:
    assert gz.SklearnOp is gz.learn.SklearnOp
    assert gz.GeoTensorEstimator is gz.learn.GeoTensorEstimator


def test_drop_vs_propagate_produce_different_shapes() -> None:
    arr = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr[:, 0, 0] = np.nan
    scene = _gt(arr)

    propagated = gz.learn.SklearnOp(
        StandardScaler(),
        nan_fit="drop",
        nan_transform="propagate",
        task="transform",
    )(scene)
    dropped = gz.learn.SklearnOp(
        StandardScaler(),
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
        SKKNNImputer(n_neighbors=2),
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
        gz.learn.SklearnOp(PCA(n_components=2), fit_mode="pre_fit")


def test_get_config_records_resolved_task() -> None:
    op = gz.learn.SklearnOp(PCA(n_components=2), mode="pixel")  # task=None
    cfg = op.get_config()
    assert cfg["task"] is None
    assert cfg["resolved_task"] == "transform"


def test_ipca_streaming_fits_via_partial_fit() -> None:
    scene = _gt(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    op = gz.learn.IPCA(IncrementalPCA(n_components=2))
    out = op(scene)
    assert out.shape == (2, 4, 5)
    assert op._geo_estimator.is_fitted


def test_kmeans_predict_labels_pixels() -> None:
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 4, 5))
    arr[:, :, 2:] += 10  # two clusters in feature space
    scene = _gt(arr)
    op = gz.learn.KMeans(SKKMeans(n_clusters=2, n_init=1, random_state=0))
    out = op(scene)
    assert out.shape == (4, 5)
    assert set(np.unique(np.asarray(out)).tolist()) == {0, 1}


def test_isolation_forest_decision_function_returns_scores() -> None:
    rng = np.random.default_rng(0)
    scene = _gt(rng.normal(size=(2, 4, 5)))
    op = gz.learn.IsolationForest(
        SKIsolationForest(n_estimators=10, random_state=0),
    )
    out = op(scene)
    assert out.shape == (4, 5)
    assert np.asarray(out).dtype.kind == "f"

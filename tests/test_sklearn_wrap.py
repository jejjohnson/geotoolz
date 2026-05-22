"""Tests for the xarray ↔ sklearn bridge."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("sklearn")


from pipekit import Sequential
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xrtoolz.geo.operators import ValidateCoords
from xrtoolz.transforms import SklearnOp
from xrtoolz.utils import XarrayEstimator


def _sample_da() -> xr.DataArray:
    values = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.nan, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, np.nan, 6.0],
            [5.0, 6.0, 7.0],
        ]
    )
    return xr.DataArray(
        values,
        dims=("time", "feature"),
        coords={"time": np.arange(values.shape[0]), "feature": ["a", "b", "c"]},
        name="ssh",
        attrs={"units": "m"},
    )


def test_nan_policy_mask_standard_scaler_preserves_sample_axis() -> None:
    da = _sample_da()
    valid = ~np.isnan(da.values).any(axis=1)

    out = XarrayEstimator(
        StandardScaler(),
        sample_dim="time",
        nan_policy="mask",
    ).fit_transform(da)

    expected = StandardScaler().fit_transform(da.values[valid])
    assert out.dims == da.dims
    assert out.attrs == da.attrs
    np.testing.assert_array_equal(out["time"], da["time"])
    np.testing.assert_allclose(out.values[valid], expected)
    assert np.isnan(out.values[~valid]).all()


def test_nan_policy_mask_pca_inverse_round_trip_preserves_nan_rows() -> None:
    da = _sample_da()
    valid = ~np.isnan(da.values).any(axis=1)
    wrap = XarrayEstimator(
        PCA(n_components=2),
        sample_dim="time",
        nan_policy="mask",
    )

    scores = wrap.fit_transform(da)
    reconstructed = wrap.inverse_transform(scores)

    assert scores.dims == ("time", "component")
    assert reconstructed.dims == da.dims
    assert np.isnan(scores.values[~valid]).all()
    assert np.isnan(reconstructed.values[~valid]).all()
    np.testing.assert_array_equal(reconstructed["time"], da["time"])
    np.testing.assert_array_equal(reconstructed["feature"], da["feature"])


def test_nan_policy_mask_kmeans_predict_preserves_nan_rows() -> None:
    da = _sample_da()
    valid = ~np.isnan(da.values).any(axis=1)
    wrap = XarrayEstimator(
        KMeans(n_clusters=2, n_init="auto", random_state=0),
        sample_dim="time",
        nan_policy="mask",
    ).fit(da)

    labels = wrap.predict(da)

    expected = wrap.estimator_.predict(da.values[valid])
    assert labels.dims == ("time",)
    np.testing.assert_array_equal(labels.values[valid], expected)
    assert np.isnan(labels.values[~valid]).all()


def test_sklearn_accessor_matches_explicit_estimator() -> None:
    da = _sample_da().drop_sel(time=[1, 4])

    explicit = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(da)
    via_accessor = da.sklearn.fit_transform(StandardScaler(), sample_dim="time")

    np.testing.assert_array_equal(via_accessor.values, explicit.values)
    assert via_accessor.dims == explicit.dims
    assert via_accessor.attrs == explicit.attrs


def test_sklearn_op_composes_in_sequential_and_config_is_json_safe() -> None:
    da = xr.DataArray(
        np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(4),
            "lat": np.array([-10.0, 0.0, 10.0]),
            "lon": np.array([150.0, 151.0]),
        },
        name="ssh",
        attrs={"long_name": "sea surface height"},
    )
    ds = xr.Dataset({"ssh": da}, attrs={"title": "synthetic ocean"})
    scaled = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(da)
    fitted_pca = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(scaled)
    seq = Sequential(
        [
            ValidateCoords(),
            SklearnOp(
                StandardScaler(),
                variable="ssh",
                sample_dim="time",
                method="fit_transform",
            ),
            SklearnOp(
                fitted_pca.estimator_,
                variable="ssh",
                output_variable="pcs",
                sample_dim="time",
            ),
        ]
    )

    out = seq(ds)

    assert out.attrs == ds.attrs
    assert "ssh" in out
    assert "pcs" in out
    assert out["pcs"].dims == ("time", "component")
    np.testing.assert_array_equal(out["time"], ds["time"])
    np.testing.assert_array_equal(out["lat"], ds["lat"])
    np.testing.assert_array_equal(out["lon"], ds["lon"])
    assert out["ssh"].attrs == da.attrs
    json.dumps(seq.get_config())


def test_sklearn_op_config_handles_circular_estimator_params() -> None:
    class CircularEstimator:
        def get_params(self, deep: bool = False) -> dict[str, object]:
            params: dict[str, object] = {}
            params["self"] = params
            return params

    config = SklearnOp(CircularEstimator()).get_config()

    json.dumps(config)


def test_nan_policy_mask_on_dataset_drops_rows_across_all_variables() -> None:
    da_a = _sample_da()
    # Var b shares the same time axis but its own NaN pattern; the union of
    # NaN rows across both variables determines what gets dropped.
    values_b = np.array(
        [
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, np.nan],
            [16.0, 17.0],
            [18.0, 19.0],
            [20.0, 21.0],
        ]
    )
    da_b = xr.DataArray(
        values_b,
        dims=("time", "channel"),
        coords={"time": da_a["time"], "channel": ["x", "y"]},
        name="other",
    )
    ds = xr.Dataset({"ssh": da_a, "other": da_b})

    union_nan = np.isnan(da_a.values).any(axis=1) | np.isnan(da_b.values).any(axis=1)
    valid = ~union_nan

    out = XarrayEstimator(
        StandardScaler(),
        sample_dim="time",
        nan_policy="mask",
    ).fit_transform(ds)

    expected_input = np.column_stack([da_a.values[valid], da_b.values[valid]])
    expected = StandardScaler().fit_transform(expected_input)

    # Stacked column count (3 + 2) doesn't match either variable's feature
    # count alone, so the wrap returns the changed-feature-count layout.
    assert out.dims == ("time", "component")
    assert out.shape == (len(da_a["time"]), expected.shape[1])
    np.testing.assert_array_equal(out["time"], ds["time"])
    np.testing.assert_allclose(out.values[valid], expected)
    assert np.isnan(out.values[~valid]).all()


def test_nan_policy_mask_all_nan_input_raises() -> None:
    bad = xr.DataArray(
        np.full((4, 2), np.nan),
        dims=("time", "feature"),
        coords={"time": np.arange(4), "feature": ["a", "b"]},
    )
    wrap = XarrayEstimator(
        StandardScaler(),
        sample_dim="time",
        nan_policy="mask",
    )

    with pytest.raises(ValueError, match="removed all sample rows"):
        wrap.fit_transform(bad)


def test_sklearn_op_dataset_without_variable_or_output_variable_raises() -> None:
    ds = xr.Dataset({"ssh": _sample_da()})
    op = SklearnOp(StandardScaler(), sample_dim="time")

    with pytest.raises(ValueError, match="neither `variable` nor `output_variable`"):
        op(ds)


def test_sklearn_op_default_method_is_transform() -> None:
    # Default switched away from fit_transform — re-fitting on every Sequential
    # call is rarely what users want.
    assert SklearnOp(StandardScaler()).method == "transform"


def test_sklearn_op_accepts_fitted_xarray_estimator_for_inverse_transform() -> None:
    da = _sample_da().drop_sel(time=[1, 4])
    fitted = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)
    scores = fitted.transform(da)

    # Wrap a *fitted* XarrayEstimator so inverse_transform recovers the
    # original feature grid (training meta is preserved).
    op = SklearnOp(fitted, method="inverse_transform")

    recon = op(scores)

    assert recon.dims == da.dims
    np.testing.assert_array_equal(recon["feature"], da["feature"])


def test_sklearn_op_get_config_for_fitted_xarray_estimator_is_json_safe() -> None:
    fitted = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(
        _sample_da().drop_sel(time=[1, 4])
    )
    op = SklearnOp(fitted, variable="ssh", method="transform")

    config = op.get_config()

    assert config["estimator"] == "PCA"
    assert config["estimator_params"]["n_components"] == 2
    json.dumps(config)


def test_json_safe_handles_numpy_scalars() -> None:
    from xrtoolz.transforms._src.sklearn_op import _json_safe

    assert _json_safe(np.int64(7)) == 7
    assert _json_safe(np.float64(1.5)) == 1.5
    assert _json_safe(np.bool_(True)) is True


def test_sklearn_accessor_predict_matches_explicit_estimator() -> None:
    da = _sample_da().drop_sel(time=[1, 4])

    fitted = XarrayEstimator(
        KMeans(n_clusters=2, n_init="auto", random_state=0), sample_dim="time"
    ).fit(da)
    explicit = fitted.predict(da)

    via_accessor = da.sklearn.predict(fitted.estimator_, sample_dim="time")

    assert via_accessor.dims == explicit.dims
    np.testing.assert_array_equal(via_accessor.values, explicit.values)


def test_sklearn_accessor_inverse_transform_via_xarray_estimator() -> None:
    # The accessor's _wrap_fitted path doesn't carry training meta; users who
    # need inverse_transform should fit an XarrayEstimator explicitly. This
    # test pins that workflow.
    da = _sample_da().drop_sel(time=[1, 4])
    fitted = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)
    scores = fitted.transform(da)

    recon = fitted.inverse_transform(scores)

    assert recon.dims == da.dims
    np.testing.assert_array_equal(recon["feature"], da["feature"])


def test_sklearn_accessor_score_returns_float() -> None:
    da = _sample_da().drop_sel(time=[1, 4])
    fitted = XarrayEstimator(
        KMeans(n_clusters=2, n_init="auto", random_state=0), sample_dim="time"
    ).fit(da)

    score = da.sklearn.score(fitted.estimator_, sample_dim="time")

    assert isinstance(score, float)


def test_sklearn_accessor_dataset_fit_transform() -> None:
    ds = xr.Dataset({"ssh": _sample_da().drop_sel(time=[1, 4])})

    explicit = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(ds)
    via_accessor = ds.sklearn.fit_transform(StandardScaler(), sample_dim="time")

    np.testing.assert_array_equal(via_accessor.values, explicit.values)
    assert via_accessor.dims == explicit.dims


def test_nan_policy_mask_passes_through_integer_dtype() -> None:
    # Integer dtypes can't carry NaN, so masking should be a no-op rather
    # than crashing on np.isnan(int_array).
    da = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
        dims=("time", "feature"),
        coords={"time": np.arange(3), "feature": ["a", "b", "c"]},
    )

    out = XarrayEstimator(
        StandardScaler(),
        sample_dim="time",
        nan_policy="mask",
    ).fit_transform(da)

    expected = StandardScaler().fit_transform(da.values)
    np.testing.assert_allclose(out.values, expected)


def test_sklearn_accessor_passes_through_fitted_xarray_estimator() -> None:
    # When a fitted XarrayEstimator is handed to the accessor, the wrapper
    # must be reused as-is so its _fitted_meta_ recovers the original grid
    # on inverse_transform.
    da = _sample_da().drop_sel(time=[1, 4])
    fitted = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)

    scores = da.sklearn.transform(fitted)
    recon = scores.sklearn.inverse_transform(fitted)

    assert scores.dims == ("time", "component")
    # If _fitted_meta_ were lost, recon.dims would be (time, component); the
    # pass-through keeps the training grid:
    assert recon.dims == da.dims
    np.testing.assert_array_equal(recon["feature"], da["feature"])

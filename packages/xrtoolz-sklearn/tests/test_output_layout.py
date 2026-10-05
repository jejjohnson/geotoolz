"""Per-method output layouts and Dataset / target round-trips."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from xrsklearn import XarrayEstimator


def _cube(seed: int = 0, name: str = "ssh") -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(30, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(30),
            "lat": [10.0, 0.0, -10.0],
            "lon": np.arange(4.0),
        },
        name=name,
        attrs={"units": "m"},
    )


# ---------- transform: one-to-one vs reducing ----------------------------


def test_pca_with_as_many_components_as_features_stays_generic() -> None:
    scores = XarrayEstimator(PCA(n_components=12), sample_dim="time").fit_transform(
        _cube()
    )

    assert scores.dims == ("time", "component")
    np.testing.assert_array_equal(scores["component"], np.arange(12))


def test_kmeans_transform_with_as_many_clusters_as_features_stays_generic() -> None:
    wrap = XarrayEstimator(
        KMeans(n_clusters=12, n_init=1, random_state=0), sample_dim="time"
    ).fit(_cube())

    assert wrap.transform(_cube()).dims == ("time", "component")


def test_transformer_without_feature_names_falls_back_to_shape() -> None:
    out = XarrayEstimator(FunctionTransformer(np.abs), sample_dim="time").fit_transform(
        _cube()
    )

    assert out.dims == ("time", "lat", "lon")


def test_reduced_dataarray_output_keeps_name_and_attrs() -> None:
    scores = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit_transform(
        _cube()
    )

    assert scores.name == "ssh"
    assert scores.attrs == {"units": "m"}


# ---------- Dataset round-trips ------------------------------------------


def _dataset() -> xr.Dataset:
    ds = xr.Dataset(
        {"u": _cube(0, "u"), "v": _cube(1, "v").isel(lon=0, drop=True)},
        attrs={"title": "synthetic"},
    )
    ds["v"].attrs = {"units": "m/s"}
    return ds


def test_one_to_one_on_dataset_returns_dataset() -> None:
    ds = _dataset()

    out = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(ds)

    assert isinstance(out, xr.Dataset)
    assert out["u"].dims == ("time", "lat", "lon")
    assert out["v"].dims == ("time", "lat")
    assert out["v"].attrs == {"units": "m/s"}
    assert out.attrs == {"title": "synthetic"}
    np.testing.assert_allclose(out["v"].mean("time"), 0.0, atol=1e-12)


def test_dataset_scaler_inverse_round_trips() -> None:
    ds = _dataset()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time")

    scaled = wrap.fit_transform(ds)
    # Variable order of the input must not matter on the way back either.
    reordered = xr.Dataset({"v": scaled["v"], "u": scaled["u"]})
    recon = wrap.inverse_transform(reordered)

    xr.testing.assert_allclose(recon, ds)


def test_dataset_pca_inverse_returns_dataset_on_training_grid() -> None:
    ds = _dataset()
    wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")

    scores = wrap.fit_transform(ds)
    recon = wrap.inverse_transform(scores)

    assert scores.dims == ("time", "component")
    assert scores.name is None
    assert isinstance(recon, xr.Dataset)
    assert set(recon.data_vars) == {"u", "v"}
    assert recon["u"].dims == ds["u"].dims
    assert recon["v"].attrs == {"units": "m/s"}
    np.testing.assert_array_equal(recon["lat"], ds["lat"])


# ---------- predict: target space ----------------------------------------


def test_predict_uses_target_name_and_attrs() -> None:
    da = _cube()
    y = da.mean(("lat", "lon")).rename("sst_index")
    y.attrs = {"units": "K"}

    pred = XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y).predict(da)

    assert pred.dims == ("time",)
    assert pred.name == "sst_index"
    assert pred.attrs == {"units": "K"}


def test_multi_output_predict_rebuilds_target_grid() -> None:
    da = _cube()
    y = da.mean("lon").rename("zonal_mean")  # (time, lat)

    pred = XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y).predict(da)

    assert pred.dims == ("time", "lat")
    np.testing.assert_array_equal(pred["lat"], da["lat"])
    xr.testing.assert_allclose(pred, y, atol=1e-8)


def test_dataset_target_predicts_dataset() -> None:
    da = _cube()
    y = xr.Dataset({"a": da.mean(("lat", "lon")), "b": da.mean("lon")})

    pred = XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y).predict(da)

    assert isinstance(pred, xr.Dataset)
    assert pred["a"].dims == ("time",)
    assert pred["b"].dims == ("time", "lat")


def test_unsupervised_predict_carries_no_input_attrs() -> None:
    wrap = XarrayEstimator(
        KMeans(n_clusters=2, n_init=1, random_state=0), sample_dim="time"
    ).fit(_cube())

    labels = wrap.predict(_cube())

    assert labels.dims == ("time",)
    assert labels.name is None
    assert labels.attrs == {}


def test_numpy_target_predict_is_generic() -> None:
    da = _cube()
    y = da.mean(("lat", "lon")).values

    pred = XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y).predict(da)

    assert pred.dims == ("time",)
    assert pred.attrs == {}


# ---------- predict_proba -------------------------------------------------


def test_predict_proba_is_labeled_by_classes() -> None:
    da = _cube()
    y = xr.where(da.mean(("lat", "lon")) > 0, "warm", "cold")

    proba = (
        XarrayEstimator(LogisticRegression(), sample_dim="time")
        .fit(da, y)
        .predict_proba(da)
    )

    assert proba.dims == ("time", "class")
    assert list(proba["class"].values) == ["cold", "warm"]
    np.testing.assert_allclose(proba.sum("class"), 1.0)


def test_multi_output_predict_proba_returns_list() -> None:
    da = _cube()
    y = np.stack(
        [(da.mean(("lat", "lon")) > 0).values, (da.isel(lat=0, lon=0) > 0).values],
        axis=1,
    )

    wrap = XarrayEstimator(
        RandomForestClassifier(n_estimators=5, random_state=0), sample_dim="time"
    ).fit(da, y)
    proba = wrap.predict_proba(da)

    assert isinstance(proba, list)
    assert len(proba) == 2
    assert all(p.dims == ("time", "class") for p in proba)


@pytest.mark.parametrize("method", ["transform", "predict"])
def test_numpy_inputs_still_pass_through(method: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    est = StandardScaler() if method == "transform" else LinearRegression()
    wrap = XarrayEstimator(est).fit(x, x[:, 0])

    assert isinstance(getattr(wrap, method)(x), np.ndarray)


# ---------- review follow-ups (#330) --------------------------------------


def test_pca_round_trip_when_feature_dim_is_named_component() -> None:
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(30, 6)),
        dims=("time", "component"),
        coords={"time": np.arange(30), "component": np.arange(10, 16)},
    )
    wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")

    recon = wrap.inverse_transform(wrap.fit_transform(da))

    assert recon.sizes == da.sizes
    np.testing.assert_array_equal(recon["component"].values, da["component"].values)


def test_predict_proba_rejects_sample_dim_named_class() -> None:
    rng = np.random.default_rng(0)
    x = xr.DataArray(rng.normal(size=(20, 3)), dims=("class", "feature"))
    y = xr.DataArray(np.arange(20) % 2, dims=("class",))
    wrap = XarrayEstimator(LogisticRegression(), sample_dim="class").fit(x, y)

    with pytest.raises(ValueError, match="clashes with sample_dim='class'"):
        wrap.predict_proba(x)


def test_unsupervised_predict_ignores_the_target_layout() -> None:
    rng = np.random.default_rng(0)
    x = xr.DataArray(rng.normal(size=(20, 3)), dims=("time", "feature"))
    y = xr.DataArray(
        rng.normal(size=20), dims=("time",), name="sst", attrs={"units": "K"}
    )
    wrap = XarrayEstimator(KMeans(n_clusters=2, n_init=1, random_state=0)).fit(x, y)

    labels = wrap.predict(x)

    assert labels.name != "sst"
    assert labels.attrs == {}
    assert wrap.target_layout_ is None

"""Tests for :class:`xr_toolz.utils.XarrayEstimator`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from xr_toolz.utils import XarrayEstimator


# ---------- fixtures -------------------------------------------------------


@pytest.fixture
def da_3d() -> xr.DataArray:
    """A small ``(time, lat, lon)`` array with reproducible noise."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2020-01-01", periods=24, freq="1ME")
    lat = np.linspace(-10.0, 10.0, 4)
    lon = np.linspace(-20.0, 20.0, 5)
    tt = np.arange(len(time))[:, None, None]
    base = np.sin(2 * np.pi * tt / 12.0) * np.ones((1, len(lat), len(lon)))
    data = base + 0.1 * rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="signal",
        attrs={"units": "m", "long_name": "synthetic signal"},
    )


# ---------- same-feature-count transforms ----------------------------------


def test_standard_scaler_round_trip_preserves_grid(da_3d):
    """A scaler keeps the feature count → output unstacks to original grid."""
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time")
    out = wrap.fit_transform(da_3d)
    assert isinstance(out, xr.DataArray)
    assert set(out.dims) == {"time", "lat", "lon"}
    np.testing.assert_array_equal(out["lat"].values, da_3d["lat"].values)
    # Mean ~0, std ~1 along the sample dim.
    np.testing.assert_allclose(out.mean("time").values, 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std("time").values, 1.0, atol=1e-10)


def test_standard_scaler_inverse_recovers_input(da_3d):
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time")
    scaled = wrap.fit_transform(da_3d)
    recon = wrap.inverse_transform(scaled)
    np.testing.assert_allclose(recon.values, da_3d.values, atol=1e-10)


def test_attrs_and_name_preserved(da_3d):
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time")
    out = wrap.fit_transform(da_3d)
    assert out.attrs == {"units": "m", "long_name": "synthetic signal"}
    assert out.name == "signal"


# ---------- changed-feature-count transforms (PCA) ------------------------


def test_pca_reduces_features_to_components(da_3d):
    wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")
    scores = wrap.fit_transform(da_3d)
    assert scores.dims == ("time", "component")
    assert scores.sizes["component"] == 3
    assert scores.sizes["time"] == da_3d.sizes["time"]


def test_pca_inverse_transform_returns_grid(da_3d):
    wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")
    scores = wrap.fit_transform(da_3d)
    recon = wrap.inverse_transform(scores)
    assert set(recon.dims) == {"time", "lat", "lon"}
    assert recon.shape == da_3d.shape


def test_pca_inverse_transform_uses_current_sample_coords(da_3d):
    """Regression for PR-15 review: inverse-transforming scores from a
    *different* sample period must stamp the new period's sample coords
    onto the result, not the training period's. The training feature
    grid (lat/lon) is reused; the sample axis comes from the input."""
    est = XarrayEstimator(PCA(n_components=2), sample_dim="time")
    est.fit(da_3d)
    # Build a fresh score array on a new time axis (different timestamps,
    # different length) and inverse-transform it.
    new_time = pd.date_range("2030-06-01", periods=5, freq="1D")
    rng = np.random.default_rng(7)
    fake_scores = xr.DataArray(
        rng.standard_normal((5, 2)),
        dims=("time", "component"),
        coords={"time": new_time, "component": np.arange(2)},
    )
    recon = est.inverse_transform(fake_scores)
    # New sample coords carried through.
    np.testing.assert_array_equal(
        recon["time"].values.astype("datetime64[ns]"),
        new_time.values.astype("datetime64[ns]"),
    )
    assert recon.sizes["time"] == 5
    # Feature grid (lat/lon) recovered from the training meta.
    assert set(recon.dims) == {"time", "lat", "lon"}
    np.testing.assert_array_equal(recon["lat"].values, da_3d["lat"].values)


def test_pca_attribute_passthrough(da_3d):
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")
    wrap.fit(da_3d)
    # ``components_`` should be proxied to the fitted PCA estimator.
    assert wrap.components_.shape == (2, da_3d.sizes["lat"] * da_3d.sizes["lon"])
    assert wrap.explained_variance_ratio_.shape == (2,)


def test_unfitted_estimator_attribute_access_raises():
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")
    with pytest.raises(AttributeError, match="has not been fitted"):
        _ = wrap.components_


def test_transform_before_fit_raises(da_3d):
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")
    with pytest.raises(RuntimeError, match="has not been fitted"):
        wrap.transform(da_3d)


# ---------- clustering (predict path) -------------------------------------


def test_kmeans_predict_returns_1d_labels(da_3d):
    wrap = XarrayEstimator(
        KMeans(n_clusters=3, n_init=4, random_state=0), sample_dim="time"
    )
    wrap.fit(da_3d)
    labels = wrap.predict(da_3d)
    assert labels.dims == ("time",)
    assert labels.sizes["time"] == da_3d.sizes["time"]
    assert set(np.unique(labels.values)).issubset({0, 1, 2})


# ---------- regression (y handling) ---------------------------------------


def test_linear_regression_with_xarray_y(da_3d):
    """Use the array's mean over (lat, lon) as a synthetic 1-D target."""
    y = da_3d.mean(dim=("lat", "lon"))
    wrap = XarrayEstimator(LinearRegression(), sample_dim="time")
    wrap.fit(da_3d, y)
    pred = wrap.predict(da_3d)
    assert pred.dims == ("time",)
    # On training data, R² should be very close to 1.
    assert wrap.score(da_3d, y) > 0.99


# ---------- Dataset input + numpy passthrough -----------------------------


def test_dataset_input_concatenates_variables():
    rng = np.random.default_rng(1)
    time = pd.date_range("2020-01-01", periods=20, freq="1D")
    lat = np.linspace(0.0, 1.0, 3)
    a = rng.standard_normal((20, 3))
    b = rng.standard_normal((20, 3))
    ds = xr.Dataset(
        {"a": (("time", "lat"), a), "b": (("time", "lat"), b)},
        coords={"time": time, "lat": lat},
    )
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")
    scores = wrap.fit_transform(ds)
    # 6 features (3 × 2 vars) reduced to 2 components.
    assert scores.dims == ("time", "component")
    assert scores.sizes["component"] == 2
    # PCA components_ should have width 6 (concat of a + b features).
    assert wrap.components_.shape == (2, 6)


def test_numpy_input_passes_through():
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((30, 5))
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time")
    out = wrap.fit_transform(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == arr.shape


def test_numpy_x_with_xarray_y_raises_clear_error():
    """When ``x`` is numpy there is no sample dim to align ``y`` against;
    accepting an xarray ``y`` would either silently misalign or surface a
    confusing ``sample_dim=''`` error from the marshaller."""
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((20, 4))
    y_xr = xr.DataArray(rng.standard_normal(20), dims=("time",))
    wrap = XarrayEstimator(LinearRegression())
    with pytest.raises(TypeError, match="NumPy"):
        wrap.fit(x_np, y_xr)


# ---------- nan policy ----------------------------------------------------


def test_nan_policy_raise_errors_on_nan(da_3d):
    da = da_3d.copy()
    da.values[0, 0, 0] = np.nan
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time", nan_policy="raise")
    with pytest.raises(ValueError, match="NaN"):
        wrap.fit(da)


def test_nan_policy_propagate_does_not_screen(da_3d):
    """``"propagate"`` must not pre-screen the input — fit succeeds even
    when a NaN is present (StandardScaler ignores NaN by default in
    modern sklearn). The contrast with ``"raise"`` is what makes this
    a meaningful test."""
    da = da_3d.copy()
    da.values[0, 0, 0] = np.nan
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time")
    wrap.fit(da)  # must not raise
    assert hasattr(wrap, "scale_")


# ---------- sample_dim resolution -----------------------------------------


def test_sample_dim_defaults_to_first_dim(da_3d):
    wrap = XarrayEstimator(StandardScaler())  # no sample_dim
    out = wrap.fit_transform(da_3d)
    np.testing.assert_allclose(out.mean("time").values, 0.0, atol=1e-10)


def test_unknown_sample_dim_raises(da_3d):
    wrap = XarrayEstimator(StandardScaler(), sample_dim="not-a-dim")
    with pytest.raises(ValueError, match="sample_dim"):
        wrap.fit(da_3d)


# ---------- repr ----------------------------------------------------------


def test_repr_does_not_crash():
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")
    assert "XarrayEstimator" in repr(wrap)
    assert "PCA" in repr(wrap)
